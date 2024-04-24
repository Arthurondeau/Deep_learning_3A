import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from torchmetrics import Metric, MetricCollection, Accuracy, Precision, Recall, AUROC, AveragePrecision, MeanMetric, MaxMetric
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

class Model(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimization_params: Dict,
        n_classes: int,
        adversarial_training: bool = False
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(optimization_params["lr"])
        self.step_size = int(optimization_params["lr_scheduler_step_size"])
        self.gamma = float(optimization_params["lr_scheduler_gamma"])
        self.weight_decay = float(optimization_params["l2_coeff"])
        self.optimizer, self.scheduler = self.configure_optimizers()
        self.logger: MLFlowLogger
        self.trainer: pl.Trainer
        self.adversarial_training = adversarial_training
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if n_classes > 2:
            metrics_params = {"task": "multiclass", "num_classes": n_classes}

            metrics = MetricCollection([
                Accuracy(**metrics_params),
                Recall(**metrics_params),
                Precision(**metrics_params),
                AUROC(**metrics_params),
                AveragePrecision(**metrics_params),
            ])

            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")
            self.train_loss = MeanMetric()
            self.train_acc_best = MaxMetric()
            self.val_loss = MeanMetric()
            self.val_acc_best = MaxMetric()
            self.test_loss = MeanMetric()
            self.test_acc_best = MaxMetric()
        else:
            raise ValueError("Number of classes must be greater than 2.")
        
        # Define loss functions
        self.criterion_sleep_stage = torch.nn.CrossEntropyLoss()
        self.criterion_subject = torch.nn.CrossEntropyLoss() 

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        # Forward pass
        sleep_stage_output, subject_output = self.forward(x)

        # Compute losses
        loss_sleep_stage = self.criterion_sleep_stage(sleep_stage_output, y[:,0].long())
        loss_subject = self.criterion_subject(subject_output, y[:,1].long())
        if self.adversarial_attack:
            # Total loss
            loss = loss_sleep_stage - loss_subject
        else :
            loss = loss_sleep_stage
        self.train_loss(loss)
        self.train_metrics(sleep_stage_output, y[:,0])
        self.train_step_outputs.append(loss)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def adversarial_attack(self, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.1, alpha: float = 0.01, num_iter: int = 10) -> torch.Tensor:
        perturbed_x = x.clone().detach()
        for _ in range(num_iter):
            perturbed_x.requires_grad = True
            logits = self.forward(perturbed_x)
            loss = F.cross_entropy(logits, y.long())
            self.model.zero_grad()
            loss.backward()
            with torch.no_grad():
                perturbed_x += alpha * torch.sign(perturbed_x.grad)
                perturbed_x = torch.max(torch.min(perturbed_x, x + epsilon), x - epsilon)
                # Clamp signal data within its valid range
                # Adjust this according to the range of your signal data
                perturbed_x = torch.clamp(perturbed_x, x.min(), x.max())
        return perturbed_x


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        # Forward pass
        sleep_stage_output, subject_output = self.forward(x)
        # Compute losses
        loss_sleep_stage = self.criterion_sleep_stage(sleep_stage_output, y[:,0].long())
        loss_subject = self.criterion_subject(subject_output, y[:,1].long())

        if self.adversarial_attack:
            # Total loss
            loss = loss_sleep_stage - loss_subject
        else :
            loss = loss_sleep_stage
        self.val_loss(loss)
        self.val_metrics(sleep_stage_output, y[:,0])
        self.validation_step_outputs.append(loss)
        self.log("val_loss:", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        # Forward pass
        sleep_stage_output, subject_output = self.forward(x)

        # Compute losses
        loss_sleep_stage = self.criterion_sleep_stage(sleep_stage_output, y[:,0].long())
        loss_subject = self.criterion_subject(subject_output, y[:,1].long())

        if self.adversarial_attack:
            # Total loss
            loss = loss_sleep_stage - loss_subject
        else :
            loss = loss_sleep_stage
        self.test_loss(loss)
        self.test_metrics(sleep_stage_output, y[:,0])
        self.test_step_outputs.append(loss)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, torch.optim.lr_scheduler._LRScheduler]]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)}
        return [optimizer], [scheduler]

    def on_train_epoch_end(self) -> None:
        epoch_average = torch.stack(self.train_step_outputs).mean()
        acc = self.train_metrics.compute()["train_MulticlassAccuracy"]
        self.train_acc_best(acc)
        self.log("train_epoch_average with adv" + str(self.adversarial_training), epoch_average)
        self.log("train_acc_best", self.train_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self) -> None:
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        acc = self.val_metrics.compute()["val_MulticlassAccuracy"]
        self.val_acc_best(acc)
        self.log("validation_epoch_average with adv" + str(self.adversarial_training), epoch_average)
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self) -> None:
        epoch_average = torch.stack(self.test_step_outputs).mean()
        acc = self.test_metrics.compute()["val_MulticlassAccuracy"]
        self.test_acc_best(acc)
        self.log("test_epoch_average with adv" + str(self.adversarial_training), epoch_average)
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_file, batch_size,num_workers):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        with open(self.data_file, 'rb') as f:
            xtrain, xvalid, ytrain, yvalid = pickle.load(f)

        split_index = len(xvalid) // 2
        xtest, ytest = xvalid[:split_index], yvalid[:split_index]
        xvalid, yvalid = xvalid[split_index:], yvalid[split_index:]

        self.train_dataset = TensorDataset(torch.tensor(xtrain), torch.tensor(ytrain))
        self.val_dataset = TensorDataset(torch.tensor(xvalid), torch.tensor(yvalid))
        self.test_dataset = TensorDataset(torch.tensor(xtest), torch.tensor(ytest))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=self.num_workers)
