import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from torchmetrics import Metric, MetricCollection, Accuracy, Precision, Recall, AUROC, AveragePrecision, MeanMetric, MaxMetric
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import numpy as np


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
        self.train_outputs = {"train_loss": [], "train_acc": []}
        self.validation_outputs = {"val_loss": [], "val_acc": []}
        self.test_outputs = {"test_loss": [], "test_acc": []}

        if n_classes > 2:
            metrics_params = {"task": "multiclass", "num_classes": n_classes}

            metrics = MetricCollection([
                Accuracy(**metrics_params),
                AUROC(**metrics_params),
                AveragePrecision(**metrics_params),
            ])

            train_metrics_collect = MetricCollection([
                Accuracy(**metrics_params)
            ])
        else:
            raise ValueError("Number of classes must be greater than 2.")
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix='test_')
        self.train_loss = MeanMetric()
        self.train_acc_best = MaxMetric()
        self.val_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        self.test_acc_best = MaxMetric()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        if self.adversarial_training : 
            # Generate adversarial examples using PGD
            with torch.enable_grad():
                perturbed_x = self.adversarial_attack(x, y)
            # Compute logits for adversarial examples
            logits = self.forward(perturbed_x)
        else : 
            y = y[:,0] #take only sleep labels
            logits = self.forward(x)
        # Compute loss using original labels
        loss = F.cross_entropy(logits, y.long())
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(y),
            sync_dist=True,
        )
        self.log("train_loss", loss.detach(),prog_bar=True)
        self.train_outputs["train_loss"].append(loss.detach().clone())
        self.train_metrics(logits.detach(), y)
        acc = self.train_metrics.compute()["train_MulticlassAccuracy"]
        auc = self.train_metrics.compute()["train_MulticlassAUROC"]
        self.log(
            "train_acc",
            acc.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(y),
            sync_dist=True,
        )
        self.log(
            "train_auc",
            auc.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(y),
            sync_dist=True,
        )

        return {"loss": loss}
    
    def adversarial_attack(self, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.1, alpha: float = 0.01, num_iter: int = 10) -> torch.Tensor:
        perturbed_x = x.clone().detach()
        for _ in range(num_iter):
            perturbed_x.requires_grad = True
            logits = self.forward(perturbed_x)
            print('logits shape',logits.size())
            print('y shape adv',y.size())
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
        if self.adversarial_training : 
            # Generate adversarial examples using PGD
            with torch.enable_grad():
                perturbed_x = self.adversarial_attack(x, y)
            # Compute logits for adversarial examples
            logits = self.forward(perturbed_x)
        else : 
            y = y[:,0] #take only sleep labels
            logits = self.forward(x)
        # Compute loss using original labels
        print('y size',y.size())
        loss = F.cross_entropy(logits, y.long())
        self.log("val_loss", loss.detach(), on_step=True, on_epoch=False,prog_bar=True)
        self.validation_outputs["val_loss"].append(loss.detach().clone())
        self.val_metrics(logits.detach(), y)
        acc = self.val_metrics.compute()["val_MulticlassAccuracy"]
        auc = self.val_metrics.compute()["val_MulticlassAUROC"]
        self.log("val_acc", acc.detach(), on_step=True, on_epoch=False,prog_bar=True)
        self.log("val_auc", auc.detach(), on_step=True, on_epoch=False,prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        if self.adversarial_training : 
            # Generate adversarial examples using PGD
            perturbed_x = self.adversarial_attack(x, y)
            # Compute logits for adversarial examples
            logits = self.forward(perturbed_x)
        else : 
            y = y[:,0] #take only sleep labels
            logits = self.forward(x)
        # Compute loss using original labels
        loss = F.cross_entropy(logits, y.long())
        self.test_outputs["test_loss"].append(loss.detach().clone())
        self.test_metrics(logits.detach(), y)
        acc = self.test_metrics.compute()["test_MulticlassAccuracy"]
        auc = self.test_metrics.compute()["test_MulticlassAUROC"]
        self.log("test_acc", acc.detach(), on_step=True, on_epoch=False,prog_bar=True)
        self.log("test_auc", auc.detach(), on_step=True, on_epoch=False,prog_bar=True)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, torch.optim.lr_scheduler._LRScheduler]]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)}
        return [optimizer], [scheduler]


    def on_train_epoch_end(self) -> None:
        acc = self.train_metrics.compute()["train_MulticlassAccuracy"]
        self.train_acc_best(acc)
        self.log("train_acc_best", self.train_acc_best.compute(), prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_metrics.compute()["val_MulticlassAccuracy"]
        self.val_acc_best(acc)
        self.log("val_acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        acc = self.test_metrics.compute()["test_MulticlassAccuracy"]
        self.test_acc_best(acc)
        self.log("test_acc_best", self.test_acc_best.compute(), prog_bar=True, sync_dist=True)





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
        print('shape ytrain',np.shape(ytrain))
        split_index = len(xvalid) // 2
        xtest, ytest = xvalid[:split_index], yvalid[:split_index]
        xvalid, yvalid = xvalid[split_index:], yvalid[split_index:]
        self.train_dataset = TensorDataset(torch.tensor(xtrain), torch.tensor(ytrain))
        self.val_dataset = TensorDataset(torch.tensor(xvalid), torch.tensor(yvalid))
        self.test_dataset = TensorDataset(torch.tensor(xtest), torch.tensor(ytest))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers,pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=self.num_workers,pin_memory=False)
