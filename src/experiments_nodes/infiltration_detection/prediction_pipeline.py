import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from src.experiments_nodes.infiltration_detection import gcn_supervised_baseline
from src.utils import gcn_classification_utils, graph_level_classification_utils
from tqdm import tqdm

log = logging.getLogger(__name__)


def infiltration_classification_pipeline(
    cfg: DictConfig, output_folder: Path, iteration: int, gts_df: pd.DataFrame
) -> None:
    """Runs the infiltration classification experiments for one iteration

    Linear classifiers are evaluated on raw_features/dgi embeddings/message passing embeddings
    Graph level representations are either Average readout or Bags-of-features
    A supervised GCN is also evaluated

    Multiple training dataset sizes are used to demonstrated the ability to learn from small datasets

    Args:
        cfg (DictConfig): hydra config
        output_folder (Path): Folder to store results
        iteration (int)
        gts_df (pd.DataFrame): Groundtruth dataframe. Should have a patient column and a infiltration column
    """
    with open(
        output_folder / f"splits/test_patients_{iteration}.pkl", "rb"
    ) as test_split_file:
        test_patients = pickle.load(test_split_file)

    log.info("Iteration %i", iteration)

    results = []
    iteration_output_folder = Path(f"global_representations_{iteration}")

    for frac in tqdm(np.arange(0.1, 1.1, 0.2)):
        log.info("Frac %f", frac)
        training_patients_ids = (
            (gts_df[~gts_df["patient"].isin(test_patients)])
            .groupby("infiltration")
            .apply(lambda group: list(group.index[: int(group.shape[0] * frac)]))
            .sum()
        )
        training_patients = gts_df.loc[training_patients_ids]["patient"].values
        if not all(
            (gts_df.loc[training_patients_ids]["infiltration"] == infiltration).sum()
            > 0
            for infiltration in gts_df.infiltration.unique()
        ):
            continue
        linear_results = (
            graph_level_classification_utils.patient_level_classification_pipeline(
                cfg,
                gts_df,
                training_patients,
                test_patients,
                iteration_output_folder / f"{frac=:.2f}",
                "infiltration",
            )
        )

        log.info("Supervised GCN")
        gcn_results = gcn_classification_utils.graph_level_multiclass_gcn(
            gcn_supervised_baseline.InfiltrationGraphDataset,
            training_patients,
            test_patients,
            cfg,
            gts_df,
            list(gts_df["infiltration"].unique()),
        )
        gcn_results["exp"] = "supervised gcn"

        ratio_results = linear_results.append(gcn_results, ignore_index=True)
        ratio_results["training_ratio"] = frac
        results += [ratio_results]
    results_df = pd.concat(results, ignore_index=True)
    results_df["iteration"] = iteration
    results_df["probas"] = results_df["probas"].apply(list)
    results_df.to_parquet(
        output_folder / "infiltration_classification.parquet",
        partition_cols=["iteration", "exp", "training_ratio"],
    )
