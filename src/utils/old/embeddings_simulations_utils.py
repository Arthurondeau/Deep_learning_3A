"""embeddings simulations utils"""
from pathlib import Path
from typing import Any, Dict, List

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics


def cluster_enrichment_distances(
    enrichments: pd.DataFrame, metric: Dict
) -> pd.DataFrame:
    """compute pairwise distance on all columns for each cluster_indication groups
    Columns should be clusters, rows are patients and the values are enrichment
    df[patient, cluster]=% cells of the cluster belonging to patient

    Score computed for each clustering then aveged to get one score for each indication"""
    enrichments = enrichments[
        [
            col
            for col in enrichments.columns
            if col in [str(el) for el in range(len(enrichments.columns))]
        ]
        + ["clustering", "indication"]
    ]
    scores = enrichments.groupby(["clustering", "indication"]).apply(
        lambda x: compute_distances(
            x.drop(columns=["clustering", "indication"]).fillna(0), metric
        )
    )
    scores = scores.reset_index().rename(columns={0: "distance"})

    scores["distance"] = scores["distance"].fillna(0)
    return scores


def compute_distances(enrichments: pd.DataFrame, metric: Dict) -> float:
    """compute pairwise euclidean distance btwn lines of enrichments df
    returns the mean of all non-zero scores (no self distance)
    Args:
        enrichments (pd.DataFrame): numeric values with no NaN

    """
    if len(enrichments) < 2:
        return np.nan

    cross_enrichments = pd.merge(enrichments, enrichments, how="cross")
    partial_metric = hydra.utils.call(metric)

    scores = cross_enrichments.apply(
        lambda row: partial_metric(
            row.filter(regex="_x"),
            row.filter(regex="_y"),
        ),
        1,
    ).values
    return scores[np.nonzero(scores)].mean()


def plot_scores(scores: pd.DataFrame, plots_path: Path, **filters: Any) -> None:
    """plot the cluster enrichment scores with indication and perturbation hue"""

    fig = sns.lineplot(
        data=scores,
        x="ratio",
        y="distance",
        hue=scores[["perturbation", "indication"]].apply(tuple, axis=1),
    )
    fig.set(title="Cluster enrichment distance by indication")
    figname = (
        plots_path
        / "cluster_enrichment_simulations"
        / "/".join(f"{key}={value}" for key, value in filters.items())
    )
    figname.mkdir(parents=True, exist_ok=True)
    fig.figure.savefig(figname / "cluster_enrichment_scores.png")
    plt.close("all")


def clustering_randscore(
    perturbation: Path, reference_perturbation: Path, indications: Dict, **filters: Any
) -> pd.DataFrame:
    """compute randscore for each indication independently
    indications dict is key:indication name, value list of patients ids

    filters :
        clustering=str(clustering),
        max_edge_size=cfg.graph_creation.graphs.max_edge_size,
        prefix=cfg.analysis.indication_level_prefix,"""
    scores = []
    for indication in indications:
        filters["indication"] = indication
        indication_score = clustering_randscore_indication(
            perturbation=perturbation,
            reference_perturbation=reference_perturbation,
            **filters,
        )
        score = [
            {"indication": indication, **filters, "randscore": score}
            for score in indication_score
        ]

        scores += score
    return pd.DataFrame(scores)


def clustering_randscore_indication(
    perturbation: Path,
    reference_perturbation: Path,
    **filters: Any,
) -> List[float]:
    """read clustering dfs for each patient in the indication and compute randscore
    btwn clustered cells after perturbation and the corresponding cells in the original graphs
    """

    # Due to parquet confusion in type for str patients that can be
    # interpreted as int, patient name can't be used as a filter
    # so open everything once and do pd.query(patient==@patient) downstream

    perturbed_patient_cells = pd.read_parquet(
        perturbation / "cells_with_cluster",
        filters=[(key, "=", value) for key, value in filters.items()],
    )
    perturbed_patient_cells["patient"] = perturbed_patient_cells["patient"].astype(str)
    original_patient_cells = pd.read_parquet(
        reference_perturbation / "cells_with_cluster",
        filters=[(key, "=", value) for key, value in filters.items()],
    )
    original_patient_cells["patient"] = original_patient_cells["patient"].astype(str)

    scores = []
    for perturbed_cluster in perturbed_patient_cells.clustering.unique():
        for original_cluster in original_patient_cells.clustering.unique():

            common_cells = pd.merge(
                perturbed_patient_cells.query("clustering==@perturbed_cluster"),
                original_patient_cells.query("clustering==@original_cluster"),
                suffixes=("_perturbed", "_original"),
                on=[
                    "Unnamed: 0",
                    "x",
                    "y",
                    "patient",
                    "indication",
                    "datadump",
                    "max_edge_size",
                ],
            )

            score = metrics.adjusted_rand_score(
                common_cells.cluster_perturbed, common_cells.cluster_original
            )
            scores += [score]
    return scores


def plot_randscores(scores: pd.DataFrame, plots_path: Path, **filters: Any) -> None:
    """plot the cluster enrichment scores with indication and perturbation hue"""
    fig = sns.lineplot(
        data=scores,
        x="ratio",
        y="randscore",
        hue=scores[["perturbation", "indication"]].apply(tuple, axis=1),
    )
    fig.set(
        title="Randscore btwn original and post-perturbation clustering by indication"
    )
    figname = (
        plots_path
        / "randscores_simulations"
        / "/".join(f"{key}={value}" for key, value in filters.items())
    )
    figname.mkdir(parents=True, exist_ok=True)
    fig.figure.savefig(figname / "randscores_scores.png")
    plt.close("all")
