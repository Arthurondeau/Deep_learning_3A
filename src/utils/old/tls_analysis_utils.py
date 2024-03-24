"""utils for analysis, might be refactored into multiple files """
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression


def density_based_clustering_tls(
    df_cells: pd.DataFrame, min_samples: int = 3, eps: int = 10
) -> pd.DataFrame:
    """apply dbscan to the df_cells
    Default parameters were tuned on IMMUCAN first 8 patients and validated by Elton
    A cluster column is added to df_cells. value of -1 means not clustered, otherwise cluster id

    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html for params tuning

    Returns:
        pd.DataFrame: df_cells with an additional "cluster" column
    """
    df_cells.insert(
        loc=df_cells.shape[1],
        column="cluster",
        value=DBSCAN(
            min_samples=min_samples,
            eps=eps,
        ).fit_predict(df_cells[["x", "y"]]),
    )
    return df_cells


def pool_level_classification(
    all_embeddings: np.ndarray,
    cells_with_labels: pd.DataFrame,
    training_patients: List[str],
    test_patients: List[str],
) -> Dict[str, Any]:
    """puts all cells of all patients together"""

    cells_train = cells_with_labels.query(f"patient in {training_patients}")
    cells_test = cells_with_labels.query(f"patient in {test_patients}")
    embeddings_train, embeddings_test = (
        all_embeddings[cells_train.index],
        all_embeddings[cells_test.index],
    )
    labels_train, labels_test = (
        cells_train["is_tls"].values,
        cells_test["is_tls"].values,
    )
    print(len(embeddings_train), len(embeddings_test))
    print(labels_train.sum(), labels_test.sum())
    if len(embeddings_test) == 0 or (labels_train.sum() == 0 or labels_test.sum() == 0):
        return {
            "probas": [None],
            "labels": [None],
            "f1": [None],
            "recall": [None],
            "accuracy": [None],
            "precision": [None],
            "auc": [None],
            "map": [None],
        }
    model = LogisticRegression(class_weight="balanced")
    model = model.fit(embeddings_train, labels_train)
    test_probas = (
        model.predict_proba(embeddings_test)[:, 1]
        if embeddings_test.size
        else np.array([])
    )

    f1_score = metrics.f1_score(
        labels_test,
        test_probas > 0.5,
    )

    recall = metrics.recall_score(
        labels_test,
        test_probas > 0.5,
    )

    accuracy = metrics.accuracy_score(labels_test, test_probas > 0.5)

    precision = metrics.precision_score(
        labels_test,
        test_probas > 0.5,
    )

    auc = metrics.roc_auc_score(
        labels_test,
        test_probas,
    )
    apre = metrics.average_precision_score(
        labels_test,
        test_probas,
    )

    results = {
        "probas": [test_probas],
        "labels": [labels_test],
        "f1": [f1_score],
        "recall": [recall],
        "accuracy": [accuracy],
        "precision": [precision],
        "auc": [auc],
        "map": [apre],
    }
    return results
