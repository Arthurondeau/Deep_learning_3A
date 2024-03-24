"""utils for model evaluation"""
from collections import Counter
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from omegaconf import DictConfig
from sklearn import metrics


def plot_umap_proj(
    high_dim_data: np.ndarray, labels: np.ndarray, title: str = "2D Umap proj"
) -> None:
    """
    Project high_dim_data into a 2D space and plot with colors associated to labels
    Args:
        high_dim_data (np.array): N*D array
        labels (np.array): N*1 categorical array
    """
    fit = umap.UMAP(n_components=2, random_state=0)
    projection = fit.fit_transform(high_dim_data)
    df_projection = pd.DataFrame(projection, columns=["x", "y"])
    df_projection["labels"] = labels
    sns.scatterplot(
        data=df_projection, x="x", y="y", hue="labels", palette="tab10"
    ).set(title=title)
    plt.show()


def plot_roc_curve(
    class_probas: np.ndarray, labels: np.ndarray, title: str = "ROC CURVE"
) -> None:
    """class_probas and labels are both 1D array"""
    fpr, tpr, _ = metrics.roc_curve(labels, class_probas)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(title)
    plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:.2f}")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()


def plot_multiple_roc_curve(
    class_probas_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    curve_names: List[str],
    title: str = "ROC CURVE",
    datapath: Path | None = None,
) -> None:
    """Plot multiple roc curves

    Args:
        class_probas_list (List[np.ndarray]): list of 1D array with prediction probas
        labels_list (List[np.ndarray]): list of 1D array with labels
        curve_names (List[str]): name for each curve, can leave {} to format with roc_auc score
        title (str, optional): plot title. Defaults to "ROC CURVE".
    """

    fig, axe = plt.subplots()
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    for labels, class_probas, curve_name in zip(
        labels_list, class_probas_list, curve_names
    ):
        display = metrics.RocCurveDisplay.from_predictions(labels, class_probas)
        display.plot(ax=axe, name=curve_name)

    if datapath:
        figname = datapath / "plots" / f"{title}.png"
        figname.parent.mkdir(parents=True, exist_ok=True)
        fig.figure.savefig(figname)
    plt.close("all")


def plot_multiple_pr_recall_curve(
    class_probas_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    curve_names: List[str],
    title: str = "PR-Recall CURVE",
    datapath: Path | None = None,
) -> None:
    """Plot multiple pr recall curves

    Args:
        class_probas_list (List[np.ndarray]): list of 1D array with prediction probas
        labels_list (List[np.ndarray]): list of 1D array with labels
        curve_names (List[str]): name for each curve, can leave {} to format with roc_auc score
        title (str, optional): plot title. Defaults to "ROC CURVE".
    """
    fig, axe = plt.subplots()
    for labels, class_probas, curve_name in zip(
        labels_list, class_probas_list, curve_names
    ):
        display = metrics.PrecisionRecallDisplay.from_predictions(labels, class_probas)
        display.plot(ax=axe, name=curve_name)
    if datapath:
        figname = datapath / "plots" / f"{title}.png"
        figname.parent.mkdir(parents=True, exist_ok=True)
        fig.figure.savefig(figname)
    plt.close("all")


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    title: str = "Confusion Matrix with labels\n\n",
    datapath: Path | None = None,
) -> None:
    """predictions and labels are 1D binary arrays"""
    plot = sns.heatmap(
        metrics.confusion_matrix(labels, predictions, normalize=None), annot=True
    )

    plot.set_title(title)
    plot.set_xlabel("\nPredicted Values")
    plot.set_ylabel("Actual Values ")

    # Ticket labels - List must be in alphabetical order
    plot.xaxis.set_ticklabels(["False", "True"])
    plot.yaxis.set_ticklabels(["False", "True"])
    # Display the visualization of the Confusion Matrix.
    plt.show()
    if datapath:
        figname = datapath / "plots" / f"{title}.png"
        figname.parent.mkdir(parents=True, exist_ok=True)
        plot.figure.savefig(figname)
    plt.close("all")


def plot_expressions_heatmap(
    expressions: pd.DataFrame,
    datapath: Path,
    channels: List[str],
    **filters: str | int,
) -> None:
    """plot expression heatmaps with the appropriate channels"""
    expressions = expressions.loc[:, channels]
    if expressions.shape[0] < 2:
        normalized_expressions = expressions
        fig = sns.heatmap(
            normalized_expressions,
            xticklabels=True,
            yticklabels=True,
            cmap="vlag",
        )
    else:
        normalized_expressions = expressions.apply(
            lambda col: (col - col.mean()) / (col.std() + np.finfo("float").eps)
        )
        fig = sns.clustermap(
            normalized_expressions,
            xticklabels=True,
            yticklabels=True,
            cmap="vlag",
        )
    plt.figure(figsize=(16, 10))

    # fig.set(title="Channel expressions for each cluster, ")

    figname = (
        datapath
        / "plots"
        / "expressions_by_cluster"
        / f"prefix={filters.get('prefix')}"
        / f"max_edge_size={filters.get('max_edge_size')}"
        / f"clustering={filters.get('clustering')}"
        / f"indication={filters.get('indication')}"
        / f"datadump={filters.get('datadump')}"
        / f"patient='{filters.get('patient')}'"
        / "expressions_by_cluster.png"
    )
    figname.parent.mkdir(parents=True, exist_ok=True)

    fig.figure.savefig(figname)
    plt.close("all")


def plot_expression_clusters_markers_covariance(
    expressions: pd.DataFrame,
    datapath: Path,
    **filters: str | int,
) -> None:
    """plot covariance of the expression features for given clustering
    eable to see which markers tend to behave the same way ie: good for sanity check"""
    fig = sns.clustermap(
        pd.DataFrame(
            np.cov(expressions.T),
            index=expressions.columns,
            columns=expressions.columns,
        ),
        xticklabels=True,
        yticklabels=True,
        cmap="vlag",
    )
    figname = (
        datapath
        / "plots"
        / "expressions_by_cluster"
        / f"prefix={filters.get('prefix')}"
        / f"max_edge_size={filters.get('max_edge_size')}"
        / f"clustering={filters.get('clustering')}"
        / f"indication={filters.get('indication')}"
        / f"patient='{filters.get('patient', 'pool')}'"
        / "expression_clusters_markers_covariance.png"
    )
    figname.parent.mkdir(parents=True, exist_ok=True)

    fig.figure.savefig(figname)
    plt.close("all")


# def plot_graph(
#     G, patch=None, labels=None, save=None, node_sizes="small", display=False
# ):
#     """plot graph, overlay on patch if provided

#     Args:
#         G (nx graph)
#         patch: np array of the source patch. Defaults to None.
#     """
#     # node size proportional to degree
#     degrees = dict(G.degree)
#     if node_sizes == "degrees":
#         sizes = [10 + v * 10 for v in degrees.values()]
#     else:
#         sizes = 5

#     figure = plt.figure(figsize=(15, 15))
#     centroids = {i: (G.nodes[i]["y"], G.nodes[i]["x"]) for i in G.nodes}
#     # REMOVED EDGE WEIGHTS BC TOO BIG SO THE PLOTTING WOULD OVERLOAD THE RAM

#     if labels is None:
#         labels = [1] * len(G.nodes)

#     maxval = len(set(labels))
#     label_dict = {list(set(labels))[i]: i for i in range(len(set(labels)))}

#     cmap = plt.cm.Spectral
#     nx.draw(
#         G,
#         centroids,
#         node_size=sizes,
#         node_color=[cmap(label_dict[v] / maxval) for v in labels],
#         ax=figure.add_subplot(111),
#     )
#     for v in label_dict.keys():
#         plt.scatter([], [], c=[cmap(label_dict[v] / maxval)], label="Group{}".format(v))
#     plt.legend()
#     if patch is not None:
#         plt.imshow(patch)
#     if save:
#         figure.savefig(save)
#     if display:
#         figure.show()


def plot_clusters_patient(
    cfg: DictConfig,
    patient_cells: pd.DataFrame,
    cluster_col: str,
    output_path: Path,
    **filters: Any,
) -> None:
    """trigger plot of cells in xy coodinates with cluster coloring
    for each roi"""
    for roi, query in cfg.analysis.rois_queries.items():
        roi_cells = patient_cells.query(query)
        plot_clusters_by_roi(roi_cells, cluster_col, output_path, roi=roi, **filters)


def plot_clusters_by_roi(
    roi_cells: pd.DataFrame,
    cluster_col: str,
    output_path: Path,
    **filters: Any,
) -> None:
    """scatter plot of xy coordinates with cluster as hue"""
    plt.close()
    plt.figure(figsize=(10, 10))
    fig = sns.scatterplot(
        data=roi_cells,
        x="x",
        y="y",
        hue=cluster_col,
        # palette=sns.color_palette("Spectral", k),
        palette=sns.color_palette("husl", roi_cells[cluster_col].nunique()),
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1))

    title = (
        Path(f"prefix={filters.get('prefix')}")
        / f"max_edge_size={filters.get('max_edge_size')}"
        / f"clustering={filters.get('clustering')}"
        / f"indication={filters.get('indication')}"
        / f"datadump={filters.get('datadump')}"
        / f"patient='{filters.get('patient')}'"
        / f"clusters_in_roi={filters.get('roi')}.png"
    )
    fig.set(title=title)

    figname = output_path / "plots" / "clusters" / title
    figname.parent.mkdir(parents=True, exist_ok=True)
    fig.figure.savefig(figname)

    plt.show()
    plt.close(fig.figure)


def avg_pr_curve_exps(results: pd.DataFrame, datapath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)
    for exp in sorted(results["exp"].unique()):
        exp_results = results.query("training_ratio==1 | training_ratio=='1'").query(
            f"exp=='{exp}'"
        )
        exp_results["ROCs"] = exp_results.apply(
            lambda row: metrics.PrecisionRecallDisplay.from_predictions(
                row.labels, row.probas
            ),
            1,
        )
        exp_results["tprs"] = exp_results["ROCs"].apply(
            lambda roc: np.interp(mean_fpr, roc.recall[::-1], roc.precision[::-1])
        )

        mean_tpr = np.mean(exp_results["tprs"].values)
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(exp_results["map"])

        # std_tpr = np.std(exp_results["tprs"].values, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.plot(
            mean_fpr,
            mean_tpr,
            label=f"{exp} "
            + r"$\pm$ 1 std "
            + f"(mAP={mean_auc:.02f} "
            + r"$\pm$"
            + f" {std_auc:.02f})",
            lw=2,
            alpha=0.8,
        )

        # ax.fill_between(
        #     mean_fpr,
        #     tprs_lower,
        #     tprs_upper,
        #     alpha=0.2,
        # )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Recall",
        ylabel="Precision",
        title="Mean PR curve",
        #  with variability",
    )
    # ax.axis("square")
    ax.legend(loc="lower left")

    fig.figure.savefig(datapath / "avg_pr_curve.png")
    plt.close("all")


def avg_roc_curve_exps(results: pd.DataFrame, datapath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)
    for exp in sorted(results["exp"].unique()):
        exp_results = results.query("training_ratio==1 | training_ratio=='1'").query(
            f"exp=='{exp}'"
        )
        exp_results["ROCs"] = exp_results.apply(
            lambda row: metrics.RocCurveDisplay.from_predictions(
                row.labels, row.probas
            ),
            1,
        )
        exp_results["tprs"] = exp_results["ROCs"].apply(
            lambda roc: np.interp(mean_fpr, roc.fpr, roc.tpr)
        )

        mean_tpr = np.mean(exp_results["tprs"].values)
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(exp_results["auc"])

        # std_tpr = np.std(exp_results["tprs"].values, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.plot(
            mean_fpr,
            mean_tpr,
            label=f"{exp} "
            + r"$\pm$ 1 std "
            + f"(AUC={mean_auc:.02f} "
            + r"$\pm$"
            + f" {std_auc:.02f})",
            lw=2,
            alpha=0.8,
        )

        # ax.fill_between(
        #     mean_fpr,
        #     tprs_lower,
        #     tprs_upper,
        #     alpha=0.2,
        # )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve",
        #  with variability",
    )
    # ax.axis("square")
    ax.legend(loc="lower right")
    fig.figure.savefig(datapath / "avg_roc_curve.png")
    plt.close("all")


def weighted_avg_roc(mean_fpr: np.array):
    def interpolated_weighted_avg_roc(row: pd.Series):
        labels = Counter(row["labels"].astype(int))
        weights = {key: value / len(row["labels"]) for key, value in labels.items()}

        tprs = []
        for i in labels:
            roc = metrics.RocCurveDisplay.from_predictions(
                (row["labels"].astype(int) == i).astype(int),
                torch.softmax(torch.tensor(np.vstack(row["probas"])), -1)[:, i],
            )
            tprs += [np.interp(mean_fpr, roc.fpr, roc.tpr)]
        mean_tpr = np.average(
            np.vstack(tprs), weights=[weights[i] for i in range(len(weights))], axis=0
        )
        return mean_tpr

    return interpolated_weighted_avg_roc


def weighted_avg_pr(mean_fpr: np.array):
    def interpolated_weighted_avg_pr(row: pd.Series):
        labels = Counter(row["labels"].astype(int))
        weights = {key: value / len(row["labels"]) for key, value in labels.items()}

        tprs = []
        for i in labels:
            pr = metrics.PrecisionRecallDisplay.from_predictions(
                (row["labels"].astype(int) == i).astype(int),
                torch.softmax(torch.tensor(np.vstack(row["probas"])), -1)[:, i],
            )
            tprs += [np.interp(mean_fpr, pr.recall[::-1], pr.precision[::-1])]
        mean_tpr = np.average(
            np.vstack(tprs), weights=[weights[i] for i in range(len(weights))], axis=0
        )
        return mean_tpr

    return interpolated_weighted_avg_pr


def multiclass_avg_roc_curve_exps(results: pd.DataFrame, datapath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)

    for exp in sorted(results["exp"].unique()):
        exp_results = results.query("training_ratio==1 | training_ratio=='1'").query(
            f"exp=='{exp}'"
        )

        exp_results["tprs"] = exp_results.apply(weighted_avg_roc(mean_fpr), 1)

        mean_tpr = np.mean(exp_results["tprs"].values)
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(exp_results["auc"])

        # std_tpr = np.std(exp_results["tprs"].values, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.plot(
            mean_fpr,
            mean_tpr,
            label=f"{exp} "
            + r"$\pm$ 1 std "
            + f"(AUC={mean_auc:.02f} "
            + r"$\pm$"
            + f" {std_auc:.02f})",
            lw=2,
            alpha=0.8,
        )

        # ax.fill_between(
        #     mean_fpr,
        #     tprs_lower,
        #     tprs_upper,
        #     alpha=0.2,
        # )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve",
        #  with variability",
    )
    # ax.axis("square")
    ax.legend(loc="lower right")
    fig.figure.savefig(datapath / "avg_roc_curve.png")
    plt.close("all")


def multiclass_avg_pr_curve_exps(results: pd.DataFrame, datapath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)

    for exp in sorted(results["exp"].unique()):
        exp_results = results.query("training_ratio==1 | training_ratio=='1'").query(
            f"exp=='{exp}'"
        )
        exp_results["tprs"] = exp_results.apply(weighted_avg_pr(mean_fpr), 1)

        mean_tpr = np.mean(exp_results["tprs"].values)
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(exp_results["map"])

        # std_tpr = np.std(exp_results["tprs"].values, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax.plot(
            mean_fpr,
            mean_tpr,
            label=f"{exp} "
            + r"$\pm$ 1 std "
            + f"(mAP={mean_auc:.02f} "
            + r"$\pm$"
            + f" {std_auc:.02f})",
            lw=2,
            alpha=0.8,
        )

        # ax.fill_between(
        #     mean_fpr,
        #     tprs_lower,
        #     tprs_upper,
        #     alpha=0.2,
        # )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Recall",
        ylabel="Precision",
        title="Mean PR curve",
        #  with variability",
    )
    # ax.axis("square")
    ax.legend(loc="lower left")

    fig.figure.savefig(datapath / "avg_pr_curve.png")
    plt.close("all")
