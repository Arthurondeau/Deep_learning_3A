# """clustering utils"""
# import statistics
# from pathlib import Path
# from typing import Any, Callable

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import phenograph
# import seaborn as sns
# from omegaconf import DictConfig
# from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics.pairwise import euclidean_distances
# from tqdm import tqdm


# class PhenographWrapper:
#     """wrapper to be able to instantiate phenograph like a sklearn clustering class"""

#     def __init__(self, n_clusters: int, random_state: int = 0):
#         """n_clusters is not actuallly n_clusters but phenograph k param
#         named n_clusters here to match with sklearn.KMeans args
#         TODO change back to k"""
#         self.k = n_clusters
#         self.random_state = random_state
#         self.name = f"phenograph_k_{self.k}"

#     def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
#         """return 1xn array phenograph clusters"""
#         return phenograph.cluster(embeddings, k=self.k, seed=self.random_state)[0]

#     def __str__(self) -> str:
#         return self.name


# def compare_clusterings_randscore_heatmap(
#     cfg: DictConfig,
#     cluster_class1: Callable,
#     cluster_class2: Callable,
#     all_embeddings: np.ndarray,
#     max_k: int = 100,
#     output_path: Path = Path("."),
# ) -> None:
#     """Runs phenograph clustering for all k<max_k
#     Compute rand score for all combinations and plot the corresponding heatmap
#     as phenograph_randscore_heatmap.png
#     """
#     k_range = np.arange(10, max_k, 5)  # This is the range of k values we will look at

#     clusters1 = [
#         [
#             cluster_class1(k).fit_predict(all_embeddings)
#             for _ in range(cfg.analysis.clustering_benchmark.num_clusterings)
#         ]
#         for k in tqdm(k_range)
#     ]

#     clusters2 = [
#         [
#             cluster_class2(k).fit_predict(all_embeddings)
#             for _ in range(cfg.analysis.clustering_benchmark.num_clusterings)
#         ]
#         for k in tqdm(k_range)
#     ]

#     rand_indices = pd.DataFrame(
#         np.zeros((len(k_range), len(k_range))), index=k_range, columns=k_range
#     )
#     rand_indices.index.name = str(cluster_class1)
#     rand_indices.columns.name = str(cluster_class2)

#     for i in range(len(k_range)):
#         for j in range(len(k_range)):
#             rand_indices.iloc[i, j] = statistics.mean(
#                 [
#                     adjusted_rand_score(
#                         clusters1[i][k],
#                         clusters2[j][m],
#                     )
#                     for k in range(len(clusters1[i]))
#                     for m in range(len(clusters2[j]))
#                 ]
#             )

#     plt.figure(figsize=(8, 8))
#     sns.heatmap(
#         rand_indices,
#         cmap="coolwarm",
#         square=True,
#         vmin=0,
#         vmax=1,
#         xticklabels=True,
#         yticklabels=True,
#     ).invert_yaxis()
#     sns.set(font_scale=1)
#     plt.xlabel("k1")
#     plt.ylabel("k2")
#     plt.title("Adjusted Rand Score")

#     path = output_path / f"max_edge_size={cfg.graph_creation.graphs.max_edge_size}/"
#     path.mkdir(parents=True, exist_ok=True)
#     plt.savefig(
#         path / f"{str(cluster_class1)}_{str(cluster_class2)}_randscore_heatmap.png"
#     )
#     plt.close()


# def find_centroids(embeddings: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
#     """compute average embedding vector for each cluster id"""
#     embeddings_df = pd.DataFrame(embeddings)
#     embeddings_df.insert(loc=0, value=clusters, column="cluster")
#     centroids = embeddings_df.groupby("cluster").mean()
#     centroids.columns = [str(column) for column in centroids.columns]
#     return centroids


# def apply_centroids(embeddings: np.ndarray, centroids: pd.DataFrame) -> np.ndarray:
#     """apply euclidean distance btwn all embeddings vectors and all centroids
#     for each embdding, return the argmin of centroids"""
#     return euclidean_distances(embeddings, centroids).argmin(1)


# def save_cells_with_cluster(
#     path: Path, dataframe: pd.DataFrame, **filters: Any
# ) -> None:
#     """save cells_df in the appropriate parquet format"""
#     path = path / "cells_with_cluster"
#     path.parent.mkdir(parents=True, exist_ok=True)
#     for key, value in filters.items():
#         if key not in dataframe.columns:
#             dataframe.insert(loc=0, column=key, value=value)
#     dataframe.to_parquet(
#         path,
#         partition_cols=[
#             "prefix",
#             "max_edge_size",
#             "clustering",
#             "indication",
#             "datadump",
#             "patient",
#         ],
#     )
