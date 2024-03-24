"""graph utils"""
import itertools
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch_geometric
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from src.utils.utils import compute_ellipse_circularity, compute_ellipse_eccentricity
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


def load_graph(filename: str) -> Tuple[sp.csr_matrix, pd.DataFrame]:
    with open(filename.format("adjacency"), "rb") as adjacency:
        adjacency = sp.csr_matrix(pickle.load(adjacency))

    with open(filename.format("features"), "rb") as file:
        features: pd.DataFrame = pickle.load(file)

    features = features.fillna(features.mean(numeric_only=True))
    return adjacency, features


def generate_adj_from_csv_torch(
    cells_df: pd.DataFrame, max_edge_size: float
) -> torch.Tensor:
    """process cell detection dataframe, 1 line = 1 cell/columns=features,
    and generate the corresponding graph as adjacency/features.
    cells_df is expected to have X, Y columns

    distance matrix computed with torch (possible cuda integration)
    Args:
        cells_df (pd.DataFrame): cell detection dataframe
        max_edge_size (sp.coo_matrix): max distance between nodes to have an edge
    Returns:
        torch.tensor: adj matrix (features are not modified so no need to return)
    """
    centroids = torch.tensor(
        [(cells_df.loc[i, "x"], cells_df.loc[i, "y"]) for i in cells_df.index]
    )

    # calulcate the distance matrix
    dist = torch.round(torch.cdist(centroids, centroids, p=2), decimals=4)
    dist = dist.fill_diagonal_(1)

    # filter the distance matrix by the treshold max_edge_size
    dist = torch.where(dist < max_edge_size, dist, 0.0)
    dist = torch.where(dist > 0, dist, 1 / dist)
    adj = 1 / dist

    return adj


def generate_graph_from_csv(
    cells_df: pd.DataFrame, max_edge_size: int
) -> Tuple[sp.coo_matrix, pd.DataFrame]:
    """process cell detection dataframe, 1 line = 1 cell/columns=features,
    and generate the corresponding graph as adjacency/features.
    cells_df is expected to have X, Y columns

    Args:
        cells_df (pd.DataFrame): cell detection dataframe
        max_edge_size (int): max distance between nodes to have an edge

    Returns:
        Tuple[sp.coo_matrix, pd.DataFrame]: sparse weighted adjacency matrix, features matrix
    """
    centroids = [(cells_df.loc[i, "x"], cells_df.loc[i, "y"]) for i in cells_df.index]

    # compute nearest neighbors to find all nodes within max_edge_size from eachothers
    nearest_neighbors = NearestNeighbors(radius=max_edge_size, algorithm="kd_tree").fit(
        centroids
    )
    dists, ids = nearest_neighbors.radius_neighbors(centroids)

    # flatten dists list
    dists_ = [j for i in dists for j in i]
    dists_ = [dist if dist > 0 else 1 for dist in dists_]

    ids_ = [ids[i][j] for i in range(ids.shape[0]) for j in range(len(dists[i]))]
    rows = [i for i, j in enumerate(ids) for _ in range(len(dists[i]))]

    # weight of each edge
    weights = 1 / np.array(dists_)

    # complete matrix according to positions
    adjacency = sp.coo_matrix(
        (weights, (rows, ids_)), shape=(len(centroids), len(centroids))
    )
    sp.coo_matrix.setdiag(adjacency, 0)
    adjacency = 1 / 2 * (adjacency + adjacency.T)

    cells_df["deg"] = np.array((adjacency > 0).sum(1)).flatten()
    return adjacency, cells_df


def add_degree_to_features(
    features_matrix: pd.DataFrame, adjacency_matrix: sp.csr_array
) -> pd.DataFrame:
    """add a degree column to the features matrix using the adjacency matrix"""
    degrees = np.array((adjacency_matrix > 0).sum(1)).flatten()
    features_matrix["degree"] = degrees
    features_matrix["normed_degree"] = StandardScaler().fit_transform(
        degrees.reshape(-1, 1)
    )
    return features_matrix


def generate_graph_from_splits(
    cells_df: pd.DataFrame, n_splits: int = 4, max_edge_size: int = 100
) -> Tuple[sp.csr_array, pd.DataFrame]:
    """compute adjacency matrix from the cell_df dataframe
    The space is divided in a n_splits*n_splits grid to reduce the memory needed for computations
    The grid panels have an overlap of max_edge_size to make sure that no edge is lost
    One adjacency matrix is computed for each panel.
    The resulting adjacency matrix is the sum of all submatrices minus the duplicates edges

    Args:
        cells_df (pd.DataFrame): cells df
        n_splits (int, optional): grid size. Defaults to 4.
        max_edge_size (int, optional): max edge size. Defaults to 100.

    Returns:
        sp.csr_array: adjacency matrix
    """
    max_x = cells_df.x.max()
    max_y = cells_df.y.max()
    splits_x = (
        cells_df["x"].quantile([i / n_splits for i in range(n_splits + 1)]).values
    )
    splits_y = (
        cells_df["y"].quantile([i / n_splits for i in range(n_splits + 1)]).values
    )
    adjacency_indices = torch.tensor([])
    adjacency_weights = torch.tensor([])
    for split_1, split_2 in itertools.product(range(n_splits), range(n_splits)):
        query = (
            f"x>{max(-1, splits_x[split_1]-max_edge_size//2-1)} &"
            f" x<{min(max_x+1, max_edge_size//2+1+splits_x[split_1+1])} &"
            f" y>{max(-1, splits_y[split_2]-max_edge_size//2-1)} &"
            f" y<{min(max_y+1,  max_edge_size//2+splits_y[split_2+1]+1)}"
        )
        split_cells = cells_df.query(query)
        if split_cells.shape[0] == 0:
            continue
        split_adjacency = generate_adj_from_csv_torch(split_cells, max_edge_size)
        adjacency_indices = torch.concat(
            (
                adjacency_indices,
                torch.nonzero(split_adjacency).apply_(lambda x: split_cells.index[x]),
            ),
            0,
        )
        adjacency_weights = torch.concat(
            (adjacency_weights, split_adjacency[split_adjacency > 0])
        )
    adjacency_indices, idxs = np.unique(
        adjacency_indices.numpy(), axis=0, return_index=True
    )
    adjacency_weights = adjacency_weights[idxs]
    adj = sp.csr_array(
        (adjacency_weights, (adjacency_indices[:, 0], adjacency_indices[:, 1])),
        shape=(cells_df.shape[0], cells_df.shape[0]),
    )
    features = add_degree_to_features(cells_df, adj)
    return adj, features


# def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
#     """Symmetrically normalize adjacency matrix.
#     FROM DGI CODE"""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def sparse_mx_to_torch_sparse_tensor(
#     sparse_mx: sp.spmatrix,
# ) -> torch.Tensor:
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
#     )
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


# def adj_to_tensor(adj: sp.spmatrix) -> torch.Tensor:
#     """turns sparse matrix into torch tensor
#     adds identity matrix
#     Args:
#         adj (sp.spmatrix): sp matrix to be turned into tensor

#     Returns:
#         torch.sparse.FloatTensor
#     """
#     adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
#     sparse_adj = sparse_mx_to_torch_sparse_tensor(adj_normalized)
#     return sparse_adj


def feat_to_tensor(features: pd.DataFrame, features_list: List[str]) -> torch.Tensor:
    """cast pd of features to torch tensor
    only keeping the features in features_list
    """
    features_tensor = torch.tensor(
        features.loc[:, features_list].values, dtype=torch.float32
    )
    return features_tensor


def sample_subgraphs_from_cells_df(
    cell_df: pd.DataFrame,
    samples_indices: List[Set[int]],
    d_to_centroid: int = 500,
    max_edge_size: int = 300,
    overlap_threshold: float = 0,
) -> Tuple[Optional[torch.Tensor], Optional[pd.DataFrame], List[Set[int]]]:
    """randomly sample a subgraph from cell_df

    a random point is selected as seed,
    all points within d_to_centroid to this seed are selected
    a graph is created with those points

    if the selected nodes are have >overlap_threshold nodes in common
    with another sampled graph in sampled_indices, this graph won't be kept

    overlap_threshold=1 => all graphs are kept (can't have >1 ratio of common nodes)
    overlap_threshold=0 => 0 nodes in common

    Args:
        cell_df (pd.DataFrame): dataframe with cells features and coordinates
        samples_indices (List[List[int]]): list of previously sampled nodes
        d_to_centroid (int, optional): max dist with seed. Defaults to 500.
        max_edge_size (int, optional): Defaults to 300.
        overlap_threshold (float, optional): 0<threshold<1. Defaults to 0.

    Returns:
    Tuple[torch.Tensor, pd.DF, List[Set[int]]]: adjacency matrix, features, updated samples_indices
    """
    seed = random.randint(0, cell_df.shape[0] - 1)
    x_id, y_id = cell_df.iloc[seed]["x"], cell_df.iloc[seed]["y"]
    seed_coords = (x_id, y_id)
    query = (
        f"x>{seed_coords[0]-d_to_centroid} &"
        f" x<{seed_coords[0]+d_to_centroid} &"
        f" y>{seed_coords[1]-d_to_centroid} &"
        f" y<{seed_coords[1]+d_to_centroid}"
    )
    sampled = cell_df.query(query)
    sampled_idx = set(sampled.index)
    # check if sampled set of nodes overlap with another sampled graph
    new_adj, new_feats = None, None
    if not any(
        len(sampled_idx & set(idx)) / len(sampled_idx) > overlap_threshold
        for idx in samples_indices
    ):
        samples_indices += [sampled_idx]
        adj = generate_adj_from_csv_torch(sampled, max_edge_size)
        if adj.sum() > 0:
            new_adj = adj
            new_feats = sampled.reset_index(drop=True)
    return new_adj, new_feats, samples_indices


def adj_feat_to_data(
    adjacency: sp.spmatrix, features: pd.DataFrame, features_list: List[str]
) -> BaseData:
    """generate torch geometric graph as data
    from adjacency and features matrix

    Args:
        adjacency (sp.spmatrix)
        features (pd.csv)

    Returns:
        torch_geometric.data.Data
    """
    features_tensor = feat_to_tensor(features, features_list)
    if adjacency[adjacency.nonzero()].shape[1] == 0:
        edge_weight = torch.tensor([])
    else:
        edge_weight = torch.tensor(adjacency[adjacency.nonzero()]).squeeze(0).float()
    data = Data(
        features_tensor,
        edge_index=torch_geometric.utils.from_scipy_sparse_matrix(adjacency)[0],
        edge_weight=edge_weight,
    )
    return data


def get_graphs_names_in_folder(
    path_to_folder: Path, limit_size: int = 10000
) -> List[Path]:
    """list objects in the provided folder that appear twice
    graphs deteted will be those where two files appear in the folder (_features.p and _adjacency.p)
    Args:
        path_to_folder Path: path to folder where features/adjacnecy files are stored
        limit_size (int, optional): max num of graphs to load. Defaults to 10000.

    Returns:
        List[Path]: list of file names with {} to be formated with features/adjacency to open
    """
    graphs_names = sorted(
        list(path for path in path_to_folder.rglob("*") if path.is_file())
    )
    graphs_names = sorted(
        [
            path.parent / path.name.replace("adjacency", "{}").replace("features", "{}")
            for path in graphs_names
        ]
    )
    graphs_names = sorted(
        list({graph for graph in graphs_names if graphs_names.count(graph) == 2})
    )
    graphs_names.sort(
        key=lambda f: (f.parent / f.name.format("adjacency")).stat().st_size
    )
    return graphs_names[:limit_size]


def clean_data(
    data: pd.DataFrame,
    desired_ftres: List,
    features_filtration: Dict[str, str],
    features_normalization: List,
    detection_probability_filter: Dict[str, float],
) -> pd.DataFrame:
    """clean data by keeping the desired ftres + the features_normalization.
    After the first step, the data is normalized accodring to the features_normalization,
    Lastly it is filtered by the mean treshold as well as the quantiles according to
    the features normalization,
    the new data is stocked in path_destination

    Args:
        detection_probability_filter: column name/threshold

    Returns:
        a data .csv stocked in the path_destination
    """
    if detection_probability_filter:

        data = data.query(
            " | ".join(
                f"{item[0]}>={item[1]}" for item in detection_probability_filter.items()
            )
        )
    # Data filtering with mean and quantile.
    if features_filtration:
        data = data.query(
            " | ".join(f"{item[0]}>={item[1]}" for item in features_filtration.items())
        )

    # Compute ellipse features
    if "eccentricity" not in data.columns and (
        "minor_axis_length" in data.columns and "major_axis_length" in data.columns
    ):
        data.loc[:, "eccentricity"] = data.apply(compute_ellipse_eccentricity, 1)
    if "circularity" not in data.columns and (
        "minor_axis_length" in data.columns and "major_axis_length" in data.columns
    ):
        data.loc[:, "circularity"] = data.apply(compute_ellipse_circularity, 1)

    if "cell_labels" not in data.columns and "cell_labels" in desired_ftres:
        data.rename(columns={"layer_2_gated": "cell_labels"}, inplace=True)

    if features_normalization:
        data.loc[:, features_normalization] = StandardScaler().fit_transform(
            data.loc[:, features_normalization]
        )

    data = data.loc[
        :, [feature for feature in desired_ftres if feature in data.columns]
    ]
    data = data.drop_duplicates()

    return data


def add_offset_to_coordinates(dataframe: pd.DataFrame, offset: int) -> pd.DataFrame:
    """
    Immucan has 4 rois per image and all are in the same csv files
    Rois are ~600*600
    This function adds offsets to coordinates of separate rois to have them separated
    You should have offset - ROI.size > max_edge_size to ensure nodes of differente rois
    are not connected

    Args:
        dataframe (pd.DataFrame): cells_dataset
        offset (int): value of offset
    """
    roi_names = sorted(dataframe["ROI"].unique())

    def offset_row(row: pd.Series) -> pd.Series:
        if roi_names.index(row["ROI"]) == 1:
            row["X"] += offset
        elif roi_names.index(row["ROI"]) == 2:
            row["Y"] += offset
        elif roi_names.index(row["ROI"]) == 3:
            row["X"] += offset
            row["Y"] += offset
        return row

    return dataframe.apply(offset_row, 1)
