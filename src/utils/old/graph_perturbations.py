"""graph perturbations"""
from typing import Any, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


class RandomNodeDrop:
    """Randomly drop nodes in the graph regardless of phenotype"""

    def __init__(self, ratio: float, **_: Any):
        """ratio is proportion of nodes to drop"""

        self.name = f"drop_{ratio=}"
        self.ratio = ratio if ratio != "None" else 0

    def apply(
        self, features: pd.DataFrame, adjacency: sp.csr_array
    ) -> Tuple[pd.DataFrame, sp.csr_array, np.ndarray]:
        """randomly remove self.ratio nodes from the graph"""
        nodes_to_keep = np.array(
            sorted(
                list(
                    np.random.choice(
                        len(features),
                        int((1 - self.ratio) * features.shape[0]),
                        replace=False,
                    )
                )
            )
        )
        features = features.iloc[nodes_to_keep]
        adjacency = adjacency[nodes_to_keep][:, nodes_to_keep]
        return features, adjacency, nodes_to_keep

    def __str__(self) -> str:
        return self.name


class PopNodeDrop:
    """Randomly drop nodes in the graph of a certain category"""

    def __init__(
        self,
        ratio: float,
        population: str,
        query: str = "{population}>{population}.mean()",
    ):
        self.name = f"pop_query={population}_{ratio=}"
        self.ratio = ratio if ratio != "None" else 0
        self.pop_query = query.format(population=population)

    def apply(
        self, features: pd.DataFrame, adjacency: np.ndarray
    ) -> Tuple[pd.DataFrame, sp.csr_array, np.ndarray]:
        """randomly remove self.ratio nodes from a specific selection of cells"""
        nodes_to_drop = np.random.choice(
            features.query(self.pop_query).index,
            int(self.ratio * len(features.query(self.pop_query))),
            replace=False,
        )
        index = list(features.index)
        nodes_to_keep = np.array(
            sorted([index.index(el) for el in index if el not in nodes_to_drop])
        )

        features = features.iloc[nodes_to_keep]
        adjacency = adjacency[nodes_to_keep][:, nodes_to_keep]
        return features, adjacency, nodes_to_keep

    def __str__(self) -> str:
        return self.name
