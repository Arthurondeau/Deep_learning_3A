"""expression utils"""
from typing import Any, Dict

import numpy as np
import pandas as pd


def transform_expression(
    expression: pd.DataFrame, channels_list: Dict[Any, Any], **filters: str | int
) -> pd.DataFrame:
    """
    Filters should point to the specific expression required for analysis
    ex: max_edge_size, k_means, prefix, patient

    Clusters are treated independently of the patient id so if pool level multiple patients cis ok
    for patient level clusters, a patient id must be specified in the filters

    output:
        pd.DataFrame:
            K-rows: one for each cluster
            cols 0->39 for mean activation in this cluster
            Additional cols with single value corresponding to metadata (patient, kmeans etc...)
    """
    # concat all pixel expression in cluster level
    # group by "roi", "cluster", "patient" to only concat the lists and not actually sum them
    groups = (
        expression[["expression", "roi", "cluster", "patient"]]
        .groupby(["patient", "cluster", "roi"])
        .sum()
        .query("expression!=0")["expression"]
    )

    # Get the avg by sum()/len()
    # Shapes for each roi etc are diffent so an easy way is
    # to bring everyting to shape (1, #expression)
    groups_df = pd.DataFrame(groups.apply(np.sum).tolist(), index=groups.index)
    groups_df.insert(loc=0, column="area", value=groups.apply(len))
    groups_df = groups_df.groupby(["cluster"]).sum()
    groups_df.loc[
        :, [col for col in groups_df.columns if col != "area"]
    ] = groups_df.drop(columns="area").div(
        groups_df.loc[:, ["area"]].values, axis="columns"
    )

    for key, value in filters.items():
        groups_df.insert(loc=0, column=key, value=value)

    groups_df = col_names_id_to_marker(groups_df, channels_list)
    return groups_df


def col_names_id_to_marker(
    dataframe: pd.DataFrame, channels_list: Dict[Any, Any]
) -> pd.DataFrame:
    """rename col names in the dataframe from channel id to the corresponding marker"""
    id_to_marker = {value: key for key, value in channels_list.items()}
    return dataframe.rename(columns=id_to_marker)
