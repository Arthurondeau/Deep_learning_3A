"""miscellaneous utils"""
import importlib
import math
import pathlib
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchmetrics
from omegaconf import DictConfig
from sklearn.preprocessing import LabelBinarizer


def get_is_tumor(cells_csv: pd.DataFrame) -> pd.DataFrame:
    if "probabilities.tumor" in cells_csv.columns:
        cells_csv["is_tumour"] = ~(cells_csv["probabilities.tumor"] < 0.5) | (
            cells_csv["celltypes"] == "Tumor"
        )
    else:
        cells_csv["is_tumour"] = ~(
            cells_csv["tumour_mask"].str.lower() == "non_tumour"
        ) | (cells_csv["celltypes"] == "Tumor")
    return cells_csv


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    if obj_path == "":
        obj_path = "builtins"
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object '{obj_name}' cannot be loaded from '{obj_path}'.")
    return getattr(module_obj, obj_name)


def filter_filenames(
    candidate_filenames: Iterable[Path], graph_creation_config: Dict[str, Any]
) -> List[Path]:
    """Apply filters to the filenames in candidate_filenames
    filter can be:
        'should_be_in' ex: '.csv' should be in the name
        'should_not_be_in': ex: 'colocalization should not be in the name

    Args:
        candidate_filenames (List[str]): list of filenames (ie: os.listdir output)
        graph_creation_config (Dict[str, Any]): graph_creation config with a filename_filters key

    Returns:
        List[str]: list of all filenames that meet the filters
    """
    filters = graph_creation_config.get(
        "filename_filters", {"should_be_in": [], "should_not_be_in": []}
    )
    should_be_in = filters.get("should_be_in", [])
    should_not_be_in = filters.get("should_not_be_in", [])
    filtered_files = [
        filename
        for filename in candidate_filenames
        if (
            all(ext in filename.name for ext in should_be_in)
            and not any(ext in filename.name for ext in should_not_be_in)
        )
    ]
    return filtered_files


def remove_keywords_from_columns_name(
    columns: List[str], keywords: List[str]
) -> List[str]:
    """Remove and lower all the requested keywords from columns name"""
    columns = [element.lower() for element in columns]
    for keyword in keywords:
        columns = [element.replace(keyword, "") for element in columns]
    return columns


def compute_ellipse_eccentricity(row: pd.Series) -> float:
    """compute eccentricity for ellipse"""
    if row["minor_axis_length"] == 0 or row["major_axis_length"] == 0:
        return 0
    return math.sqrt(
        1
        - math.pow(row["minor_axis_length"], 2) / math.pow(row["major_axis_length"], 2)
    )


def compute_ellipse_circularity(row: pd.Series) -> float:
    """compute circularity for ellipse"""
    area = math.pi * (row["major_axis_length"] / 2) * (row["minor_axis_length"] / 2)
    perimeter = (
        2
        * math.pi
        * math.sqrt(
            (
                math.pow(row["major_axis_length"] / 2, 2)
                + math.pow(row["minor_axis_length"] / 2, 2)
            )
            / 2
        )
    )
    if area == 0:
        return 0
    return math.pow(perimeter, 2) / (4 * math.pi * area)


def load_analysis_data(
    config: DictConfig, dataset: str, rglob_pattern: str = "*"
) -> Tuple[List[str], List[np.ndarray], List[pd.DataFrame], List[pd.DataFrame]]:
    """load patients, embeddings, filtered_cells, unfiltered_cells for dataset dataset"""
    immucan_embeddings_path = sorted(
        list(
            file
            for file in Path(config["paths"][dataset]["embeddings"]).rglob(
                rglob_pattern
            )
            if file.is_file()
        )
    )
    patients = [el.name.split("_embeddings")[0] for el in immucan_embeddings_path]
    indications = [
        patient_file.parents[1].name.split("=")[1].upper()
        if "=" in patient_file.parents[1].name
        else None
        for patient_file in immucan_embeddings_path
    ]
    datadumps = [
        patient_file.parents[0].name.split("=")[1]
        if "=" in patient_file.parents[0].name
        else None
        for patient_file in immucan_embeddings_path
    ]

    embeddings = []
    filtered_cells = []
    unfiltered_cells = []

    for indication, datadump, patient in zip(indications, datadumps, patients):
        if indication:
            filename = (
                Path(config["paths"][dataset]["embeddings"])
                / f"indication={indication}"
                / f"datadump={datadump}"
                / f"{patient}_embeddings.p"
            )
        else:
            filename = (
                Path(config["paths"][dataset]["embeddings"]) / f"{patient}_embeddings.p"
            )
        with open(
            filename,
            "rb",
        ) as file:
            embeddings += [pickle.load(file)]

        if indication:
            filename = (
                Path(config["paths"][dataset]["patient_graphs"])
                / f"indication={indication}"
                / f"datadump={datadump}"
                / f"{patient}_features.p"
            )
        else:
            filename = (
                Path(config["paths"][dataset]["patient_graphs"])
                / f"{patient}_features.p"
            )
        with open(
            filename,
            "rb",
        ) as file:
            cells_df = pickle.load(file)
            cells_df.insert(column="patient", loc=0, value=patient)
            cells_df.insert(column="datadump", loc=0, value=datadump)
            cells_df.insert(column="indication", loc=0, value=indication)
            filtered_cells += [cells_df]

        if indication:
            filename = (
                Path(config["paths"][dataset]["input_cells"])
                / f"indication={indication}"
                / f"datadump={datadump}"
                / f"{patient}_cells.csv"
            )
        else:
            filename = (
                Path(config["paths"][dataset]["input_cells"]) / f"{patient}_cells.csv"
            )
        unfiltered_cells += [pd.read_csv(filename).rename(columns={"X": "x", "Y": "y"})]
    return patients, embeddings, filtered_cells, unfiltered_cells


def swap_root_directory(
    source_directory: Path, target_directory: Path, file: Path
) -> Path:
    """replace source_directory by target_directory in file"""
    index = file.parts.index(source_directory.parts[-1])
    new_path = pathlib.Path(target_directory).joinpath(*file.parts[index + 1 :])
    return new_path


def read_patients_to_indications_datadump(
    cfg: DictConfig, dataset: str, rglob_pattern: str = "*"
) -> Dict:
    """read all patients that have embeddings
    and return a mapping {patient_name:{indication, datadump"""
    files = list(
        file
        for file in Path(cfg["paths"][dataset]["embeddings"]).rglob(rglob_pattern)
        if file.is_file()
    )
    mapping = {
        "_".join(file.name.split("_")[:-1]): {
            "indication": file.parents[1].name.split("=")[1],
            "datadump": file.parents[0].name.split("=")[1],
        }
        for file in files
    }

    return mapping


def get_reference_perturbations_from_all_perturbations(
    perturbations: List[Path],
) -> List[Path]:
    """Find in a list of paths of perturbation the references with ratio=None
    There might be multiple perturbations with the same ratio (ex in distinct iteration_i parent folder)
    As the sampling are done randomly from original graphs anyway, the perturbation None can be of any iteration
    Args:
        perturbations (List[Path]): paths are ***/type_ratio=Y/

    Returns:
        List[Path]: list of paths with one for each type and ratio=None
    """

    references_perturbations = [
        perturbation
        for perturbation in perturbations
        if perturbation.name.endswith("_ratio='None'")
    ]

    perturbation_types = set(
        perturbation.name.replace("_ratio='None'", "")
        for perturbation in references_perturbations
    )
    references_perturbations_one_by_type = [
        [
            perturbation
            for perturbation in references_perturbations
            if perturbation.name == f"{perturbation_type}_ratio='None'"
        ][0]
        for perturbation_type in perturbation_types
    ]
    return references_perturbations_one_by_type


def find_patient_indications(cfg: DictConfig) -> pd.DataFrame:
    """read embeddings folder and find all patients along with their indication"""
    files = [
        file for file in Path(cfg.paths.immucanembeddings).rglob("*") if file.is_file()
    ]
    indications = [
        re.search(  # type: ignore
            r"(?<=indication=)(.*)(?=/datadump=)", file.as_posix()
        ).group(1)
        for file in files
    ]
    datadumps = [
        re.search(r"(?<=/datadump=)(.*)(?=/)", file.as_posix()).group(1)  # type: ignore
        for file in files
    ]
    patients = ["_".join(file.name.split("_")[:-1]) for file in files]
    dataset = pd.DataFrame(
        {"indication": indications, "patient": patients, "datadump": datadumps}
    )
    return dataset[dataset["indication"].isin(["NSCLC", "SCCHN", "BC"])]


def get_binary_classif_results(preds: torch.Tensor, gts: torch.Tensor):
    """inputs are 1D tensors, one with probas predictions, the other binary label"""
    preds = preds.cpu()
    gts = gts.cpu()
    results = {
        "accuracy": torchmetrics.Accuracy("binary")(
            torch.softmax(preds, 1)[:, 1] > 0.5, gts
        ).item(),
        "recall": torchmetrics.Recall("binary")(
            torch.softmax(preds, 1)[:, 1] > 0.5, gts
        ).item(),
        "precision": torchmetrics.Precision("binary")(
            torch.softmax(preds, 1)[:, 1] > 0.5, gts
        ).item(),
        "f1": torchmetrics.F1Score("binary")(
            torch.softmax(preds, 1)[:, 1] > 0.5, gts
        ).item(),
        "auc": torchmetrics.AUROC("binary")(torch.softmax(preds, 1)[:, 1], gts).item(),
        "map": torchmetrics.AveragePrecision("binary")(
            torch.softmax(preds, 1)[:, 1], gts
        ).item(),
        "probas": np.array(torch.softmax(preds, 1)[:, 1].cpu()).astype(np.float64),
        "labels": np.array(gts.cpu()).astype(np.float64),
    }

    return results


def get_multiclass_classif_results(
    preds: torch.Tensor, gts: torch.Tensor, n_classes: int
):
    """inputs are 1D tensors, one with probas predictions, the other multiclass label"""
    preds = preds.cpu()
    gts = gts.cpu()
    results = {
        "accuracy": torchmetrics.Accuracy(
            "multiclass", num_classes=n_classes, average="macro"
        )(preds, gts).item(),
        "recall": torchmetrics.Recall(
            "multiclass", num_classes=n_classes, average="macro"
        )(preds, gts).item(),
        "precision": torchmetrics.Precision(
            "multiclass", num_classes=n_classes, average="macro"
        )(preds, gts).item(),
        "f1": torchmetrics.F1Score(
            "multiclass", num_classes=n_classes, average="macro"
        )(preds, gts).item(),
        "auc": torchmetrics.AUROC("multiclass", num_classes=n_classes, average="macro")(
            preds, gts
        ).item(),
        "map": torchmetrics.AveragePrecision(
            "multiclass",
            average="macro",
            num_classes=n_classes,
        )(preds, gts).item(),
        "probas": np.array(preds.cpu()).astype(np.float64),
        "labels": np.array(gts.cpu()).astype(np.float64),
    }

    return results


def update_results(row: pd.Series) -> pd.Series:
    results = get_multiclass_classif_results(
        torch.tensor(np.vstack(row["probas"]), dtype=float),
        torch.tensor(row["labels"], dtype=torch.long),
        n_classes=len(row["probas"][0]),
    )
    for key, value in results.items():
        row[key] = value
    return row


def generate_test_sets_tls(
    config: DictConfig, output_path: Path, n_iterations: int
) -> None:
    patients_df = find_patient_indications(config)
    tls_masks = [
        el for el in Path(config.paths.immucan.tls_masks).rglob("*") if el.is_file()
    ]
    patients_with_tls = ["_".join(patient.stem.split("_")[1:]) for patient in tls_masks]
    patients_df["is_tls"] = patients_df["patient"].isin(patients_with_tls)
    (output_path / "splits").mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        test_patients = list(
            patients_df.groupby("is_tls", group_keys=False)
            .apply(lambda x: x.sample(frac=0.25))["patient"]
            .values
        )
        with open(output_path / f"splits/test_patients_{iteration}.pkl", "wb") as file:
            pickle.dump(test_patients, file)


def generate_test_sets(
    config: DictConfig, output_path: Path, n_iterations: int
) -> None:
    patients_df = find_patient_indications(config)
    (output_path / "splits").mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        test_patients = list(
            patients_df.groupby("indication", group_keys=False)
            .apply(lambda x: x.sample(frac=0.25))["patient"]
            .values
        )
        with open(output_path / f"splits/test_patients_{iteration}.pkl", "wb") as file:
            pickle.dump(test_patients, file)


def find_patient_clinical(config: DictConfig) -> pd.DataFrame:
    clinical = pd.read_parquet(
        "/cloud-data/cloud-pipeline-aida-mihc-storage/autoencoder-clustering/IMMUCAN/CLINICAL/"
    )
    clinical["patient"] = "P" + clinical["IMMUcan Sample ID"]
    patients = [
        file.name.split("_")[0]
        for file in Path(config.paths.immucan.embeddings).rglob("*")
        if file.is_file()
    ]
    clinical = clinical[clinical["patient"].isin(patients)].rename(
        columns={"Lymphocyte infiltration": "infiltration", "Fibrosis": "fibrosis"}
    )
    clinical.loc[clinical["fibrosis"] == "Yes", "fibrosis"] = 0
    clinical.loc[clinical["fibrosis"] == "No", "fibrosis"] = 1
    return clinical


def generate_test_sets_by_col(
    config: DictConfig, output_path: Path, n_iterations: int, col: str
) -> None:
    patients_df = find_patient_clinical(config)
    if col == "infiltration":
        patients_df = patients_df[
            ~patients_df["infiltration"].isin(["0", "Not applicable"])
        ]

    (output_path / "splits").mkdir(parents=True, exist_ok=True)
    for iteration in range(n_iterations):
        test_patients = list(
            patients_df.groupby(col, group_keys=False)
            .apply(lambda x: x.sample(frac=0.25))["patient"]
            .values
        )
        with open(output_path / f"splits/test_patients_{iteration}.pkl", "wb") as file:
            pickle.dump(test_patients, file)
