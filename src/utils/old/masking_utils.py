"""Utils related to binary mask creation, usage"""
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from skimage import io, measure, morphology


def roi_masks_from_patient_mask(
    patient_mask: np.ndarray, roi_size: int
) -> List[np.ndarray]:
    """Assumes 4 non overlapping roi, one in each corner"""
    masks = [
        patient_mask[:roi_size, :roi_size],
        patient_mask[-roi_size:, :roi_size],
        patient_mask[:roi_size, -roi_size:],
        patient_mask[-roi_size:, -roi_size:],
    ]
    return masks


def apply_mask_cells_df(cells_df: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    """apply mask to cells_df
    clls_df must have x, y columns"""

    return cells_df.apply(lambda row: int(0 < mask[int(row["x"]), int(row["y"])]), 1)


def apply_mask_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """returns the values in the image for all nonzero pixels in the mask

    Args:
        image (np.ndarray): (H*W*C) array
        mask (np.ndarray): (H*W) array

    Returns:
        np.ndarray: N*C array with N = #nonzero pixels in mask
    """
    return image[mask.nonzero()]


def contour_coordinates(no_holes: np.ndarray) -> np.ndarray:
    """Compute contour coordinates from a binary mask of one single object
    ie: should only be one ROI or at let one tls
    Args:
        no_holes (np.ndarray): binary mask of one single object, 1s are the object the rest is 0

    Returns:
        (np.ndarray): Nx2 array coordinates of contour points
    """
    lab_img = measure.label(no_holes)
    allcoords = measure.find_contours((lab_img == 1).astype(np.uint8), level=0)
    coords = allcoords[0]
    return coords


def get_tls_mask(
    roi_cells: pd.DataFrame,
    tls_cluster_index: int,
    mask_size: Tuple[int, int] = (1000, 1000),
) -> np.ndarray:
    """Generate binary mask from cell df and cluster index
    Arbitrary morphology operations, worked for immucan first 8 patients
    Args:
        roi_cells (pd.DataFrame): cells df. should have a cluster col andn x,y
        tls_cluster_index (int): id of the cluster to contour
        mask_size (Tuple[int, int]): size of the image ~offset+roi_size
    Returns:
        np.ndarray: binary mask aroung the cluster with id tls_cluster_id
    """
    zeros = np.zeros(mask_size)
    cells = roi_cells.query(f"cluster=={tls_cluster_index}")[["x", "y"]].values.astype(
        int
    )
    zeros[cells[:, 0], cells[:, 1]] = 1

    dilated = morphology.binary_dilation(zeros, np.ones((10, 10)))

    closed = morphology.binary_closing(dilated, morphology.disk(3))

    opened = morphology.binary_dilation(closed, morphology.disk(2))

    no_holes = morphology.remove_small_holes(opened, 200)
    return no_holes


def get_cluster_mask(
    cells: pd.DataFrame, cluster_id: int, mask_size: Tuple = (1600, 1600)
) -> np.ndarray:
    """morpho operations to get binary mask around cluster"""
    filtered_cells = cells.query(f"x<{mask_size[0]}").query(f"y<{mask_size[1]}")
    print(f"DROPPED {cells.shape[0]-filtered_cells.shape[0]} CELLS OUT OF MASK")
    zeros = np.zeros(mask_size)
    cells = filtered_cells.query(f"cluster=={cluster_id}")[["x", "y"]].values.astype(
        int
    )
    zeros[cells[:, 0], cells[:, 1]] = 1
    dilated = morphology.binary_dilation(zeros, np.ones((20, 20)))

    closed = morphology.binary_closing(dilated, morphology.disk(10))

    opened = morphology.binary_dilation(closed, morphology.disk(10))

    no_holes = morphology.remove_small_holes(opened, 500)

    no_objects = morphology.remove_small_objects(no_holes, 4000)
    return no_objects


def save_cluster_masks(
    output_path: Path,
    patient_level_dataframe: pd.DataFrame,
    min_mask_size: int,
    **filters: Any,
) -> None:
    """generate mask for reach cluster in cells_df and save it to the appropriate folder"""
    target_path = (
        output_path
        / "masks"
        / f"prefix={filters.get('prefix')}"
        / f"max_edge_size={filters.get('max_edge_size')}"
        / f"clustering={filters.get('clustering')}"
        / f"indication={filters.get('indication')}"
        / f"datadump={filters.get('datadump')}"
        / f"patient='{filters.get('patient')}'"
    )
    for cluster in sorted(list(patient_level_dataframe["cluster"].unique())):
        mask = get_cluster_mask(patient_level_dataframe, cluster)
        if mask.sum() > min_mask_size:
            path = target_path / f"cluster_{cluster}_mask.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            io.imsave(
                path,
                255 * mask.astype(np.uint8),
            )
