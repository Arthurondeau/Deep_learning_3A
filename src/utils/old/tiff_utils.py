"""tiff utils"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageSequence


def preprocess_tiff(img: np.ndarray, output_shape: Tuple) -> np.ndarray:
    """pad the image to the right shape + zscore each channel"""
    img = np.pad(
        img,
        (
            (max(0, output_shape[0] - img.shape[0]), 0),
            (max(0, output_shape[1] - img.shape[1]), 0),
            (0, 0),
        ),
        mode="edge",
    )
    return img


def load_tiff_with_channels(path: Path, channels: List[int]) -> np.ndarray:
    """load the tiff file and returns the array with channels in channels
    the values are rescaled to 0-255 using the top values for each channels

    Args:
        path (Path): path to tiff
        channels (List[int]): indices of channels to keep, see conf['immucan_tiff_channels']

    Returns:
        np.array: np.uint8 array with values btwn 0 and 255
    """
    image = Image.open(path)

    stacked_pages = np.stack(
        [np.array(page) for page in ImageSequence.Iterator(image)], axis=-1
    )
    selected_channels = stacked_pages[:, :, channels]
    return ((selected_channels / selected_channels.max(0).max(0)) * 255).astype(
        np.uint8
    )
