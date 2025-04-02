import numpy as np
from skimage.transform import rescale
import tifffile as tf
from pathlib import Path
import magnify

import magnify


def bin2x2(
    array: np.array,
) -> np.array:
    """Takes an unbinned image and performs a correction to match the
    2x2 binning performed by the camera."""

    # Downsize
    array_2x2 = rescale(array, 0.5)

    # Get shapes for intensity scaling
    x, y = array.shape
    new_x, new_y = array_2x2.shape

    # Scale intensity by difference in image size
    array_2x2 = array_2x2 * (x / new_x) * (y / new_y)

    # Adjust to 16-bit int
    array_2x2 = np.clip(array_2x2, 0, 65535).astype('uint16')

    return array_2x2


def make_binned_tiff(
    file: str | Path,
    output: str | Path,
    overlap: int = int(96*2),
    rotation: int = 3,
    verbose: bool = True,
) -> None:
    """Converts a magnify-importable file to a 2x2 binned tiff file
    if 1x1 binning was used.
    """

    data = magnify.image(
        file,
        overlap=overlap,
        rotation=rotation,
    )

    array = bin2x2(data.image.to_numpy())

    array = array.astype('int16')

    tf.imwrite(output, array)
    
    if verbose:
        print(f'File written to {output}.')