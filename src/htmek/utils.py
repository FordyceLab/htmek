import numpy as np
import pandas as pd
from skimage.transform import rescale
import tifffile as tf
from pathlib import Path
import xarray as xr
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
    overlap: int = int(102*2),
    rotation: int = 0,
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

    tf.imwrite(output, array)
    
    if verbose:
        print(f'File written to {output}.')

def subtract_bg(
    egfp: str | Path,
    egfp_bg: str | Path,
    overlap: int,
    output: str | Path,
    verbose: bool = True,
) -> None:
    """Substracts the egfp background image from the egfp image."""
    egfp_image = magnify.image(
        egfp,
        overlap=overlap,
    ).image.values.astype('int16')

    bg_image = magnify.image(
        egfp_bg,
        overlap=overlap,
    ).image.values.astype('int16')

    bg_sub = egfp_image - bg_image
    bg_sub = np.clip(bg_sub, 0, 65535).astype('uint16')

    tf.imwrite(output, bg_sub)
    if verbose:
        print(f'File written to {output}.')


def to_df(
    chip: xr.Dataset,
    value: str = 'roi',
    agg_func: str = 'median',
    mask: None | xr.Dataset = None,
    make_coord: None | str | list[str] = None,
) -> pd.DataFrame:
    """Collapses a chip dataset into a pd.DataFrame.

    Parameters
    ----------
    chip :
        The chip object from a magnify.microfluidic_chip_pipe.
    value :
        What value to plot. Defaults to roi.
    agg_func :
        How to aggregate the value. Defaults to median.
    mask :
        How to select what is used as data from the image. Defaults to
        the foreground mask determined by magnify (chip.fg), but can be 
        overwritten with a different mask, including from a different
        chip entirely.
    make_coord :
        Any data variables in the chip to convert to a coordinate, which
        will then get carried through to the output dataframe. Can be a
        string or list of strings.

    Returns
    -------
    df :
        pandas DataFrame with aggregated data.

    Examples
    --------
    >>> htmek.to_df(chip)

    This will return the median of the roi values within chip.fg.

    >>> htmek.to_df(chip, agg_func='sum', mask=chip2.fg)

    This will return the sum of the roi values within a fg mask provided
    by chip2.
    """
    chip = chip.copy()
    
    if mask is None:
        mask = chip.fg
    else:
        if isinstance(mask, xr.Dataset):
            # Reduces over-indexing and dropping columns
            mask = mask.to_numpy()

    if make_coord is not None:
        if isinstance(make_coord, str):
            make_coord = [make_coord]
        for coord in make_coord:
            chip.coords[coord] = chip[coord]

    # Aggregate to DataFrame
    collapse = [f'{value}_x', f'{value}_y']
    df = getattr(chip[value].where(mask), agg_func)(collapse)
    df = df.to_dataframe().reset_index()

    return df


def make_df_fit(
    fit_dict,
    params,
    col,
    row,
    z=None,
):
    args = [col, row]
    if z is not None:
        args.append(z)
    
    # Convert to DataFrame
    df_fit = pd.DataFrame(fit_dict).T
    df_fit.index = df_fit.index.rename(args)
    df_fit.columns = ['popt', 'perr']
    df_fit = df_fit.reset_index()

    # Unpack
    df_fit[params] = pd.DataFrame(df_fit['popt'].to_list())

    params_std = [f'err_{param}' for param in params]
    df_fit[params_std] = pd.DataFrame(df_fit['perr'].to_list())

    # Delete old columns
    del df_fit['popt']
    del df_fit['perr']

    return df_fit