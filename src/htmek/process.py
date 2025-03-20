import numpy as np
import pandas as pd
import xarray as xr


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