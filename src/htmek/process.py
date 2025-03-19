import numpy as np
import pandas as pd
import xarray as xr


def to_df(
    chip: xr.Dataset,
    value: str = 'roi',
    agg_func: str = 'median',
    mask: None | xr.Dataset = None,
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
    if mask is None:
        mask = chip.fg
    else:
        if isinstance(mask, xr.Dataset):
            # Reduces over-indexing and dropping columns
            mask = mask.to_numpy()

    # Aggregate to DataFrame
    collapse = [f'{value}_x', f'{value}_y']
    df = getattr(chip[value].where(mask), agg_func)(collapse)
    df = df.to_dataframe().reset_index()

    return df