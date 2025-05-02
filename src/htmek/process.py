import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv

from .assays.standards import fit_PBP
from .viz import fit_map


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


def standards(
    df: pd.DataFrame,
    fit_func = fit_PBP,
    x_label: str = 'standard_conc',
    y_label: str =  'median_intensity',
    row: str = 'mark_row',
    col: str = 'mark_col',
) -> dict | hv.DynamicMap:
    """Processes a standards_df to fit parameters.

    Parameters
    ----------
    df :
        A standards dataframe that contains per-chamber RFU values for
        different substrate concentrations.
    fit_func :
        The function used to fit the data, usually a wrapper around the
        curve_fit function. Must return pcov, popt, and model
        (the function used to model the data).
    x_label :
        Name of the column containing substrate concentrations (x-axis).
    y_label :
        Name of the column containing the RFU values (y-axis).
    row :
        Name of the column for chamber row information.
    col :
        Name of the column for chamber column information.

    Returns
    -------
    fit_dict :
        Dictionary containing fits.
    p_fit_map :
        DynamicMap of the fits. Can be used for downstream plotting.
    
    TODO: Add option for global intial fit of KD and PS, which should be
    the same for the full chip? A and I_0 are the parameters that really
    should change.
    """
    fit_dict = {}
    for chamber, sub_df in df.groupby(['mark_row', 'mark_col']):
        xs, ys = sub_df[x_label].values, sub_df[y_label].values

        # Try fit with error catching
        try:
            out = fit_func(xs, ys)

            # Make sure the fit_func parameter outputs the right thing.
            try:
                popt, pcov, model = out
            except ValueError:
                raise ValueError(
                    'Parameter `fit` must return three args, popt, pcov, and ' 'the function used to model the data.'
                )

            # Store fit
            fit_dict[chamber] = popt, pcov
        
        # Store error in dict
        except (ValueError, RuntimeError) as e:
            fit_dict[chamber] = f'Failed: {e}'

    # Make the fit map
    p_fit_map = fit_map(
        df,
        model,
        x_label,
        y_label,
        fit_dict = fit_dict,
    )

    return fit_dict, p_fit_map
