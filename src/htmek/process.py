import numpy as np
import pandas as pd
import holoviews as hv

from .assays.standards import fit_PBP
from .assays.kinetics import fit_RFU_progress
from .viz import fit_map
from .utils import make_df_fit


def standards(
    df: pd.DataFrame,
    fit_func = fit_PBP,
    x_label: str = 'standard_conc',
    y_label: str = 'median_intensity',
    col: str = 'mark_col',
    row: str = 'mark_row',
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
    col :
        Name of the column for chamber column information.
    row :
        Name of the column for chamber row information.

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
    for chamber, sub_df in df.groupby([col, row]):
        xs, ys = sub_df[x_label].values, sub_df[y_label].values

        # Make sure the fit_func parameter outputs the right thing.
        try:
            out = fit_func(xs, ys)
            popt, pcov, model = out
        except ValueError:
            raise ValueError(
                'Parameter `fit` must return three args, popt, pcov, and '\
                'the function used to model the data.'
            )

        # Store fit
        fit_dict[chamber] = popt, np.sqrt(np.diag(pcov))

    # Make the fit map
    p_fit_map = fit_map(
        df,
        model,
        x_label,
        y_label,
        fit_dict = fit_dict,
    )

    return fit_dict, p_fit_map

def RFU_progress(
    df: pd.DataFrame,
    fit_func = fit_RFU_progress,
    x_label: str = 'acq_time',
    y_label: str = 'median_intensity',
    conc_label: str = 'substrate_conc',
    col: str = 'mark_col',
    row: str = 'mark_row',
) -> dict | hv.DynamicMap:
    """Processes a standards_df to fit parameters.

    Parameters
    ----------
    df :
        An assay dataframe that contains per-chamber RFU values for
        different timepoints different substrate concentrations.
    fit_func :
        The function used to fit the data, usually a wrapper around the
        curve_fit function. Must return pcov, popt, and model
        (the function used to model the data).
    x_label :
        Name of the column containing time information (x-axis).
    y_label :
        Name of the column containing the RFU values (y-axis).
    conc_label :
        Name of the column containing substrate concentration values.
    col :
        Name of the column for chamber column information.
    row :
        Name of the column for chamber row information.

    Returns
    -------
    fit_dict :
        Dictionary containing fits.
    p_fit_map :
        DynamicMap of the fits. Can be used for downstream plotting.
    p_corr :
        A plot correlating chamber rates across [substrate].
    """
    fit_dict = {}
    for group, sub_df in df.groupby([col, row, conc_label]):
        xs, ys = sub_df[x_label].values, sub_df[y_label].values

        # Make sure the fit_func parameter outputs the right thing.
        try:
            out = fit_func(xs, ys)
            popt, pcov, model = out
        except ValueError:
            raise ValueError(
                'Parameter `fit` must return three args, popt, pcov, and '\
                'the function used to model the data.'
            )

        # Store fit
        fit_dict[group] = popt, np.sqrt(np.diag(pcov))

    # Make the fit map
    p_fit_map = fit_map(
        df,
        model,
        x_label,
        y_label,
        conc_label,
        fit_dict = fit_dict,
    )

    # Convert to DataFrame
    params = ['A', 'k', 'y0']
    df_fit = make_df_fit(fit_dict, params, col, row, conc_label)

    # Make the correlation map
    # p_corr = plot_correlation()

    return fit_dict, p_fit_map#, p_corr