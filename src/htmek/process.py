import numpy as np
import pandas as pd
import holoviews as hv
from collections import namedtuple

from .fitting import (ChamberFit, fit_parameters, fit_PBP_isotherm,
                      fit_PBP_linear, fit_RFU_expon, fit_RFU_linear, 
                      fit_kinetics_expon, fit_kinetics_linear)
from .assays.standards import fit_PBP, PBP_isotherm, linear_PBP
from .assays.kinetics import fit_RFU_progress
from .viz import fit_map
from .utils import make_df_fit


def fit(
    df: pd.DataFrame,
    fit_params: fit_parameters,
    x_label: str,
    y_label: str,
    z_label: None | str = None,
    col: str = 'mark_col',
    row: str = 'mark_row',
    r2_thresh: float = 0.8,
    skip_col: None | str = None,
) -> dict | hv.DynamicMap:
    """Generic function to fit data to a function.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe that contains per-chamber data to fit.
    fit_params : fit parameters or list of fit parameters
        Function to attempt fit. If list, will try the next one only if
        the first fit fails.
    x_label : str
        Name of the column containing x-axis information, e.g., substrate
        concentrations, time, etc.
    y_label : str
        Name of the column containing y-axis information, e.g., fluorescence
        intensity, etc.
    z_label : None or str, default None
        If there is an additional column containing additional grouping info,
        e.g., substrate concentrations for progress curves.
        TODO: let this take a list of more than one column for multiple
        additional conditions.
    col : str, default 'mark_col'
        Name of the column for chamber column information.
    row : str, default 'mark_row'
        Name of the column for chamber row information.
    r2_thresh : float, default 0.8
        Minimum r2 for a fit to be considered a success.
    skip_col : None or str, default None
        If 'df' contains a column where chambers have been flagged to be
        skipped (True or False), this should be the name of that column.

    Returns
    -------
    fit_dict :
        Dictionary containing fits.
    p_fit_map :
        DynamicMap of the fits. Can be used for downstream plotting.

    Notes
    -----

    """
    fit_dict = {}
    group = [col, row]
    if z_label is not None:
        group = [*group, z_label]
    for group, sub_df in df.groupby(group):
        xs, ys = sub_df[x_label].values, sub_df[y_label].values
        if skip_col is not None:
            skip = sub_df[skip_col].values[0]
        else:
            skip = False

        chamber_fit = ChamberFit(
            xs,
            ys,
            fit_params,
            r2_thresh=r2_thresh,
            autoskip=skip
        )

        # Store fit
        fit_dict[group] = chamber_fit

    # Make the fit map
    p_fit_map = fit_map(
        df,
        fit_dict,
        x_label,
        y_label,
        z_label,
    )

    return fit_dict, p_fit_map


def standards(
    df: pd.DataFrame,
    fit_params = [fit_PBP_isotherm, fit_PBP_linear],
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
    fit_params : fit parameters or list of fit parameters
        Function to attempt fit. If list, will try the next one only if
        the first fit fails.
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

        chamber_fit = ChamberFit(
            xs,
            ys,
            fit_params,
        )

        # Store fit
        fit_dict[chamber] = chamber_fit

    # Make the fit map
    p_fit_map = fit_map(
        df,
        fit_dict,
        x_label,
        y_label,
    )

    return fit_dict, p_fit_map

def RFU_progress(
    df: pd.DataFrame,
    fit_func = fit_RFU_expon,
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
        different timepoints and different substrate concentrations.
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
    # params = ['A', 'k', 'y0']
    # df_fit = make_df_fit(fit_dict, params, col, row, conc_label)

    # Make the correlation map
    # p_corr = plot_correlation()

    return fit_dict, p_fit_map#, p_corr

def kinetic_fit(
    df: pd.DataFrame,
    fit_func = fit_kinetics_expon,
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
        An assay dataframe that contains per-chamber Pi values for
        different timepoints and different substrate concentrations.
    fit_func :
        The function used to fit the data, usually a wrapper around the
        curve_fit function. Must return pcov, popt, and model
        (the function used to model the data).
    x_label :
        Name of the column containing time information (x-axis).
    y_label :
        Name of the column containing the product values (y-axis).
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