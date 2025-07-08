import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import magnify
import sympy as sp

from numpy.typing import ArrayLike


def standards_pipe(
    data,
    pinlist,
    overlap=96,
    rotation=3,
    blank='BLANK',
    post_hflip=True,
    pre_hflip=False,
    basic_correct=False,
    pipes=None,
    return_pipe=False,
    **kwargs,
):
    """Pipeline for obtaining a chip object specific for Standards.

    Notes
    -----
    For `assay` pipelines (Standards and Kinetics), it is expected that
    there will be some signal in the blank chambers. `magnify` does not
    adjust blank chambers when circle-finding, so these would be poorly
    identified. In Standards and Kinetics images, there is almost always
    enough signal to find the chambers correctly. Therefore, the magnify
    pipeline is performed with a nonsense string first, and then these
    'tag' values are adjusted in the final chip.
    
    See magnify.microfluidic_chip_pipe for docs:
    
    https://github.com/FordyceLab/magnify/blob/main/src/magnify/registry.py#L35
    """
    pipe = magnify.microfluidic_chip_pipe(
        chip_type='ps',
        shape=(56,32),
        pinlist=pinlist,
        overlap=overlap,
        rotation=rotation,
        blank='bananas', # nonsense string
        high_edge_quantile=0.990,
        min_button_diameter=65,
        max_button_diameter=70,
        chamber_diameter=75,
        min_roundness=0.10, # lower for reaction chambers
        roi_length=None,
        **kwargs,
    )

    if pipes is not None:   
        for new_pipe in pipes:
            pipe.add_pipe(new_pipe)

    if post_hflip:
        pipe.add_pipe("horizontal_flip", after="stitch")
    if pre_hflip:
        pipe.add_pipe("horizontal_flip", before="stitch")
    if basic_correct:
        pipe.add_pipe("basic_correct", before="stitch")

    if return_pipe:
        return pipe

    chip = pipe(data)

    # Adjust pinlist blanks to magnify blank value (empty string)
    chip['tag'] = chip.tag.str.replace(blank, '')

    return chip

def PBP_isotherm(
    P_i: float | ArrayLike,
    A: float,
    KD: float,
    PS: float,
    I0: float,
) -> float | ArrayLike:
    """Isotherm for single-site Pi:PBP binding.

    Parameters
    ----------
    P_i : float or np.array
        Free phosphate concentration(s).
    A : float
        Amplitude of the isotherm.
    KD : float
        Dissociation constant.
    PS : float
        PBP concentration.
    I0 : float
        RFU value at 0 uM Pi.

    Returns
    -------
    float or np.array
        Predicted RFU value(s) for P_i(s).
    """
    return 0.5 * A * (KD + P_i + PS - ((KD + PS + P_i)**2 - 4*PS*P_i)**(1/2)) + I0

def fit_PBP(
    P_is: ArrayLike,
    RFUs: ArrayLike,
):
    """Curve fit PBP isotherm with intial guesses and bounds.

    This function is prescribed – it uses known best-guess parameters
    and physical bounds for fitting PBP data.
    
    NOTE: This function returns the fitting function.

    Parameters
    ----------
    P_is : np.array
        Array of phosphate concentrations.
    RFUs : np.array
        Array of RFUs.

    Returns
    -------
    popt : np.array
        The optimal values for parameters A, KD, PS, I_0uMP_i.
    pcov : np.array
        The covariance matrix for parameter fits. To convert to standard
        deviation, run `np.sqrt(np.diag(pcov))`.
    PBP_isotherm : callable
        The function used within curve fit. Returned so that it can be
        used directly and unambiguously to plot the result of the fit
        that used this function.
    """
    try:
        popt, pcov = curve_fit(
            PBP_isotherm,
            P_is,
            RFUs,
            p0 = [100, 0.1, 100, 1000],
            bounds = ([0]*4, [np.inf]*4),
        )
    except (ValueError, RuntimeError):
        popt = np.empty(4)*np.nan
        pcov = np.empty((4,4))*np.nan

    return popt, pcov, PBP_isotherm

def PBP_isotherm_inverse(
    RFU,
    A,
    KD,
    PS,
    I0
):
    """Inverse of single-site Pi:PBP binding isotherm.
    
    Parameters
    ----------
    RFU : float
        RFU value.
    A : float
        Amplitude of the isotherm.
    KD : float
        Dissociation constant.
    PS : float
        PBP concentration.
    I0 : float
        RFU value at 0 uM Pi.

    Returns
    -------
    float
        Predicted free phosphate concentration.
    """
    return (-A*I0*KD - A*I0*PS + A*KD*RFU + A*PS*RFU - I0**2 + 2.0*I0*RFU - RFU**2)/(A*(A*PS + I0 - RFU))

def compute_PBP_product(
    RFU,
    popt,
):
    """Compute [P_i] values from RFU from inverted PBP function."""
    A, KD, PS, I0 = popt
    
    return PBP_isotherm_inverse(RFU, A, KD, PS, I0)

def fit_standard_curve(df, mark_row, mark_col):
    dat = df[(df['mark_col'] == mark_col) & (df['mark_row'] == mark_row)]
    concs = dat.standard_conc.values
    median_intensities = dat.median_intensity.values
    try:
        popt, pcov = fit_PBP(concs, median_intensities)
        return popt
    except:
        return np.nan, np.nan, np.nan, np.nan

