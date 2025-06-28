import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from numpy.typing import ArrayLike

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import magnify

from .standards import compute_PBP_product, PBP_isotherm


def kinetics_pipe(
    data,
    pinlist,
    overlap=96,
    rotation=3,
    blank='BLANK',
    post_hflip=True,
    pre_hflip=False,
    pipes=None,
    return_pipe=False,
    **kwargs,
):
    """Pipeline for obtaining a chip object specific for Kinetics.

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
        high_edge_quantile=0.995,
        min_button_diameter=65,
        max_button_diameter=70,
        chamber_diameter=75,
        min_roundness=0.10, # lower for reaction chambers
        roi_length=None,
        drop_tiles=False,
        **kwargs,
    )

    if pipes is not None:   
        for new_pipe in pipes:
            pipe.add_pipe(new_pipe)

    if post_hflip:
        pipe.add_pipe("horizontal_flip", after="stitch")
    if pre_hflip:
        pipe.add_pipe("horizontal_flip", before="stitch")

    if return_pipe:
        return pipe

    chip = pipe(data)
    
    # Adjust pinlist blanks to magnify blank value (empty string)
    chip['tag'] = chip.tag.str.replace(blank, '')

    return chip

def exponential(
    t: float | ArrayLike,
    A: float,
    k: float,
    y0: float,
) -> float | ArrayLike:
    """Exponential fit for progress curve.

    Parameters
    ----------
    t : float or np.array
        Time(s).
    A : float
        Maximum change in signal.
    k : float
        Rate of change of signal over time.
    y0 : float
        Intercept.

    Returns
    -------
    float or np.array
        Predicted value at time t.
    """
    return A*(1-np.exp(-k*t))+y0

def fit_RFU_progress(
    ts: ArrayLike,
    RFUs: ArrayLike,
):
    """Curve fit RFU progress curve with intial guesses and bounds.

    This function is prescribed – it uses known best-guess parameters
    and physical bounds for fitting timecourse data.
    
    NOTE: This function returns the fitting function.

    Parameters
    ----------
    ts : np.array
        Array of times.
    RFUs : np.array
        Array of RFUs.

    Returns
    -------
    popt : np.array
        The optimal values for parameters A, k, and y0.
    pcov : np.array
        The covariance matrix for parameter fits. To convert to standard
        deviation, run `np.sqrt(np.diag(pcov))`.
    exponential : callable
        The function used within curve fit. Returned so that it can be
        used directly and unambiguously to plot the result of the fit
        that used this function.
    """
    try:
        popt, pcov = curve_fit(
            exponential,
            ts,
            RFUs,
            p0 = [np.max(RFUs), 0.01, np.min(RFUs)],
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]),
        )
    except (ValueError, RuntimeError):
        popt = np.empty(3)*np.nan
        pcov = np.empty((3,3))*np.nan

    return popt, pcov, exponential


def get_product_conc_PBP(x, standards_df, pbp_conc=None, cutoff=None):
    RFU = x['median_intensity']
    col = x['mark_col']
    row = x['mark_row']

    standard_rows = standards_df[(standards_df['mark_row'] == row) & (standards_df['mark_col'] == col)]
    
    exceeds_PBP_cutoff = False

    if not standard_rows.empty:
        popt = standard_rows[['A', 'KD', 'PS',	'I_0uMP_i']].iloc[0].values
        product_conc = compute_PBP_product(RFU, popt)

        if cutoff == None:
            cutoff = pbp_conc * (2/3)

        # If RFU above 50uM Pi RFU, set a flag
        if RFU > PBP_isotherm(cutoff, *popt):
            exceeds_PBP_cutoff = True

        return round(float(product_conc), 2), exceeds_PBP_cutoff
    else:
        popt = np.nan, np.nan, np.nan, np.nan
        return np.nan, exceeds_PBP_cutoff

def single_exponential(x, A, k, y0):
    return A*(1-np.exp(-k*x))+y0

def fit_single_exponential_turnover(assays_df, remove_RFUs_above_PBP_cutoff=True):
     
    # Copy to new df
    assays_df.sort_values(['mark_col', 'mark_row', 'substrate_conc'], inplace=True)

    # Drop rows with NaNs in 'seconds' or 'product_conc'
    assays_df = assays_df.dropna(subset=['acq_time', 'product_conc'])

    # Drop RFUs above PBP cutoff
    if remove_RFUs_above_PBP_cutoff:
        assays_df = assays_df[assays_df['exceeds_PBP_cutoff'] == False]

    # Group by 'mark_row', 'mark_col', and 'substrate_conc'
    grouped = assays_df.groupby(['mark_row', 'mark_col', 'substrate_conc'])

    # Iterate over the groups with tqdm to show progress
    for (r,c,s), group in tqdm(grouped, total=len(grouped)):
        seconds = group['acq_time'].to_numpy()
        product_concs = group['product_conc'].to_numpy()

        # Mask rows that match the current group key
        row_mask = (
            (assays_df["mark_row"] == r) &
            (assays_df["mark_col"] == c) &
            (assays_df["substrate_conc"] == s)
        )

        if len(seconds) > 2:
            p0 = [s, 0.01, product_concs[0]] # A, k, y0
            try:
                popt, _ = curve_fit(single_exponential, 
                                    seconds, product_concs, 
                                    maxfev=100000, p0=p0,
                                    bounds=([0, 0, -np.inf],[s*1.5, 1, np.inf])
                                    )
                A, k, y0 = popt[0], popt[1], popt[2]
            except RuntimeError:
                A, k, y0 = np.nan, np.nan, np.nan
        else:
                A, k, y0 = np.nan, np.nan, np.nan

        assays_df.loc[row_mask, 'A'] = A # Assign 'A' to the group in the original assays_dfaFrame
        assays_df.loc[row_mask, 'k'] = k # Assign 'k' to the group in the original assays_dfaFrame
        assays_df.loc[row_mask, 'y0'] = y0 # Assign 'y0' to the group in the original assays_dfaFrame
        assays_df.loc[row_mask, 'init_rate'] = A*k # Assign 'k' to the group in the original assays_dfaFrame

    return assays_df

def calculate_enzyme_conc(expression_df, assays_df):
    expression_df['EnzymeConc'] = expression_df['summed_button_egfp_intensity']/100000

    # drop summed if exists
    for c in assays_df.columns:
        if 'summed_button_egfp_intensity' in c:
            assays_df = assays_df.drop(columns=[c])
        if 'EnzymeConc' in c:
            assays_df = assays_df.drop(columns=[c])

    # merge in 
    assays_df = pd.merge(assays_df, expression_df[['mark_col', 'mark_row', 'summed_button_egfp_intensity', 'EnzymeConc']], on=['mark_col', 'mark_row'])

    # normalize rate to expression
    assays_df['normalized_rate'] = (assays_df['init_rate']/assays_df['EnzymeConc'])

    return assays_df

def michaelis_menten(S, vmax, KM):
    return (vmax*S)/(KM+S)

def fit_michaelis_menten(assays_df, p0):

    assays_df.sort_values(['mark_col', 'mark_row', 'substrate_conc'], inplace=True)

    # Drop rows with NaNs
    assays_df = assays_df.dropna(subset=['substrate_conc', 'k'])

    # Group by 'mark_row', 'mark_col', and 'substrate_conc'
    grouped = assays_df.groupby(['mark_row', 'mark_col'])

    # Iterate over the groups with tqdm to show progress
    for (r,c), group in tqdm(grouped, total=len(grouped)):
        substrate_concs = group['substrate_conc'].to_numpy()
        initial_rates = group['init_rate'].to_numpy()
        enzyme_conc = group['EnzymeConc'].iloc[0]
        num_points_fit = len(initial_rates)

        # Initialize two point fit flag
        two_point_fit = False
        if len(initial_rates) == 2:
            two_point_fit = True

        # Mask rows that match the current group key
        row_mask = (
            (assays_df["mark_row"] == r) &
            (assays_df["mark_col"] == c)
        )

        if len(substrate_concs) > 1:
            try:
                popt, _ = curve_fit(michaelis_menten, 
                                    substrate_concs, 
                                    initial_rates, 
                                    maxfev=100000, 
                                    p0=p0, 
                                    bounds=([0, 0], [np.inf, np.inf])
                                    )
                vmax, KM = popt[0], popt[1]
            except RuntimeError:
                vmax, KM = np.nan, np.nan
        else:
                vmax, KM = np.nan, np.nan

        kcat = (vmax/(enzyme_conc/1000)) # Convert nM to uM
        kcat_over_KM = kcat/KM # In uM^-1 s^-1 
        kcat_over_KM = kcat_over_KM * 1e6 # In M^-1 s^-1 
        
        residuals = initial_rates - michaelis_menten(substrate_concs, vmax, KM)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((initial_rates - np.mean(initial_rates))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        assays_df.loc[row_mask, 'vmax'] = vmax # Assign 'vmax' to the group in the original assays_dfaFrame
        assays_df.loc[row_mask, 'KM'] = KM # Assign 'KM' to the group in the original assays_dfaFrame
        assays_df.loc[row_mask, 'kcat'] = kcat # Assign 'kcat' to the group in the original assays_dfaFrame
        assays_df.loc[row_mask, 'kcat_over_KM'] = kcat_over_KM # Assign 'kcat_over_KM' to the group in the original assays_dfaFrame
        assays_df.loc[row_mask, 'MM_R2'] = r_squared
        assays_df.loc[row_mask, 'MM_points_fit'] = num_points_fit
        assays_df.loc[row_mask, 'MM_two_point_fit'] = two_point_fit

    return assays_df