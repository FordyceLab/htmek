import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import magnify


def standards_pipe(
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
    """Pipeline for obtaining a chip object specific for Standards.
    
    See magnify.microfluidic_chip_pipe for docs:
    
    https://github.com/FordyceLab/magnify/blob/main/src/magnify/registry.py#L35
    """
    pipe = magnify.microfluidic_chip_pipe(
        chip_type='ps',
        shape=(56,32),
        pinlist=pinlist,
        overlap=overlap,
        rotation=rotation,
        blank=blank,
        high_edge_quantile=0.990,
        min_button_diameter=65,
        max_button_diameter=70,
        chamber_diameter=75,
        min_roundness=0.20,
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

    if return_pipe:
        return pipe

    return pipe(data)


######################
# PBP
######################

def PBP_isotherm(
    P_i,
    RFU,
    KD,
    PS,
    I_0uMP_i,
):
    """Isotherm for PBP binding Pi."""
    radicand = (KD + PS + P_i)**2 - 4*PS*P_i
    product = KD + P_i + PS - np.sqrt(radicand)
    return 0.5*RFU*product + I_0uMP_i


def fit_PBP(
    P_is,
    RFUs,
):
    """Curve fit PBP isotherm with intial guesses and bounds."""
    popt, pcov = curve_fit(
        PBP_isotherm,
        P_is,
        RFUs,
        p0 = [40, 1, 50, 500],
        bounds = ([0]*4, [np.inf]*4),
    )

    return popt, pcov