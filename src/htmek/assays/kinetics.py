import numpy as np
import pandas as pd

import magnify


def kinetics_pipe(
    data,
    pinlist,
    blank='BLANK',
    hflip=True,
    pipes=None,
    return_pipe=False,
    **kwargs,
):
    """Pipeline for obtaining a chip object specific for Kinetics.
    
    See magnify.microfluidic_chip_pipe for docs:
    
    https://github.com/FordyceLab/magnify/blob/main/src/magnify/registry.py#L35
    """
    pipe = magnify.microfluidic_chip_pipe(
        chip_type='ps',
        shape=(56,32),
        overlap=96,
        rotation=3,
        pinlist=pinlist,
        blank=blank,
        high_edge_quantile=0.995,
        min_button_diameter=65,
        max_button_diameter=70,
        chamber_diameter=75,
        min_roundness=0.20,
        roi_length=None,
        drop_tiles=False,
        **kwargs,
    )

    if pipes is not None:   
        for new_pipe in pipes:
            pipe.add_pipe(new_pipe)

    if hflip:
        pipe.add_pipe("horizontal_flip", after="stitch")
        
    if return_pipe:
        return pipe

    return pipe(data)