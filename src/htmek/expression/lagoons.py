import numpy as np
import pandas as pd

import magnify


def lagoons_pipe(
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
    """Pipeline for obtaining a chip object specific for Lagoons.
    
    See magnify.microfluidic_chip_pipe for docs:
    
    https://github.com/FordyceLab/magnify/blob/main/src/magnify/registry.py#L35
    """
    default_kwargs = dict(
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
    )

    # Update default kwargs with any provided kwargs, overwriting defaults
    default_kwargs.update(kwargs)

    pipe = magnify.microfluidic_chip_pipe(
        **default_kwargs,
    )

    if pipes is not None:   
        for new_pipe in pipes:
            pipe.add_pipe(new_pipe)

    if post_hflip:
        pipe.add_pipe("horizontal_flip", after="stitch")
    if pre_hflip:
        pipe.add_pipe("horizontal_flip", before="stitch")

    pipe.remove_pipe("filter_leaky")

    if return_pipe:
        return pipe
    
    return pipe(data)