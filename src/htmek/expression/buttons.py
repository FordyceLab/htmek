import numpy as np
import pandas as pd

import magnify


def buttons_pipe(
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
    """Pipeline for obtaining a chip object specific for Buttons.
    
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
        high_edge_quantile=0.999,
        min_button_diameter=20, # new 20
        max_button_diameter=32, # new 30
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
    if basic_correct:
        pipe.add_pipe("basic_correct", before="stitch")

    if return_pipe:
        return pipe

    return pipe(data)