import numpy as np
import pandas as pd

import magnify


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