import numpy as np
import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')

import magnify
import xarray as xr

from .assays.standards import PBP_isotherm
from .process import to_df

######################
# Default opts
######################

hv_defaults = (
    hv.opts.HeatMap(
        tools=['hover'],
        active_tools=[],
    ),
    hv.opts.Scatter(
        size=6,
        line_color='black',
        fill_alpha=0.8,
        frame_width=350,
        frame_height=350,
        tools=['hover'],
        active_tools=[],
    ),
    hv.opts.Points(
        size=6,
        line_color='black',
        fill_alpha=0.8,
        frame_width=350,
        frame_height=350,
        tools=['hover'],
        active_tools=[],
    ),
    hv.opts.Curve(
        line_color='black',
        frame_width=350,
        frame_height=350,
        tools=['hover'],
        active_tools=[],
    ),
    hv.opts.Image(
        frame_width=350,
        frame_height=350,
        tools=['hover'],
        active_tools=[],
    ),
)

hv.opts.defaults(*hv_defaults)


######################
# General plotting
######################
def view(
    chip: xr.Dataset,
    chamber: None | tuple[int, int] = None,
    rastered: bool = True,
    imscale: float = None,
    limits: None | tuple[int, int] = None,
) -> hv.Image:
    """Plot an image of the full chip or a chamber from the chip object.
    
    Parameters
    ----------
    chip :
        The chip object from a magnify.microfluidic_chip_pipe.
    chamber :
        A tuple of ints that indicates the chamber (col, row; zero-indexed)
        to view in more detail.
    rastered :
        Whether to raster the image for quicker rendering. Automatically
        switches to False if chamber is not None (it's not necessary).
    imscale :
        Scale of the image. Defaults: scaled down to 0.06 for full chip,
        scaled up to 2 for chamber.
    limits :
        A tuple of ints for intensity limits.

    Returns
    -------
    p :
        hv.Image object of chip/chamber.

    Examples
    --------
    >>> htmek.viz.view(chip, imscale=0.04, limits=(0, 5000))

    This will show an image of the full chip at 0.04 scale, setting the
    intensity bounds to 0 and 5000.

    >>> htmek.viz.view(chip, chamber=(0,0), limits=(500, 1200))

    This will show an image of the chamber (0,0) (column, row), setting the
    intensity bounds to 0 and 5000.
    """

    if chamber:
        subset = chip.sel(mark_col=chamber[0], mark_row=chamber[1])
        data = subset.roi.to_numpy()
        if imscale is None:
            imscale = 2
        title=f'Chamber {chamber}'
        x = subset.x.data
        y = subset.y.data
        xs = x - len(subset.roi_x), x + len(subset.roi_x)
        ys = y - len(subset.roi_y), y + len(subset.roi_y)
        
        # No need to raster the small image
        rastered=False
        
    else:
        data = chip.image.to_numpy()
        if imscale is None:
            imscale = 0.06
        title=''
        xs = chip.image.im_x.to_numpy()
        ys = chip.image.im_y.to_numpy()

    if limits is None:
        limits = (data.min(), data.max())

    # Need to flip vertically to align with hv.Image bounds behavior
    # Requires "invert_yaxis" to be True below
    data = np.flipud(data)

    p = hv.Image(
        data,
        bounds = (xs[0], ys[0], xs[-1], ys[-1]),
        vdims = 'intensity',
    ).opts(
        title=title,
        frame_width=int(len(data[0])*imscale),
        frame_height=int(len(data)*imscale),
        clim=limits,
        invert_xaxis=True,
        invert_yaxis=True
    )

    if rastered:
        p = rasterize(p)

    return p


def chip_hm(
    chip: xr.Dataset,
    value: str = 'roi',
    agg_func: str = 'median',
    mask: None | xr.Dataset = None,
    xy: list[str, str] = ['mark_col', 'mark_row'],
    attrs: str | list[str] = ['tag'],
    scale=1,
) -> hv.HeatMap:
    """Plot a heatmap of a chip's value of interest from identified
    regions of interest.
    
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
        the mask determined by magnify, but can be overwritten with a
        mask from a different chip object.
    xy :
        What to use for the x and y of the heatmap. Defaults to
        ['mark_col', 'mark_row'].
    attrs :
        Additional information to access with the 'hover' tool. Defaults
        to 'tag' to provide pinlist information.
    scale :
        Arbitrary size of the rendered heatmap. Default 1.
    """

    df = to_df(chip, value, agg_func, mask)

    p = hv.HeatMap(
        df,
        kdims=xy,
        vdims=[value, *attrs],
    ).opts(
        invert_yaxis=True,
        invert_xaxis=True,
        frame_width=int(300*scale),
        frame_height=int(150*3.25*scale),
    )

    return p


def chamber_map(
    chip: xr.Dataset,
    imscale: float = None,
    limits: None | tuple[int, int] = None,
) -> hv.DynamicMap:
    """Plot an image of the full chip or a chamber from the chip object.
    
    Parameters
    ----------
    chip :
        The chip object from a magnify.microfluidic_chip_pipe.
    imscale :
        Scale of the image. Defaults to 2.
    limits :
        A tuple of ints for intensity limits. Defaults to (min, max) of
        a given chamber. Hot spots can make chambers look less bright
        than others if left as None.

    Returns
    -------
    dmap :
        hv.DynamicMap object of all chambers (accessed by sliders).

    Examples
    --------
    >>> htmek.viz.chamber_map(chip)
    """
    def chamber(col, row):
        return view(chip, chamber=(col, row), imscale=imscale, limits=limits)

    col = hv.Dimension('col', values=range(32))
    row = hv.Dimension('row', values=range(56))

    dmap = hv.DynamicMap(chamber, kdims=[col, row])
    dmap.opts(framewise=True)

    return dmap


######################
# Standards plotting
######################

def plot_PBP(
    P_is,
    RFUs,
    popt,
):
    """Plot a PBP isotherm fit given data (P_is, RFUs) and popt from
    htmek.assays.standards.fit_PBP.
    """
    xs = np.linspace(0, np.max(P_is)*1.1, 1000)

    p_data = hv.Scatter(
        (P_is, RFUs),
        kdims = '[Pi] (µM)',
        vdims = 'RFU',
    )

    p_fit = hv.Curve(
        (xs, PBP_isotherm(xs, *popt)),
        kdims = '[Pi] (µM)',
        vdims = 'RFU',
    )

    return p_fit*p_data