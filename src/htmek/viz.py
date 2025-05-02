import numpy as np
import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')

import random

import matplotlib.pyplot as plt

import magnify
import xarray as xr

from .assays.kinetics import single_exponential, michaelis_menten
from .assays.standards import PBP_isotherm, fit_PBP
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
    hv.opts.BoxWhisker(
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
def show_mask(
    chip: xr.Dataset,
    chamber: None | tuple[int, int] = None,
    imscale: float = None,
) -> hv.Image:
    """Show the mask for a given chip or chamber."""
    if chamber:
        subset = chip.sel(mark_col=chamber[0], mark_row=chamber[1])
        data = subset.fg.to_numpy()
        if imscale is None:
            imscale = 2
        title=f'Chamber {chamber}'
        x = subset.x.data
        y = subset.y.data
        xs = x - len(subset.roi_x), x + len(subset.roi_x)
        ys = y - len(subset.roi_y), y + len(subset.roi_y)

    # Need to flip vertically to align with hv.Image bounds behavior
    # Requires "invert_yaxis" to be True below
    data = np.flipud(data)
    
    # Set the mask to nans so that it can be "clipped" in the image
    data = np.select([data == True], [np.nan], data)

    p = hv.Image(
        data,
        bounds = (xs[0], ys[0], xs[-1], ys[-1]),
        vdims = 'intensity',
    ).opts(
        title=title,
        frame_width=int(len(data[0])*imscale),
        frame_height=int(len(data)*imscale),
        clipping_colors={'NaN': (0, 0, 0, 0)},
        cmap='gray',
        alpha=0.3,
        invert_xaxis=True,
        invert_yaxis=True,
    )

    return p


def view(
    chip: xr.Dataset,
    chamber: None | tuple[int, int] = None,
    tag: None | str = None,
    rastered: bool = True,
    imscale: float = None,
    limits: None | tuple[int, int] = None,
    chamber_mask: float = True,
) -> hv.Image:
    """Plot an image of the full chip or a chamber from the chip object.
    
    Parameters
    ----------
    chip :
        The chip object from a magnify.microfluidic_chip_pipe.
    chamber :
        A tuple of ints that indicates the chamber (col, row; zero-indexed)
        to view in more detail.
    tag : 
        The name of a group of chambers to view in more detail.
    rastered :
        Whether to raster the image for quicker rendering. Automatically
        switches to False if chamber is not None (it's not necessary).
    imscale :
        Scale of the image. Defaults: scaled down to 0.06 for full chip,
        scaled up to 2 for chamber.
    limits :
        A tuple of ints for intensity limits.
    chamber_mask :
        Only relevant if chamber is not None. Shows the mask over it.

    Returns
    -------
    p :
        hv.Image object of chip/chamber(s).

    Examples
    --------
    >>> htmek.viz.view(chip, imscale=0.04, limits=(0, 5000))

    This will show an image of the full chip at 0.04 scale, setting the
    intensity bounds to 0 and 5000.

    >>> htmek.viz.view(chip, chamber=(0,0), limits=(500, 1200))

    This will show an image of the chamber (0,0) (column, row), setting the
    intensity bounds to 0 and 5000.
    """
    # Check for other dimension besides row and column
    if len(chip.image.shape) > 2:
        raise NotImplementedError('Chip contains multiple images.')

    chambers = [chamber]
    plots = []

    if tag is not None:
        chip = chip.stack(chamber=("mark_col", "mark_row"))
        chambers = chip.where(chip.tag == tag, drop=True).chamber.values

    for chamber in chambers:
        if chamber is not None:
            subset = chip.sel(mark_col=chamber[0], mark_row=chamber[1])
            data = subset.roi.to_numpy()
            if imscale is None:
                imscale = 2
            title=f'Chamber {chamber}'
            if tag is None:
                title = f'{title} ({subset.tag.values})'
            x = subset.x.data
            y = subset.y.data
            xs = x - len(subset.roi_x), x + len(subset.roi_x)
            ys = y - len(subset.roi_y), y + len(subset.roi_y)
            
            # No need to raster the small image
            rastered = False
            
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
            invert_yaxis=True,
            xlabel='',
            ylabel='',
            colorbar=True,
            colorbar_opts=dict(title='intensity'),
        )

        if rastered:
            p = rasterize(p)

        if chamber_mask and chamber is not None:
            p = p*show_mask(chip, chamber, imscale).opts(framewise=True)

        plots.append(p)

    if tag:
        p = hv.Layout(
            plots
        ).opts(
            title=f'Tag: {tag}',
            shared_axes=False
        ).cols(3)

    return p


def chip_hm(
    data: xr.Dataset | pd.DataFrame,
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
    data :
        Data container. Could be the chip object from 
        magnify.microfluidic_chip_pipe or a DataFrame.
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

    if isinstance(data, xr.Dataset):
        df = to_df(data, value, agg_func, mask)
    else:
        df = data

    p = hv.HeatMap(
        df,
        kdims=xy,
        vdims=[value, *attrs],
    ).opts(
        invert_yaxis=True,
        invert_xaxis=True,
        frame_width=int(300*scale),
        frame_height=int(150*3.25*scale),
        colorbar=True,
        colorbar_opts=dict(title=value),
    )

    return p


def chamber_map(
    chip: xr.Dataset,
    imscale: float = None,
    limits: None | tuple[int, int] = None,
    chamber_mask: bool = True,
    time_dim: None | str = None,
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
    chamber_mask :
        Whether to show the mask over the chamber.
    time_dim :
        The name of the dimension that indicates the time.

    Returns
    -------
    dmap :
        hv.DynamicMap object of all chambers (accessed by sliders).

    Examples
    --------
    >>> htmek.viz.chamber_map(chip)
    """
    if time_dim is None:
        time_dim = 'time'
    else:
        if time_dim not in chip.dims:
            raise ValueError(f'Time dimension `{time_dim}` not found.')

    kdims = []
    
    if time_dim in chip.dims:
        def timecourse(time, col, row):
            p = view(
                chip.sel({time_dim: time}),
                chamber = (col, row),
                imscale = imscale,
                limits = limits,
                chamber_mask = chamber_mask,
            )

            return p
        
        f = timecourse

        kdims.append(hv.Dimension('time', values=chip[time_dim].values))

    else:
        def chamber(col, row):
            p = view(
                chip,
                chamber = (col, row),
                imscale = imscale,
                limits = limits,
                chamber_mask = chamber_mask,
            )

            return p
        
        f = chamber

    kdims.append(hv.Dimension('col', values=range(32)))
    kdims.append(hv.Dimension('row', values=range(56)))

    dmap = hv.DynamicMap(f, kdims=kdims)
    dmap.opts(framewise=True)

    return dmap

#########################
# General fitting
#########################
def plot_fit(
    xs,
    ys,
    model,
    popt,
    x_label = 'x',
    y_label = 'y',
    col = None,
    row = None,
    tag = None,
):
    """Plot a fit given data (xs, ys), model for data, and popt from
    a previous fit.

    TODO: Write a better docstring...
    """
    xs_full = np.linspace(0, np.max(xs)*1.1, 1000)

    p_data = hv.Scatter(
        (xs, ys, row, col, tag),
        kdims = x_label,
        vdims = [y_label, 'row', 'col', 'tag'],
    )

    if not np.isnan(np.sum(popt)):
        p_fit = hv.Curve(
            (xs_full, model(xs_full, *popt), row, col, tag),
            kdims = x_label,
            vdims = [y_label, 'row', 'col', 'tag'],
        )
        p = p_fit*p_data
    else:
        # Need to add empty plot or else it won't show in DynamicMap
        # with other overlay plots.
        p_empty = hv.Curve(
            (np.min(xs), np.min(ys))
        )
        p = p_data*p_empty

    return p


def fit_map(
    df: pd.DataFrame,
    func,
    x: str,
    y: str,
    z: None | str = None,
    col: str = 'mark_col',
    row: str = 'mark_row',
    tag: str = 'tag',
    fit_dict: None | dict = None,
):
    """Creates a hv.DynamicMap to display fits across all chambers.

    Parameters
    ----------
    df :
        A tidy pandas DataFrame containing PBP fluorescence data (y) across
        multiple concentrations (x) for all chambers (mark_row, mark_col).
    func :
        Function related to the fit. If no fit_dict is passed, it should be
        the function used to fit the data (e.g., that uses curve_fit) and
        must return the modeling function as the third argument. If you
        pass a fit_dict, it should be the modeling function itself (the
        function passed to curve_fit).
    x :
        Name of the x axis (e.g., standard concentration).
    y :
        Name of the y axis (e.g., fluorescence signal).
    z :
        Optional extra dimension (e.g., time, [substrate]).
    col :
        Name of df column containing column information ('mark_col' from
        magnify).
    row :
        Name of df column containing row information ('mark_row' from
        magnify).
    tag :
        Name of df column containing tag information ('tag' from magnify).
    fit_dict :
        Optionally add a dictionary of fit popt/pcovs. This function is
        generally very fast to compute and plot, but in some instances
        a dictionary lookup will be noticably faster.
    
    Returns
    -------
    dmap :
        hv.DynamicMap of fits for all chambers.
    """
    cols = df[col].unique()
    rows = df[row].unique()

    indexers = [col, row]
    if z is not None:
        zs = df[z].unique()
        indexers.append(z)

    indexed_df = df.set_index(indexers).sort_index().copy()

    def fit(*args):
        # Decide if doing two or three dimensions
        if len(args) == 2:
            col_val, row_val = args
        else:
            col_val, row_val, z_val = args
        
        # Subset dataframe
        sub_df = indexed_df.loc[*args].copy()

        xs, ys = sub_df[x].values, sub_df[y].values
        tag_value = sub_df[tag].values[0]

        if fit_dict is None:
            fit_func = func
            popt, pcov, model = fit_func(xs, ys)
        else:
            popt, pcov = fit_dict[row_val, col_val]
            model = func
        ylim = (0, df[y].max())

        p = plot_fit(
            xs,
            ys,
            model,
            popt,
            x,
            y,
            col_val,
            row_val,
            tag_value,
        ).opts(ylim=ylim)

        return p

    col_dim = hv.Dimension('column', values=cols)
    row_dim = hv.Dimension('row', values=rows)
    kdims = [col_dim, row_dim]

    if z is not None:
        z_dim = hv.Dimension(z, values=zs)
        kdims.append(z_dim)

    dmap = hv.DynamicMap(fit, kdims=kdims)

    return dmap


def sample_fit_map(
    fit_map: hv.DynamicMap,
    n: int = 100,
    col: str = 'mark_col',
    row: str = 'mark_row',
    z = None,
    point_alpha: float = 0.3,
    line_alpha: float = 0.1,
) -> hv.Overlay:
    """Samples a fit_map to show a sample of the data"""
    plots = []

    # Get random sample
    kdims = fit_map.kdims

    if len(kdims) > 2:
        if z is not None:
            if z not in kdims[2].values:
                raise ValueError(
                    f'z arg `{z}` not found in kdim `{kdims[2]}`.'
                )
        combos = [
            (i, j, k)
            for i in kdims[0].values
            for j in kdims[1].values
            for k in kdims[2].values
        ]
    else:
        combos = [
            (i, j)
            for i in kdims[0].values
            for j in kdims[1].values
        ]

    sample = random.sample(combos, k=n)

    # Select and overlay sampled plots
    for args in sample:

        p = fit_map[*args]

        plots.append(p.opts({
            'Scatter': dict(
                color='#1f77b4',
                line_alpha=point_alpha,
                fill_alpha=point_alpha,
            ),
            'Curve': dict(alpha=line_alpha),
        }))

    return hv.Overlay(plots)

######################
# Standards plotting
######################

def plot_PBP(
    P_is,
    RFUs,
    popt,
    row = None,
    col = None,
):
    """Plot a PBP isotherm fit given data (P_is, RFUs) and popt from
    htmek.assays.standards.fit_PBP.
    """
    xs = np.linspace(0, np.max(P_is)*1.1, 1000)

    p_data = hv.Scatter(
        (P_is, RFUs, row, col),
        kdims = '[Pi] (µM)',
        vdims = ['RFU', 'mark_row', 'mark_col'],
    )

    if 'Failed' not in popt:
        p_fit = hv.Curve(
            (xs, PBP_isotherm(xs, *popt), row, col),
            kdims = '[Pi] (µM)',
            vdims = ['RFU', 'mark_row', 'mark_col'],
        )
        p = p_fit*p_data
    else:
        p = p_data

    return p_fit*p_data


def PBP_map(
    df: pd.DataFrame,
    row: str = 'mark_row',
    col: str = 'mark_col',
    x: str = '[Pi] (µM)',
    y: str = 'roi',
    fit_dict: None | dict = None,
):
    """Creates a hv.DynamicMap to display PBP fits across all chambers.

    Parameters
    ----------
    df :
        A tidy pandas DataFrame containing PBP fluorescence data (y) across
        multiple concentrations (x) for all chambers (mark_row, mark_col).
    row :
        Name of df column containing row information ('mark_row' from
        magnify.
    col :
        Name of df column containing column information ('mark_col' from
        magnify.
    x :
        Name of the x axis (standard concentration). Defaults to '[Pi] (µM)'.
    y :
        Name of the y axis (fluorescence signal). Defaults to 'roi', but
        is plotted as RFU by viz.plot_PBP.
    fit_dict :
        Optionally add a dictionary of fit popt/pcovs. This function is
        generally very fast to compute and plot, but in some instances
        a dictionary lookup will be noticably faster.
    
    Returns
    -------
    dmap :
        hv.DynamicMap of PBP fits for all chambers.
    
    """
    def fit(mark_col, mark_row):
        sub_df = df[(df[row]==mark_row) & (df[col]==mark_col)].copy()

        P_is, RFUs = sub_df[x].values, sub_df[y].values
        if fit_dict is None:
            popt, pcov = fit_PBP(P_is, RFUs)
        else:
            popt, pcov = fit_dict[mark_row, mark_col]
        ylim = (0, df[y].max())

        return plot_PBP(P_is, RFUs, popt, mark_row, mark_col).opts(ylim=ylim)

    mark_col = hv.Dimension('column', values=range(32))
    mark_row = hv.Dimension('row', values=range(56))

    dmap = hv.DynamicMap(fit, kdims=[mark_col, mark_row])

    return dmap

def plot_all_std_curves(standards_df):
    ax = plt.subplot()
    n_chambers = 200
    for i in range(n_chambers):
        row, col = np.random.choice(standards_df['mark_row'].unique()), np.random.choice(standards_df['mark_col'].unique())
        plotting_dat = standards_df[(standards_df['mark_col'] == col) & (standards_df['mark_row'] == row)]
        ax.scatter(plotting_dat.standard_conc, plotting_dat.median_intensity, color='blue', alpha=0.1)
        ax.set_title(f"Standard curves from {n_chambers} chambers")

def plot_sample_progress_curves(assays_df):
    dat = assays_df.dropna()

    fig, ax = plt.subplots(figsize=(3,3),dpi=200)
    mutants = []
    for i in range(4):
        row, col = np.random.choice(dat['mark_row'].unique()), np.random.choice(dat['mark_col'].unique())

        substrate_conc = 50
        plotting_dat = dat[(dat['mark_col'] == col) & (dat['mark_row'] == row) & (dat['substrate_conc'] == substrate_conc)]
        seconds, rois = plotting_dat.acq_time, plotting_dat.product_conc

        if not np.isnan(plotting_dat.A.iloc[0]):
            xlin = np.arange(0,2000,1)
            A, k, y0 = plotting_dat.A.iloc[0], plotting_dat.k.iloc[0], plotting_dat.y0.iloc[0],
            ax.plot(xlin, single_exponential(xlin,A,k,y0))

        mutantID = plotting_dat['tag'].iloc[0]

        ax.scatter(seconds, rois, label=f"{mutantID} ({row},{col})")
        mutants.append(mutantID)

    plt.xlabel("Time (s)")
    plt.ylabel("Pi Conc (uM)")
    plt.title(f"{substrate_conc}uM BzP")
    plt.legend(title='Variant', bbox_to_anchor=(1,1), loc="upper left")
    plt.show()

def compare_sample_progress_curves_vs_MM(expression_df, assays_df):
    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    row, col = expression_df[(expression_df['tag'] == 'Y11F')][['mark_row', 'mark_col']].iloc[0]
    row, col = np.random.randint(56), np.random.randint(32)

    dat = assays_df[(assays_df['mark_col'] == col) & (assays_df['mark_row'] == row)]

    # Define a color dictionary for substrate concentrations
    substrate_colors = {s: plt.cm.viridis(i / len(dat['substrate_conc'].unique())) for i, s in enumerate(sorted(dat['substrate_conc'].unique()))}

    for s in dat['substrate_conc'].unique():
        sub_dat = dat[dat['substrate_conc'] == s]
        times = sub_dat['acq_time']
        product_concs = sub_dat['product_conc']
        A = sub_dat['A'].iloc[0]
        k = sub_dat['k'].iloc[0]
        y0 = sub_dat['y0'].iloc[0]
        
        axs[0].scatter(times, product_concs, alpha=0.5, color=substrate_colors[s])
        axs[0].plot(times, single_exponential(times, A, k, y0), label=f"[S] = {s} µM", color=substrate_colors[s])

    axs[0].set_ylabel("[Product] (uM)")
    axs[0].set_xlabel("Time (s)")

    kcat = dat['kcat'].iloc[0]
    KM = dat['KM'].iloc[0]
    vmax = dat['vmax'].iloc[0]
    kcat_over_KM = dat['kcat_over_KM'].iloc[0]

    xlin = np.arange(0, max(substrate_colors), 1)

    axs[1].scatter(dat['substrate_conc'], dat['init_rate'], alpha=0.5, color=[substrate_colors[s] for s in dat['substrate_conc']])
    axs[1].plot(xlin, michaelis_menten(xlin, vmax, KM), label=f"KM = {KM:.2f}\nvmax = {vmax:.3f}")
    axs[1].set_ylabel("Initial Rate")
    axs[1].set_xlabel("[Substrate] (uM)")
    axs[1].legend()
    axs[1].set_ylim(-0.01, 0.2)
    axs

    fig.suptitle(f"{dat['tag'].iloc[0]} ({row},{col})\nkcat/KM: {kcat_over_KM:,.0f}")
    plt.tight_layout()

def plot_kcat_KM_by_sequence_boxplot(assays_df):
    dat = assays_df.copy()

    dat['position'] = dat['tag'].apply(lambda x: int(x[1:-1]) if x[1:-1].isdigit() else x)
    dat['library'] = dat['tag'].apply(lambda x: x[-1] if x[1:-1].isdigit() else x)
    dat['log_kcat_over_KM'] = np.log10(dat['kcat_over_KM'])
    dat = dat.drop_duplicates(subset=['mark_col', 'mark_row'])

    dat.sort_values(by=['position', 'library'], inplace=True)

    def hook(plot, element):
        plot.handles['x_range'].factors = [str(value) for value in range(1,99)]

    # Create a Holoviews boxplot
    boxplot = hv.BoxWhisker(dat, kdims=['tag'], vdims=['log_kcat_over_KM', 'library'], label='kcat/KM Boxplot')
    boxplot.opts(
        width=1200,
        height=400,
        xlabel='Tag',
        ylabel='kcat/KM',
        title='kcat/KM Boxplot',
        xrotation=45,
        tools=['hover'],
        # hooks=[hook]
    )

    # Overlay points as scatter
    scatter = hv.Scatter(dat, kdims=['tag'], vdims=['log_kcat_over_KM', 'mark_row', 'mark_col', 'EnzymeConc', 'library'], label='kcat/KM Scatter')
    scatter.opts(
        size=5,
        alpha=0.5,
        color='red',
        tools=['hover'],
        # hooks=[hook]
    )

    # Combine boxplot and scatter
    overlay = boxplot * scatter
    
    return overlay