# htmek
General utilities for HT-MEK data processing with [`magnify`](https://github.com/FordyceLab/magnify).

### Installation
Install pip, then pip install magnify and PyQt5
```
conda install pip
pip install magnify pyqt5 scikit-image
```
Next, clone and install `htmek` from this repository
```
git clone https://github.com/FordyceLab/htmek.git
cd htmek
pip install -e .
```

## Processing Pipelines
### Buttons
#### Subtract egfp background
```
htmek.utils.subtract_bg(
    path/to/final_egfp_image,
    path/to/background_egfp_image,
    overlap = 82, # use 82 for 0.08 overlap, 102 for 0.1 overlap
    output = 'egfp_bg_sub.tif'
)
```

#### Button finding and analysis
```
# Make a chip object
egfp_chip = htmek.buttons_pipe(
    'egfp_bg_sub.tif',
    path/to/pinlist,
    overlap = 0, # assuming using pre-stitched bg image from above
)

# Quickly view the chip interactively
htmek.viz.view(egfp_chip, limits=(1000, 10000))

# Look at each chamber individually with mask
htmek.viz.chamber_map(egfp_chip, limits=(1000, 16000))

# Look at all chambers for one variant/pinlist tag
htmek.viz.chamber_map(egfp_chip, tag='N41A', limits=(1000, 16000))

# Make a heatmap of all summed button intensitites
htmek.viz.chip_hm(egfp_chip, agg_func='sum')
```

#### Get dataframe of summed egfp intensity
```
# Convert to dataframe
egfp_df = htmek.utils.to_df(egfp_chip, agg_func='sum')

# Rename generic 'roi' column
egfp_df = egfp_df.rename({'roi': 'summed_egfp'}, axis=1)

# Export to csv
egfp_df.to_csv('egfp_data.csv', index=False)
```

### Checking assay progress curves
It can be beneficial to check progress curves as they are collected to ensure that the data look as expected/high quality, prior to collected stanard curves to make the results more quantitative. These pipelines are similar to those used for analyzing standards and properly analyzing kinetics data below.

## Fitting process
Fitting data using `htmek` requires a `fit_parameters` variable, which is just a special `namedtuple` contaning information about how to fit the data and what exactly is being fit.

The main goal of this is to create _flexible_ but _well-defined_ processes by which we can fit data and retain information about how the function was fit.

A simple example for fitting a line is given below:
```
# First define the function you are fitting
def linear(x, m, b):
    """Fits a line given the slope and y-intercept."""
    return m*x+b

# Pack this into `fit_parameters` with additonal info
linear_fit_params = fit_parameters(
    'linear',
    linear,
    param_names = ['m', 'b'],
    p0 = [1, 0], 
    param_bounds = ([0]*2, [np.inf]*2),
    xlimits = 50,
    ylimits = None,
)

# Fit the data
fit_dict, fit_map = htmek.process.fit(
    df = df,
    fit_params = [linear_fit_params],
    x_label = 'time (s)',
    y_label = 'median intensity',
)
```

Description of each parameter:
```
fit_params = fit_parameters(
    name: name of the function
    func: the function itself
    param_names: names for each fit parameter
    p0: initial guesses for each parameter (see scipy.optimize.curve_fit)
    param_bounds: ounds for each parameter (see scipy.optimize.curve_fit)
    xlimits: limits on the x-data used for the fit. Can be None, int (upper limit), or tuple (lower, upper), and can contain np.infs.
    ylimits: same as xlimits, but for y-data.
)
