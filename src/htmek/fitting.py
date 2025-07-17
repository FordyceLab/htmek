import numpy as np
import pandas as pd
from collections import namedtuple
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import copy

#######################
# Fitting Functions
#######################
def linear(
    x: float | ArrayLike,
    slope: float,
    y0: float,
) -> float | ArrayLike:
    """Linear fit parameters for arbitrary data."""
    return slope*x+y0

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

#####################################################
# Collection of useful defaults for fitting functions
#####################################################
# Make a default namedtuple object
fit_parameters = namedtuple(
    'fit_parameters',
    [   
        # Name of the function
        'name',
        # Function itself
        'func',
        # Names of each parameter in the fit
        'param_names',
        # Initial guesses for each parameter (see scipy.optimize.curve_fit)
        'p0',
        # Bounds for each parameter (see scipy.optimize.curve_fit)
        'param_bounds',
        # Limits on the x data used, can be None, int, or tuple, and can
        # contain np.infs. Single int is assumed to be upper limit (inclusive).
        'xlimits',
        # Limits on the y data used, same as x
        'ylimits',
    ]
)

# Full PBP isotherm
fit_PBP_isotherm = fit_parameters(
    'PBP_isotherm',
    PBP_isotherm,
    param_names = ['A', 'KD', 'PS', 'I0'],
    p0 = [100, 0.1, 100, 1000],
    param_bounds = ([0]*4, [np.inf]*4),
    xlimits = None,
    ylimits = None,
)

# Linear PBP fit for limited data
fit_PBP_linear = fit_parameters(
    'linear',
    linear,
    param_names = ['slope', 'I0'],
    p0 = [20, 1000],
    param_bounds = ([0]*2, [np.inf]*2),
    xlimits = 50,
    ylimits = None,
)

# RFU progress curves
fit_RFU_expon = fit_parameters(
    'exponential',
    exponential,
    param_names = ['A', 'k', 'I0'],
    p0 = [6000, 0.1, 1000],
    param_bounds = ([0]*3, [np.inf]*3),
    xlimits = None,
    ylimits = None,
)

# RFU progress curve linear fit
fit_RFU_linear = fit_parameters(
    'linear',
    linear,
    param_names = ['slope', 'I0'],
    p0 = [10, 1000],
    param_bounds = ([0]*2, [np.inf]*2),
    xlimits = None,
    ylimits = None,
)

fit_kinetics_expon = fit_parameters(
    'exponential',
    exponential,
    param_names = ['A', 'k', 'P0'],
    p0 = [100, 0.001, 1],
    param_bounds = ([0]*3, [np.inf]*3),
    xlimits = None,
    ylimits = None,
)

fit_kinetics_linear = fit_parameters(
    'linear',
    linear,
    param_names = ['slope', 'P0'],
    p0 = [0.1, 1],
    param_bounds = ([0]*2, [np.inf]*2),
    xlimits = None,
    ylimits = None,
)


class ChamberFit():
    """Fit a chamber given data and fit parameters.

    Parameters:
    -----------
    xs : ArrayLike
        The x data for the fit
    ys : ArrayLike
        The y data for the fit
    fit_params : namedtuple or list of namedtuples
        The functions and parameters to fit, in order of preference.
        Should match the format of `fit_parameters` namedtuple object in
        this module.
    autoskip: bool
        Whether to try to fit at all. Useful for consistency when processing
        all chambers equivalently after flagging some to be skipped.
    """
    def __init__(
        self,
        xs: ArrayLike,
        ys: ArrayLike,
        fit_params,
        r2_thresh: float = 0.8,
        autoskip: bool = False,
    ):
        self.xs = xs
        self.ys = ys
        self.r2_thresh = r2_thresh
        self.autoskip = autoskip

        # _init_fit_params is a list of all initial fit params
        self._init_fit_params = fit_params
        if not isinstance(self._init_fit_params, list):
            self._init_fit_params = [self._init_fit_params]

        # Instantiate all other parameters
        self.success = None
        self.fit_type = None
        self.params = None
        self.fit_data = (None, None)
        self.popt = None
        self.pcov = None
        self.perr = None
        self.r2 = None
        self.fit_params = None
        if self.autoskip:
            return

        # TODO: Check the fit parameters? Same length, etc.

        # TODO: Check x and y data? Need to not include nans!

        # There's got to be a better way to do this lol
        # Need to add properties, I think
        for fit_params in self._init_fit_params:
            chamber_fit = self.fit(fit_params)
            self.success = chamber_fit.success
            if self.success:
                self.fit_type = chamber_fit.fit_params.name
                self.params = chamber_fit.fit_params.param_names
                self.fit_data = chamber_fit.fit_data
                self.popt = chamber_fit.popt
                self.pcov = chamber_fit.pcov
                self.perr = chamber_fit.perr
                self.r2 = chamber_fit.r2
                self.fit_params = chamber_fit.fit_params
                break


    def __repr__(self):
        if self.success:
            repr_text = f"""ChamberFit:
  fit_type: {self.fit_params.name}
  params: {'\t'.join(self.params)}
  popt: {'\t'.join([str(np.round(v, 2)) for v in self.popt])}
  perr: {'\t'.join([str(np.round(v, 2)) for v in self.perr])}
  r2: {np.round(self.r2, 6)}
  data: fit_xs: {list(self.fit_data[0])}, fit_ys: {list(self.fit_data[1])}"""
        elif self.autoskip:
            repr_text = """ChamberFit: Not attempted."""
        else:
            repr_text = """ChamberFit: No successful fit(s)."""
        return repr_text


    def fit(self, fit_params):
        """Try a fitting function."""
        # Copy self
        chamber_fit = copy.copy(self)

        # Format data to fit based on limits
        fit_xs = self.xs[:]
        fit_ys = self.ys[:]

        # TODO: break these into hidden functions
        if fit_params.xlimits is not None:
            xlimits = fit_params.xlimits
            if isinstance(xlimits, (int, float)):
                xlimits = (-np.inf, xlimits)
            mask = (xlimits[0] <= self.xs) & (self.xs <= xlimits[1])
            fit_xs = self.xs[mask]
            fit_ys = self.ys[mask]
        
        if fit_params.ylimits is not None:
            ylimits = fit_params.ylimits
            if isinstance(ylimits, (int, float)):
                ylimits = (-np.inf, ylimits)
            mask = (ylimits[0] <= self.xs) & (self.xs <= ylimits[1])
            fit_xs = self.xs[mask]
            fit_ys = self.ys[mask]

        # Attempt to fit
        try:
            popt, pcov = curve_fit(
                fit_params.func,
                fit_xs,
                fit_ys,
                p0 = fit_params.p0,
                bounds = fit_params.param_bounds,
            )
            success = True

        except (ValueError, RuntimeError):
            success = False
            n = len(fit_params.param_names)
            popt = np.empty(n)*np.nan
            pcov = np.empty((n,n))*np.nan

        perr = np.sqrt(np.diag(pcov))

        # Calculate r2
        if success:
            r2 = r2_score(fit_ys, fit_params.func(fit_xs, *popt))
            # Check if passes
            if r2 < self.r2_thresh:
                success = False
        else:
            r2 = np.nan

        # Update attributes and return object
        chamber_fit.success = success
        chamber_fit.fit_params = fit_params
        chamber_fit.fit_data = fit_xs, fit_ys
        chamber_fit.popt = popt
        chamber_fit.pcov = pcov
        chamber_fit.perr = perr
        chamber_fit.r2 = r2

        return chamber_fit