"""One-compartment IV bolus pharmacokinetic model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class OneCompartmentFit:
    """Estimated parameters for a one-compartment IV bolus model.

    Attributes:
        c0: Concentration at time zero, in the same units as the input data.
        ke: First-order elimination rate constant, in inverse time units.
        covariance: Covariance matrix returned by scipy.optimize.curve_fit.
    """

    c0: float
    ke: float
    covariance: np.ndarray


@dataclass(frozen=True)
class GoodnessOfFit:
    """Diagnostics for how well a fitted model matches observed data.

    Attributes:
        r_squared: Coefficient of determination (1 - SS_res/SS_tot). Closer to
            1.0 indicates the model explains more of the variance in the
            observed data. Can be negative for a very poor fit.
        aic: Akaike Information Criterion for the fit (lower is better when
            comparing models fit to the same data). Undefined (NaN) for a
            residual sum of squares of ~0, which only occurs with noiseless
            synthetic data.
        residuals: Observed minus predicted concentration at each time point.
        se_c0: Standard error of the fitted C0, from the fit covariance matrix.
        se_ke: Standard error of the fitted ke, from the fit covariance matrix.
    """

    r_squared: float
    aic: float
    residuals: np.ndarray
    se_c0: float
    se_ke: float


def concentration_time_model(
    time: np.ndarray | float,
    c0: float,
    ke: float,
) -> np.ndarray | float:
    """Return concentration for a one-compartment IV bolus model.

    The model assumes instantaneous distribution and first-order elimination:
    C(t) = C0 * exp(-ke * t).
    """

    return c0 * np.exp(-ke * time)


def fit_one_compartment_model(
    time: np.ndarray,
    concentration: np.ndarray,
) -> OneCompartmentFit:
    """Fit C(t) = C0 * exp(-ke * t) to concentration-time data.

    Args:
        time: Increasing sampling times.
        concentration: Non-negative plasma concentrations.

    Returns:
        Estimated C0, ke, and covariance matrix.

    Raises:
        ValueError: If the data are insufficient for nonlinear fitting.
        RuntimeError: If scipy cannot estimate model parameters.
    """

    if time.size < 3:
        raise ValueError("At least three concentration-time points are required for model fitting.")

    positive = concentration > 0
    if positive.sum() < 3:
        raise ValueError("At least three positive concentrations are required for model fitting.")

    initial_c0 = float(max(concentration[0], concentration.max()))
    terminal_time = time[positive]
    terminal_concentration = concentration[positive]

    # Estimate an initial ke from a log-linear slope to help nonlinear fitting converge.
    slope, _intercept = np.polyfit(terminal_time, np.log(terminal_concentration), 1)
    initial_ke = float(max(-slope, 1e-6))

    bounds = ([0.0, 0.0], [np.inf, np.inf])
    popt, pcov = curve_fit(
        concentration_time_model,
        time,
        concentration,
        p0=[initial_c0, initial_ke],
        bounds=bounds,
        maxfev=10_000,
    )

    return OneCompartmentFit(c0=float(popt[0]), ke=float(popt[1]), covariance=pcov)


def evaluate_goodness_of_fit(
    time: np.ndarray,
    concentration: np.ndarray,
    fit: OneCompartmentFit,
) -> GoodnessOfFit:
    """Compute fit diagnostics for a fitted one-compartment model.

    Args:
        time: The same time values used to produce `fit`.
        concentration: The same observed concentrations used to produce `fit`.
        fit: Result of `fit_one_compartment_model`.

    Returns:
        GoodnessOfFit with R-squared, AIC, residuals, and parameter standard
        errors.
    """

    predicted = concentration_time_model(time, fit.c0, fit.ke)
    residuals = concentration - predicted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((concentration - np.mean(concentration)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n = time.size
    k = 2  # number of fitted parameters: C0, ke
    aic = float(n * np.log(ss_res / n) + 2 * k) if ss_res > 0 else float("nan")

    standard_errors = np.sqrt(np.diag(fit.covariance))

    return GoodnessOfFit(
        r_squared=r_squared,
        aic=aic,
        residuals=residuals,
        se_c0=float(standard_errors[0]),
        se_ke=float(standard_errors[1]),
    )
