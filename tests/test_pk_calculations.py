"""Tests for noncompartmental calculations and one-compartment model fitting.

Drop this file in tests/test_pk_calculations.py. Run with:
    pytest tests/ -v

Reference values here were computed independently (numpy/scipy directly) and
verified before being hardcoded, not copied from the app's own output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calculator import (
    calculate_auc,
    calculate_auc_extrapolated,
    calculate_pk_parameters,
    estimate_terminal_ke,
    validate_concentration_data,
    AUC_EXTRAPOLATION_RELIABILITY_THRESHOLD_PCT,
)
from models.one_compartment import (
    concentration_time_model,
    evaluate_goodness_of_fit,
    fit_one_compartment_model,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data with known, exact analytical answers
# ---------------------------------------------------------------------------

KE_TRUE = 0.2
C0_TRUE = 100.0


def exact_exponential_short():
    """Short sampling window: terminal phase is under-characterized on purpose."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    concentration = C0_TRUE * np.exp(-KE_TRUE * time)
    return time, concentration


def exact_exponential_long():
    """Longer sampling window, well into the terminal phase."""
    time = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    concentration = C0_TRUE * np.exp(-KE_TRUE * time)
    return time, concentration


def noisy_exponential_long():
    """Same long window with small deterministic perturbation, for GOF tests."""
    time, clean = exact_exponential_long()
    noise = np.array([0.5, -0.3, 0.4, -0.2, 0.15, -0.1, 0.05, -0.02])
    return time, clean + noise


# ---------------------------------------------------------------------------
# calculate_auc
# ---------------------------------------------------------------------------

def test_calculate_auc_matches_hand_computed_trapezoid():
    # Two trapezoids: (10+20)/2*1 + (20+10)/2*1 = 15 + 15 = 30
    time = np.array([0.0, 1.0, 2.0])
    concentration = np.array([10.0, 20.0, 10.0])
    assert calculate_auc(time, concentration) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# estimate_terminal_ke
# ---------------------------------------------------------------------------

def test_estimate_terminal_ke_recovers_known_rate_constant():
    time, concentration = exact_exponential_short()
    ke = estimate_terminal_ke(time, concentration, terminal_points=3)
    assert ke == pytest.approx(KE_TRUE, rel=1e-6)


# ---------------------------------------------------------------------------
# calculate_auc_extrapolated / reliability flag
# ---------------------------------------------------------------------------

def test_auc_extrapolation_flags_undersampled_terminal_phase():
    time, concentration = exact_exponential_short()
    auc_last = calculate_auc(time, concentration)
    auc_inf, tail, pct, reliable = calculate_auc_extrapolated(
        time, concentration, KE_TRUE, auc_last=auc_last
    )
    assert auc_last == pytest.approx(317.113112, rel=1e-6)
    assert tail == pytest.approx(183.939721, rel=1e-6)
    assert auc_inf == pytest.approx(501.052833, rel=1e-6)
    assert pct == pytest.approx(36.710644, rel=1e-6)
    assert pct > AUC_EXTRAPOLATION_RELIABILITY_THRESHOLD_PCT
    assert reliable is False


def test_auc_extrapolation_passes_for_well_sampled_terminal_phase():
    time, concentration = exact_exponential_long()
    auc_last = calculate_auc(time, concentration)
    auc_inf, tail, pct, reliable = calculate_auc_extrapolated(
        time, concentration, KE_TRUE, auc_last=auc_last
    )
    assert pct == pytest.approx(1.783789, rel=1e-4)
    assert pct <= AUC_EXTRAPOLATION_RELIABILITY_THRESHOLD_PCT
    assert reliable is True


def test_auc_extrapolation_rejects_nonpositive_ke():
    time, concentration = exact_exponential_short()
    with pytest.raises(ValueError):
        calculate_auc_extrapolated(time, concentration, ke=0.0)


def test_auc_extrapolation_rejects_zero_last_concentration():
    time = np.array([0.0, 1.0, 2.0])
    concentration = np.array([10.0, 5.0, 0.0])
    with pytest.raises(ValueError):
        calculate_auc_extrapolated(time, concentration, ke=KE_TRUE)


# ---------------------------------------------------------------------------
# calculate_pk_parameters (end-to-end)
# ---------------------------------------------------------------------------

def test_calculate_pk_parameters_end_to_end_known_values():
    time, concentration = exact_exponential_long()
    data = pd.DataFrame({"time": time, "concentration": concentration})
    dose = 1000.0

    result = calculate_pk_parameters(data, dose=dose)

    assert result.cmax == pytest.approx(C0_TRUE)
    assert result.tmax == pytest.approx(0.0)
    assert result.ke == pytest.approx(KE_TRUE, rel=1e-6)
    assert result.half_life == pytest.approx(np.log(2) / KE_TRUE, rel=1e-6)
    # vd = dose / cmax = 1000 / 100 = 10; clearance = ke * vd = 0.2 * 10 = 2
    assert result.vd == pytest.approx(10.0, rel=1e-6)
    assert result.clearance == pytest.approx(2.0, rel=1e-6)
    assert result.auc_extrapolation_reliable is True


def test_calculate_pk_parameters_rejects_nonpositive_dose():
    time, concentration = exact_exponential_long()
    data = pd.DataFrame({"time": time, "concentration": concentration})
    with pytest.raises(ValueError):
        calculate_pk_parameters(data, dose=0.0)


# ---------------------------------------------------------------------------
# validate_concentration_data
# ---------------------------------------------------------------------------

def test_validate_rejects_missing_column():
    data = pd.DataFrame({"time": [0, 1, 2], "conc": [10, 5, 2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_concentration_data(data)


def test_validate_rejects_non_increasing_time():
    data = pd.DataFrame({"time": [0, 2, 1], "concentration": [10, 5, 2]})
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_concentration_data(data)


def test_validate_rejects_negative_concentration():
    data = pd.DataFrame({"time": [0, 1, 2], "concentration": [10, -5, 2]})
    with pytest.raises(ValueError, match="non-negative"):
        validate_concentration_data(data)


def test_validate_rejects_fewer_than_three_observations():
    data = pd.DataFrame({"time": [0, 1], "concentration": [10, 5]})
    with pytest.raises(ValueError, match="At least three"):
        validate_concentration_data(data)


def test_validate_rejects_non_numeric_concentration():
    data = pd.DataFrame({"time": [0, 1, 2], "concentration": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="numeric"):
        validate_concentration_data(data)


# ---------------------------------------------------------------------------
# fit_one_compartment_model
# ---------------------------------------------------------------------------

def test_fit_recovers_known_parameters_on_noiseless_data():
    time, concentration = exact_exponential_long()
    fit = fit_one_compartment_model(time, concentration)
    assert fit.c0 == pytest.approx(C0_TRUE, rel=1e-4)
    assert fit.ke == pytest.approx(KE_TRUE, rel=1e-4)


def test_fit_rejects_fewer_than_three_points():
    time = np.array([0.0, 1.0])
    concentration = np.array([100.0, 80.0])
    with pytest.raises(ValueError):
        fit_one_compartment_model(time, concentration)


# ---------------------------------------------------------------------------
# evaluate_goodness_of_fit
# ---------------------------------------------------------------------------

def test_goodness_of_fit_on_noisy_data_matches_reference():
    time, concentration = noisy_exponential_long()
    fit = fit_one_compartment_model(time, concentration)
    gof = evaluate_goodness_of_fit(time, concentration, fit)

    assert fit.c0 == pytest.approx(100.27380078, rel=1e-5)
    assert fit.ke == pytest.approx(0.20068359, rel=1e-5)
    assert gof.r_squared == pytest.approx(0.9999554133, rel=1e-5)
    assert gof.aic == pytest.approx(-18.902372379, rel=1e-4)
    assert gof.se_c0 == pytest.approx(0.22020975, rel=1e-4)
    assert gof.se_ke == pytest.approx(0.00107055, rel=1e-3)
    assert gof.residuals.shape == time.shape


def test_goodness_of_fit_r_squared_near_one_for_clean_data():
    time, concentration = exact_exponential_long()
    fit = fit_one_compartment_model(time, concentration)
    gof = evaluate_goodness_of_fit(time, concentration, fit)
    assert gof.r_squared > 0.999
