"""Core data loading, validation, and noncompartmental PK calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("time", "concentration")

# Per standard NCA convention (e.g. FDA/EMA bioequivalence guidance), AUC0-inf
# is considered unreliable for dosing/exposure decisions when more than 20% of
# it is extrapolated beyond the last observed point rather than measured.
AUC_EXTRAPOLATION_RELIABILITY_THRESHOLD_PCT = 20.0


@dataclass(frozen=True)
class PKParameters:
    """Calculated pharmacokinetic parameters.

    Attributes:
        cmax: Maximum observed concentration.
        tmax: Time at which Cmax is observed.
        auc: Area under the observed concentration-time curve (AUC0-last), by
            the linear trapezoidal rule.
        ke: First-order elimination rate constant.
        half_life: Elimination half-life, calculated as ln(2) / ke.
        vd: Apparent volume of distribution after IV bolus, calculated as dose / C0.
        clearance: Clearance, calculated as ke * Vd.
        auc_inf: AUC extrapolated to infinity (AUC0-last + Clast/ke).
        auc_extrapolated_tail: The extrapolated portion of auc_inf (Clast/ke).
        percent_auc_extrapolated: Percentage of auc_inf that is extrapolated,
            not observed.
        auc_extrapolation_reliable: False when percent_auc_extrapolated exceeds
            AUC_EXTRAPOLATION_RELIABILITY_THRESHOLD_PCT (i.e. sampling didn't run
            long enough for AUC0-inf, ke, t1/2, Vd, and CL to be trustworthy).
    """

    cmax: float
    tmax: float
    auc: float
    ke: float
    half_life: float
    vd: float
    clearance: float
    auc_inf: float
    auc_extrapolated_tail: float
    percent_auc_extrapolated: float
    auc_extrapolation_reliable: bool


def load_concentration_data(csv_path: str | Path) -> pd.DataFrame:
    """Load concentration-time data from a CSV file and validate it.

    The file must contain exactly the required analytical columns: `time` and
    `concentration`. Extra columns are allowed but ignored by downstream code.
    """

    data = pd.read_csv(csv_path)
    validate_concentration_data(data)
    return data.loc[:, REQUIRED_COLUMNS].copy()


def validate_concentration_data(data: pd.DataFrame) -> None:
    """Validate concentration-time input data.

    Raises:
        ValueError: If required columns are missing, values are missing, time is
        non-numeric or not strictly increasing, or concentrations are negative.
    """

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    analytical_data = data.loc[:, REQUIRED_COLUMNS]
    if analytical_data.isna().any().any():
        raise ValueError("Input data must not contain missing time or concentration values.")

    for column in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(analytical_data[column]):
            raise ValueError(f"{column!r} values must be numeric.")

    time = analytical_data["time"].to_numpy(dtype=float)
    concentration = analytical_data["concentration"].to_numpy(dtype=float)

    if time.size < 3:
        raise ValueError("At least three observations are required for PK analysis.")

    if not np.all(np.diff(time) > 0):
        raise ValueError("Time values must be strictly increasing.")

    if np.any(concentration < 0):
        raise ValueError("Concentrations must be non-negative.")


def calculate_auc(time: np.ndarray, concentration: np.ndarray) -> float:
    """Calculate AUC using the linear trapezoidal rule.

    This computes observed AUC from the first to last sampled time point
    (AUC0-last), without extrapolating to infinity.
    """

    return float(np.trapezoid(concentration, time))


def calculate_auc_extrapolated(
    time: np.ndarray,
    concentration: np.ndarray,
    ke: float,
    auc_last: float | None = None,
) -> tuple[float, float, float, bool]:
    """Extrapolate AUC0-last to AUC0-infinity and flag reliability.

    AUC0-inf = AUC0-last + Clast/ke, where Clast is the last observed
    concentration and ke is the terminal elimination rate constant. The
    extrapolated tail should generally be a small fraction of the total; a
    large fraction means sampling stopped before the terminal phase was well
    characterized, and downstream parameters (t1/2, Vd, CL) should be treated
    with caution.

    Returns:
        Tuple of (auc_inf, auc_extrapolated_tail, percent_auc_extrapolated,
        auc_extrapolation_reliable).

    Raises:
        ValueError: If ke is not positive, or the last observed concentration
            is not positive (extrapolation is undefined).
    """

    if ke <= 0:
        raise ValueError("ke must be positive to extrapolate AUC to infinity.")

    clast = float(concentration[-1])
    if clast <= 0:
        raise ValueError("Last observed concentration must be positive to extrapolate AUC.")

    resolved_auc_last = float(auc_last) if auc_last is not None else calculate_auc(time, concentration)

    auc_extrapolated_tail = clast / ke
    auc_inf = resolved_auc_last + auc_extrapolated_tail
    percent_auc_extrapolated = (auc_extrapolated_tail / auc_inf) * 100.0 if auc_inf > 0 else 0.0
    auc_extrapolation_reliable = percent_auc_extrapolated <= AUC_EXTRAPOLATION_RELIABILITY_THRESHOLD_PCT

    return auc_inf, auc_extrapolated_tail, percent_auc_extrapolated, auc_extrapolation_reliable


def estimate_terminal_ke(
    time: np.ndarray,
    concentration: np.ndarray,
    terminal_points: int = 3,
) -> float:
    """Estimate terminal elimination rate constant from log-linear data.

    The final positive concentration points are treated as the terminal phase.
    A straight line is fit to ln(C) versus time, where slope = -ke.
    """

    positive = concentration > 0
    terminal_time = time[positive][-terminal_points:]
    terminal_concentration = concentration[positive][-terminal_points:]

    if terminal_time.size < terminal_points:
        raise ValueError(
            f"At least {terminal_points} positive terminal concentrations are required to estimate ke."
        )

    slope, _intercept = np.polyfit(terminal_time, np.log(terminal_concentration), 1)
    ke = -float(slope)

    if ke <= 0:
        raise ValueError("Terminal concentration profile does not show positive elimination.")

    return ke


def calculate_pk_parameters(
    data: pd.DataFrame,
    dose: float,
    c0: float | None = None,
    ke: float | None = None,
    terminal_points: int = 3,
) -> PKParameters:
    """Calculate common PK parameters from concentration-time data.

    Args:
        data: Validated DataFrame with `time` and `concentration` columns.
        dose: IV bolus dose in mass units. Dose units determine Vd and CL units.
        c0: Optional model-estimated initial concentration. If omitted, Cmax is
            used as a pragmatic approximation for IV bolus data sampled at t=0.
        ke: Optional elimination rate constant. If omitted, ke is estimated from
            the terminal log-linear phase.
        terminal_points: Number of final positive points used for terminal ke.

    Returns:
        PKParameters with observed and derived parameters, including AUC
        extrapolated to infinity and its reliability flag.
    """

    validate_concentration_data(data)

    if dose <= 0:
        raise ValueError("Dose must be positive.")

    time = data["time"].to_numpy(dtype=float)
    concentration = data["concentration"].to_numpy(dtype=float)

    cmax_index = int(np.argmax(concentration))
    cmax = float(concentration[cmax_index])
    tmax = float(time[cmax_index])
    auc = calculate_auc(time, concentration)

    estimated_ke = float(ke) if ke is not None else estimate_terminal_ke(time, concentration, terminal_points)
    if estimated_ke <= 0:
        raise ValueError("ke must be positive.")

    half_life = float(np.log(2) / estimated_ke)

    reference_c0 = float(c0) if c0 is not None else cmax
    if reference_c0 <= 0:
        raise ValueError("C0 or Cmax must be positive to calculate Vd.")

    # IV bolus assumption: initial plasma concentration is dose divided by Vd.
    vd = float(dose / reference_c0)

    # For linear first-order elimination, systemic clearance equals ke * Vd.
    clearance = float(estimated_ke * vd)

    auc_inf, auc_extrapolated_tail, percent_auc_extrapolated, auc_extrapolation_reliable = (
        calculate_auc_extrapolated(time, concentration, estimated_ke, auc_last=auc)
    )

    return PKParameters(
        cmax=cmax,
        tmax=tmax,
        auc=auc,
        ke=estimated_ke,
        half_life=half_life,
        vd=vd,
        clearance=clearance,
        auc_inf=auc_inf,
        auc_extrapolated_tail=auc_extrapolated_tail,
        percent_auc_extrapolated=percent_auc_extrapolated,
        auc_extrapolation_reliable=auc_extrapolation_reliable,
    )
