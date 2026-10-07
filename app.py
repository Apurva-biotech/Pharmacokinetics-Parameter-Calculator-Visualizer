"""Streamlit web application for PK analysis."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculator import (
    REQUIRED_COLUMNS,
    PKParameters,
    calculate_pk_parameters,
    validate_concentration_data,
)
from models.one_compartment import (
    GoodnessOfFit,
    OneCompartmentFit,
    concentration_time_model,
    evaluate_goodness_of_fit,
    fit_one_compartment_model,
)


st.set_page_config(
    page_title="PK Parameter Calculator",
    page_icon="PK",
    layout="wide",
)


def load_uploaded_data(uploaded_file: object | None) -> pd.DataFrame:
    """Load uploaded CSV data or fall back to the bundled sample dataset."""

    if uploaded_file is None:
        data = pd.read_csv("data/sample_data.csv")
    else:
        data = pd.read_csv(uploaded_file)

    validate_concentration_data(data)
    return data.loc[:, REQUIRED_COLUMNS].copy()


def create_results_table(
    parameters: PKParameters,
    model_fit: OneCompartmentFit,
    goodness_of_fit: GoodnessOfFit,
) -> pd.DataFrame:
    """Convert PK, extrapolation, model-fit, and GOF results into a download-friendly table."""

    rows = [
        ("Cmax", parameters.cmax),
        ("Tmax", parameters.tmax),
        ("AUC0-last", parameters.auc),
        ("AUC0-inf", parameters.auc_inf),
        ("AUC extrapolated tail", parameters.auc_extrapolated_tail),
        ("% AUC extrapolated", parameters.percent_auc_extrapolated),
        ("AUC0-inf reliable (<=20% extrapolated)", parameters.auc_extrapolation_reliable),
        ("ke", parameters.ke),
        ("Half-life", parameters.half_life),
        ("Vd", parameters.vd),
        ("Clearance", parameters.clearance),
        ("Fitted C0", model_fit.c0),
        ("Fitted ke", model_fit.ke),
        ("Fitted ke SE", goodness_of_fit.se_ke),
        ("Fitted C0 SE", goodness_of_fit.se_c0),
        ("R-squared", goodness_of_fit.r_squared),
        ("AIC", goodness_of_fit.aic),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value"])


def create_interactive_plot(data: pd.DataFrame, model_fit: OneCompartmentFit) -> go.Figure:
    """Create an interactive concentration-time Plotly figure."""

    time = data["time"].to_numpy(dtype=float)
    concentration = data["concentration"].to_numpy(dtype=float)
    model_time = np.linspace(float(time.min()), float(time.max()), 300)
    model_concentration = concentration_time_model(model_time, model_fit.c0, model_fit.ke)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=time,
            y=concentration,
            mode="markers",
            name="Observed data",
            marker={"size": 9, "color": "#1f77b4"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=model_time,
            y=model_concentration,
            mode="lines",
            name="One-compartment fit",
            line={"width": 3, "color": "#d62728"},
        )
    )
    figure.update_layout(
        title="Plasma Concentration-Time Profile",
        xaxis_title="Time",
        yaxis_title="Plasma concentration",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
    )
    return figure


def create_residuals_plot(data: pd.DataFrame, goodness_of_fit: GoodnessOfFit) -> go.Figure:
    """Create a residuals-vs-time plot to visually assess fit quality."""

    time = data["time"].to_numpy(dtype=float)

    figure = go.Figure()
    figure.add_hline(y=0, line={"width": 1, "color": "#888888", "dash": "dash"})
    figure.add_trace(
        go.Scatter(
            x=time,
            y=goodness_of_fit.residuals,
            mode="markers",
            name="Residual (observed - predicted)",
            marker={"size": 9, "color": "#2ca02c"},
        )
    )
    figure.update_layout(
        title="Residuals vs. Time",
        xaxis_title="Time",
        yaxis_title="Residual",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
    )
    return figure


def results_to_csv(results: pd.DataFrame) -> bytes:
    """Serialize results table as CSV bytes for Streamlit downloads."""

    buffer = StringIO()
    results.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def render_metrics(
    parameters: PKParameters,
    model_fit: OneCompartmentFit,
    goodness_of_fit: GoodnessOfFit,
) -> None:
    """Display calculated PK, AUC extrapolation, model-fit, and goodness-of-fit results.

    Fields are grouped and formatted explicitly rather than looped over
    generically, because PKParameters mixes floats with a bool
    (auc_extrapolation_reliable) that can't be formatted the same way.
    """

    core_cols = st.columns(4)
    core_cols[0].metric("Cmax", f"{parameters.cmax:.4f}")
    core_cols[1].metric("Tmax", f"{parameters.tmax:.4f}")
    core_cols[2].metric("ke", f"{parameters.ke:.4f}")
    core_cols[3].metric("Half-life", f"{parameters.half_life:.4f}")

    derived_cols = st.columns(2)
    derived_cols[0].metric("Vd", f"{parameters.vd:.4f}")
    derived_cols[1].metric("Clearance", f"{parameters.clearance:.4f}")

    st.subheader("AUC: Observed vs. Extrapolated to Infinity")
    auc_cols = st.columns(3)
    auc_cols[0].metric("AUC0-last (observed)", f"{parameters.auc:.4f}")
    auc_cols[1].metric("AUC0-inf (extrapolated)", f"{parameters.auc_inf:.4f}")
    auc_cols[2].metric("% extrapolated", f"{parameters.percent_auc_extrapolated:.2f}%")

    if parameters.auc_extrapolation_reliable:
        st.success(
            "AUC0-inf is reliable: the extrapolated tail is "
            f"{parameters.percent_auc_extrapolated:.1f}% of total AUC (<=20%)."
        )
    else:
        st.warning(
            "AUC0-inf may not be reliable: the extrapolated tail is "
            f"{parameters.percent_auc_extrapolated:.1f}% of total AUC (>20%). "
            "Sampling likely stopped before the terminal phase was well characterized; "
            "treat half-life, Vd, and clearance with caution."
        )

    st.subheader("One-Compartment Model Fit")
    fit_cols = st.columns(4)
    fit_cols[0].metric("Fitted C0", f"{model_fit.c0:.4f}")
    fit_cols[1].metric("Fitted ke", f"{model_fit.ke:.4f}")
    fit_cols[2].metric("Fitted C0 SE", f"{goodness_of_fit.se_c0:.4f}")
    fit_cols[3].metric("Fitted ke SE", f"{goodness_of_fit.se_ke:.4f}")

    st.subheader("Goodness of Fit")
    gof_cols = st.columns(2)
    gof_cols[0].metric("R-squared", f"{goodness_of_fit.r_squared:.4f}")
    if np.isnan(goodness_of_fit.aic):
        gof_cols[1].metric("AIC", "undefined (near-zero residuals)")
    else:
        gof_cols[1].metric("AIC", f"{goodness_of_fit.aic:.4f}")

    if goodness_of_fit.r_squared < 0.9:
        st.warning(
            f"R-squared is {goodness_of_fit.r_squared:.3f} - the one-compartment model "
            "may not describe this data well. Check the residuals plot below for structure "
            "(e.g. curvature) that would suggest a two-compartment model instead."
        )


def main() -> None:
    """Run the Streamlit PK analysis app."""

    st.title("PK Parameter Calculator & Visualizer")
    st.caption(
        "Upload plasma concentration-time data, estimate PK parameters, "
        "and fit a one-compartment IV bolus model."
    )

    with st.sidebar:
        st.header("Analysis Inputs")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        dose = st.number_input("Dose", min_value=0.0001, value=1000.0, step=10.0)
        terminal_points = st.number_input(
            "Terminal points for ke",
            min_value=3,
            max_value=10,
            value=3,
            step=1,
        )

    try:
        data = load_uploaded_data(uploaded_file)
        model_fit = fit_one_compartment_model(
            data["time"].to_numpy(dtype=float),
            data["concentration"].to_numpy(dtype=float),
        )
        goodness_of_fit = evaluate_goodness_of_fit(
            data["time"].to_numpy(dtype=float),
            data["concentration"].to_numpy(dtype=float),
            model_fit,
        )
        parameters = calculate_pk_parameters(
            data,
            dose=dose,
            c0=model_fit.c0,
            terminal_points=int(terminal_points),
        )
    except Exception as exc:
        st.error(f"Unable to analyze dataset: {exc}")
        st.stop()

    st.subheader("Input Data")
    st.dataframe(data, width="stretch", hide_index=True)

    st.subheader("Calculated Parameters")
    render_metrics(parameters, model_fit, goodness_of_fit)

    st.subheader("Interactive Concentration-Time Plot")
    st.plotly_chart(create_interactive_plot(data, model_fit), width="stretch")

    st.subheader("Residuals Plot")
    st.plotly_chart(create_residuals_plot(data, goodness_of_fit), width="stretch")

    results = create_results_table(parameters, model_fit, goodness_of_fit)
    st.subheader("Download Results")
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        label="Download results CSV",
        data=results_to_csv(results),
        file_name="pk_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
