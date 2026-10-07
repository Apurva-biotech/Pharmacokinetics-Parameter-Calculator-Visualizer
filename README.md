# PK Parameter Calculator & Visualizer

A production-quality Python portfolio project for pharmacokinetic analysis of plasma concentration-time data. The tool reads a CSV file, validates analytical data, calculates core pharmacokinetic parameters (including AUC extrapolated to infinity with a reliability flag), fits a one-compartment IV bolus model with goodness-of-fit diagnostics, and exports a publication-ready PNG visualization.

## Preview

![Metrics and Model Fit](assets/metrics_model_fit.png)
![Plot and Download](assets/plot_download_table.png)

**🔗 Live Demo:** [PK Parameter Calculator](https://pk-parameter-calculator-visualizer-c3w2mydgfwf6tmzbaobtpj.streamlit.app/)

## Project Overview

This project is designed for biotechnology and computational pharmacokinetics portfolios. It demonstrates:

- Robust CSV data ingestion and validation
- Noncompartmental pharmacokinetic calculations, including AUC extrapolated to infinity with an automatic reliability flag
- One-compartment IV bolus model fitting with `scipy.optimize.curve_fit`, including goodness-of-fit diagnostics (R², AIC, parameter standard errors, residuals)
- Modular Python code with type hints and docstrings
- A pytest suite covering the core calculations against hand-verified reference values
- Reproducible visualization with `matplotlib`
- Interactive web analysis with `streamlit` and `plotly`

## Project Structure

```text
pk_calculator/
├── data/
│   └── sample_data.csv
├── models/
│   └── one_compartment.py
├── tests/
│   └── test_pk_calculations.py
├── calculator.py
├── visualizer.py
├── app.py
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Pharmacokinetic Theory

The input data must contain plasma concentration measurements over time:

```csv
time,concentration
0.0,50.0
0.25,45.9
...
```

The calculator estimates:

- **Cmax**: maximum observed plasma concentration.
- **Tmax**: time at which Cmax occurs.
- **AUC0-last**: area under the observed concentration-time curve, calculated with the linear trapezoidal rule.
- **AUC0-inf**: AUC extrapolated to infinity as `AUC0-last + Clast/ke`. The app flags AUC0-inf as unreliable whenever the extrapolated tail exceeds 20% of the total — the standard NCA convention for deciding whether sampling ran long enough into the terminal phase.
- **ke**: terminal elimination rate constant estimated from the slope of the log-linear terminal phase.
- **t1/2**: elimination half-life, calculated as `ln(2) / ke`.
- **Vd**: apparent volume of distribution for IV bolus dosing, calculated as `Dose / C0`.
- **CL**: clearance, calculated as `ke * Vd`.

The one-compartment IV bolus model fit additionally reports:

- **R²** and **AIC** for the fit.
- **Standard errors** on fitted C0 and ke, from the fit's covariance matrix.
- A **residuals plot** (observed minus predicted vs. time) for visually checking whether a one-compartment model is actually appropriate for the data.

The one-compartment IV bolus model is:

```text
C(t) = C0 * exp(-ke * t)
```

Key assumptions:

- The dose is administered as an IV bolus.
- Distribution is instantaneous and can be represented by a single well-mixed compartment.
- Elimination follows first-order kinetics.
- The terminal phase is log-linear.

## Installation

From the project directory:

```bash
cd pk_calculator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

Run the Streamlit web app:

```bash
streamlit run app.py
```

Run the sample analysis:

```bash
python main.py --input data/sample_data.csv --dose 1000 --output pk_profile.png
```

Use your own CSV file:

```bash
python main.py --input path/to/your_data.csv --dose 250 --output results/my_pk_plot.png
```

Your CSV must include:

```csv
time,concentration
0,50.0
1,35.8
2,26.4
4,14.9
8,4.9
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Coverage includes the trapezoidal AUC calculation, terminal ke estimation, AUC0-inf extrapolation (both the reliable and flagged-unreliable cases), end-to-end parameter calculation against hand-verified reference values, every input validation error path, one-compartment model fitting, and goodness-of-fit diagnostics.

## Example Output

Console output:

```text
PK Parameter Calculator & Visualizer
========================================
Cmax:       50.0000
Tmax:       0.0000
AUC0-last:  163.6375
AUC0-inf:   169.3815 (3.4% extrapolated — reliable)
ke:         0.2786
t1/2:       2.4884
Vd:         20.2727
CL:         5.6470

One-compartment model fit
C0:         49.3273
ke_fit:     0.3046
R-squared:  0.9992
AIC:        -13.3214

Saved plot: pk_profile.png
```

The generated PNG overlays observed concentration-time data with the fitted one-compartment curve.

## Streamlit Web App

The Streamlit interface provides:

- CSV upload for files containing `time` and `concentration` columns.
- Dose input from the sidebar.
- Calculated PK parameters displayed as metrics.
- AUC0-last vs. AUC0-inf, with a color-coded banner flagging whether the extrapolation is reliable.
- Fitted one-compartment model parameters and their standard errors.
- Goodness-of-fit diagnostics (R², AIC), with an automatic warning when R² suggests a poor fit.
- Interactive Plotly concentration-time visualization.
- A residuals-vs-time plot for visually assessing fit quality.
- Downloadable `pk_results.csv` containing all calculated parameters, extrapolation results, and model-fit diagnostics.

If no CSV is uploaded, the app analyzes `data/sample_data.csv` so the project is immediately demo-ready.

## Deployment

### Local macOS Deployment

From the project directory:

```bash
cd pk_calculator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

### Streamlit Community Cloud

1. Push the `pk_calculator` project to a GitHub repository.
2. Confirm that `requirements.txt`, `app.py`, and the `data/` folder are committed.
3. Create a new app in Streamlit Community Cloud.
4. Set the app entry point to `app.py`.
5. Deploy the app.

For a repository where `pk_calculator` is a subfolder, set the app path to `pk_calculator/app.py`.

## Data Validation

The program checks that:

- Required columns `time` and `concentration` are present.
- There are no missing values.
- Time and concentration values are numeric.
- Time values are strictly increasing.
- Concentrations are non-negative.
- At least three observations are present.

## Notes on Units

The code is unit-consistent but unit-agnostic. For example, if dose is in mg and concentration is in mg/L, then:

- Vd is reported in L.
- CL is reported in L per time unit.
- ke is reported as inverse time.
- half-life uses the same time unit as the input time column.

## Limitations

- **Validated on synthetic data only.** The test suite and example outputs use clean, noiseless or lightly perturbed exponential decay data. Real assay data has more complex noise structure, and parameter estimates (especially ke and half-life) should be interpreted with more caution on real datasets than on the synthetic cases this project has been checked against.
- **One-compartment model only.** The residuals plot for the bundled sample data shows a slight systematic wave (not pure random scatter) even at R² = 0.9992, which is a classic early sign that a two-compartment model might fit the terminal phase better. Always check the residuals plot, not just R², before trusting the single-compartment ke and half-life for a given dataset.
- **AUC0-inf extrapolation assumes true first-order terminal elimination.** If the terminal phase is not genuinely log-linear (e.g. flip-flop kinetics, multi-phasic elimination), both ke and the AUC0-inf extrapolation built on it will be unreliable even when the 20%-rule flag says otherwise.
- **CSV-loading path and sidebar parameter interactions are not covered by the current test suite** — only the core calculation functions are. `load_concentration_data`'s file-reading behavior and the effect of changing "terminal points for ke" in the Streamlit sidebar are exercised manually, not by automated tests.

## Future Improvements

- Add two-compartment IV bolus and oral absorption models.
- Support replicate observations and summary statistics.
- Add weighted regression options for heteroscedastic concentration data.
- Add confidence intervals on AUC0-inf (currently only a point estimate and a reliability flag).
- Add authentication and project persistence for the Streamlit interface.
- Export results to Excel and PDF reports.
- Extend the test suite to cover CSV loading and sidebar parameter interactions.
