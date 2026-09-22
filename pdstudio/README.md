# PD Insight Studio

A desktop application for exploratory analysis of Parkinson's disease real-world data.

Load a CSV, map your columns to roles, apply transparent quality control, get descriptive statistics and group comparisons with effect sizes, and optionally fit an exploratory predictive model. Everything runs locally. No SPSS, no cloud upload, no data leaving your machine.

![Status](https://img.shields.io/badge/status-research%20prototype-orange) ![Python](https://img.shields.io/badge/python-3.11-blue) ![Licence](https://img.shields.io/badge/licence-MIT-green)

---

## Honest scope

Read this before you read anything else.

**What this is.** A working research prototype, built as an undergraduate dissertation project at the University of Birmingham. It automates a workflow that is otherwise done by hand in SPSS or Excel: cleaning real-world tapping-test data, flagging bad rows with a visible reason, comparing groups properly, and producing a report.

**What this is not.** It is not a clinical tool. It is not validated against any clinical endpoint. It has not been used in a regulated study, it is not a medical device, and nothing it outputs should inform care for any individual. The predictive tab is exploratory: it will happily fit a model to data that does not support one, and it is your job to know the difference.

**What it has been tested on.** The mPower public tapping dataset, and the synthetic sample included here. It has not been run against a large linked clinical dataset.

If you are evaluating this, that paragraph is the one that matters. I would rather you know the limits up front than find them yourself.

---

## Quick start

Requires Python 3.11.

```bash
git clone https://github.com/cdibie7/PD-Insight-Studio.git
cd PD-Insight-Studio
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Then in the app: **Load Data** and open `sample_synthetic_tapping.csv`, included in this repo. That file is synthetic, generated for demonstration, and contains no real participant data. It exists so you can see the whole pipeline work in about two minutes without sourcing a dataset first.

---

## What it does

| Tab | Purpose |
|-----|---------|
| 1. Load Data | Load one or more CSVs, preview, auto-normalise column names |
| 2. Configure | Map columns to roles (participant ID, time, group, outcome), build derived-variable rules, set QC thresholds |
| 3. QC and Clean | Quality control where every flagged row shows *why* it was flagged, not just that it was |
| 4. Insights | Descriptive statistics, group comparisons with effect sizes, time trends, progression analysis, narrative report |
| 5. Predict | Exploratory classification or regression with feature importance |
| 6. Export | Cleaned CSV, markdown report, plots, model results |

The design principle throughout is that nothing is hidden. Every exclusion is attributable to a rule you set, and the cleaned dataset that comes out the other end can be traced back row by row. That came from a real problem during the project: a silent failure in the source data was corrupting rows without surfacing an error, and the fix was a loader that refuses to fail quietly.

---

## Project layout

| File | Role |
|---|---|
| `main.py` | The application. Tkinter GUI, wires the modules together |
| `standardise.py` | Column-name normalisation and type coercion |
| `rules.py` | User-defined derived-variable rules |
| `qc.py` | Quality control checks and flagging, with reasons |
| `insights.py` | Descriptive statistics, group comparisons, effect sizes |
| `modeling.py` | Exploratory classification and regression |
| `plotting.py` | Chart generation |
| `sample_synthetic_tapping.csv` | Synthetic demonstration data, safe to share |

---

## Contributing

Contributions are genuinely welcome, particularly on the modelling side. Issues tagged `help wanted` are scoped so you can pick one up without reading the whole codebase first.

The areas where help would make the most difference:

- **Model validation.** The predictive tab needs proper cross-validation, calibration, and honest performance reporting. Right now it is easier than it should be to get a flattering number from a model that would not generalise.
- **Leakage guards.** Repeated-measures data from the same participant must not straddle a train/test split. This needs enforcing rather than documenting.
- **Interpretability.** Feature importance is currently the built-in kind. SHAP or permutation importance would be more defensible.
- **Test coverage.** There are no tests. The QC and rules modules are the highest-value place to start.

If you are unsure whether something fits, open an issue and ask. A short question is welcome and I will reply.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

## Background

Built as a final-year dissertation at the University of Birmingham (BSc Biomedical Science with Biomedical Entrepreneurship), awarded a First. The project paired the software with a full commercialisation assessment covering market sizing, competitor benchmarking, and an NHS data governance review.

Questions, criticism, or interest in collaborating: open an issue, or contact me through the profile linked above.
