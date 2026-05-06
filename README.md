# PD Insight Studio

**A desktop application for Parkinson's disease real-world data analysis.**

Built for dissertation-grade research — accepts any dataset, produces meaningful insights, comparisons, and optional predictions — all inside the app. No SPSS required.

---

## What It Does

| Tab | Purpose |
|-----|---------|
| ① Load Data | Load one or more CSVs, preview, auto-normalise column names |
| ② Configure | Map columns to roles (patient ID, time, group, outcome), build derived-variable rules, set QC thresholds |
| ③ QC & Clean | Transparent quality control — every flagged row shows *why* it was flagged |
| ④ Insights | Descriptive stats, group comparisons (effect sizes), time trends, progression analysis, narrative report |
| ⑤ Predict | Exploratory classification or regression with feature importance |
| ⑥ Export | Save cleaned CSV, markdown report, plots, and model results |

---

## How to Run in VS Code on Windows

### 1. Prerequisites

Make sure you have **Python 3.11** installed:
```
python --version
```
Should print `Python 3.11.x`. If not, download from https://python.org.

---

### 2. Clone / Extract the Project

Place the `pd_insight_studio` folder somewhere on your machine, e.g.:
```
C:\Users\YourName\Documents\pd_insight_studio
```

---

### 3. Open in VS Code

```
File > Open Folder > select pd_insight_studio
```

---

### 4. Create a Virtual Environment

Open the VS Code Terminal (`` Ctrl+` ``):

```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in the terminal prompt.

---

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs pandas, matplotlib, seaborn, scikit-learn, scipy, and other required packages.

> **Note:** `tkinter` is included with standard Python on Windows — no separate install needed.

---

### 6. Run the App

```bash
python main.py
```

The **PD Insight Studio** window will open.

---

## Typical Workflow with mPower Data

1. **Load Data** → Add your mPower tapping CSV(s). The app auto-normalises column names.

2. **Configure** →
   - Set `Patient/Subject ID` → e.g. `healthcode` or `recordid`
   - Set `Time/Visit column` → e.g. `createdon` or `timestamp`
   - Set `Primary group variable` → e.g. `medtimingstring` (medication timing)
   - Set `Outcome` → e.g. `tap_count` or derived tap rate

3. **Rule Builder** →
   - Add a **Keyword Map Rule**: Source = `medtimingstring`, map keywords:
     ```
     before,immediately before → BEFORE
     after,just after,a while after → AFTER
     none,na,no → NO_MEDS
     ```
     Output column: `med_state`
   - Add a **Formula Rule**: `tap_count / duration_seconds` → `tap_rate`

4. **QC** → Run QC to see how many rows pass/fail and why.

5. **Insights** → Click "Generate All Insights" to see:
   - Distribution of tap rate per med group
   - Group comparison: BEFORE vs AFTER vs NO_MEDS effect sizes
   - Progression over time if visit timestamps are mapped

6. **Predict** → Optionally run a classification (PD vs Control proxy) or regression (tap rate).

7. **Export** → Save cleaned data, plots, and reports to your chosen folder.

---

## Using with PPMI Data

- `entity_id` → `patno`
- `time_col` → `infodt` or `visdate`
- `group_col` → `cohort` or `sex`
- `outcome_col` → `updrs_totscore` or similar

Add a **Months Since Baseline** rule to convert dates to numeric visit month.

---

## Project Structure

```
pd_insight_studio/
├── main.py          # GUI entrypoint
├── standardise.py   # Column normalisation, type inference, time parsing
├── rules.py         # Rule builder — keyword map, formula, months-since-baseline
├── qc.py            # Quality control — transparent flagging
├── insights.py      # Descriptive stats, group comparisons, progression, narrative
├── plotting.py      # All matplotlib/seaborn figures
├── modeling.py      # Logistic/Ridge regression + Random Forest with metrics
├── requirements.txt
└── README.md
```

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0
pillow>=10.0.0
openpyxl>=3.1.0
```

---

## Important Notes

- **Not hard-coded to mPower**: any CSV dataset works — you configure the column roles yourself.
- **No silent deletions**: QC flags rows with reasons; you decide what to keep.
- **Predictions are exploratory**: All modelling output is clearly labelled as having no clinical validity.
- **Large files**: The app uses threading for QC and modelling to avoid freezing the UI.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Make sure `.venv` is activated and you ran `pip install -r requirements.txt` |
| Window doesn't open | Check Python version is 3.11; try `python -m tkinter` to verify Tkinter works |
| Plots don't show | Ensure matplotlib is installed; restart app |
| Large file is slow to load | This is normal for >100k rows. Wait for progress or split the file |
