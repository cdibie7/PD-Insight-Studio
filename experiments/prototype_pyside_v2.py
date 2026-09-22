# main.py
# Generic RWD Insight + Prediction Desktop App (CSV -> QC/Standardise -> Insights -> Optional Model)
# Works for ANY dataset (mPower, PPMI, etc.) via user-configurable column mapping.
#
# Requires: PySide6, pandas, numpy, scikit-learn
# Run: py main.py

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox, QTextEdit,
    QTableView, QGroupBox, QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, mean_absolute_error, r2_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# -----------------------------
# Utilities
# -----------------------------
def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-zA-Z0-9_]+", "_", str(c).strip()).lower()[:64]
        for c in df.columns
    ]
    return df


def try_parse_datetime(s: pd.Series) -> pd.Series:
    # Robust: handles epoch ms/sec strings, ISO, mixed
    x = s.copy()
    # If numeric-like: guess epoch
    num = pd.to_numeric(x, errors="coerce")
    if num.notna().mean() > 0.7:
        # decide seconds vs ms by magnitude
        med = np.nanmedian(num.to_numpy())
        if np.isfinite(med) and med > 1e12:  # ms
            return pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
        if np.isfinite(med) and med > 1e9:   # sec
            return pd.to_datetime(num, unit="s", errors="coerce", utc=True)
    # else attempt normal parsing
    return pd.to_datetime(x, errors="coerce", utc=True)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if sp == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp


def safe_value_counts(s: pd.Series, top: int = 15) -> pd.Series:
    vc = s.value_counts(dropna=False)
    if len(vc) > top:
        head = vc.iloc[:top].copy()
        head.loc["(others)"] = vc.iloc[top:].sum()
        return head
    return vc


# -----------------------------
# Qt Table Model for pandas
# -----------------------------
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if self._df is None else self._df.shape[1]

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None
        if role == Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if pd.isna(val):
                return ""
            return str(val)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or self._df is None:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section)

    def set_df(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()


# -----------------------------
# Config
# -----------------------------
@dataclass
class AppConfig:
    id_col: Optional[str] = None
    time_col: Optional[str] = None
    group_col: Optional[str] = None  # e.g. sex
    outcome_col: Optional[str] = None  # classification target
    target_col: Optional[str] = None   # regression target
    drop_cols_text: str = ""           # comma separated
    strict_drop_high_missing: bool = False
    drop_missing_threshold: float = 0.8
    test_size: float = 0.2
    random_state: int = 42
    model_type: str = "Auto"  # Auto / Logistic / RF Classifier / Ridge / RF Regressor


# -----------------------------
# Core processing
# -----------------------------
def profile_df(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["shape"] = df.shape
    out["dtypes"] = df.dtypes.astype(str).to_dict()

    miss_frac = df.isna().mean().sort_values(ascending=False)
    out["missing_top"] = miss_frac.head(25)

    dup_rows = int(df.duplicated().sum())
    out["dup_rows_exact"] = dup_rows

    # Identify likely categorical vs numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    out["num_cols"] = num_cols
    out["cat_cols"] = cat_cols

    return out


def standardise_df(df: pd.DataFrame, cfg: AppConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Generic standardisation:
    - Normalize column names (already done upstream)
    - Optional: parse time column into datetime_utc
    - Drop user-specified columns
    - Optional: drop columns with very high missingness
    - Add qc flags + reasons (generic)
    """
    qc_info: Dict[str, object] = {}
    x = df.copy()

    # Drop columns user specified
    drop_cols = []
    if cfg.drop_cols_text.strip():
        drop_cols = [normalize_key(s) for s in cfg.drop_cols_text.split(",") if s.strip()]
        # Map normalized keys to actual columns best-effort
        actual_drop = []
        for d in drop_cols:
            if d in x.columns:
                actual_drop.append(d)
        x = x.drop(columns=actual_drop, errors="ignore")
        qc_info["dropped_cols_user"] = actual_drop
    else:
        qc_info["dropped_cols_user"] = []

    # Parse time
    if cfg.time_col and cfg.time_col in x.columns:
        dt = try_parse_datetime(x[cfg.time_col])
        x["datetime_utc"] = dt
        qc_info["parsed_time_col"] = cfg.time_col
    else:
        x["datetime_utc"] = pd.NaT
        qc_info["parsed_time_col"] = None

    # Drop very high missing columns
    miss = x.isna().mean()
    if cfg.strict_drop_high_missing:
        to_drop = miss[miss >= cfg.drop_missing_threshold].index.tolist()
        x = x.drop(columns=to_drop, errors="ignore")
        qc_info["dropped_cols_high_missing"] = to_drop
        qc_info["drop_missing_threshold"] = cfg.drop_missing_threshold
    else:
        qc_info["dropped_cols_high_missing"] = []
        qc_info["drop_missing_threshold"] = cfg.drop_missing_threshold

    # Generic QC row flags
    reasons = []
    keep = []

    for i in range(len(x)):
        r = []
        if cfg.id_col and cfg.id_col in x.columns:
            if pd.isna(x.at[x.index[i], cfg.id_col]) or str(x.at[x.index[i], cfg.id_col]).strip() == "":
                r.append("missing_id")
        if cfg.outcome_col and cfg.outcome_col in x.columns:
            if pd.isna(x.at[x.index[i], cfg.outcome_col]):
                r.append("missing_outcome")
        if cfg.target_col and cfg.target_col in x.columns:
            if pd.isna(x.at[x.index[i], cfg.target_col]):
                r.append("missing_target")
        # time QC only if user mapped time
        if cfg.time_col and "datetime_utc" in x.columns:
            if pd.isna(x.at[x.index[i], "datetime_utc"]):
                r.append("unparseable_time")
        reason = "OK" if len(r) == 0 else ",".join(r)
        reasons.append(reason)
        keep.append(1 if reason == "OK" else 0)

    x["qc_reason"] = reasons
    x["qc_keep"] = keep

    # Breakdown
    removed = x[x["qc_keep"] == 0].copy()
    breakdown = (
        removed["qc_reason"]
        .str.split(",", expand=True)
        .stack()
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="n_removed")
    )

    qc_info["row_removed_count"] = int((x["qc_keep"] == 0).sum())
    qc_info["row_kept_count"] = int((x["qc_keep"] == 1).sum())
    qc_info["row_breakdown"] = breakdown

    return x, qc_info


def normalize_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s).strip()).lower()[:64]


def auto_insights(df: pd.DataFrame, cfg: AppConfig) -> str:
    """
    Generate relevant insights for Parkinson's disease analysis, focusing on:
    - Medication ON/OFF effects
    - PD vs Control group comparisons
    - Missingness summary
    - Outcome vs group (if selected)
    """
    lines: List[str] = []
    lines.append("AUTO INSIGHTS\n")

    # Missingness
    miss = df.isna().mean().sort_values(ascending=False)
    top_miss = miss.head(10)
    lines.append("Top missing columns:")
    for c, v in top_miss.items():
        lines.append(f"  - {c}: {v:.1%}")
    lines.append("")

    # Rows after QC
    kept = df[df.get("qc_keep", 1) == 1]
    lines.append(f"Rows kept (qc_keep=1): {len(kept)} / {len(df)}")
    lines.append("")

    # Outcome distribution (only if outcome column is selected)
    if cfg.outcome_col and cfg.outcome_col in kept.columns:
        s = kept[cfg.outcome_col]
        lines.append(f"Outcome '{cfg.outcome_col}' distribution (top):")
        vc = safe_value_counts(s, top=12)
        for idx, val in vc.items():
            lines.append(f"  - {idx}: {int(val)}")
        lines.append("")

    # Medication ON/OFF effects: Compare control vs PD or ON/OFF meds
    if cfg.group_col and cfg.outcome_col and cfg.group_col in kept.columns and cfg.outcome_col in kept.columns:
        group = kept[cfg.group_col].astype("object")
        outcome = kept[cfg.outcome_col].astype("object")
        ct = pd.crosstab(group, outcome, dropna=False)
        lines.append(f"Group '{cfg.group_col}' vs Outcome '{cfg.outcome_col}' (counts):")
        lines.append(ct.to_string())
        lines.append("")

        # Group-based metrics (e.g., medication ON/OFF effects)
        if len(kept[cfg.outcome_col].dropna().unique()) == 2:
            pos_class = kept[cfg.outcome_col].dropna().unique()[1]  # if binary classification
            lines.append(f"Medication effect comparison for {cfg.outcome_col}:")
            lines.append(ct.div(ct.sum(axis=1), axis=0)[pos_class].replace([np.inf, -np.inf], np.nan).to_string())
            lines.append("")

    # Time trends (e.g., symptom progression)
    if "datetime_utc" in kept.columns and kept["datetime_utc"].notna().any():
        tmp = kept.copy()
        tmp = tmp[tmp["datetime_utc"].notna()]
        tmp["month"] = tmp["datetime_utc"].dt.to_period("M").astype(str)
        counts = tmp["month"].value_counts().sort_index()
        lines.append("Time trend: rows per month (first 12 shown):")
        for m, c in counts.head(12).items():
            lines.append(f"  - {m}: {int(c)}")
        lines.append("")

    lines.append("Note: For Parkinson’s-specific insights (e.g., medication ON/OFF effects), map the relevant columns as group/outcome/target and/or include them as features for the model.")
    return "\n".join(lines)


def run_prediction(df: pd.DataFrame, cfg: AppConfig) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Runs either:
    - Classification if outcome_col provided
    - Regression if target_col provided
    Returns (report_text, optional_predictions_df)
    """
    kept = df[df.get("qc_keep", 1) == 1].copy()
    if len(kept) < 20:
        return "Not enough kept rows to train a model (need ~20+).", None

    # Decide problem
    problem = None
    ycol = None
    if cfg.outcome_col and cfg.outcome_col in kept.columns and kept[cfg.outcome_col].notna().any():
        problem = "classification"
        ycol = cfg.outcome_col
    elif cfg.target_col and cfg.target_col in kept.columns and kept[cfg.target_col].notna().any():
        problem = "regression"
        ycol = cfg.target_col
    else:
        return "Select an outcome (classification) or target (regression) column first.", None

    # Build features: drop obvious non-features
    drop_feature_cols = set(["qc_reason", "qc_keep"])
    if "datetime_utc" in kept.columns:
        # derive basic time features
        dt = kept["datetime_utc"]
        kept["year"] = dt.dt.year
        kept["month"] = dt.dt.month
        kept["dow"] = dt.dt.dayofweek
        kept["hour"] = dt.dt.hour
        drop_feature_cols.add("datetime_utc")

    # Remove ID/time columns from features (kept for joining/pred output only)
    if cfg.id_col and cfg.id_col in kept.columns:
        drop_feature_cols.add(cfg.id_col)
    if cfg.time_col and cfg.time_col in kept.columns:
        drop_feature_cols.add(cfg.time_col)

    X = kept.drop(columns=[c for c in drop_feature_cols if c in kept.columns] + ([ycol] if ycol in kept.columns else []), errors="ignore")
    y = kept[ycol].copy()

    # Clean y
    if problem == "classification":
        # if many unique categories, reduce risk of nonsense
        y = y.astype("object")
        # Drop rows where y missing
        m = y.notna()
        X = X.loc[m]
        y = y.loc[m]
        # If too many classes with too few samples: warn
        cls_counts = y.value_counts()
        if len(cls_counts) < 2:
            return "Outcome has <2 classes after cleaning.", None
        if (cls_counts < 5).sum() > 0 and len(X) < 200:
            # still run, but warn in report
            warn_rare = True
        else:
            warn_rare = False
    else:
        y = pd.to_numeric(y, errors="coerce")
        m = y.notna()
        X = X.loc[m]
        y = y.loc[m]
        warn_rare = False

    if len(X) < 20:
        return "Not enough rows after removing missing outcome/target.", None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg.test_size), random_state=int(cfg.random_state),
        stratify=y if problem == "classification" and len(pd.Series(y).unique()) < 50 else None
    )

    pipe = build_model_pipeline(X_train, y_train, problem, cfg)
    pipe.fit(X_train, y_train)

    report_lines = []
    report_lines.append("MODEL REPORT\n")
    report_lines.append(f"Problem: {problem}")
    report_lines.append(f"Target: {ycol}")
    report_lines.append(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    report_lines.append(f"Model choice: {cfg.model_type}")
    if warn_rare:
        report_lines.append("Warning: rare classes detected; metrics may be unstable.")

    if problem == "classification":
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report_lines.append(f"Accuracy: {acc:.3f}")
        report_lines.append(f"F1 (weighted): {f1:.3f}")

        # AUC only for binary
        uniq = pd.Series(y_train).dropna().unique()
        if len(uniq) == 2 and hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score((y_test == uniq[1]).astype(int), proba)
                report_lines.append(f"ROC AUC (binary): {auc:.3f}")
            except Exception:
                report_lines.append("ROC AUC: (could not compute)")

        # Predictions output
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=y_test.index)
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            try:
                proba_all = pipe.predict_proba(X_test)
                # if binary show prob of class 1
                if proba_all.shape[1] == 2:
                    pred_df["p_class1"] = proba_all[:, 1]
            except Exception:
                pass
        return "\n".join(report_lines), pred_df

    # regression
    y_hat = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    report_lines.append(f"MAE: {mae:.3f}")
    report_lines.append(f"R^2: {r2:.3f}")

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_hat}, index=y_test.index)
    return "\n".join(report_lines), pred_df


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RWD Insight Studio (Generic) — Clean • Standardise • Insights • Predict")
        self.resize(1200, 760)

        self.cfg = AppConfig()
        self.raw_df: Optional[pd.DataFrame] = None
        self.std_df: Optional[pd.DataFrame] = None
        self.qc_info: Dict[str, object] = {}
        self.pred_df: Optional[pd.DataFrame] = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_tab_load()
        self._build_tab_config()
        self._build_tab_insights()
        self._build_tab_export()

    # -------- Tabs --------
    def _build_tab_load(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.btn_load = QPushButton("Load CSV…")
        self.lbl_file = QLabel("No file loaded.")
        self.lbl_shape = QLabel("")
        top.addWidget(self.btn_load)
        top.addWidget(self.lbl_file, 1)
        top.addWidget(self.lbl_shape)
        layout.addLayout(top)

        self.table_raw = QTableView()
        self.raw_model = PandasModel(pd.DataFrame())
        self.table_raw.setModel(self.raw_model)
        layout.addWidget(QLabel("Preview (first 25 rows):"))
        layout.addWidget(self.table_raw, 1)

        self.txt_profile = QTextEdit()
        self.txt_profile.setReadOnly(True)
        layout.addWidget(QLabel("Quick profile:"))
        layout.addWidget(self.txt_profile, 1)

        self.btn_load.clicked.connect(self.load_csv)

        self.tabs.addTab(tab, "1) Load Data")

    def _build_tab_config(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Column Mapping (works for any dataset)")
        form = QFormLayout(box)

        self.cb_id = QComboBox()
        self.cb_time = QComboBox()
        self.cb_group = QComboBox()
        self.cb_outcome = QComboBox()
        self.cb_target = QComboBox()

        form.addRow("Entity ID column (optional)", self.cb_id)
        form.addRow("Time column (optional)", self.cb_time)
        form.addRow("Group column (optional, e.g. sex)", self.cb_group)
        form.addRow("Outcome for classification (optional)", self.cb_outcome)
        form.addRow("Target for regression (optional)", self.cb_target)

        layout.addWidget(box)

        box2 = QGroupBox("Standardisation + QC settings")
        form2 = QFormLayout(box2)

        self.ed_dropcols = QTextEdit()
        self.ed_dropcols.setFixedHeight(52)
        self.ed_dropcols.setPlaceholderText("Columns to drop (comma separated), e.g. row_id,row_version,phoneinfo")

        self.chk_drop_high_missing = QCheckBox("Drop columns with very high missingness")
        self.sp_missing = QDoubleSpinBox()
        self.sp_missing.setRange(0.0, 1.0)
        self.sp_missing.setSingleStep(0.05)
        self.sp_missing.setValue(0.80)

        self.sp_test = QDoubleSpinBox()
        self.sp_test.setRange(0.05, 0.5)
        self.sp_test.setSingleStep(0.05)
        self.sp_test.setValue(0.20)

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 999999)
        self.sp_seed.setValue(42)

        self.cb_model = QComboBox()
        self.cb_model.addItems(["Auto", "Logistic", "RF Classifier", "Ridge", "RF Regressor"])

        form2.addRow("Drop columns", self.ed_dropcols)
        form2.addRow(self.chk_drop_high_missing)
        form2.addRow("Drop missingness threshold", self.sp_missing)
        form2.addRow("Test size (model)", self.sp_test)
        form2.addRow("Random seed", self.sp_seed)
        form2.addRow("Model choice", self.cb_model)

        layout.addWidget(box2)

        btns = QHBoxLayout()
        self.btn_apply = QPushButton("Apply Standardisation + QC")
        self.lbl_apply = QLabel("")
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.lbl_apply, 1)
        layout.addLayout(btns)

        self.table_std = QTableView()
        self.std_model = PandasModel(pd.DataFrame())
        self.table_std.setModel(self.std_model)

        layout.addWidget(QLabel("Standardised dataset preview (first 25 rows):"))
        layout.addWidget(self.table_std, 1)

        self.btn_apply.clicked.connect(self.apply_standardise)

        self.tabs.addTab(tab, "2) Configure + Standardise")

    def _build_tab_insights(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.txt_insights = QTextEdit()
        self.txt_insights.setReadOnly(True)

        btns = QHBoxLayout()
        self.btn_insights = QPushButton("Generate Insights")
        self.btn_model = QPushButton("Run Prediction Model")
        self.lbl_model = QLabel("")
        btns.addWidget(self.btn_insights)
        btns.addWidget(self.btn_model)
        btns.addWidget(self.lbl_model, 1)

        layout.addLayout(btns)
        layout.addWidget(QLabel("Insights / Model report:"))
        layout.addWidget(self.txt_insights, 2)

        self.table_pred = QTableView()
        self.pred_model = PandasModel(pd.DataFrame())
        self.table_pred.setModel(self.pred_model)

        layout.addWidget(QLabel("Model predictions preview (test split):"))
        layout.addWidget(self.table_pred, 1)

        self.btn_insights.clicked.connect(self.generate_insights)
        self.btn_model.clicked.connect(self.run_model)

        self.tabs.addTab(tab, "3) Insights + Predict")

    def _build_tab_export(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.btn_export_clean = QPushButton("Export Standardised Dataset (CSV)…")
        self.btn_export_qc = QPushButton("Export QC Log (CSV)…")
        self.btn_export_pred = QPushButton("Export Predictions (CSV)…")
        self.lbl_export = QLabel("")

        layout.addWidget(self.btn_export_clean)
        layout.addWidget(self.btn_export_qc)
        layout.addWidget(self.btn_export_pred)
        layout.addWidget(self.lbl_export, 1)

        self.btn_export_clean.clicked.connect(self.export_clean)
        self.btn_export_qc.clicked.connect(self.export_qc)
        self.btn_export_pred.clicked.connect(self.export_pred)

        self.tabs.addTab(tab, "4) Export")

    # -------- Actions --------
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV files (*.csv);;All files (*.*)")
        if not path:
            return
        try:
            df = pd.read_csv(path, engine="python")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not read CSV:\n{e}")
            return

        df = normalize_colnames(df)

        self.raw_df = df
        self.std_df = None
        self.qc_info = {}
        self.pred_df = None

        self.lbl_file.setText(path)
        self.lbl_shape.setText(f"Rows: {len(df):,} | Cols: {df.shape[1]}")

        self.raw_model.set_df(df.head(25))
        self._refresh_mapping_dropdowns(df.columns.tolist())

        prof = profile_df(df)
        prof_txt = []
        prof_txt.append(f"Shape: {prof['shape']}")
        prof_txt.append(f"Exact duplicate rows: {prof['dup_rows_exact']}")
        prof_txt.append("")
        prof_txt.append("Top missingness:")
        for c, v in prof["missing_top"].items():
            prof_txt.append(f"  - {c}: {v:.1%}")
        self.txt_profile.setText("\n".join(prof_txt))

        self.lbl_apply.setText("")
        self.txt_insights.setText("")
        self.std_model.set_df(pd.DataFrame())
        self.pred_model.set_df(pd.DataFrame())

        self.tabs.setCurrentIndex(1)

    def _refresh_mapping_dropdowns(self, cols: List[str]):
        def set_items(cb: QComboBox):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("(none)")
            cb.addItems(cols)
            cb.blockSignals(False)

        for cb in (self.cb_id, self.cb_time, self.cb_group, self.cb_outcome, self.cb_target):
            set_items(cb)

        # simple guesses (generic)
        guess_id = self._guess(cols, ["participant_id", "healthcode", "subject_id", "patno", "patient_id", "id"])
        guess_time = self._guess(cols, ["createdon", "visit_date", "date", "timestamp", "created_dt", "event_time"])
        guess_group = self._guess(cols, ["sex", "gender", "site", "cohort", "arm"])
        guess_outcome = self._guess(cols, ["diagnosis", "dx", "pd", "case_control", "group", "status"])
        guess_target = self._guess(cols, ["updrs", "moca", "score", "severity", "progression", "hn_yahr"])

        self._set_combo(self.cb_id, guess_id)
        self._set_combo(self.cb_time, guess_time)
        self._set_combo(self.cb_group, guess_group)
        self._set_combo(self.cb_outcome, guess_outcome)
        self._set_combo(self.cb_target, guess_target)

    def _guess(self, cols: List[str], opts: List[str]) -> Optional[str]:
        for o in opts:
            if o in cols:
                return o
        return None

    def _set_combo(self, cb: QComboBox, val: Optional[str]):
        if val is None:
            cb.setCurrentIndex(0)
            return
        idx = cb.findText(val)
        cb.setCurrentIndex(idx if idx >= 0 else 0)

    def apply_standardise(self):
        if self.raw_df is None:
            QMessageBox.information(self, "No data", "Load a CSV first.")
            return

        self.cfg.id_col = self._cb_val(self.cb_id)
        self.cfg.time_col = self._cb_val(self.cb_time)
        self.cfg.group_col = self._cb_val(self.cb_group)
        self.cfg.outcome_col = self._cb_val(self.cb_outcome)
        self.cfg.target_col = self._cb_val(self.cb_target)

        self.cfg.drop_cols_text = self.ed_dropcols.toPlainText().strip()
        self.cfg.strict_drop_high_missing = self.chk_drop_high_missing.isChecked()
        self.cfg.drop_missing_threshold = float(self.sp_missing.value())
        self.cfg.test_size = float(self.sp_test.value())
        self.cfg.random_state = int(self.sp_seed.value())
        self.cfg.model_type = str(self.cb_model.currentText())

        try:
            std, qc = standardise_df(self.raw_df, self.cfg)
        except Exception as e:
            QMessageBox.critical(self, "Standardisation Error", f"Failed:\n{e}")
            return

        self.std_df = std
        self.qc_info = qc

        self.std_model.set_df(std.head(25))
        kept = int(qc.get("row_kept_count", 0))
        removed = int(qc.get("row_removed_count", 0))
        self.lbl_apply.setText(f"Done. Kept {kept} / Removed {removed}.")
        self.tabs.setCurrentIndex(2)

    def _cb_val(self, cb: QComboBox) -> Optional[str]:
        v = cb.currentText().strip()
        return None if v == "(none)" else v

    def generate_insights(self):
        if self.std_df is None:
            QMessageBox.information(self, "No standardised data", "Go to Configure + Standardise first.")
            return
        txt = auto_insights(self.std_df, self.cfg)

        # Append QC breakdown table if present
        bd = self.qc_info.get("row_breakdown", None)
        if isinstance(bd, pd.DataFrame) and len(bd) > 0:
            txt += "\n\nQC REMOVAL BREAKDOWN:\n"
            txt += bd.to_string(index=False)

        self.txt_insights.setText(txt)

    def run_model(self):
        if self.std_df is None:
            QMessageBox.information(self, "No standardised data", "Go to Configure + Standardise first.")
            return
        self.lbl_model.setText("Running…")
        QApplication.processEvents()

        try:
            report, pred_df = run_prediction(self.std_df, self.cfg)
        except Exception as e:
            self.lbl_model.setText("")
            QMessageBox.critical(self, "Model Error", f"Failed:\n{e}")
            return

        self.lbl_model.setText("Done.")
        current = self.txt_insights.toPlainText().strip()
        merged = (current + "\n\n" if current else "") + report
        self.txt_insights.setText(merged)

        if pred_df is not None:
            self.pred_df = pred_df.copy()
            self.pred_model.set_df(pred_df.head(25))
        else:
            self.pred_df = None
            self.pred_model.set_df(pd.DataFrame())

    def export_clean(self):
        if self.std_df is None:
            QMessageBox.information(self, "No data", "Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save standardised CSV", "standardised.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            self.std_df.to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed:\n{e}")
            return
        self.lbl_export.setText(f"Saved: {path}")

    def export_qc(self):
        if self.std_df is None:
            QMessageBox.information(self, "No data", "Nothing to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save QC log CSV", "qc_log.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            # QC log is the standardised df including qc_reason/qc_keep (already there)
            self.std_df.to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed:\n{e}")
            return
        self.lbl_export.setText(f"Saved: {path}")

    def export_pred(self):
        if self.pred_df is None or len(self.pred_df) == 0:
            QMessageBox.information(self, "No predictions", "Run a model first to export predictions.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save predictions CSV", "predictions.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            self.pred_df.to_csv(path, index=True)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed:\n{e}")
            return
        self.lbl_export.setText(f"Saved: {path}")


# -----------------------------
# Entry
# -----------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()