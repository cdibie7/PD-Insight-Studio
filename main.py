import sys
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QMessageBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QFormLayout, QLineEdit
)


# ----------------------------
# Core cleaning functions
# ----------------------------
def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-zA-Z0-9_]+", "_", str(c).strip()).lower()[:64]
        for c in df.columns
    ]
    return df

def guess_default(colnames, options):
    for o in options:
        if o in colnames:
            return o
    return None

def parse_listlike_count(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        arr = ast.literal_eval(s)
        if isinstance(arr, list) and len(arr) >= 2:
            return len(arr)
    except Exception:
        return None
    return None

def derive_med_and_proxy_dx(med_series: pd.Series):
    t = med_series.astype(str).fillna("").str.lower()

    med_state = np.select(
        [
            t.str.contains("just after") | t.str.contains("after parkinson"),
            t.str.contains("immediately before") | t.str.contains("before parkinson"),
            t.str.contains("don't take parkinson"),
        ],
        ["AFTER", "BEFORE", "NO_MEDS"],
        default="OTHER",
    )

    med_state_s = pd.Series(med_state, dtype="object")
    med_code = med_state_s.map({"NO_MEDS": 0, "BEFORE": 1, "AFTER": 2, "OTHER": 3}).astype("Int64")

    dx_proxy = pd.Series(np.where(med_state_s == "NO_MEDS", 0, 1), dtype="Int64")
    pd_state = pd.Series(np.where(dx_proxy == 1, med_state_s, pd.NA), dtype="object")
    pd_code = pd_state.map({"BEFORE": 1, "AFTER": 2, "OTHER": 3}).astype("Int64")

    return med_state_s, med_code, dx_proxy, pd_state, pd_code

def derive_tap_features(df: pd.DataFrame, tap_col: str, duration_col: str) -> pd.DataFrame:
    tap_cnt = []
    filehandle = []

    for v in df[tap_col].tolist():
        n = parse_listlike_count(v)
        if n is not None:
            tap_cnt.append(n)
            filehandle.append(0)
            continue

        try:
            x = float(v)
            if np.isfinite(x) and 0 <= x <= 300:
                tap_cnt.append(int(x))
                filehandle.append(0)
            else:
                tap_cnt.append(pd.NA)
                filehandle.append(1)
        except Exception:
            tap_cnt.append(pd.NA)
            filehandle.append(1)

    df["tap_cnt"] = pd.Series(tap_cnt, dtype="Int64")
    df["tap_samples_is_filehandle"] = pd.Series(filehandle, dtype="Int64")

    df["tap_hz"] = np.where(
        (df["tap_cnt"].notna()) & (df[duration_col].notna()) & (df[duration_col] > 0),
        df["tap_cnt"].astype(float) / df[duration_col].astype(float),
        np.nan
    )
    df["tap_per10s"] = df["tap_hz"] * 10.0
    return df

def qc_with_reasons(df: pd.DataFrame, require_tap_rate: bool, min_dur: float, max_dur: float, max_tap_hz: float):
    qc_reason = []

    for _, r in df.iterrows():
        reasons = []

        if pd.isna(r.get("pid")) or str(r.get("pid")).strip() == "":
            reasons.append("missing_pid")
        if pd.isna(r.get("session_id")) or str(r.get("session_id")).strip() == "":
            reasons.append("missing_session_id")

        dur = r.get("duration_sec")
        if pd.isna(dur):
            reasons.append("missing_duration")
        else:
            if dur <= 0:
                reasons.append("nonpositive_duration")
            if dur < min_dur or dur > max_dur:
                reasons.append(f"duration_out_of_range({min_dur}-{max_dur})")

        if require_tap_rate:
            if pd.isna(r.get("tap_cnt")):
                reasons.append("missing_tap_cnt_for_rate_qc")
            else:
                hz = r.get("tap_hz")
                if pd.isna(hz):
                    reasons.append("missing_tap_hz")
                elif hz > max_tap_hz:
                    reasons.append(f"tap_hz_too_high(>{max_tap_hz})")

        qc_reason.append(",".join(reasons) if reasons else "OK")

    out = df.copy()
    out["qc_reason"] = qc_reason
    out["qc_keep"] = (out["qc_reason"] == "OK").astype(int)

    removed = out[out["qc_keep"] == 0].copy()
    kept = out[out["qc_keep"] == 1].copy()

    breakdown = (
        removed["qc_reason"]
        .str.split(",", expand=True)
        .stack()
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="n_removed")
    )

    return kept, removed, breakdown, out

def clean_mpower(raw_df: pd.DataFrame, mapping: dict, qc_params: dict):
    df = normalize_colnames(raw_df)

    pid_col = mapping.get("pid")
    sid_col = mapping.get("session_id")
    start_col = mapping.get("start_ms")
    end_col = mapping.get("end_ms")
    med_col = mapping.get("med_timepoint")
    tap_col = mapping.get("tap_data")

    out = pd.DataFrame()
    out["pid"] = df[pid_col].astype(str) if pid_col else ""
    out["session_id"] = df[sid_col].astype(str) if sid_col else ""

    if start_col and end_col:
        start_ms = pd.to_numeric(df[start_col], errors="coerce")
        end_ms = pd.to_numeric(df[end_col], errors="coerce")
        out["duration_sec"] = (end_ms - start_ms) / 1000.0
    else:
        out["duration_sec"] = np.nan

    if med_col:
        out["med_timepoint_raw"] = df[med_col].astype(str)
        _, _, out["dx_proxy"], out["pd_state"], out["pd_code"] = derive_med_and_proxy_dx(out["med_timepoint_raw"])
    else:
        out["med_timepoint_raw"] = ""
        out["dx_proxy"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["pd_state"] = pd.Series([pd.NA] * len(out), dtype="object")
        out["pd_code"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    if tap_col:
        out["tap_samples_raw"] = df[tap_col]
        out = derive_tap_features(out, "tap_samples_raw", "duration_sec")
    else:
        out["tap_cnt"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["tap_hz"] = np.nan
        out["tap_per10s"] = np.nan
        out["tap_samples_is_filehandle"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    # final columns (SPSS-ready + qc)
    analysis_df = out[
        ["pid","session_id","med_timepoint_raw","dx_proxy","pd_state","pd_code",
         "duration_sec","tap_cnt","tap_hz","tap_per10s","tap_samples_is_filehandle"]
    ].copy()

    kept, removed, breakdown, qc_full = qc_with_reasons(
        analysis_df,
        require_tap_rate=qc_params["strict_rate_qc"],
        min_dur=qc_params["min_dur"],
        max_dur=qc_params["max_dur"],
        max_tap_hz=qc_params["max_tap_hz"],
    )

    return kept, removed, breakdown, qc_full


# ----------------------------
# Simple table helper
# ----------------------------
def df_to_tablewidget(df: pd.DataFrame, table: QTableWidget, max_rows: int = 20):
    table.clear()
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return

    view = df.head(max_rows).copy()
    table.setColumnCount(len(view.columns))
    table.setRowCount(len(view))
    table.setHorizontalHeaderLabels([str(c) for c in view.columns])

    for r in range(len(view)):
        for c, col in enumerate(view.columns):
            val = view.iloc[r, c]
            item = QTableWidgetItem("" if pd.isna(val) else str(val))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            table.setItem(r, c, item)

    table.resizeColumnsToContents()


# ----------------------------
# GUI
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mPower Cleaner → SPSS-ready")
        self.resize(1100, 700)

        self.raw_df = None
        self.norm_df = None
        self.kept = None
        self.removed = None
        self.breakdown = None
        self.qc_full = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_load = QWidget()
        self.tab_map = QWidget()
        self.tab_out = QWidget()

        self.tabs.addTab(self.tab_load, "Load Data")
        self.tabs.addTab(self.tab_map, "Column Mapping")
        self.tabs.addTab(self.tab_out, "Output + QC")

        self.build_load_tab()
        self.build_map_tab()
        self.build_out_tab()

    def build_load_tab(self):
        layout = QVBoxLayout()

        top = QHBoxLayout()
        self.btn_load = QPushButton("Load CSV")
        self.btn_load.clicked.connect(self.load_csv)
        self.lbl_shape = QLabel("Rows: - | Cols: -")
        top.addWidget(self.btn_load)
        top.addStretch()
        top.addWidget(self.lbl_shape)
        layout.addLayout(top)

        self.load_preview = QTableWidget()
        layout.addWidget(QLabel("Preview (first 20 rows)"))
        layout.addWidget(self.load_preview)

        self.tab_load.setLayout(layout)

    def build_map_tab(self):
        layout = QVBoxLayout()

        self.map_info = QLabel("Load a CSV first.")
        layout.addWidget(self.map_info)

        grid = QGroupBox("Map columns")
        form = QFormLayout()

        self.cmb_pid = QComboBox()
        self.cmb_sid = QComboBox()
        self.cmb_start = QComboBox()
        self.cmb_end = QComboBox()
        self.cmb_med = QComboBox()
        self.cmb_tap = QComboBox()

        form.addRow("pid", self.cmb_pid)
        form.addRow("session_id", self.cmb_sid)
        form.addRow("start_ms", self.cmb_start)
        form.addRow("end_ms", self.cmb_end)
        form.addRow("med_timepoint", self.cmb_med)
        form.addRow("tap_data", self.cmb_tap)

        grid.setLayout(form)
        layout.addWidget(grid)

        self.btn_run = QPushButton("Run cleaning + QC")
        self.btn_run.clicked.connect(self.run_cleaning)
        layout.addWidget(self.btn_run)

        self.tab_map.setLayout(layout)

    def build_out_tab(self):
        layout = QVBoxLayout()

        # QC controls
        qc_box = QGroupBox("QC settings")
        qc_form = QFormLayout()
        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(0, 9999)
        self.spin_min_dur.setValue(5.0)

        self.spin_max_dur = QDoubleSpinBox()
        self.spin_max_dur.setRange(0, 9999)
        self.spin_max_dur.setValue(60.0)

        self.chk_strict = QCheckBox("Strict QC (require tap count + tap rate checks)")
        self.chk_strict.setChecked(False)

        self.spin_max_hz = QDoubleSpinBox()
        self.spin_max_hz.setRange(0, 9999)
        self.spin_max_hz.setValue(10.0)

        qc_form.addRow("Min duration (sec)", self.spin_min_dur)
        qc_form.addRow("Max duration (sec)", self.spin_max_dur)
        qc_form.addRow("", self.chk_strict)
        qc_form.addRow("Max tap rate (taps/sec)", self.spin_max_hz)
        qc_box.setLayout(qc_form)
        layout.addWidget(qc_box)

        # counts
        self.lbl_counts = QLabel("Kept: - | Removed: -")
        layout.addWidget(self.lbl_counts)

        # tables
        self.tbl_clean = QTableWidget()
        self.tbl_breakdown = QTableWidget()
        self.tbl_removed = QTableWidget()

        layout.addWidget(QLabel("Cleaned SPSS-ready dataset preview"))
        layout.addWidget(self.tbl_clean)

        layout.addWidget(QLabel("Removal breakdown (reasons + counts)"))
        layout.addWidget(self.tbl_breakdown)

        layout.addWidget(QLabel("Removed rows preview (with qc_reason)"))
        layout.addWidget(self.tbl_removed)

        # export
        exp = QHBoxLayout()
        self.btn_export_kept = QPushButton("Export kept rows (SPSS-ready CSV)")
        self.btn_export_kept.clicked.connect(self.export_kept)
        self.btn_export_qc = QPushButton("Export QC log (all rows CSV)")
        self.btn_export_qc.clicked.connect(self.export_qc)
        exp.addWidget(self.btn_export_kept)
        exp.addWidget(self.btn_export_qc)
        exp.addStretch()
        layout.addLayout(exp)

        # help
        self.btn_help = QPushButton("Help (dx_proxy rules)")
        self.btn_help.clicked.connect(self.show_help)
        layout.addWidget(self.btn_help)

        self.tab_out.setLayout(layout)

    def show_help(self):
        msg = (
            "dx_proxy rule:\n"
            "- If med_timepoint contains \"I don't take Parkinson medications\" => dx_proxy=0 (Control proxy)\n"
            "- Else dx_proxy=1 (PD proxy)\n\n"
            "Within PD (dx_proxy=1):\n"
            "- \"Just after Parkinson medication\" => pd_state=AFTER (pd_code=2)\n"
            "- \"Immediately before Parkinson medication\" => pd_state=BEFORE (pd_code=1)\n"
            "- Anything else (incl \"Another time\") => pd_state=OTHER (pd_code=3)\n"
        )
        QMessageBox.information(self, "Help", msg)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path, engine="python")
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Could not read CSV:\n{e}")
            return

        self.raw_df = df
        self.norm_df = normalize_colnames(df)
        self.lbl_shape.setText(f"Rows: {len(self.raw_df):,} | Cols: {self.raw_df.shape[1]}")
        df_to_tablewidget(self.raw_df, self.load_preview, max_rows=20)

        self.populate_mapping_dropdowns()
        self.map_info.setText("Columns loaded. Confirm mapping then run cleaning.")
        self.tabs.setCurrentIndex(1)

    def populate_mapping_dropdowns(self):
        if self.norm_df is None:
            return
        cols = list(self.norm_df.columns)

        def fill_combo(cmb: QComboBox, default_guess: str | None):
            cmb.clear()
            cmb.addItem("(none)")
            for c in cols:
                cmb.addItem(c)
            if default_guess in cols:
                cmb.setCurrentText(default_guess)
            else:
                cmb.setCurrentIndex(0)

        fill_combo(self.cmb_pid, guess_default(cols, ["healthcode", "participant_id"]))
        fill_combo(self.cmb_sid, guess_default(cols, ["recordid", "session_id", "event_id"]))
        fill_combo(self.cmb_start, guess_default(cols, ["tapping_results_json_startdate", "start_ms"]))
        fill_combo(self.cmb_end, guess_default(cols, ["tapping_results_json_enddate", "end_ms"]))
        fill_combo(self.cmb_med, guess_default(cols, ["medtimepoint", "med_timepoint"]))
        fill_combo(self.cmb_tap, guess_default(cols, ["tapping_results_json_tappingsamples", "tap_samples", "tap_timestamps", "tap_times"]))

    def run_cleaning(self):
        if self.raw_df is None or self.norm_df is None:
            QMessageBox.warning(self, "No data", "Load a CSV first.")
            return

        mapping = {
            "pid": None if self.cmb_pid.currentText() == "(none)" else self.cmb_pid.currentText(),
            "session_id": None if self.cmb_sid.currentText() == "(none)" else self.cmb_sid.currentText(),
            "start_ms": None if self.cmb_start.currentText() == "(none)" else self.cmb_start.currentText(),
            "end_ms": None if self.cmb_end.currentText() == "(none)" else self.cmb_end.currentText(),
            "med_timepoint": None if self.cmb_med.currentText() == "(none)" else self.cmb_med.currentText(),
            "tap_data": None if self.cmb_tap.currentText() == "(none)" else self.cmb_tap.currentText(),
        }

        qc_params = {
            "min_dur": float(self.spin_min_dur.value()),
            "max_dur": float(self.spin_max_dur.value()),
            "strict_rate_qc": bool(self.chk_strict.isChecked()),
            "max_tap_hz": float(self.spin_max_hz.value()),
        }

        try:
            kept, removed, breakdown, qc_full = clean_mpower(self.raw_df, mapping, qc_params)
        except Exception as e:
            QMessageBox.critical(self, "Run error", f"Cleaning failed:\n{e}")
            return

        self.kept = kept
        self.removed = removed
        self.breakdown = breakdown
        self.qc_full = qc_full

        self.lbl_counts.setText(f"Kept: {len(kept):,} | Removed: {len(removed):,}")

        df_to_tablewidget(self.qc_full, self.tbl_clean, max_rows=20)
        df_to_tablewidget(self.breakdown, self.tbl_breakdown, max_rows=200)
        df_to_tablewidget(self.removed, self.tbl_removed, max_rows=20)

        self.tabs.setCurrentIndex(2)

    def export_kept(self):
        if self.qc_full is None:
            QMessageBox.warning(self, "Nothing to export", "Run cleaning first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save kept rows CSV", "mpower_spss_ready_kept.csv", "CSV Files (*.csv)")
        if not out_path:
            return
        try:
            kept = self.qc_full[self.qc_full["qc_keep"] == 1].copy()
            kept.to_csv(out_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not save file:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Saved:\n{out_path}")

    def export_qc(self):
        if self.qc_full is None:
            QMessageBox.warning(self, "Nothing to export", "Run cleaning first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save QC log CSV", "mpower_qc_log.csv", "CSV Files (*.csv)")
        if not out_path:
            return
        try:
            self.qc_full.to_csv(out_path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not save file:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Saved:\n{out_path}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
