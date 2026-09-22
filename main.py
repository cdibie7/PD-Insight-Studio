"""
main.py — PD Insight Studio
Parkinson's Disease Real-World Data Analysis

FIXES in this version:
- Combobox dropdowns now correctly store widget references so values populate
- Combobox selected text is visible (foreground forced on widget, not just style)
- Rules ARE applied before QC and insights run (was silently skipped)
- Descriptive stats, Time Trends and Report tabs now populate correctly
- Predict target dropdown now populates from live data columns
- PD-specific interpretations added to group comparison results
- Mann-Whitney U explained in-app with what the numbers mean
"""

import os, sys, json, threading, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import standardise
import qc as qc_mod
import insights as ins
import plotting as plot
import rules as rules_mod
import modeling as mdl

# ─── Theme ──────────────────────────────────────────────────────────────────
BG      = '#0f0f1a'
BG2     = '#1a1a2e'
BG3     = '#16213e'
ACCENT  = '#4a90d9'
ACCENT2 = '#7ec8e3'
TEXT    = '#e0e0e0'
SUBTEXT = '#8a9bb5'
WARN    = '#e85d04'
OK      = '#52b788'
FONT    = ('Segoe UI', 10)
FONT_B  = ('Segoe UI', 10, 'bold')
FONT_H  = ('Segoe UI', 14, 'bold')
FONT_S  = ('Segoe UI', 9)

# PD-specific interpretation dictionary (metric keyword → (pd_direction, explanation))
PD_METRIC_GUIDE = {
    'velocity':       ('LOWER in PD', 'Slower walking speed is a hallmark of PD — reflects bradykinesia (slowness of movement).'),
    'cadence':        ('LOWER in PD', 'Fewer steps per minute — PD patients take shorter, slower steps (shuffling gait).'),
    'step_length':    ('LOWER in PD', 'Shorter step length is one of the most consistent gait markers of PD due to reduced leg swing.'),
    'stride_length':  ('LOWER in PD', 'Reduced stride length mirrors step length — a core PD gait characteristic.'),
    'stride_width':   ('VARIABLE',    'Width varies; PD patients may widen their base for stability compensation.'),
    'stride_time':    ('HIGHER in PD','Each stride takes longer — reflects overall slowing of gait cycle.'),
    'step_time':      ('HIGHER in PD','Longer step time reflects slowed movement execution in PD.'),
    'gait_cycle':     ('HIGHER in PD','A longer gait cycle indicates slower overall walking rhythm.'),
    'stance':         ('HIGHER in PD','More time with foot on ground — PD patients compensate for instability by prolonging stance.'),
    'swing':          ('LOWER in PD', 'Less time in the air per step — linked to PD motor freezing and bradykinesia.'),
    'd._support':     ('HIGHER in PD','Increased double support % (both feet on ground) is a key PD indicator — reflects poor balance.'),
    'double':         ('HIGHER in PD','More time with both feet planted — indicates cautious, unstable gait in PD.'),
    'single_support': ('LOWER in PD', 'Less time balancing on one foot — reflects balance difficulties in PD.'),
    'egvi':           ('LOWER in PD', 'eGVI (Gait Variability Index) — lower scores = more irregular gait, consistent with PD motor dysfunction.'),
    'fap':            ('LOWER in PD', 'FAP (Functional Ambulation Profile) — lower scores indicate more impaired walking ability.'),
    'ambulation':     ('HIGHER in PD','Longer time to walk the walkway — reflects slower gait speed in PD.'),
    'cop':            ('VARIABLE',    'Centre of Pressure metrics reflect weight distribution and balance during stance.'),
    'foot_angle':     ('VARIABLE',    'Foot angle changes in PD; often reduced foot clearance and toe-out angle.'),
    'dop':            ('VARIABLE',    'Direction of Progression — deviations indicate asymmetric or dysregulated gait.'),
    'pressure':       ('VARIABLE',    'Integrated pressure (foot loading) — can change with PD motor impairment.'),
    'step_ratio':     ('LOWER in PD', 'Step ratio combines length and cadence — lower values indicate impaired gait economy.'),
}


def get_pd_interpretation(feature_name: str) -> tuple:
    fn = feature_name.lower()
    for key, val in PD_METRIC_GUIDE.items():
        if key in fn:
            return val
    return ('UNKNOWN', 'No specific PD interpretation available for this metric.')


class PDInsightApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PD Insight Studio")
        self.geometry("1340x860")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        self.dataframes: list = []
        self.filenames:  list = []
        self.combined_df = pd.DataFrame()
        self.qc_df       = pd.DataFrame()
        self.working_df  = pd.DataFrame()
        self.config:     dict = {}
        self.rules:      list = []
        self.plots_generated: list = []

        # Store combobox WIDGETS (not StringVars) for config mappings
        self._cfg_cb: dict = {}

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use('default')
        s.configure('TNotebook',     background=BG,  borderwidth=0)
        s.configure('TNotebook.Tab', background=BG3, foreground=TEXT,
                    font=FONT_B, padding=[14, 6], borderwidth=0)
        s.map('TNotebook.Tab',
              background=[('selected', ACCENT), ('active', BG2)],
              foreground=[('selected', 'white')])
        s.configure('TFrame',      background=BG2)
        s.configure('TLabel',      background=BG2, foreground=TEXT, font=FONT)
        s.configure('TLabelframe', background=BG2, foreground=ACCENT2)
        s.configure('TLabelframe.Label', background=BG2, foreground=ACCENT2, font=FONT_B)
        s.configure('Treeview',    background=BG3, foreground=TEXT,
                    fieldbackground=BG3, rowheight=22, font=FONT_S)
        s.configure('Treeview.Heading', background=BG, foreground=ACCENT2,
                    font=FONT_B, borderwidth=0)
        s.map('Treeview', background=[('selected', ACCENT)])
        s.configure('TScrollbar', background=BG3, troughcolor=BG)
        # Fix combobox dropdown list colours
        self.option_add('*TCombobox*Listbox.background',       BG3)
        self.option_add('*TCombobox*Listbox.foreground',       TEXT)
        self.option_add('*TCombobox*Listbox.selectBackground', ACCENT)
        self.option_add('*TCombobox*Listbox.selectForeground', 'white')
        self.option_add('*TCombobox*Listbox.font',             FONT_S)

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG, height=50)
        hdr.pack(fill='x')
        tk.Label(hdr, text='⚗  PD Insight Studio', bg=BG, fg=ACCENT2,
                 font=('Segoe UI', 15, 'bold')).pack(side='left', padx=20, pady=10)
        tk.Label(hdr, text="Parkinson's Disease Real-World Data Analysis",
                 bg=BG, fg=SUBTEXT, font=FONT_S).pack(side='left', pady=14)
        self.status_var = tk.StringVar(value='Ready.')
        tk.Label(hdr, textvariable=self.status_var, bg=BG, fg=OK,
                 font=FONT_S).pack(side='right', padx=20)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        self.tab_load     = ttk.Frame(self.nb)
        self.tab_config   = ttk.Frame(self.nb)
        self.tab_qc       = ttk.Frame(self.nb)
        self.tab_insights = ttk.Frame(self.nb)
        self.tab_predict  = ttk.Frame(self.nb)
        self.tab_export   = ttk.Frame(self.nb)

        self.nb.add(self.tab_load,     text='① Load Data')
        self.nb.add(self.tab_config,   text='② Configure')
        self.nb.add(self.tab_qc,       text='③ QC & Clean')
        self.nb.add(self.tab_insights, text='④ Insights')
        self.nb.add(self.tab_predict,  text='⑤ Predict')
        self.nb.add(self.tab_export,   text='⑥ Export')

        self._build_load_tab()
        self._build_config_tab()
        self._build_qc_tab()
        self._build_insights_tab()
        self._build_predict_tab()
        self._build_export_tab()

    # ═══════════════ TAB 1 ═══════════════════════════════════════════════════

    def _build_load_tab(self):
        top = ttk.Frame(self.tab_load); top.pack(fill='x', padx=15, pady=12)
        tk.Label(top, text='Load CSV Files', font=FONT_H, bg=BG2, fg=ACCENT2).pack(side='left')
        bf = ttk.Frame(top); bf.pack(side='right')
        self._btn(bf, '+ Add CSV',           self._load_csv,     ACCENT).pack(side='left', padx=4)
        self._btn(bf, '✕ Clear All',         self._clear_data,   WARN  ).pack(side='left', padx=4)
        self._btn(bf, '▶ Combine & Preview', self._combine_data, OK    ).pack(side='left', padx=4)

        ff = ttk.LabelFrame(self.tab_load, text='Loaded Files')
        ff.pack(fill='x', padx=15, pady=(0, 8))
        self.files_listbox = tk.Listbox(ff, bg=BG3, fg=TEXT, font=FONT_S,
                                         height=4, selectbackground=ACCENT, bd=0)
        self.files_listbox.pack(fill='x', padx=6, pady=6)

        self.load_info_var = tk.StringVar(value='No data loaded.')
        tk.Label(self.tab_load, textvariable=self.load_info_var,
                 bg=BG2, fg=SUBTEXT, font=FONT_S, anchor='w').pack(fill='x', padx=15)

        pf = ttk.LabelFrame(self.tab_load, text='Data Preview (first 20 rows)')
        pf.pack(fill='both', expand=True, padx=15, pady=8)
        self.preview_tree = self._make_treeview(pf)

    def _load_csv(self):
        paths = filedialog.askopenfilenames(
            title='Select CSV file(s)',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not paths: return
        for path in paths:
            self._set_status(f'Loading {os.path.basename(path)}...')
            try:
                df = self._smart_load_csv(path)
                df = standardise.normalise_column_names(df)
                df = standardise.infer_and_cast_types(df)
                self.dataframes.append(df)
                self.filenames.append(path)
                self.files_listbox.insert(
                    'end', f'  {os.path.basename(path)}  [{df.shape[0]:,} rows × {df.shape[1]} cols]')
                self._set_status(f'Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols')
            except Exception as e:
                messagebox.showerror('Load Error', f'Failed:\n{e}')

    def _smart_load_csv(self, path):
        """
        Intelligently load CSV files, handling multi-header formats like PKMAS
        where row 1 = metric name and row 2 = sub-metric (Mean, SD, etc).
        Flattens multi-headers into combined column names like
        'step_length_cm_mean', 'step_length_cm_sd' etc.
        """
        # First peek at the file to detect multi-header
        peek = pd.read_csv(path, nrows=3, low_memory=False, on_bad_lines='skip', header=None)

        # Check if second row looks like sub-headers (contains words like Mean, SD, %CV etc)
        sub_header_keywords = {'mean', 'sd', '%cv', 'samples', 'median', 'min', 'max',
                               'unnamed', '#samples', 'mean - ratio', 'mean - asi'}
        row1_vals = [str(v).strip().lower() for v in peek.iloc[1] if pd.notna(v) and str(v).strip()]
        is_multi_header = sum(1 for v in row1_vals if any(kw in v for kw in sub_header_keywords)) > 3

        if is_multi_header:
            # Load with two header rows and flatten
            df = pd.read_csv(path, header=[0, 1], low_memory=False, on_bad_lines='skip')
            flat_cols = []
            for col in df.columns:
                top = str(col[0]).strip()
                sub = str(col[1]).strip()
                # Skip unnamed sub-headers (single-value columns like velocity, FAP etc)
                if 'unnamed' in sub.lower() or sub == '':
                    flat_cols.append(top)
                else:
                    flat_cols.append(f"{top} {sub}")
            df.columns = flat_cols
            return df
        else:
            return pd.read_csv(path, low_memory=False, on_bad_lines='skip')

    def _clear_data(self):
        self.dataframes.clear(); self.filenames.clear()
        self.combined_df = pd.DataFrame()
        self.qc_df       = pd.DataFrame()
        self.working_df  = pd.DataFrame()
        self.files_listbox.delete(0, 'end')
        self.load_info_var.set('No data loaded.')
        self._clear_treeview(self.preview_tree)

    def _combine_data(self):
        if not self.dataframes:
            messagebox.showwarning('No Data', 'Load at least one CSV first.'); return
        try:
            self.combined_df = standardise.merge_datasets(self.dataframes)
            s = self.combined_df.shape
            cp = ', '.join(self.combined_df.columns[:12].tolist())
            if s[1] > 12: cp += '...'
            self.load_info_var.set(f'Combined: {s[0]:,} rows × {s[1]} columns  |  {cp}')
            self._fill_treeview(self.preview_tree, self.combined_df.head(20))
            self._populate_all_dropdowns()
            self._set_status('Data ready. Proceed to ② Configure.')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    # ═══════════════ TAB 2 ═══════════════════════════════════════════════════

    def _build_config_tab(self):
        canvas = tk.Canvas(self.tab_config, bg=BG2, highlightthickness=0)
        sb = ttk.Scrollbar(self.tab_config, orient='vertical', command=canvas.yview)
        self._cfg_inner = ttk.Frame(canvas)
        self._cfg_inner.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self._cfg_inner, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        inn = self._cfg_inner

        tk.Label(inn, text='Configure Variables', font=FONT_H,
                 bg=BG2, fg=ACCENT2).pack(anchor='w', padx=15, pady=12)
        tk.Label(inn,
                 text='Map your dataset columns to the roles below. Load and Combine data first — dropdowns will fill automatically.',
                 bg=BG2, fg=SUBTEXT, font=FONT_S, wraplength=800).pack(anchor='w', padx=15)

        # ── Column Mapping ──
        mf = ttk.LabelFrame(inn, text='Column Mapping')
        mf.pack(fill='x', padx=15, pady=8)

        self._cfg_cb['entity_id']   = self._map_row(mf, 'Patient / Subject ID column *', 0)
        self._cfg_cb['time_col']    = self._map_row(mf, 'Time / Visit / Condition column  (e.g. task, visit_month)', 1)
        self._cfg_cb['group_col']   = self._map_row(mf, 'Group variable  (e.g. PD vs Control, medication state)', 2)
        self._cfg_cb['outcome_col'] = self._map_row(mf, 'Primary outcome / target column  (e.g. velocity, UPDRS)', 3)

        # ── QC Settings ──
        qf = ttk.LabelFrame(inn, text='QC Settings')
        qf.pack(fill='x', padx=15, pady=8)
        self._col_miss_var    = self._spin_row(qf, 'Column missingness threshold — flag column if % missing exceeds this', 0, 80)
        self._row_miss_var    = self._spin_row(qf, 'Row missingness threshold — flag row if % missing exceeds this',       1, 80)
        self._outlier_thr_var = self._spin_row(qf, 'Outlier threshold  (IQR multiplier or z-score cutoff)',               2,  3, step=0.5)
        self._outlier_method_var = self._radio_row(qf, 'Outlier detection method', 3,
                                                    ['iqr', 'zscore'],
                                                    ['IQR (robust, recommended for clinical data)', 'Z-score'])

        # ── Rule Builder ──
        rf = ttk.LabelFrame(inn, text='Rule Builder — Create Derived Columns (optional)')
        rf.pack(fill='x', padx=15, pady=8)
        tk.Label(rf, bg=BG2, fg=SUBTEXT, font=FONT_S,
                 text='Create new columns from existing ones without writing code.\n'
                      'e.g. map a medication text column → BEFORE / AFTER / NO_MEDS labels.').pack(anchor='w', padx=8, pady=(4,2))
        rb = ttk.Frame(rf); rb.pack(fill='x', padx=8)
        self._btn(rb, '+ Keyword Map',           self._add_keyword_rule, ACCENT).pack(side='left', padx=4, pady=4)
        self._btn(rb, '+ Formula',               self._add_formula_rule, ACCENT).pack(side='left', padx=4, pady=4)
        self._btn(rb, '+ Months Since Baseline', self._add_months_rule,  ACCENT).pack(side='left', padx=4, pady=4)

        self.rule_listbox = tk.Listbox(rf, bg=BG3, fg=TEXT, font=FONT_S,
                                        height=4, selectbackground=ACCENT, bd=0)
        self.rule_listbox.pack(fill='x', padx=8, pady=4)
        self._btn(rf, '✕ Remove Selected Rule', self._remove_rule, WARN).pack(anchor='w', padx=8, pady=2)

        # ── Save / Load ──
        scr = ttk.Frame(inn); scr.pack(fill='x', padx=15, pady=6)
        tk.Label(scr, text='Save / load config as JSON:', bg=BG2, fg=SUBTEXT, font=FONT_S).pack(side='left')
        self._btn(scr, '💾 Save', self._save_config, BG3).pack(side='left', padx=6)
        self._btn(scr, '📂 Load', self._load_config, BG3).pack(side='left', padx=4)

        self._btn(inn, '▶  Apply Configuration & Proceed to QC',
                  self._apply_config, OK, large=True).pack(pady=14)

    def _map_row(self, parent, label, row):
        """Return a Combobox widget (not StringVar)."""
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky='ew', padx=8, pady=6)
        parent.columnconfigure(0, weight=1)
        tk.Label(f, text=label, bg=BG2, fg=TEXT, font=FONT_S,
                 width=58, anchor='w').pack(side='left')
        cb = ttk.Combobox(f, values=['— not mapped —'], state='readonly', width=35)
        cb.set('— not mapped —')
        cb.configure(foreground=TEXT)
        cb.pack(side='left', padx=6)
        return cb

    def _spin_row(self, parent, label, row, default, step=1):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky='ew', padx=8, pady=4)
        parent.columnconfigure(0, weight=1)
        tk.Label(f, text=label, bg=BG2, fg=TEXT, font=FONT_S,
                 width=58, anchor='w').pack(side='left')
        var = tk.DoubleVar(value=default)
        ttk.Spinbox(f, from_=0, to=100, increment=step,
                    textvariable=var, width=8).pack(side='left', padx=6)
        return var

    def _radio_row(self, parent, label, row, values, display):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky='ew', padx=8, pady=4)
        parent.columnconfigure(0, weight=1)
        tk.Label(f, text=label, bg=BG2, fg=TEXT, font=FONT_S,
                 width=58, anchor='w').pack(side='left')
        var = tk.StringVar(value=values[0])
        for val, disp in zip(values, display):
            tk.Radiobutton(f, text=disp, variable=var, value=val,
                           bg=BG2, fg=TEXT, selectcolor=ACCENT, font=FONT_S,
                           activebackground=BG2, activeforeground=ACCENT2).pack(side='left', padx=6)
        return var

    def _cb_val(self, key):
        """Get string value from a config combobox, None if placeholder."""
        cb = self._cfg_cb.get(key)
        if cb is None: return None
        v = cb.get()
        return None if '— not mapped —' in v or v.strip() == '' else v

    def _populate_all_dropdowns(self):
        """Refresh every combobox/listbox with current column names."""
        df = self.combined_df if not self.combined_df.empty else None
        if df is None: return
        all_cols   = list(df.columns)
        map_cols   = ['— not mapped —'] + all_cols
        num_cols   = list(df.select_dtypes(include='number').columns)

        # Config mapping rows
        for key in ('entity_id', 'time_col', 'group_col', 'outcome_col'):
            cb = self._cfg_cb.get(key)
            if cb and isinstance(cb, ttk.Combobox):
                cur = cb.get()
                cb['values'] = map_cols
                cb.set(cur if cur in map_cols else map_cols[0])

        # Insights group combobox
        if hasattr(self, 'grp_cb'):
            cur = self.grp_cb.get()
            opts = ['— use configured group —'] + all_cols
            self.grp_cb['values'] = opts
            self.grp_cb.set(cur if cur in opts else opts[0])

        # Insights progression outcome
        if hasattr(self, 'prog_outcome_cb'):
            cur = self.prog_outcome_cb.get()
            opts = ['— select outcome —'] + num_cols
            self.prog_outcome_cb['values'] = opts
            self.prog_outcome_cb.set(cur if cur in opts else opts[0])

        # Predict target  ← was never populated in original
        if hasattr(self, 'pred_target_cb'):
            cur = self.pred_target_cb.get()
            opts = ['— select target —'] + all_cols
            self.pred_target_cb['values'] = opts
            self.pred_target_cb.set(cur if cur in opts else opts[0])

    def _apply_config(self):
        if self.combined_df.empty:
            messagebox.showwarning('No Data', 'Load and combine data first (Tab ①).'); return

        self.config = {
            'entity_id':       self._cb_val('entity_id'),
            'time_col':        self._cb_val('time_col'),
            'group_col':       self._cb_val('group_col'),
            'outcome_col':     self._cb_val('outcome_col'),
            'col_missingness': float(self._col_miss_var.get()) / 100,
            'row_missingness': float(self._row_miss_var.get()) / 100,
            'outlier_method':  self._outlier_method_var.get(),
            'outlier_thr':     float(self._outlier_thr_var.get()),
            'rules':           self.rules,
        }

        # Apply rules NOW and refresh dropdowns
        if self.rules:
            try:
                self.combined_df = rules_mod.apply_rules(self.combined_df, self.rules)
                self._populate_all_dropdowns()
                self._set_status(f'Applied {len(self.rules)} rule(s). New columns added to dropdowns.')
            except Exception as e:
                messagebox.showerror('Rule Error', str(e))
                return

        self._set_status('Config applied. Proceed to ③ QC.')
        self.nb.select(self.tab_qc)

    def _add_keyword_rule(self):
        if self.combined_df.empty: messagebox.showinfo('No Data', 'Load data first.'); return
        dlg = KeywordRuleDialog(self, list(self.combined_df.columns))
        self.wait_window(dlg)
        if dlg.result:
            self.rules.append(dlg.result)
            self.rule_listbox.insert('end',
                f"  Keyword map: {dlg.result['source_col']} → {dlg.result['output_col']}")

    def _add_formula_rule(self):
        if self.combined_df.empty: messagebox.showinfo('No Data', 'Load data first.'); return
        dlg = FormulaRuleDialog(self, list(self.combined_df.columns))
        self.wait_window(dlg)
        if dlg.result:
            self.rules.append(dlg.result)
            self.rule_listbox.insert('end',
                f"  Formula: {dlg.result['expression']} → {dlg.result['output_col']}")

    def _add_months_rule(self):
        if self.combined_df.empty: messagebox.showinfo('No Data', 'Load data first.'); return
        dlg = MonthsRuleDialog(self, list(self.combined_df.columns))
        self.wait_window(dlg)
        if dlg.result:
            self.rules.append(dlg.result)
            self.rule_listbox.insert('end',
                f"  Months since baseline: {dlg.result['time_col']} → {dlg.result['output_col']}")

    def _remove_rule(self):
        for i in reversed(self.rule_listbox.curselection()):
            self.rule_listbox.delete(i)
            if i < len(self.rules): self.rules.pop(i)

    def _save_config(self):
        p = filedialog.asksaveasfilename(defaultextension='.json',
                                          filetypes=[('JSON','*.json')])
        if p:
            with open(p, 'w') as f: json.dump(self.config, f, indent=2, default=str)
            messagebox.showinfo('Saved', f'Saved to {p}')

    def _load_config(self):
        p = filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if p:
            try:
                with open(p) as f: self.config = json.load(f)
                self.rules = self.config.get('rules', [])
                messagebox.showinfo('Loaded', 'Config loaded. Review and click Apply.')
            except Exception as e:
                messagebox.showerror('Error', str(e))

    # ═══════════════ TAB 3 ═══════════════════════════════════════════════════

    def _build_qc_tab(self):
        top = ttk.Frame(self.tab_qc); top.pack(fill='x', padx=15, pady=12)
        tk.Label(top, text='Quality Control', font=FONT_H, bg=BG2, fg=ACCENT2).pack(side='left')
        self._btn(top, '▶ Run QC', self._run_qc, OK).pack(side='right', padx=4)

        self.qc_summary_text = scrolledtext.ScrolledText(
            self.tab_qc, height=14, bg=BG3, fg=TEXT, font=FONT_S, wrap='word', bd=0)
        self.qc_summary_text.pack(fill='x', padx=15, pady=8)
        self.qc_summary_text.insert('end', 'Run QC to see results here.')
        self.qc_summary_text.configure(state='disabled')

        fl = ttk.LabelFrame(self.tab_qc, text='Flagged Rows Preview')
        fl.pack(fill='both', expand=True, padx=15, pady=8)
        self.qc_flagged_tree = self._make_treeview(fl)

    def _run_qc(self):
        if self.combined_df.empty:
            messagebox.showwarning('No Data', 'Load and configure data first.'); return
        if not self.config:
            messagebox.showwarning('No Config', 'Apply configuration in Tab ② first.'); return

        cfg = self.config
        num_cols = list(self.combined_df.select_dtypes(include='number').columns)[:25]

        def _do():
            try:
                self.qc_df = qc_mod.run_qc(
                    self.combined_df,
                    required_cols=[c for c in [cfg.get('entity_id')] if c],
                    col_missingness_threshold=cfg.get('col_missingness', 0.8),
                    row_missingness_threshold=cfg.get('row_missingness', 0.8),
                    duplicate_keys=[c for c in [cfg.get('entity_id'), cfg.get('time_col')] if c],
                    outlier_method=cfg.get('outlier_method', 'iqr'),
                    outlier_threshold=cfg.get('outlier_thr', 3.0),
                    numeric_cols=num_cols,
                )
                self.working_df = (self.qc_df[self.qc_df['qc_keep']]
                                   .drop(columns=['qc_keep','qc_reason'], errors='ignore')
                                   .reset_index(drop=True))
                summary = qc_mod.qc_summary(self.qc_df)
                self.after(0, lambda: self._populate_all_dropdowns())
                self.after(0, lambda: self._display_qc_results(summary))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('QC Error',
                    str(e) + '\n' + traceback.format_exc()))

        threading.Thread(target=_do, daemon=True).start()
        self._set_status('Running QC...')

    def _display_qc_results(self, summary):
        lines = ['='*55, '  QC RESULTS', '='*55,
                 f"  Total rows:     {summary['total']:>8,}",
                 f"  Rows kept:      {summary['kept']:>8,}  ({summary['kept_pct']}%)",
                 f"  Rows flagged:   {summary['removed']:>8,}",
                 '', '  Removal Reasons:']
        for r, c in list(summary['reason_counts'].items())[:10]:
            lines.append(f"    • {r:<44} {c:>6,} rows")
        if not summary['reason_counts']:
            lines.append('    (none — all rows passed)')
        lines += ['', '  Columns with >20% missingness:']
        for col, pct in list(summary['high_missingness_cols'].items())[:10]:
            lines.append(f"    • {col:<44} {pct:>5}% missing")
        if not summary['high_missingness_cols']:
            lines.append('    (none)')

        self.qc_summary_text.configure(state='normal')
        self.qc_summary_text.delete('1.0', 'end')
        self.qc_summary_text.insert('end', '\n'.join(lines))
        self.qc_summary_text.configure(state='disabled')

        flagged = self.qc_df[~self.qc_df['qc_keep']].head(50)
        self._fill_treeview(self.qc_flagged_tree, flagged)
        self._set_status(f"QC done: {summary['kept']:,} kept, {summary['removed']:,} flagged.")
        self.nb.select(self.tab_insights)

    # ═══════════════ TAB 4 ═══════════════════════════════════════════════════

    def _build_insights_tab(self):
        top = ttk.Frame(self.tab_insights); top.pack(fill='x', padx=15, pady=10)
        tk.Label(top, text='Insights & Analysis', font=FONT_H,
                 bg=BG2, fg=ACCENT2).pack(side='left')
        self._btn(top, '▶ Generate All Insights', self._run_insights, OK).pack(side='right', padx=4)

        self.ins_nb = ttk.Notebook(self.tab_insights)
        self.ins_nb.pack(fill='both', expand=True, padx=10, pady=6)

        self.ins_desc_tab   = ttk.Frame(self.ins_nb)
        self.ins_groups_tab = ttk.Frame(self.ins_nb)
        self.ins_time_tab   = ttk.Frame(self.ins_nb)
        self.ins_report_tab = ttk.Frame(self.ins_nb)

        self.ins_nb.add(self.ins_desc_tab,   text='Descriptive Stats')
        self.ins_nb.add(self.ins_groups_tab, text='Group Comparisons')
        self.ins_nb.add(self.ins_time_tab,   text='Time Trends')
        self.ins_nb.add(self.ins_report_tab, text='Report')

        # ── Descriptive ──
        df_fr = ttk.LabelFrame(self.ins_desc_tab, text='Numeric Summary Table')
        df_fr.pack(fill='x', padx=10, pady=6)
        self.desc_tree = self._make_treeview(df_fr, height=8)

        leg_fr = ttk.LabelFrame(self.ins_desc_tab, text='Column meanings')
        leg_fr.pack(fill='x', padx=10, pady=(0,6))
        tk.Label(leg_fr, bg=BG2, fg=SUBTEXT, font=FONT_S, anchor='w', justify='left',
                 text='  n = non-missing count   |   missing_pct = % rows without a value   |   '
                      'mean = average value   |   sd = standard deviation (how spread out values are)   |   '
                      'median = middle value (not affected by extreme outliers)   |   '
                      'iqr = interquartile range (spread of the middle 50% of values)',
                 wraplength=900).pack(padx=8, pady=4)

        self.desc_plot_frame = ttk.LabelFrame(self.ins_desc_tab, text='Distributions (split by group)')
        self.desc_plot_frame.pack(fill='both', expand=True, padx=10, pady=6)

        # ── Group Comparisons ──
        g_ctrl = ttk.Frame(self.ins_groups_tab); g_ctrl.pack(fill='x', padx=10, pady=8)
        tk.Label(g_ctrl, text='Compare by group:', bg=BG2, fg=TEXT, font=FONT_S).pack(side='left', padx=4)
        self.grp_cb = ttk.Combobox(g_ctrl, values=['— use configured group —'],
                                    width=34, state='readonly')
        self.grp_cb.set('— use configured group —')
        self.grp_cb.configure(foreground=TEXT)
        self.grp_cb.pack(side='left', padx=4)
        self._btn(g_ctrl, '▶ Compare Groups', self._run_group_comparison, ACCENT).pack(side='left', padx=6)

        stat_leg = ttk.LabelFrame(self.ins_groups_tab, text='How to read the comparison table')
        stat_leg.pack(fill='x', padx=10, pady=(0,4))
        tk.Label(stat_leg, bg=BG2, fg=SUBTEXT, font=FONT_S, anchor='w', justify='left',
                 wraplength=1000,
                 text='  test = Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups) — non-parametric tests '
                      'that do NOT assume data is normally distributed, making them appropriate for gait data.\n'
                      '  p_value = probability the difference occurred by chance. p < 0.05 is typically considered statistically significant.\n'
                      '  effect_size = Cohen\'s d: the practical size of the difference. '
                      '0.2 = small,  0.5 = medium,  0.8 = large.  Effect size matters more than p-value for clinical relevance.\n'
                      '  pd_expected = the direction of this metric that is typically seen in Parkinson\'s disease patients.'
                 ).pack(padx=8, pady=4)

        cmp_fr = ttk.LabelFrame(self.ins_groups_tab,
                                 text='Comparison Results  (sorted by effect size — biggest differences first)')
        cmp_fr.pack(fill='x', padx=10, pady=4)
        self.cmp_tree = self._make_treeview(cmp_fr, height=7)

        self.pd_interp_text = scrolledtext.ScrolledText(
            self.ins_groups_tab, height=7, bg=BG3, fg=TEXT, font=FONT_S, bd=0, wrap='word')
        self.pd_interp_text.pack(fill='x', padx=10, pady=4)
        self.pd_interp_text.insert('end', 'PD interpretations will appear here after running comparisons.')
        self.pd_interp_text.configure(state='disabled')

        self.effect_plot_frame = ttk.LabelFrame(self.ins_groups_tab, text='Effect Size Chart')
        self.effect_plot_frame.pack(fill='both', expand=True, padx=10, pady=4)

        # ── Time Trends ──
        t_ctrl = ttk.Frame(self.ins_time_tab); t_ctrl.pack(fill='x', padx=10, pady=8)
        tk.Label(t_ctrl, text='Outcome to analyse:', bg=BG2, fg=TEXT, font=FONT_S).pack(side='left', padx=4)
        self.prog_outcome_cb = ttk.Combobox(t_ctrl, values=['— select outcome —'],
                                             width=30, state='readonly')
        self.prog_outcome_cb.set('— select outcome —')
        self.prog_outcome_cb.configure(foreground=TEXT)
        self.prog_outcome_cb.pack(side='left', padx=4)
        self._btn(t_ctrl, '▶ Analyse', self._run_time_trends, ACCENT).pack(side='left', padx=6)

        tk.Label(self.ins_time_tab, bg=BG2, fg=SUBTEXT, font=FONT_S, anchor='w', wraplength=900,
                 text='Time Trends compare your outcome across the time/condition column set in Configure. '
                      'For this gait dataset that would be HurriedPace vs SelfPace conditions. '
                      'For longitudinal data (e.g. PPMI) it shows change over months/visits.').pack(
                      anchor='w', padx=10)

        self.prog_text = scrolledtext.ScrolledText(
            self.ins_time_tab, height=10, bg=BG3, fg=TEXT, font=FONT_S, bd=0)
        self.prog_text.pack(fill='x', padx=10, pady=6)
        self.prog_text.insert('end', 'No time/condition analysis yet. Click Analyse above.')
        self.prog_text.configure(state='disabled')

        self.prog_plot_frame = ttk.LabelFrame(self.ins_time_tab, text='Condition / Time Plots')
        self.prog_plot_frame.pack(fill='both', expand=True, padx=10, pady=4)

        # ── Report ──
        self.report_text = scrolledtext.ScrolledText(
            self.ins_report_tab, bg=BG3, fg=TEXT, font=FONT_S, bd=0, wrap='word')
        self.report_text.pack(fill='both', expand=True, padx=10, pady=8)
        self.report_text.insert('end', 'Click "Generate All Insights" to build this report.')
        self.report_text.configure(state='disabled')

    def _get_working_df(self):
        if not self.working_df.empty:
            return self.working_df
        if not self.qc_df.empty:
            return (self.qc_df[self.qc_df['qc_keep']]
                    .drop(columns=['qc_keep','qc_reason'], errors='ignore')
                    .reset_index(drop=True))
        return self.combined_df

    def _best_numeric_cols(self, df, max_n=20):
        """Prefer mean columns for gait data; fall back to all numeric."""
        num = list(df.select_dtypes(include="number").columns)
        mean_cols = [c for c in num
                     if "mean" in c.lower()
                     and "ratio" not in c.lower()
                     and "asi" not in c.lower()
                     and "sample" not in c.lower()
                     and "%cv" not in c.lower()
                     and "_cv" not in c.lower()]
        if mean_cols:
            return mean_cols[:max_n]
        clean_num = [c for c in num
                     if "%cv" not in c.lower()
                     and "unnamed" not in c.lower()
                     and "sample" not in c.lower()]
        return (clean_num[:max_n] if clean_num else num[:max_n])

    def _run_insights(self):
        df = self._get_working_df()
        if df.empty:
            messagebox.showwarning('No Data', 'Load data and run QC first.'); return

        cfg       = self.config
        use_cols  = self._best_numeric_cols(df)
        group_col = cfg.get('group_col')
        out_col   = cfg.get('outcome_col')

        def _do():
            try:
                desc_df = ins.describe_numeric(df, use_cols)
                cmp_df  = pd.DataFrame()
                if group_col and group_col in df.columns:
                    cmp_df = ins.compare_groups(df, use_cols, group_col)

                qc_sum = qc_mod.qc_summary(self.qc_df) if not self.qc_df.empty else \
                         {'total': len(df), 'kept': len(df), 'removed': 0,
                          'kept_pct': 100, 'reason_counts': {}, 'high_missingness_cols': {}}

                narrative = self._build_narrative(df, desc_df, cmp_df, qc_sum, group_col, out_col)

                self.after(0, lambda: self._display_insights(
                    desc_df, cmp_df, df, narrative, group_col, use_cols))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Insights Error',
                    str(e) + '\n' + traceback.format_exc()))

        threading.Thread(target=_do, daemon=True).start()
        self._set_status('Generating insights...')

    def _display_insights(self, desc_df, cmp_df, df, narrative, group_col, use_cols):
        # ── Descriptive Stats tab ──
        self._fill_treeview(self.desc_tree, desc_df)

        for w in self.desc_plot_frame.winfo_children():
            w.destroy()
        for col in use_cols[:4]:
            try:
                fig = plot.make_histogram(df, col, group_col=group_col, figsize=(5, 3))
                self._embed_figure(fig, self.desc_plot_frame, side='left')
            except Exception:
                pass

        # ── Group Comparisons tab ──
        self._display_comparison_results(cmp_df, df, group_col)

        # ── Report tab ──
        self.report_text.configure(state='normal')
        self.report_text.delete('1.0', 'end')
        self.report_text.insert('end', narrative)
        self.report_text.configure(state='disabled')

        self._populate_all_dropdowns()
        self._set_status('Insights generated!')
        self.ins_nb.select(self.ins_groups_tab)

    def _display_comparison_results(self, cmp_df, df, group_col):
        if cmp_df is None or cmp_df.empty:
            return

        cmp_ann = cmp_df.copy()
        cmp_ann['pd_expected'] = cmp_ann['feature'].apply(
            lambda f: get_pd_interpretation(f)[0])
        sorted_cmp = (cmp_ann.sort_values('effect_size', ascending=False)
                      if 'effect_size' in cmp_ann.columns else cmp_ann)
        self._fill_treeview(self.cmp_tree, sorted_cmp.head(25))

        # PD interpretation panel
        interp = ['PD INDICATORS — What each metric tells us about Parkinson\'s Disease\n' + '─'*65]
        for _, row in sorted_cmp.head(12).iterrows():
            feat      = row.get('feature', '')
            es        = row.get('effect_size', None)
            pval      = row.get('p_value', None)
            direction, explanation = get_pd_interpretation(feat)
            sig       = '★ SIGNIFICANT' if (pval is not None and pval < 0.05) else '  not significant'
            es_lbl    = ''
            if es is not None and not pd.isna(es):
                ae = abs(es)
                if ae >= 0.8:   es_lbl = '[LARGE]'
                elif ae >= 0.5: es_lbl = '[medium]'
                elif ae >= 0.2: es_lbl = '[small]'
                else:           es_lbl = '[negligible]'
            interp.append(
                f"\n▸ {feat}\n"
                f"  Expected direction in PD: {direction}\n"
                f"  Effect size d={round(es,3) if es is not None and not pd.isna(es) else 'n/a'} {es_lbl}   "
                f"p={round(pval,4) if pval is not None else 'n/a'}  {sig}\n"
                f"  What it means: {explanation}"
            )
        self.pd_interp_text.configure(state='normal')
        self.pd_interp_text.delete('1.0', 'end')
        self.pd_interp_text.insert('end', '\n'.join(interp))
        self.pd_interp_text.configure(state='disabled')

        try:
            fig = plot.make_group_effect_chart(cmp_df)
            for w in self.effect_plot_frame.winfo_children():
                w.destroy()
            self._embed_figure(fig, self.effect_plot_frame)
            self.plots_generated.append(('effect_sizes', fig))
        except Exception:
            pass

    def _run_group_comparison(self):
        df = self._get_working_df()
        if df.empty:
            messagebox.showwarning('No Data', 'No data available.'); return

        group_col = self.grp_cb.get()
        if '— use configured' in group_col:
            group_col = self.config.get('group_col')
        if not group_col or group_col not in df.columns:
            messagebox.showwarning('No Group',
                f'Column "{group_col}" not found. Select a valid group column.'); return

        use_cols = self._best_numeric_cols(df)
        cmp_df   = ins.compare_groups(df, use_cols, group_col)

        if cmp_df.empty:
            messagebox.showinfo('No Results',
                'No comparisons computed. Check that the group column has 2+ groups and numeric columns exist.')
            return

        self._display_comparison_results(cmp_df, df, group_col)
        self._set_status(f'Comparison done: {group_col}')

    def _run_time_trends(self):
        df = self._get_working_df()
        if df.empty:
            messagebox.showwarning('No Data', 'No data available.'); return

        cfg       = self.config
        time_col  = cfg.get('time_col')
        outcome   = self.prog_outcome_cb.get()
        if '— select' in outcome:
            outcome = cfg.get('outcome_col')
        entity_id = cfg.get('entity_id')

        if not time_col or time_col not in df.columns:
            messagebox.showwarning('Missing Config',
                'No time/condition column mapped. Set it in ② Configure.'); return
        if not outcome or outcome not in df.columns:
            messagebox.showwarning('No Outcome', 'Select an outcome from the dropdown.'); return

        is_cat = df[time_col].dtype == object or df[time_col].nunique() < 15
        lines  = [
            f'TIME / CONDITION ANALYSIS\n{"─"*55}',
            f'  Condition column : {time_col}',
            f'  Outcome          : {outcome}',
            f'  Conditions found : {sorted(df[time_col].dropna().unique().tolist())}',
            ''
        ]

        if is_cat:
            cmp_df = ins.compare_groups(df, [outcome], time_col)
            for grp in df[time_col].dropna().unique():
                vals = pd.to_numeric(df.loc[df[time_col]==grp, outcome], errors='coerce').dropna()
                if len(vals):
                    lines.append(f'  {str(grp):<28} mean={vals.mean():.4f}  sd={vals.std():.4f}  n={len(vals)}')
            if not cmp_df.empty:
                r  = cmp_df.iloc[0]
                es = r.get('effect_size', None)
                pv = r.get('p_value', None)
                lines += ['',
                    f'  Effect size (Cohen\'s d): {round(es,3) if es is not None else "n/a"}',
                    f'  p-value: {round(pv,4) if pv is not None else "n/a"}',
                    '',
                    f'  PD note: {get_pd_interpretation(outcome)[1]}']
        else:
            if entity_id and entity_id in df.columns:
                df2 = df.copy()
                df2['_t'] = pd.to_numeric(df[time_col], errors='coerce')
                slopes_df = ins.compute_slopes(df2, entity_id, '_t', outcome)
                prog = ins.progression_summary(slopes_df)
                for k, v in prog.items():
                    lines.append(f'  {k.replace("_"," ").title():<35} {v}')
            else:
                lines.append('  (Map a patient ID column in Configure to see per-subject slopes)')

        self.prog_text.configure(state='normal')
        self.prog_text.delete('1.0', 'end')
        self.prog_text.insert('end', '\n'.join(lines))
        self.prog_text.configure(state='disabled')

        for w in self.prog_plot_frame.winfo_children():
            w.destroy()
        try:
            if is_cat:
                fig1 = plot.make_boxplot(df, [outcome], group_col=time_col, figsize=(7, 4))
                self._embed_figure(fig1, self.prog_plot_frame, side='left')
                self.plots_generated.append(('condition_boxplot', fig1))
                fig2 = plot.make_histogram(df, outcome, group_col=time_col, figsize=(6, 4))
                self._embed_figure(fig2, self.prog_plot_frame, side='left')
                self.plots_generated.append(('condition_histogram', fig2))
        except Exception:
            pass

        self._set_status('Time/condition analysis done.')

    def _build_narrative(self, df, desc_df, cmp_df, qc_sum, group_col, outcome_col):
        dsname = ', '.join([os.path.basename(f) for f in self.filenames]) or 'Dataset'
        L = [
            '# PD Insight Studio — Analysis Report', '',
            f'**Dataset:** {dsname}',
            f'**Shape after QC:** {df.shape[0]:,} rows × {df.shape[1]} columns',
            '', '---', '## 1. Quality Control', '',
            f'- Raw rows: **{qc_sum["total"]:,}**',
            f'- Rows passed QC: **{qc_sum["kept"]:,}** ({qc_sum["kept_pct"]}%)',
            f'- Rows flagged: **{qc_sum["removed"]:,}**',
        ]
        if qc_sum['reason_counts']:
            L.append('\n**Top flagging reasons:**')
            for r,c in list(qc_sum['reason_counts'].items())[:5]:
                L.append(f'- `{r}`: {c} rows')

        if desc_df is not None and not desc_df.empty:
            L += ['', '---', '## 2. Descriptive Statistics', '',
                  'Key gait metrics across all participants:']
            for _, r in desc_df.head(8).iterrows():
                L.append(f'- **{r["column"]}**: mean={r["mean"]}, SD={r["sd"]}, '
                         f'median={r["median"]}, missing={r["missing_pct"]}%')

        if cmp_df is not None and not cmp_df.empty and group_col:
            L += ['', '---', f'## 3. Group Comparisons — {group_col}', '',
                  '### Statistical Method Used',
                  'The **Mann-Whitney U test** is a non-parametric test that compares whether values from '
                  'one group tend to be higher or lower than another, without assuming normal distribution. '
                  'This is appropriate for clinical gait data which is often skewed.',
                  '',
                  '**Effect size (Cohen\'s d)** measures the practical significance of the difference:',
                  '- d = 0.2 → small difference', '- d = 0.5 → medium difference',
                  '- d = 0.8 → large difference (clinically meaningful)',
                  '', '### Top Findings:']
            top5 = (cmp_df.sort_values('effect_size', ascending=False).head(5)
                    if 'effect_size' in cmp_df.columns else cmp_df.head(5))
            for _, r in top5.iterrows():
                feat = r.get('feature','')
                es   = r.get('effect_size', None)
                pv   = r.get('p_value', None)
                direction, expl = get_pd_interpretation(feat)
                sig = '✓ statistically significant' if (pv and pv < 0.05) else '✗ not significant'
                L += [f'\n**{feat}**',
                      f'- Effect size: {round(es,3) if es is not None else "n/a"} | p-value: {round(pv,4) if pv is not None else "n/a"} ({sig})',
                      f'- Expected in PD: {direction}',
                      f'- Interpretation: {expl}']

        L += ['', '---', '## 4. Limitations', '',
              '- These are **observational** comparisons — they show differences, not causes.',
              '- Small sample sizes reduce statistical power.',
              '- Missing data may introduce bias.',
              '- **No clinical conclusions** should be drawn from this exploratory analysis.']
        return '\n'.join(L)

    # ═══════════════ TAB 5 ═══════════════════════════════════════════════════

    def _build_predict_tab(self):
        warn = tk.Frame(self.tab_predict, bg='#2a1000')
        warn.pack(fill='x')
        tk.Label(warn,
                 text='⚠  EXPLORATORY ONLY — Predictions have NO clinical validity. For research use only.',
                 bg='#2a1000', fg='#ffaa33', font=FONT_S).pack(padx=15, pady=8)

        tk.Label(self.tab_predict, text='Predictive Modelling', font=FONT_H,
                 bg=BG2, fg=ACCENT2).pack(anchor='w', padx=15, pady=10)

        ctrl = ttk.LabelFrame(self.tab_predict, text='Model Settings')
        ctrl.pack(fill='x', padx=15, pady=8)

        r0 = ttk.Frame(ctrl); r0.pack(fill='x', padx=8, pady=6)
        tk.Label(r0, text='Target column (what to predict):',
                 bg=BG2, fg=TEXT, font=FONT_S, width=35, anchor='w').pack(side='left')
        self.pred_target_cb = ttk.Combobox(r0, values=['— select target —'],
                                            width=35, state='readonly')
        self.pred_target_cb.set('— select target —')
        self.pred_target_cb.configure(foreground=TEXT)
        self.pred_target_cb.pack(side='left', padx=6)

        r1 = ttk.Frame(ctrl); r1.pack(fill='x', padx=8, pady=4)
        tk.Label(r1, text='Task:', bg=BG2, fg=TEXT, font=FONT_S,
                 width=35, anchor='w').pack(side='left')
        self.pred_task_var = tk.StringVar(value='classification')
        tk.Radiobutton(r1, text='Classification  (e.g. PD vs Control)',
                       variable=self.pred_task_var, value='classification',
                       bg=BG2, fg=TEXT, selectcolor=ACCENT, font=FONT_S,
                       activebackground=BG2).pack(side='left', padx=6)
        tk.Radiobutton(r1, text='Regression  (e.g. predict a numeric score)',
                       variable=self.pred_task_var, value='regression',
                       bg=BG2, fg=TEXT, selectcolor=ACCENT, font=FONT_S,
                       activebackground=BG2).pack(side='left', padx=6)

        self._btn(ctrl, '▶ Run Models (Exploratory)',
                  self._run_prediction, OK, large=True).pack(pady=10)

        leg = ttk.LabelFrame(self.tab_predict, text='Understanding model metrics')
        leg.pack(fill='x', padx=15, pady=(0,8))
        tk.Label(leg, bg=BG2, fg=SUBTEXT, font=FONT_S, anchor='w', justify='left',
                 wraplength=1000,
                 text='  Classification:  Accuracy = % correctly classified  |  '
                      'F1 = balanced measure of precision & recall (good for uneven group sizes)  |  '
                      'AUC = ability to rank PD above Control (0.5=random chance, 1.0=perfect)  |  '
                      'Confusion matrix = exact breakdown of correct vs incorrect predictions\n'
                      '  Regression:  MAE = average error in same units as your outcome  |  '
                      'RMSE = error measure that penalises large mistakes more heavily  |  '
                      'R² = proportion of variance explained (1.0=perfect, 0=no better than just using the mean)'
                 ).pack(padx=8, pady=4)

        self.pred_results_text = scrolledtext.ScrolledText(
            self.tab_predict, bg=BG3, fg=TEXT, font=FONT_S, bd=0, wrap='word')
        self.pred_results_text.pack(fill='both', expand=True, padx=15, pady=8)
        self.pred_results_text.insert('end', 'Run models to see results here.\n\n'
            'Tip for this dataset: set Target = pd_vs_control_unnamed_2_level_1, '
            'Task = Classification to see how well gait features distinguish PD from Control.')
        self.pred_results_text.configure(state='disabled')

    def _run_prediction(self):
        df = self._get_working_df()
        if df.empty:
            messagebox.showwarning('No Data', 'No data available.'); return

        target = self.pred_target_cb.get()
        if '— select' in target or not target:
            target = self.config.get('outcome_col')
        if not target or target not in df.columns:
            messagebox.showwarning('No Target',
                'Select a valid target column.\n'
                'For PD vs Control classification use the PD vs Control column.'); return

        num_cols = list(df.select_dtypes(include='number').columns)
        feats    = [c for c in num_cols if c != target][:30]
        task     = self.pred_task_var.get()
        time_col = self.config.get('time_col')

        def _do():
            try:
                if task == 'classification':
                    result = mdl.run_classification(df, feats, target, time_col)
                else:
                    result = mdl.run_regression(df, feats, target, time_col)
                if 'error' in result:
                    self.after(0, lambda: messagebox.showerror('Model Error', result['error']))
                    return
                report = mdl.model_report_md(result,
                    ', '.join([os.path.basename(f) for f in self.filenames]))
                self.pred_model_result = result
                self.pred_report_md    = report
                self.after(0, lambda: self._display_prediction(report))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Prediction Error',
                    str(e) + '\n' + traceback.format_exc()))

        threading.Thread(target=_do, daemon=True).start()
        self._set_status('Running models...')

    def _display_prediction(self, report):
        self.pred_results_text.configure(state='normal')
        self.pred_results_text.delete('1.0', 'end')
        self.pred_results_text.insert('end', report)
        self.pred_results_text.configure(state='disabled')
        self._set_status('Modelling complete!')

    # ═══════════════ TAB 6 ═══════════════════════════════════════════════════

    def _build_export_tab(self):
        tk.Label(self.tab_export, text='Export Results',
                 font=FONT_H, bg=BG2, fg=ACCENT2).pack(anchor='w', padx=15, pady=12)

        dr = ttk.Frame(self.tab_export); dr.pack(fill='x', padx=15, pady=8)
        tk.Label(dr, text='Output directory:', bg=BG2, fg=TEXT, font=FONT_S).pack(side='left')
        self.export_dir_var = tk.StringVar(value=os.path.expanduser('~'))
        tk.Entry(dr, textvariable=self.export_dir_var, bg=BG3, fg=TEXT, font=FONT_S,
                 width=55, relief='flat', insertbackground=TEXT).pack(side='left', padx=6)
        self._btn(dr, '📂 Browse', lambda: self.export_dir_var.set(
            filedialog.askdirectory() or self.export_dir_var.get()), BG3).pack(side='left')

        of = ttk.LabelFrame(self.tab_export, text='Export Options')
        of.pack(fill='x', padx=15, pady=8)
        self.exp_cleaned = tk.BooleanVar(value=True)
        self.exp_report  = tk.BooleanVar(value=True)
        self.exp_plots   = tk.BooleanVar(value=True)
        self.exp_model   = tk.BooleanVar(value=True)
        ttk.Checkbutton(of, text='Cleaned QC dataset  (cleaned_qc_dataset.csv)',
                        variable=self.exp_cleaned).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(of, text='Insights narrative report  (insights_report.md)',
                        variable=self.exp_report).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(of, text='All generated plots  (plots/*.png)',
                        variable=self.exp_plots).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(of, text='Predictive model report  (model_report.md)',
                        variable=self.exp_model).pack(anchor='w', padx=10, pady=2)

        self._btn(self.tab_export, '▶ Export All', self._run_export, OK, large=True).pack(pady=10)

        self.export_log = scrolledtext.ScrolledText(self.tab_export, height=10,
                                                     bg=BG3, fg=TEXT, font=FONT_S, bd=0)
        self.export_log.pack(fill='both', expand=True, padx=15, pady=8)

    def _run_export(self):
        out = self.export_dir_var.get()
        if not os.path.isdir(out):
            messagebox.showerror('Invalid Directory', out); return
        log = []
        try:
            if self.exp_cleaned.get() and not self.qc_df.empty:
                p = os.path.join(out, 'cleaned_qc_dataset.csv')
                self.qc_df.to_csv(p, index=False); log.append(f'✓ {p}')
            if self.exp_report.get():
                txt = self.report_text.get('1.0', 'end').strip()
                if txt and not txt.startswith('Click'):
                    p = os.path.join(out, 'insights_report.md')
                    with open(p, 'w', encoding='utf-8') as f: f.write(txt)
                    log.append(f'✓ {p}')
            if self.exp_plots.get() and self.plots_generated:
                pd_dir = os.path.join(out, 'plots')
                os.makedirs(pd_dir, exist_ok=True)
                for name, fig in self.plots_generated:
                    p = os.path.join(pd_dir, f'{name}.png')
                    try: plot.save_fig(fig, p); log.append(f'✓ {p}')
                    except Exception: pass
            if self.exp_model.get() and hasattr(self, 'pred_report_md'):
                p = os.path.join(out, 'model_report.md')
                with open(p, 'w', encoding='utf-8') as f: f.write(self.pred_report_md)
                log.append(f'✓ {p}')
                if hasattr(self, 'pred_model_result'):
                    p2 = os.path.join(out, 'model_result.pkl')
                    mdl.save_model(self.pred_model_result, p2)
                    log.append(f'✓ {p2}')
            if not log:
                log.append('Nothing exported — run QC and Insights first.')
        except Exception as e:
            log.append(f'✗ Error: {e}')
        self.export_log.configure(state='normal')
        self.export_log.delete('1.0', 'end')
        self.export_log.insert('end', '\n'.join(log))
        self.export_log.configure(state='disabled')
        self._set_status('Export done.')

    # ─── Shared Helpers ───────────────────────────────────────────────────────

    def _btn(self, parent, text, command, bg=BG3, large=False):
        font = ('Segoe UI', 11, 'bold') if large else FONT
        b = tk.Button(parent, text=text, command=command, bg=bg, fg='white',
                      font=font, relief='flat', cursor='hand2',
                      activebackground=ACCENT, activeforeground='white',
                      padx=12 if large else 8, pady=5 if large else 3)
        b.bind('<Enter>', lambda e, btn=b, c=bg: btn.configure(bg=ACCENT))
        b.bind('<Leave>', lambda e, btn=b, c=bg: btn.configure(bg=c))
        return b

    def _make_treeview(self, parent, height=10):
        tree = ttk.Treeview(parent, show='headings', height=height)
        vsb  = ttk.Scrollbar(parent, orient='vertical',   command=tree.yview)
        hsb  = ttk.Scrollbar(parent, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right',  fill='y')
        hsb.pack(side='bottom', fill='x')
        tree.pack(fill='both', expand=True)
        return tree

    def _clear_treeview(self, tree):
        tree.delete(*tree.get_children())
        tree['columns'] = []

    def _fill_treeview(self, tree, df):
        tree.delete(*tree.get_children())
        if df is None or df.empty: return
        cols = list(df.columns)
        tree['columns'] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=max(70, min(160, len(str(col)) * 9)), anchor='w')
        for _, row in df.iterrows():
            vals = [str(v)[:80] if pd.notna(v) else '' for v in row]
            tree.insert('', 'end', values=vals)

    def _embed_figure(self, fig, parent, side='top'):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=side, fill='both', expand=True, padx=4, pady=4)
        plt.close(fig)

    def _set_status(self, msg):
        self.status_var.set(msg)
        print(f'[PD Insight] {msg}')


# ─── Rule Dialogs ─────────────────────────────────────────────────────────────

class BaseDialog(tk.Toplevel):
    def __init__(self, parent, title, cols):
        super().__init__(parent)
        self.title(title); self.configure(bg=BG2)
        self.resizable(False, False)
        self.result = None; self.cols = cols
        self._build(); self.grab_set()

    def _lbl(self, p, t):
        tk.Label(p, text=t, bg=BG2, fg=TEXT, font=FONT_S).pack(anchor='w', pady=2)

    def _ent(self, p, d=''):
        v = tk.StringVar(value=d)
        tk.Entry(p, textvariable=v, bg=BG3, fg=TEXT, font=FONT_S,
                 width=38, relief='flat', insertbackground=TEXT).pack(anchor='w', pady=2)
        return v

    def _dd(self, p, options, default=None):
        v = tk.StringVar(value=default or (options[0] if options else ''))
        cb = ttk.Combobox(p, textvariable=v, values=options, state='readonly', width=36)
        cb.configure(foreground=TEXT); cb.pack(anchor='w', pady=2)
        return v

    def _build(self): pass


class KeywordRuleDialog(BaseDialog):
    def _build(self):
        f = tk.Frame(self, bg=BG2, padx=14, pady=14); f.pack()
        tk.Label(f, text='Keyword Map Rule', font=FONT_B, bg=BG2, fg=ACCENT2).pack(pady=(0,8))
        self._lbl(f, 'Source column (text column to classify):')
        self.src = self._dd(f, self.cols)
        self._lbl(f, 'Output column name:')
        self.out = self._ent(f, 'med_state')
        self._lbl(f, 'Mappings — one per line:\n  keywords, separated, by, commas → LABEL')
        self.map_text = tk.Text(f, bg=BG3, fg=TEXT, font=FONT_S, width=44, height=6,
                                 insertbackground=TEXT, relief='flat')
        self.map_text.insert('end',
            'before, pre, off → BEFORE\nafter, post, on → AFTER\nnone, no med, control → NO_MEDS')
        self.map_text.pack(pady=4)
        self._lbl(f, 'Default label (if nothing matches):')
        self.default = self._ent(f, 'OTHER')
        tk.Button(f, text='Add Rule', command=self._confirm,
                  bg=OK, fg='white', font=FONT_B, relief='flat').pack(pady=8)

    def _confirm(self):
        mappings = []
        for line in self.map_text.get('1.0', 'end').strip().splitlines():
            sep = '→' if '→' in line else '->'
            if sep in line:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    kws   = [k.strip() for k in parts[0].split(',') if k.strip()]
                    label = parts[1].strip()
                    if kws and label:
                        mappings.append({'keywords': kws, 'label': label})
        if not mappings:
            messagebox.showwarning('Empty', 'Enter at least one mapping.'); return
        self.result = {
            'type': 'keyword_map',
            'source_col': self.src.get(),
            'output_col': self.out.get().strip(),
            'mappings':   mappings,
            'default':    self.default.get().strip(),
        }
        self.destroy()


class FormulaRuleDialog(BaseDialog):
    def _build(self):
        f = tk.Frame(self, bg=BG2, padx=14, pady=14); f.pack()
        tk.Label(f, text='Formula Rule', font=FONT_B, bg=BG2, fg=ACCENT2).pack(pady=(0,8))
        self._lbl(f, 'Output column name:')
        self.out  = self._ent(f, 'tap_rate')
        self._lbl(f, 'Expression (use exact column names  +  -  *  /):\n  e.g.  tap_count / duration_seconds')
        self.expr = self._ent(f, 'col_a / col_b')
        tk.Label(f, text='Available columns:\n' + ', '.join(self.cols[:20]) +
                 ('...' if len(self.cols) > 20 else ''),
                 bg=BG2, fg=SUBTEXT, font=FONT_S, wraplength=360).pack(pady=4)
        tk.Button(f, text='Add Rule', command=self._confirm,
                  bg=OK, fg='white', font=FONT_B, relief='flat').pack(pady=8)

    def _confirm(self):
        self.result = {'type': 'formula',
                       'output_col': self.out.get().strip(),
                       'expression': self.expr.get().strip()}
        self.destroy()


class MonthsRuleDialog(BaseDialog):
    def _build(self):
        f = tk.Frame(self, bg=BG2, padx=14, pady=14); f.pack()
        tk.Label(f, text='Months Since Baseline Rule', font=FONT_B,
                 bg=BG2, fg=ACCENT2).pack(pady=(0,8))
        self._lbl(f, 'Date/time column:')
        self.time_v = self._dd(f, self.cols)
        self._lbl(f, 'Subject/patient ID column:')
        self.eid_v  = self._dd(f, self.cols)
        self._lbl(f, 'Output column name:')
        self.out_v  = self._ent(f, 'months_since_baseline')
        tk.Button(f, text='Add Rule', command=self._confirm,
                  bg=OK, fg='white', font=FONT_B, relief='flat').pack(pady=8)

    def _confirm(self):
        self.result = {
            'type':          'months_since_baseline',
            'time_col':      self.time_v.get(),
            'entity_id_col': self.eid_v.get(),
            'output_col':    self.out_v.get().strip(),
        }
        self.destroy()


if __name__ == '__main__':
    app = PDInsightApp()
    app.mainloop()
