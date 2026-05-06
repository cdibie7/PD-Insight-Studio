"""
main.py — PD Insight Studio
A desktop application for Parkinson's disease real-world data analysis.
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

# Local modules
import standardise
import qc as qc_mod
import insights as ins
import plotting as plot
import rules as rules_mod
import modeling as mdl

# ─── Theme colours ─────────────────────────────────────────────────────────
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


def style_widget(w):
    try:
        w.configure(bg=BG2, fg=TEXT, font=FONT)
    except Exception:
        pass


# ─── Main App ───────────────────────────────────────────────────────────────

class PDInsightApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PD Insight Studio")
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(bg=BG)

        # State
        self.dataframes: list[pd.DataFrame] = []
        self.filenames: list[str] = []
        self.combined_df: pd.DataFrame = pd.DataFrame()
        self.qc_df: pd.DataFrame = pd.DataFrame()
        self.config: dict = {}
        self.rules: list = []
        self.plots_generated: list = []

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('default')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=BG3, foreground=TEXT,
                        font=FONT_B, padding=[14, 6], borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT), ('active', BG2)],
                  foreground=[('selected', 'white')])
        style.configure('TFrame', background=BG2)
        style.configure('TLabel', background=BG2, foreground=TEXT, font=FONT)
        style.configure('TButton', background=BG3, foreground=TEXT, font=FONT)
        style.configure('TCombobox', fieldbackground=BG3, background=BG3,
                        foreground=TEXT, selectbackground=ACCENT)
        style.configure('TCheckbutton', background=BG2, foreground=TEXT, font=FONT)
        style.configure('Treeview', background=BG3, foreground=TEXT,
                        fieldbackground=BG3, rowheight=22, font=FONT_S)
        style.configure('Treeview.Heading', background=BG, foreground=ACCENT2,
                        font=FONT_B, borderwidth=0)
        style.map('Treeview', background=[('selected', ACCENT)])
        style.configure('TScrollbar', background=BG3, troughcolor=BG)
        style.configure('TEntry', fieldbackground=BG3, foreground=TEXT)
        style.configure('TLabelframe', background=BG2, foreground=ACCENT2)
        style.configure('TLabelframe.Label', background=BG2, foreground=ACCENT2, font=FONT_B)

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG, height=50)
        header.pack(fill='x')
        tk.Label(header, text="⚗  PD Insight Studio", bg=BG, fg=ACCENT2,
                 font=('Segoe UI', 15, 'bold')).pack(side='left', padx=20, pady=10)
        tk.Label(header, text="Parkinson's Disease Real-World Data Analysis",
                 bg=BG, fg=SUBTEXT, font=FONT_S).pack(side='left', pady=14)

        # Notebook
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

    # ═══════════════ TAB 1: Load Data ═══════════════════════════════════════

    def _build_load_tab(self):
        tab = self.tab_load
        top = ttk.Frame(tab)
        top.pack(fill='x', padx=15, pady=12)

        tk.Label(top, text='Load CSV Files', font=FONT_H, bg=BG2, fg=ACCENT2).pack(side='left')
        btn_frame = ttk.Frame(top)
        btn_frame.pack(side='right')

        self._btn(btn_frame, '+ Add CSV', self._load_csv, ACCENT).pack(side='left', padx=4)
        self._btn(btn_frame, '✕ Clear All', self._clear_data, WARN).pack(side='left', padx=4)
        self._btn(btn_frame, '▶ Combine & Preview', self._combine_data, OK).pack(side='left', padx=4)

        # File list
        files_frame = ttk.LabelFrame(tab, text='Loaded Files')
        files_frame.pack(fill='x', padx=15, pady=(0, 8))

        self.files_listbox = tk.Listbox(files_frame, bg=BG3, fg=TEXT, font=FONT_S,
                                         height=4, selectbackground=ACCENT, bd=0)
        self.files_listbox.pack(fill='x', padx=6, pady=6)

        # Info bar
        self.load_info_var = tk.StringVar(value='No data loaded.')
        tk.Label(tab, textvariable=self.load_info_var, bg=BG2, fg=SUBTEXT, font=FONT_S,
                 anchor='w').pack(fill='x', padx=15)

        # Preview table
        prev_frame = ttk.LabelFrame(tab, text='Data Preview (first 20 rows)')
        prev_frame.pack(fill='both', expand=True, padx=15, pady=8)

        self.preview_tree = self._make_treeview(prev_frame)
        self.preview_tree.pack(fill='both', expand=True)

    def _load_csv(self):
        paths = filedialog.askopenfilenames(
            title='Select CSV file(s)',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not paths:
            return
        for path in paths:
            self._status(f'Loading {os.path.basename(path)}...')
            try:
                df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
                df = standardise.normalise_column_names(df)
                df = standardise.infer_and_cast_types(df)
                self.dataframes.append(df)
                self.filenames.append(path)
                self.files_listbox.insert('end', f'  {os.path.basename(path)}  [{df.shape[0]:,} rows × {df.shape[1]} cols]')
                self._status(f'Loaded {os.path.basename(path)}: {df.shape[0]:,} rows')
            except Exception as e:
                messagebox.showerror('Load Error', f'Failed to load file:\n{e}')

    def _clear_data(self):
        self.dataframes.clear()
        self.filenames.clear()
        self.combined_df = pd.DataFrame()
        self.files_listbox.delete(0, 'end')
        self.load_info_var.set('No data loaded.')
        self._clear_treeview(self.preview_tree)

    def _combine_data(self):
        if not self.dataframes:
            messagebox.showwarning('No Data', 'Please load at least one CSV file first.')
            return
        try:
            self.combined_df = standardise.merge_datasets(self.dataframes)
            shape = self.combined_df.shape
            self.load_info_var.set(
                f'Combined: {shape[0]:,} rows × {shape[1]} columns | '
                f'Columns: {", ".join(self.combined_df.columns[:12].tolist())}{"..." if shape[1] > 12 else ""}')
            self._fill_treeview(self.preview_tree, self.combined_df.head(20))
            self._populate_config_dropdowns()
            self._status('Data combined successfully. Proceed to ② Configure.')
        except Exception as e:
            messagebox.showerror('Combine Error', str(e))

    # ═══════════════ TAB 2: Configure ════════════════════════════════════════

    def _build_config_tab(self):
        tab = self.tab_config

        # Scroll container
        canvas = tk.Canvas(tab, bg=BG2, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
        self.config_frame = ttk.Frame(canvas)
        self.config_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.config_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        inner = self.config_frame

        tk.Label(inner, text='Configure Variables', font=FONT_H, bg=BG2, fg=ACCENT2).pack(anchor='w', padx=15, pady=12)
        tk.Label(inner, text='Map your dataset columns to the analysis roles below. All fields are optional except where noted.',
                 bg=BG2, fg=SUBTEXT, font=FONT_S, wraplength=700).pack(anchor='w', padx=15)

        # ── Mapping section ──
        map_frame = ttk.LabelFrame(inner, text='Column Mapping')
        map_frame.pack(fill='x', padx=15, pady=8)

        self.cfg_entity   = self._dropdown_row(map_frame, 'Patient/Subject ID column*', row=0)
        self.cfg_time     = self._dropdown_row(map_frame, 'Time / Visit column (optional)', row=1)
        self.cfg_group    = self._dropdown_row(map_frame, 'Primary group variable (e.g. diagnosis, sex)', row=2)
        self.cfg_outcome  = self._dropdown_row(map_frame, 'Outcome / Target column (for comparisons & prediction)', row=3)

        # ── Group variable extras ──
        extra_frame = ttk.LabelFrame(inner, text='Additional Group Variables (optional)')
        extra_frame.pack(fill='x', padx=15, pady=8)
        tk.Label(extra_frame, text='Select additional columns to use as group/stratification variables:',
                 bg=BG2, fg=SUBTEXT, font=FONT_S).pack(anchor='w', padx=8, pady=(4, 2))
        self.extra_groups_lb = tk.Listbox(extra_frame, bg=BG3, fg=TEXT, font=FONT_S,
                                           height=5, selectmode='multiple',
                                           selectbackground=ACCENT, bd=0)
        self.extra_groups_lb.pack(fill='x', padx=8, pady=4)

        # ── QC settings ──
        qc_frame = ttk.LabelFrame(inner, text='QC Settings')
        qc_frame.pack(fill='x', padx=15, pady=8)

        self.qc_col_miss = self._spinner_row(qc_frame, 'Column missingness threshold (flag if > this %)', 0, 80, 0, 100)
        self.qc_row_miss = self._spinner_row(qc_frame, 'Row missingness threshold (flag if > this %)',    1, 80, 0, 100)
        self.qc_outlier_method = self._radio_row(qc_frame, 'Outlier detection method', 2,
                                                  ['iqr', 'zscore'], ['IQR', 'Z-score'])
        self.qc_outlier_thr = self._spinner_row(qc_frame, 'Outlier threshold (IQR multiplier or z-score)', 3, 3, 1, 10, step=0.5)

        # ── Rule Builder ──
        rule_frame = ttk.LabelFrame(inner, text='Rule Builder — Derive New Columns')
        rule_frame.pack(fill='x', padx=15, pady=8)

        tk.Label(rule_frame, text='Add rules to create derived variables (e.g. med_state from medication column):',
                 bg=BG2, fg=SUBTEXT, font=FONT_S).pack(anchor='w', padx=8, pady=(4,2))

        rule_btn_row = ttk.Frame(rule_frame)
        rule_btn_row.pack(fill='x', padx=8)
        self._btn(rule_btn_row, '+ Keyword Map Rule', self._add_keyword_rule, ACCENT).pack(side='left', padx=4, pady=4)
        self._btn(rule_btn_row, '+ Formula Rule',     self._add_formula_rule, ACCENT).pack(side='left', padx=4, pady=4)
        self._btn(rule_btn_row, '+ Months-since-baseline', self._add_months_rule, ACCENT).pack(side='left', padx=4, pady=4)

        self.rule_listbox = tk.Listbox(rule_frame, bg=BG3, fg=TEXT, font=FONT_S,
                                        height=5, selectbackground=ACCENT, bd=0)
        self.rule_listbox.pack(fill='x', padx=8, pady=4)
        self._btn(rule_frame, '✕ Remove Selected Rule', self._remove_rule, WARN).pack(anchor='w', padx=8, pady=2)

        # ── Save/Load config ──
        cfg_btn_row = ttk.Frame(inner)
        cfg_btn_row.pack(fill='x', padx=15, pady=8)
        tk.Label(cfg_btn_row, text='Save/load this configuration as JSON to reuse with similar datasets:',
                 bg=BG2, fg=SUBTEXT, font=FONT_S).pack(side='left', padx=(0, 10))
        self._btn(cfg_btn_row, '💾 Save Config', self._save_config, BG3).pack(side='left', padx=4)
        self._btn(cfg_btn_row, '📂 Load Config', self._load_config, BG3).pack(side='left', padx=4)

        # Apply button
        self._btn(inner, '▶ Apply Configuration & Go to QC', self._apply_config, OK,
                  large=True).pack(pady=12)

    def _dropdown_row(self, parent, label, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky='ew', padx=8, pady=4)
        parent.columnconfigure(0, weight=1)
        tk.Label(frame, text=label, bg=BG2, fg=TEXT, font=FONT_S, width=42, anchor='w').pack(side='left')
        var = tk.StringVar(value='— not mapped —')
        cb = ttk.Combobox(frame, textvariable=var, values=['— not mapped —'], state='readonly', width=30)
        cb.pack(side='left', padx=6)
        return var

    def _spinner_row(self, parent, label, row, default, from_, to, step=1):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky='ew', padx=8, pady=4)
        parent.columnconfigure(0, weight=1)
        tk.Label(frame, text=label, bg=BG2, fg=TEXT, font=FONT_S, width=52, anchor='w').pack(side='left')
        var = tk.DoubleVar(value=default)
        sb = ttk.Spinbox(frame, from_=from_, to=to, increment=step, textvariable=var, width=8)
        sb.pack(side='left', padx=6)
        return var

    def _radio_row(self, parent, label, row, values, display):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky='ew', padx=8, pady=4)
        parent.columnconfigure(0, weight=1)
        tk.Label(frame, text=label, bg=BG2, fg=TEXT, font=FONT_S, width=52, anchor='w').pack(side='left')
        var = tk.StringVar(value=values[0])
        for val, disp in zip(values, display):
            tk.Radiobutton(frame, text=disp, variable=var, value=val,
                           bg=BG2, fg=TEXT, selectcolor=ACCENT, font=FONT_S,
                           activebackground=BG2, activeforeground=ACCENT2).pack(side='left', padx=6)
        return var

    def _populate_config_dropdowns(self):
        if self.combined_df.empty:
            return
        cols = ['— not mapped —'] + list(self.combined_df.columns)
        for var in [self.cfg_entity, self.cfg_time, self.cfg_group, self.cfg_outcome]:
            # Find the associated combobox widget and update its values
            pass
        # Re-bind dropdowns
        self._refresh_all_dropdowns(cols)
        self.extra_groups_lb.delete(0, 'end')
        for c in self.combined_df.columns:
            self.extra_groups_lb.insert('end', c)

    def _refresh_all_dropdowns(self, cols):
        """Update all config comboboxes with available column names."""
        for widget in self.config_frame.winfo_children():
            self._update_comboboxes_in(widget, cols)

    def _update_comboboxes_in(self, widget, cols):
        if isinstance(widget, ttk.Combobox):
            current = widget.get()
            widget['values'] = cols
            if current not in cols:
                widget.set(cols[0])
        for child in widget.winfo_children():
            self._update_comboboxes_in(child, cols)

    def _apply_config(self):
        if self.combined_df.empty:
            messagebox.showwarning('No Data', 'Please load and combine data first (Tab ①).')
            return

        def v(var):
            val = var.get()
            return None if '— not mapped —' in val or val == '' else val

        self.config = {
            'entity_id':       v(self.cfg_entity),
            'time_col':        v(self.cfg_time),
            'group_col':       v(self.cfg_group),
            'outcome_col':     v(self.cfg_outcome),
            'extra_groups':    [self.extra_groups_lb.get(i) for i in self.extra_groups_lb.curselection()],
            'col_missingness': float(self.qc_col_miss.get()) / 100,
            'row_missingness': float(self.qc_row_miss.get()) / 100,
            'outlier_method':  self.qc_outlier_method.get(),
            'outlier_thr':     float(self.qc_outlier_thr.get()),
            'rules':           self.rules,
        }

        # Apply rules
        if self.rules:
            self.combined_df = rules_mod.apply_rules(self.combined_df, self.rules)

        self._status('Configuration applied. Proceed to ③ QC & Clean.')
        self.nb.select(self.tab_qc)

    def _add_keyword_rule(self):
        """Open a dialog to build a keyword map rule."""
        if self.combined_df.empty:
            messagebox.showinfo('No Data', 'Load data first.')
            return
        dlg = KeywordRuleDialog(self, list(self.combined_df.columns))
        self.wait_window(dlg)
        if dlg.result:
            self.rules.append(dlg.result)
            self.rule_listbox.insert('end', f"  Keyword map: {dlg.result['source_col']} → {dlg.result['output_col']}")

    def _add_formula_rule(self):
        if self.combined_df.empty:
            messagebox.showinfo('No Data', 'Load data first.')
            return
        dlg = FormulaRuleDialog(self, list(self.combined_df.columns))
        self.wait_window(dlg)
        if dlg.result:
            self.rules.append(dlg.result)
            self.rule_listbox.insert('end', f"  Formula: {dlg.result['expression']} → {dlg.result['output_col']}")

    def _add_months_rule(self):
        if self.combined_df.empty:
            messagebox.showinfo('No Data', 'Load data first.')
            return
        dlg = MonthsRuleDialog(self, list(self.combined_df.columns))
        self.wait_window(dlg)
        if dlg.result:
            self.rules.append(dlg.result)
            self.rule_listbox.insert('end', f"  Months since baseline: {dlg.result['time_col']} → {dlg.result['output_col']}")

    def _remove_rule(self):
        sel = self.rule_listbox.curselection()
        for i in reversed(sel):
            self.rule_listbox.delete(i)
            if i < len(self.rules):
                self.rules.pop(i)

    def _save_config(self):
        path = filedialog.asksaveasfilename(defaultextension='.json',
                                             filetypes=[('JSON', '*.json')],
                                             title='Save Configuration')
        if path:
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            messagebox.showinfo('Saved', f'Configuration saved to {path}')

    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')], title='Load Configuration')
        if path:
            try:
                with open(path) as f:
                    self.config = json.load(f)
                self.rules = self.config.get('rules', [])
                messagebox.showinfo('Loaded', 'Configuration loaded. Review settings and click Apply.')
            except Exception as e:
                messagebox.showerror('Error', str(e))

    # ═══════════════ TAB 3: QC ════════════════════════════════════════════════

    def _build_qc_tab(self):
        tab = self.tab_qc

        top = ttk.Frame(tab)
        top.pack(fill='x', padx=15, pady=12)
        tk.Label(top, text='Quality Control & Standardisation', font=FONT_H, bg=BG2, fg=ACCENT2).pack(side='left')
        self._btn(top, '▶ Run QC', self._run_qc, OK).pack(side='right', padx=4)

        # Summary
        self.qc_summary_text = scrolledtext.ScrolledText(tab, height=14, bg=BG3, fg=TEXT,
                                                          font=FONT_S, wrap='word', bd=0)
        self.qc_summary_text.pack(fill='x', padx=15, pady=8)
        self.qc_summary_text.insert('end', 'Run QC to see results here.')
        self.qc_summary_text.configure(state='disabled')

        # Flagged rows preview
        flagged_frame = ttk.LabelFrame(tab, text='Flagged Rows Preview (first 50 removed rows)')
        flagged_frame.pack(fill='both', expand=True, padx=15, pady=8)
        self.qc_flagged_tree = self._make_treeview(flagged_frame)
        self.qc_flagged_tree.pack(fill='both', expand=True)

    def _run_qc(self):
        if self.combined_df.empty:
            messagebox.showwarning('No Data', 'Please load and configure data first.')
            return

        cfg = self.config
        numeric_cols = list(self.combined_df.select_dtypes(include='number').columns)[:20]

        def _do_qc():
            try:
                self.qc_df = qc_mod.run_qc(
                    self.combined_df,
                    required_cols=[c for c in [cfg.get('entity_id'), cfg.get('outcome_col')] if c],
                    col_missingness_threshold=cfg.get('col_missingness', 0.8),
                    row_missingness_threshold=cfg.get('row_missingness', 0.8),
                    duplicate_keys=[c for c in [cfg.get('entity_id'), cfg.get('time_col')] if c],
                    outlier_method=cfg.get('outlier_method', 'iqr'),
                    outlier_threshold=cfg.get('outlier_thr', 3.0),
                    numeric_cols=numeric_cols,
                )
                summary = qc_mod.qc_summary(self.qc_df)
                self.after(0, lambda: self._display_qc_results(summary))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('QC Error', str(e) + '\n' + traceback.format_exc()))

        threading.Thread(target=_do_qc, daemon=True).start()
        self._status('Running QC...')

    def _display_qc_results(self, summary):
        lines = []
        lines.append(f"{'='*55}")
        lines.append(f"  QC RESULTS")
        lines.append(f"{'='*55}")
        lines.append(f"  Total rows:     {summary['total']:>8,}")
        lines.append(f"  Rows kept:      {summary['kept']:>8,}  ({summary['kept_pct']}%)")
        lines.append(f"  Rows flagged:   {summary['removed']:>8,}")
        lines.append(f"\n  Removal Reasons:")
        if summary['reason_counts']:
            for reason, cnt in list(summary['reason_counts'].items())[:10]:
                lines.append(f"    • {reason:<40} {cnt:>6,}")
        else:
            lines.append('    (none)')
        lines.append(f"\n  High-missingness columns (>20%):")
        if summary['high_missingness_cols']:
            for col, pct in list(summary['high_missingness_cols'].items())[:10]:
                lines.append(f"    • {col:<40} {pct:>5}% missing")
        else:
            lines.append('    (none)')

        self.qc_summary_text.configure(state='normal')
        self.qc_summary_text.delete('1.0', 'end')
        self.qc_summary_text.insert('end', '\n'.join(lines))
        self.qc_summary_text.configure(state='disabled')

        # Flagged rows preview
        flagged = self.qc_df[~self.qc_df['qc_keep']].head(50)
        self._fill_treeview(self.qc_flagged_tree, flagged)

        self._status(f'QC complete: {summary["kept"]:,} kept, {summary["removed"]:,} flagged.')
        self.nb.select(self.tab_insights)

    # ═══════════════ TAB 4: Insights ════════════════════════════════════════

    def _build_insights_tab(self):
        tab = self.tab_insights

        top = ttk.Frame(tab)
        top.pack(fill='x', padx=15, pady=12)
        tk.Label(top, text='Insights & Analysis', font=FONT_H, bg=BG2, fg=ACCENT2).pack(side='left')
        self._btn(top, '▶ Generate All Insights', self._run_insights, OK).pack(side='right', padx=4)

        # Sub-notebook inside insights
        self.ins_nb = ttk.Notebook(tab)
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
        d = self.ins_desc_tab
        df = ttk.LabelFrame(d, text='Numeric Summary')
        df.pack(fill='x', padx=10, pady=6)
        self.desc_tree = self._make_treeview(df, height=8)
        self.desc_tree.pack(fill='x', pady=4)

        plots_row = ttk.LabelFrame(d, text='Distributions')
        plots_row.pack(fill='both', expand=True, padx=10, pady=6)
        self.desc_plot_frame = plots_row

        # ── Group comparisons ──
        g = self.ins_groups_tab
        ctrl_row = ttk.Frame(g)
        ctrl_row.pack(fill='x', padx=10, pady=6)
        tk.Label(ctrl_row, text='Compare by group:', bg=BG2, fg=TEXT, font=FONT_S).pack(side='left', padx=4)
        self.grp_var = tk.StringVar(value='— use configured group —')
        self.grp_cb = ttk.Combobox(ctrl_row, textvariable=self.grp_var, width=28, state='readonly')
        self.grp_cb.pack(side='left', padx=4)
        self._btn(ctrl_row, '▶ Compare', self._run_group_comparison, ACCENT).pack(side='left', padx=6)

        cmp_frame = ttk.LabelFrame(g, text='Comparison Results (sorted by effect size)')
        cmp_frame.pack(fill='x', padx=10, pady=6)
        self.cmp_tree = self._make_treeview(cmp_frame, height=8)
        self.cmp_tree.pack(fill='x')

        self.effect_plot_frame = ttk.LabelFrame(g, text='Effect Size Chart')
        self.effect_plot_frame.pack(fill='both', expand=True, padx=10, pady=6)

        # ── Time trends ──
        t = self.ins_time_tab
        t_ctrl = ttk.Frame(t)
        t_ctrl.pack(fill='x', padx=10, pady=6)
        tk.Label(t_ctrl, text='Outcome for progression:', bg=BG2, fg=TEXT, font=FONT_S).pack(side='left', padx=4)
        self.prog_outcome_var = tk.StringVar(value='— select outcome —')
        self.prog_outcome_cb = ttk.Combobox(t_ctrl, textvariable=self.prog_outcome_var, width=25, state='readonly')
        self.prog_outcome_cb.pack(side='left', padx=4)
        self._btn(t_ctrl, '▶ Analyse Progression', self._run_progression, ACCENT).pack(side='left', padx=6)

        self.prog_text = scrolledtext.ScrolledText(t, height=8, bg=BG3, fg=TEXT, font=FONT_S, bd=0)
        self.prog_text.pack(fill='x', padx=10, pady=4)
        self.prog_text.insert('end', 'No progression analysis yet.')
        self.prog_text.configure(state='disabled')

        self.prog_plot_frame = ttk.LabelFrame(t, text='Progression Plots')
        self.prog_plot_frame.pack(fill='both', expand=True, padx=10, pady=4)

        # ── Report ──
        r = self.ins_report_tab
        self.report_text = scrolledtext.ScrolledText(r, bg=BG3, fg=TEXT, font=FONT_S, bd=0, wrap='word')
        self.report_text.pack(fill='both', expand=True, padx=10, pady=8)
        self.report_text.insert('end', 'Generate insights to see the narrative report here.')
        self.report_text.configure(state='disabled')

    def _run_insights(self):
        if self.qc_df.empty and self.combined_df.empty:
            messagebox.showwarning('No Data', 'Please load data and run QC first.')
            return
        df = self.qc_df[self.qc_df['qc_keep']].drop(columns=['qc_keep', 'qc_reason'], errors='ignore') \
             if not self.qc_df.empty else self.combined_df

        cfg = self.config
        numeric_cols = list(df.select_dtypes(include='number').columns)[:20]
        cat_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() < 50][:10]

        def _do():
            try:
                # Descriptive
                desc_df = ins.describe_numeric(df, numeric_cols)
                cat_desc = ins.describe_categorical(df, cat_cols[:5])

                # Group comparison
                group_col = cfg.get('group_col')
                cmp_df = pd.DataFrame()
                if group_col and group_col in df.columns:
                    cmp_df = ins.compare_groups(df, numeric_cols[:15], group_col)

                # QC summary for report
                qc_summary_dict = qc_mod.qc_summary(self.qc_df) if not self.qc_df.empty else {}

                # Update dropdowns
                self.after(0, lambda: self._update_insight_dropdowns(list(df.columns)))

                # Generate report
                narrative = ins.generate_narrative(
                    dataset_name=', '.join([os.path.basename(f) for f in self.filenames]) or 'Dataset',
                    shape=df.shape,
                    qc_summary=qc_summary_dict,
                    descriptive_df=desc_df if not desc_df.empty else None,
                    group_comparison_df=cmp_df if not cmp_df.empty else None,
                    group_col=group_col,
                    progression_summary_dict=None,
                    outcome_col=cfg.get('outcome_col'),
                )

                self.after(0, lambda: self._display_insights(desc_df, cmp_df, cat_desc, df, narrative, group_col))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Insights Error', str(e) + '\n' + traceback.format_exc()))

        threading.Thread(target=_do, daemon=True).start()
        self._status('Generating insights...')

    def _display_insights(self, desc_df, cmp_df, cat_desc, df, narrative, group_col):
        # Descriptive table
        self._fill_treeview(self.desc_tree, desc_df)

        # Histogram plot (first 3 numeric)
        for widget in self.desc_plot_frame.winfo_children():
            widget.destroy()
        numeric_cols = list(df.select_dtypes(include='number').columns)[:3]
        for col in numeric_cols:
            fig = plot.make_histogram(df, col, group_col=group_col)
            self._embed_figure(fig, self.desc_plot_frame, side='left')

        # Group comparison
        if not cmp_df.empty:
            sorted_cmp = cmp_df.sort_values('effect_size', ascending=False) if 'effect_size' in cmp_df.columns else cmp_df
            self._fill_treeview(self.cmp_tree, sorted_cmp.head(20))
            fig2 = plot.make_group_effect_chart(cmp_df)
            for w in self.effect_plot_frame.winfo_children():
                w.destroy()
            self._embed_figure(fig2, self.effect_plot_frame)
            self.plots_generated.append(('effect_sizes', fig2))

        # Report
        self.report_text.configure(state='normal')
        self.report_text.delete('1.0', 'end')
        self.report_text.insert('end', narrative)
        self.report_text.configure(state='disabled')

        self._status('Insights generated!')

    def _run_group_comparison(self):
        df = self.qc_df[self.qc_df['qc_keep']].drop(columns=['qc_keep', 'qc_reason'], errors='ignore') \
             if not self.qc_df.empty else self.combined_df
        if df.empty:
            messagebox.showwarning('No Data', 'No data available.')
            return
        group_col = self.grp_var.get()
        if '— use configured' in group_col:
            group_col = self.config.get('group_col')
        if not group_col:
            messagebox.showwarning('No Group', 'Please select a group variable.')
            return
        numeric_cols = list(df.select_dtypes(include='number').columns)[:20]
        cmp_df = ins.compare_groups(df, numeric_cols, group_col)
        if not cmp_df.empty:
            sorted_cmp = cmp_df.sort_values('effect_size', ascending=False) if 'effect_size' in cmp_df.columns else cmp_df
            self._fill_treeview(self.cmp_tree, sorted_cmp.head(20))
            fig = plot.make_group_effect_chart(cmp_df)
            for w in self.effect_plot_frame.winfo_children():
                w.destroy()
            self._embed_figure(fig, self.effect_plot_frame)
        self._status(f'Group comparison done for: {group_col}')

    def _run_progression(self):
        df = self.qc_df[self.qc_df['qc_keep']].drop(columns=['qc_keep', 'qc_reason'], errors='ignore') \
             if not self.qc_df.empty else self.combined_df
        if df.empty:
            messagebox.showwarning('No Data', 'No data available.')
            return

        cfg = self.config
        entity_id = cfg.get('entity_id')
        time_col   = cfg.get('time_col')
        outcome    = self.prog_outcome_var.get()
        if '— select' in outcome:
            outcome = cfg.get('outcome_col')

        if not all([entity_id, time_col, outcome]):
            messagebox.showwarning('Config Missing', 'Need entity_id, time_col, and outcome in configuration.')
            return

        # Parse time
        df = df.copy()
        df['_months'] = standardise.compute_months_since_baseline(df, entity_id, time_col)

        slopes_df = ins.compute_slopes(df, entity_id, '_months', outcome)
        prog_sum  = ins.progression_summary(slopes_df)

        # Display
        lines = ['PROGRESSION ANALYSIS\n' + '='*45]
        for k, v in prog_sum.items():
            lines.append(f"  {k.replace('_',' ').title():35} {v}")

        self.prog_text.configure(state='normal')
        self.prog_text.delete('1.0', 'end')
        self.prog_text.insert('end', '\n'.join(lines))
        self.prog_text.configure(state='disabled')

        # Plots
        for w in self.prog_plot_frame.winfo_children():
            w.destroy()

        fig1 = plot.make_time_trend(df, entity_id, '_months', outcome,
                                     group_col=cfg.get('group_col'))
        fig2 = plot.make_slope_distribution(slopes_df, group_data=df,
                                             entity_id_col=entity_id,
                                             group_col=cfg.get('group_col'))
        self._embed_figure(fig1, self.prog_plot_frame, side='left')
        self._embed_figure(fig2, self.prog_plot_frame, side='left')
        self.plots_generated.extend([('time_trend', fig1), ('slope_dist', fig2)])
        self._status('Progression analysis done.')

    def _update_insight_dropdowns(self, cols):
        cols_no_none = ['— use configured group —'] + cols
        self.grp_cb['values'] = cols_no_none
        outcome_cols = ['— select outcome —'] + cols
        self.prog_outcome_cb['values'] = outcome_cols

    # ═══════════════ TAB 5: Predict ══════════════════════════════════════════

    def _build_predict_tab(self):
        tab = self.tab_predict

        warn_frame = tk.Frame(tab, bg='#3a1a00')
        warn_frame.pack(fill='x', padx=0, pady=0)
        tk.Label(warn_frame,
                 text='⚠  EXPLORATORY ONLY — These predictions have NO clinical validity and are intended for research purposes only.',
                 bg='#3a1a00', fg='#ffbb55', font=FONT_S).pack(padx=15, pady=8)

        top = ttk.Frame(tab)
        top.pack(fill='x', padx=15, pady=10)
        tk.Label(top, text='Predictive Modelling', font=FONT_H, bg=BG2, fg=ACCENT2).pack(side='left')

        ctrl = ttk.LabelFrame(tab, text='Model Settings')
        ctrl.pack(fill='x', padx=15, pady=8)

        row0 = ttk.Frame(ctrl); row0.pack(fill='x', padx=8, pady=4)
        tk.Label(row0, text='Target column:', bg=BG2, fg=TEXT, font=FONT_S, width=22, anchor='w').pack(side='left')
        self.pred_target_var = tk.StringVar(value='— not set —')
        self.pred_target_cb  = ttk.Combobox(row0, textvariable=self.pred_target_var, width=28, state='readonly')
        self.pred_target_cb.pack(side='left', padx=4)

        row1 = ttk.Frame(ctrl); row1.pack(fill='x', padx=8, pady=4)
        tk.Label(row1, text='Task type:', bg=BG2, fg=TEXT, font=FONT_S, width=22, anchor='w').pack(side='left')
        self.pred_task_var = tk.StringVar(value='classification')
        tk.Radiobutton(row1, text='Classification', variable=self.pred_task_var, value='classification',
                       bg=BG2, fg=TEXT, selectcolor=ACCENT, font=FONT_S,
                       activebackground=BG2).pack(side='left', padx=6)
        tk.Radiobutton(row1, text='Regression', variable=self.pred_task_var, value='regression',
                       bg=BG2, fg=TEXT, selectcolor=ACCENT, font=FONT_S,
                       activebackground=BG2).pack(side='left', padx=6)

        self._btn(ctrl, '▶ Run Models (Exploratory)', self._run_prediction, OK,
                  large=True).pack(pady=8)

        self.pred_results_text = scrolledtext.ScrolledText(tab, bg=BG3, fg=TEXT,
                                                            font=FONT_S, bd=0, wrap='word')
        self.pred_results_text.pack(fill='both', expand=True, padx=15, pady=8)
        self.pred_results_text.insert('end', 'Run models to see results here.')
        self.pred_results_text.configure(state='disabled')

    def _run_prediction(self):
        df = self.qc_df[self.qc_df['qc_keep']].drop(columns=['qc_keep', 'qc_reason'], errors='ignore') \
             if not self.qc_df.empty else self.combined_df
        if df.empty:
            messagebox.showwarning('No Data', 'No QC data available.')
            return

        target = self.pred_target_var.get()
        if '— not set —' in target or not target:
            target = self.config.get('outcome_col')
        if not target or target not in df.columns:
            messagebox.showwarning('No Target', 'Please select a valid target column.')
            return

        feature_cols = [c for c in df.select_dtypes(include=['number', 'object']).columns
                        if c != target and c not in ('qc_reason', 'qc_keep')][:30]
        time_col = self.config.get('time_col')
        task = self.pred_task_var.get()

        def _do():
            try:
                if task == 'classification':
                    result = mdl.run_classification(df, feature_cols, target, time_col)
                else:
                    result = mdl.run_regression(df, feature_cols, target, time_col)

                if 'error' in result:
                    self.after(0, lambda: messagebox.showerror('Model Error', result['error']))
                    return

                report_md = mdl.model_report_md(result, ', '.join(self.filenames))
                self.pred_model_result = result
                self.pred_report_md = report_md
                self.after(0, lambda: self._display_prediction_results(report_md))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Prediction Error',
                                                            str(e) + '\n' + traceback.format_exc()))

        threading.Thread(target=_do, daemon=True).start()
        self._status('Running models (this may take a moment)...')

    def _display_prediction_results(self, report_md: str):
        self.pred_results_text.configure(state='normal')
        self.pred_results_text.delete('1.0', 'end')
        self.pred_results_text.insert('end', report_md)
        self.pred_results_text.configure(state='disabled')
        self._status('Modelling complete!')

    # ═══════════════ TAB 6: Export ═══════════════════════════════════════════

    def _build_export_tab(self):
        tab = self.tab_export
        tk.Label(tab, text='Export Results', font=FONT_H, bg=BG2, fg=ACCENT2).pack(anchor='w', padx=15, pady=12)
        tk.Label(tab, text='Choose an output directory and export your results.',
                 bg=BG2, fg=SUBTEXT, font=FONT_S).pack(anchor='w', padx=15)

        dir_row = ttk.Frame(tab)
        dir_row.pack(fill='x', padx=15, pady=8)
        tk.Label(dir_row, text='Output directory:', bg=BG2, fg=TEXT, font=FONT_S).pack(side='left', padx=4)
        self.export_dir_var = tk.StringVar(value=os.path.expanduser('~'))
        tk.Entry(dir_row, textvariable=self.export_dir_var, bg=BG3, fg=TEXT, font=FONT_S,
                 width=50, relief='flat', insertbackground=TEXT).pack(side='left', padx=4)
        self._btn(dir_row, '📂 Browse', self._browse_export_dir, BG3).pack(side='left')

        opt_frame = ttk.LabelFrame(tab, text='Export Options')
        opt_frame.pack(fill='x', padx=15, pady=8)

        self.exp_cleaned = tk.BooleanVar(value=True)
        self.exp_report  = tk.BooleanVar(value=True)
        self.exp_plots   = tk.BooleanVar(value=True)
        self.exp_model   = tk.BooleanVar(value=True)

        ttk.Checkbutton(opt_frame, text='Cleaned QC dataset  (cleaned_qc_dataset.csv)',
                        variable=self.exp_cleaned).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(opt_frame, text='Insights narrative report  (insights_report.md)',
                        variable=self.exp_report).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(opt_frame, text='All generated plots  (plots/*.png)',
                        variable=self.exp_plots).pack(anchor='w', padx=10, pady=2)
        ttk.Checkbutton(opt_frame, text='Predictive model report  (model_report.md)',
                        variable=self.exp_model).pack(anchor='w', padx=10, pady=2)

        self._btn(tab, '▶ Export', self._run_export, OK, large=True).pack(pady=12)

        self.export_log = scrolledtext.ScrolledText(tab, height=10, bg=BG3, fg=TEXT,
                                                     font=FONT_S, bd=0)
        self.export_log.pack(fill='both', expand=True, padx=15, pady=8)

    def _browse_export_dir(self):
        d = filedialog.askdirectory(title='Select output directory')
        if d:
            self.export_dir_var.set(d)

    def _run_export(self):
        out_dir = self.export_dir_var.get()
        if not os.path.isdir(out_dir):
            messagebox.showerror('Invalid Directory', f'Not a valid directory: {out_dir}')
            return
        log = []
        try:
            if self.exp_cleaned.get() and not self.qc_df.empty:
                path = os.path.join(out_dir, 'cleaned_qc_dataset.csv')
                self.qc_df.to_csv(path, index=False)
                log.append(f'✓ Saved: {path}')

            if self.exp_report.get():
                report_text = self.report_text.get('1.0', 'end')
                if report_text.strip() and 'Generate insights' not in report_text:
                    path = os.path.join(out_dir, 'insights_report.md')
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(report_text)
                    log.append(f'✓ Saved: {path}')

            if self.exp_plots.get() and self.plots_generated:
                plots_dir = os.path.join(out_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                for name, fig in self.plots_generated:
                    path = os.path.join(plots_dir, f'{name}.png')
                    plot.save_fig(fig, path)
                    log.append(f'✓ Saved plot: {path}')

            if self.exp_model.get() and hasattr(self, 'pred_report_md'):
                path = os.path.join(out_dir, 'model_report.md')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.pred_report_md)
                log.append(f'✓ Saved: {path}')
                if hasattr(self, 'pred_model_result'):
                    path2 = os.path.join(out_dir, 'model_result.pkl')
                    mdl.save_model(self.pred_model_result, path2)
                    log.append(f'✓ Saved model: {path2}')

            if not log:
                log.append('Nothing to export. Make sure you have run QC and insights first.')

        except Exception as e:
            log.append(f'✗ Error: {e}')

        self.export_log.configure(state='normal')
        self.export_log.delete('1.0', 'end')
        self.export_log.insert('end', '\n'.join(log))
        self.export_log.configure(state='disabled')
        self._status('Export complete.')

    # ═══════════════ Helpers ═════════════════════════════════════════════════

    def _btn(self, parent, text, command, bg=BG3, large=False):
        font = ('Segoe UI', 11, 'bold') if large else FONT
        pad = 12 if large else 8
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg='white',
                        font=font, relief='flat', cursor='hand2',
                        activebackground=ACCENT, activeforeground='white',
                        padx=pad, pady=4 if large else 3)
        btn.bind('<Enter>', lambda e, b=btn, c=bg: b.configure(bg=ACCENT))
        btn.bind('<Leave>', lambda e, b=btn, c=bg: b.configure(bg=c))
        return btn

    def _make_treeview(self, parent, height=10):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True)
        tree = ttk.Treeview(frame, show='headings', height=height)
        vsb = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        tree.pack(fill='both', expand=True)
        return tree

    def _clear_treeview(self, tree):
        tree.delete(*tree.get_children())
        tree['columns'] = []

    def _fill_treeview(self, tree, df: pd.DataFrame):
        tree.delete(*tree.get_children())
        if df is None or df.empty:
            return
        cols = list(df.columns)
        tree['columns'] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=max(80, min(160, len(col) * 9)), anchor='w')
        for _, row in df.iterrows():
            values = [str(v)[:80] if pd.notna(v) else '' for v in row]
            tree.insert('', 'end', values=values)

    def _embed_figure(self, fig, parent, side='top'):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=side, fill='both', expand=True, padx=4, pady=4)
        plt.close(fig)

    def _status(self, msg: str):
        print(f'[PD Insight Studio] {msg}')


# ─── Rule Dialogs ─────────────────────────────────────────────────────────────

class BaseDialog(tk.Toplevel):
    def __init__(self, parent, title, cols):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG2)
        self.resizable(False, False)
        self.result = None
        self.cols = cols
        self._build()
        self.grab_set()

    def _label(self, parent, text):
        tk.Label(parent, text=text, bg=BG2, fg=TEXT, font=FONT_S).pack(anchor='w', pady=2)

    def _entry(self, parent, default=''):
        var = tk.StringVar(value=default)
        tk.Entry(parent, textvariable=var, bg=BG3, fg=TEXT, font=FONT_S,
                 width=35, relief='flat', insertbackground=TEXT).pack(anchor='w', pady=2)
        return var

    def _dropdown(self, parent, options, default=None):
        var = tk.StringVar(value=default or (options[0] if options else ''))
        cb = ttk.Combobox(parent, textvariable=var, values=options, state='readonly', width=33)
        cb.pack(anchor='w', pady=2)
        return var

    def _build(self):
        pass


class KeywordRuleDialog(BaseDialog):
    def _build(self):
        f = tk.Frame(self, bg=BG2, padx=14, pady=14)
        f.pack()
        tk.Label(f, text='Keyword Map Rule', font=FONT_B, bg=BG2, fg=ACCENT2).pack(pady=(0, 8))
        self._label(f, 'Source column (text column to search):')
        self.src_var = self._dropdown(f, self.cols)
        self._label(f, 'Output column name:')
        self.out_var = self._entry(f, 'med_state')
        self._label(f, 'Mapping (format: keywords→LABEL, one per line):\n  e.g.  before,pre,off → BEFORE')
        self.map_text = tk.Text(f, bg=BG3, fg=TEXT, font=FONT_S, width=40, height=6,
                                 insertbackground=TEXT, relief='flat')
        self.map_text.insert('end', 'before,pre,off → BEFORE\nafter,post,on → AFTER\nnone,no med,control → NO_MEDS')
        self.map_text.pack(pady=4)
        self._label(f, 'Default label (if no keywords match):')
        self.default_var = self._entry(f, 'OTHER')
        tk.Button(f, text='Add Rule', command=self._confirm, bg=OK, fg='white',
                  font=FONT_B, relief='flat').pack(pady=8)

    def _confirm(self):
        mappings = []
        for line in self.map_text.get('1.0', 'end').strip().splitlines():
            if '→' in line or '->' in line:
                sep = '→' if '→' in line else '->'
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    keywords = [k.strip() for k in parts[0].split(',') if k.strip()]
                    label = parts[1].strip()
                    if keywords and label:
                        mappings.append({'keywords': keywords, 'label': label})
        if not mappings:
            messagebox.showwarning('Empty', 'Please enter at least one keyword mapping.')
            return
        self.result = {
            'type': 'keyword_map',
            'source_col': self.src_var.get(),
            'output_col': self.out_var.get().strip(),
            'mappings': mappings,
            'default': self.default_var.get().strip(),
        }
        self.destroy()


class FormulaRuleDialog(BaseDialog):
    def _build(self):
        f = tk.Frame(self, bg=BG2, padx=14, pady=14)
        f.pack()
        tk.Label(f, text='Formula Rule', font=FONT_B, bg=BG2, fg=ACCENT2).pack(pady=(0, 8))
        self._label(f, 'Output column name:')
        self.out_var = self._entry(f, 'tap_rate')
        self._label(f, 'Expression (use column names, +, -, *, /):\n  e.g.  tap_count / duration_seconds')
        self.expr_var = self._entry(f, 'tap_count / duration_seconds')
        tk.Label(f, text='Available columns: ' + ', '.join(self.cols[:15]) + ('...' if len(self.cols) > 15 else ''),
                 bg=BG2, fg=SUBTEXT, font=FONT_S, wraplength=350).pack(pady=4)
        tk.Button(f, text='Add Rule', command=self._confirm, bg=OK, fg='white',
                  font=FONT_B, relief='flat').pack(pady=8)

    def _confirm(self):
        self.result = {
            'type': 'formula',
            'output_col': self.out_var.get().strip(),
            'expression': self.expr_var.get().strip(),
        }
        self.destroy()


class MonthsRuleDialog(BaseDialog):
    def _build(self):
        f = tk.Frame(self, bg=BG2, padx=14, pady=14)
        f.pack()
        tk.Label(f, text='Months Since Baseline Rule', font=FONT_B, bg=BG2, fg=ACCENT2).pack(pady=(0, 8))
        self._label(f, 'Time/Date column:')
        self.time_var = self._dropdown(f, self.cols)
        self._label(f, 'Subject/Patient ID column:')
        self.eid_var = self._dropdown(f, self.cols)
        self._label(f, 'Output column name:')
        self.out_var = self._entry(f, 'months_since_baseline')
        tk.Button(f, text='Add Rule', command=self._confirm, bg=OK, fg='white',
                  font=FONT_B, relief='flat').pack(pady=8)

    def _confirm(self):
        self.result = {
            'type': 'months_since_baseline',
            'time_col': self.time_var.get(),
            'entity_id_col': self.eid_var.get(),
            'output_col': self.out_var.get().strip(),
        }
        self.destroy()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = PDInsightApp()
    app.mainloop()
