# app.py
# Streamlit Budget vs Actual (uses st.secrets["GOOGLE_CREDS"] for Google Sheets auth)

import os
import re
import unicodedata
import calendar
from datetime import datetime
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe

# =========================
# Constants / Mappings
# =========================
CAT_ORDER = ["Groceries", "Medical expenses", "Gas", "Taylor's flex"]

# Use CATEGORY_SUM for expenses
EXP_MAP = {
    'groceries': 'Groceries',
    'medical expenses': 'Medical expenses',
    'gas expenses': 'Gas',
    'gas by femi': 'Gas',
    'gas': 'Gas',
    "taylor's flex": "Taylor's flex",
    'taylors flex': "Taylor's flex",
}

# Fold budget lines -> Groceries + keep direct Groceries
BUD_MAP = {
    'groceries': 'Groceries',
    'food by femi': 'Groceries',
    'baby': 'Groceries',
    'other household': 'Groceries',
    'medical expenses': 'Medical expenses',
    'gas by femi': 'Gas',
    'gas': 'Gas',
    "taylor's flex": "Taylor's flex",
    'taylors flex': "Taylor's flex",
}

# =========================
# Utilities
# =========================
def ensure_cols(df: pd.DataFrame, required: Iterable[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {missing}. Present: {list(df.columns)}")

def to_number(x):
    if pd.isna(x): return np.nan
    s = unicodedata.normalize('NFKC', str(x))
    neg = s.startswith('(') and s.endswith(')')
    if neg: s = s[1:-1]
    s = s.replace(',', '').replace(' ', '')
    s = re.sub(r'[^0-9.\-]', '', s)
    v = pd.to_numeric(s, errors='coerce')
    return -v if neg and pd.notna(v) else v

ABBR = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
FULL = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
def parse_month_any(s) -> pd.Period:
    if pd.isna(s): return pd.NaT
    t = unicodedata.normalize('NFKC', str(s)).strip().lower()
    m = re.search(r'([a-z]{3,9})\s*([12]\d{3})', t)  # oct2025 / october 2025
    if m:
        mon, year = m.group(1), int(m.group(2))
        if mon[:3] in ABBR:  return pd.Period(f"{year}-{ABBR[mon[:3]]:02d}", freq="M")
        if mon in FULL:      return pd.Period(f"{year}-{FULL[mon]:02d}",    freq="M")
    m = re.search(r'([12]\d{3})[-/](\d{1,2})', t)     # yyyy-mm, yyyy/mm, yyyy-mm-dd
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return pd.Period(f"{year}-{month:02d}", freq="M")
    try:
        return pd.to_datetime(t, errors='raise').to_period('M')
    except Exception:
        return pd.NaT

def norm_text(s):
    if pd.isna(s): return None
    s = unicodedata.normalize('NFKC', str(s))
    s = s.replace('’', "'").replace('‘', "'").replace('´', "'").replace('`', "'")
    s = re.sub(r'\s+', ' ', s.strip()).lower()
    return s

# =========================
# Custom budget windows
# =========================
def map_transaction_date(posting_date):
    if pd.isnull(posting_date): return np.nan
    p = pd.Timestamp(posting_date)
    rules = [
        ("2025-01-02","2025-01-29","2025-01-01"),
        ("2025-01-30","2025-02-26","2025-02-01"),
        ("2025-02-27","2025-03-26","2025-03-01"),
        ("2025-03-27","2025-04-23","2025-04-01"),
        ("2025-04-24","2025-05-21","2025-05-01"),
        ("2025-05-22","2025-06-18","2025-06-01"),
        ("2025-06-19","2025-07-30","2025-07-01"),
        ("2025-07-31","2025-08-27","2025-08-01"),
        ("2025-08-28","2025-09-24","2025-09-01"),
        ("2025-09-25","2025-10-22","2025-10-01"),
        ("2025-10-23","2025-11-19","2025-11-01"),
        ("2025-11-20","2025-12-31","2025-12-01"),
    ]
    for lo, hi, start in rules:
        if pd.Timestamp(lo) <= p <= pd.Timestamp(hi):
            return pd.Timestamp(start)
    return np.nan

def current_bucket_label(today: Optional[pd.Timestamp] = None) -> Optional[str]:
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    start = map_transaction_date(today)
    return start.strftime('%b%Y') if pd.notna(start) else None

# =========================
# Google Sheets (Streamlit secrets)
# =========================

import json
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe
import pandas as pd


@st.cache_data(show_spinner=True)
def load_sheet_to_df(sheet_url: str, tab_name: str) -> pd.DataFrame:
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    # ✅ Fix: parse JSON string from Streamlit secrets
    raw = st.secrets["GOOGLE_CREDS"]
    creds_dict = json.loads(raw) if isinstance(raw, str) else dict(raw)

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    ws = client.open_by_url(sheet_url).worksheet(tab_name)
    df = get_as_dataframe(ws, evaluate_formulas=True).dropna(how='all')
    return df

# =========================
# Core pipeline
# =========================
def build_report(sum_exp: pd.DataFrame, budget_df: pd.DataFrame) -> pd.DataFrame:
    ensure_cols(sum_exp, ['Month', 'AMOUNTS', 'CATEGORY_SUM'], 'sum_exp')
    ensure_cols(budget_df, ['Month', 'Budgeted_Amount', 'category'], 'budget_df')

    # Expenses
    exp = sum_exp.copy()
    exp['AMOUNTS'] = exp['AMOUNTS'].apply(to_number)
    exp['MonthKey'] = exp['Month'].apply(parse_month_any)
    exp['Category'] = exp['CATEGORY_SUM'].apply(norm_text).map(EXP_MAP)
    exp_ok = exp[exp['MonthKey'].notna() & exp['Category'].isin(CAT_ORDER)]
    actual_m = (exp_ok.groupby(['MonthKey','Category'], as_index=False)['AMOUNTS']
                    .sum().rename(columns={'AMOUNTS':'Actual'}))

    # Budget
    bud = budget_df.copy()
    bud['Budgeted_Amount'] = bud['Budgeted_Amount'].apply(to_number)
    bud['MonthKey'] = bud['Month'].apply(parse_month_any)
    bud['Category'] = bud['category'].apply(norm_text).map(BUD_MAP)
    bud_ok = bud[bud['MonthKey'].notna() & bud['Category'].isin(CAT_ORDER)]
    budget_m = (bud_ok.groupby(['MonthKey','Category'], as_index=False)['Budgeted_Amount']
                    .sum().rename(columns={'Budgeted_Amount':'Budget'}))

    # Merge 4×N grid
    months = pd.PeriodIndex(sorted(set(budget_m['MonthKey']) | set(actual_m['MonthKey']))).sort_values()
    idx = pd.MultiIndex.from_product([months, CAT_ORDER], names=['MonthKey','Category'])
    report = (budget_m.merge(actual_m, on=['MonthKey','Category'], how='outer')
                        .set_index(['MonthKey','Category'])
                        .reindex(idx)
                        .reset_index())

    # Metrics
    report['Budget'] = pd.to_numeric(report['Budget'], errors='coerce').fillna(0)
    report['Actual'] = pd.to_numeric(report['Actual'], errors='coerce').fillna(0)
    report['Variance'] = report['Budget'] - report['Actual']
    report['Pct_Used'] = np.where(report['Budget'] != 0, report['Actual']/report['Budget'], np.nan)
    report['Month'] = report['MonthKey'].dt.strftime('%b%Y')

    # Tidy
    report_tidy = (report[['Month','Category','Budget','Actual','Variance','Pct_Used']]
                   .sort_values(['Month','Category'])
                   .reset_index(drop=True))
    return report_tidy

# =========================
# Plotting (returns a Matplotlib figure for st.pyplot)
# =========================
def make_grouped_bar_figure(report_tidy: pd.DataFrame, month_label: str) -> Optional[plt.Figure]:
    dfp = (report_tidy[report_tidy['Month'] == month_label]
           .set_index('Category')
           .reindex(CAT_ORDER)[['Budget','Actual']])
    if dfp.empty or dfp.isna().all().all():
        return None
    dfp = dfp.fillna(0.0)

    x = np.arange(len(CAT_ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_budget = ax.bar(x - width/2, dfp['Budget'].values, width, label='Budget')
    bars_actual = ax.bar(x + width/2, dfp['Actual'].values, width, label='Actual')

    # Value labels
    ymax = float(max(dfp['Budget'].max(), dfp['Actual'].max()))
    offset = 0.02 * (ymax if ymax > 0 else 1.0)
    for bars in (bars_budget, bars_actual):
        for b in bars:
            h = float(b.get_height())
            ax.text(b.get_x() + b.get_width()/2, h + offset, f"{h:,.0f}",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER, rotation=20, ha='right')
    ax.set_ylabel('Amount ($)') 
    ax.set_title(f'Budget vs Actual — {month_label}')
    ax.legend()
    fig.tight_layout()
    return fig

# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="Budget vs Actual", page_icon="📊", layout="wide")
    st.title("📊 Budget vs Actual — Monthly")

    # Sheet URLs / Tabs (you can also move these to st.secrets if you prefer)
    SUM_EXP_URL = "https://docs.google.com/spreadsheets/d/1PDFMR7Xdv3VqnpwqeftS_d81HNVD3SUGs9MccXRwdOs"
    SUM_EXP_TAB = "Sheet14"
    BUDGET_URL  = "https://docs.google.com/spreadsheets/d/1t4a-OWpHNNiwkxPw-bH-71CUO512li1P9wVME2KQqQI"
    BUDGET_TAB  = "Sheet6"

    # 🔁 Auto + Manual refresh
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600_000, key="auto_refresh")  # auto every 10 min
    if st.button("🔄 Refresh data now"):
        st.cache_data.clear()
        st.experimental_rerun()

    with st.spinner("Loading Google Sheets…"):
        sum_exp = load_sheet_to_df(SUM_EXP_URL, SUM_EXP_TAB)
        budget_df = load_sheet_to_df(BUDGET_URL, BUDGET_TAB)

    # Build report
    try:
        report_tidy = build_report(sum_exp, budget_df)
    except Exception as e:
        st.error(f"Failed to build report: {e}")
        st.stop()

    # Default month = current bucket (falls back to first month if not found)
    default_label = current_bucket_label()
    month_options = sorted(report_tidy['Month'].unique(), key=lambda x: pd.to_datetime(x, format='%b%Y'))
    if default_label not in month_options and month_options:
        default_label = month_options[-1]  # latest month in data

    selected_month = st.selectbox("Select month", month_options, index=month_options.index(default_label) if default_label in month_options else 0)

    # Chart
    fig = make_grouped_bar_figure(report_tidy, selected_month)
    if fig is None:
        st.info(f"No data for {selected_month}")
    else:
        st.pyplot(fig)

    # Table below
    st.subheader("Detail")
    st.dataframe(
        report_tidy[report_tidy['Month'] == selected_month]
        .set_index('Category')
        .reindex(CAT_ORDER)
        .assign(Pct_Used=lambda d: (d['Pct_Used']*100).round(1))
        .rename(columns={'Pct_Used':'Pct_Used (%)'})
    )

    # Download CSV
    csv = report_tidy.to_csv(index=False).encode('utf-8')
    st.download_button("Download full report CSV", data=csv, file_name="budget_vs_actual_report.csv", mime="text/csv")

if __name__ == "__main__":
    main()
