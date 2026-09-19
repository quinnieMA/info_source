# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:10:59 2026

@author: Lenovo
"""

"""

05_sample_filter.py
===================
Apply sequential sample filters to produce analysis-ready dataset.

Input  : data/merged/04b_deal_firm_country.csv    [raw post-match sample]
Output : data/merged/05_deal_firm_filtered.csv     [final analysis sample]
         data/merged/05_filter_log.txt             [filter count log]

Filter sequence (empirical_design.md §3.2):
Note: operational order follows standard journal practice.
Core mandatory filters (executed in this exact order):
  1. sample period 2000–2024
  2. deal_type starts with "Acquisition"
  3. stake_acq_pct >= 50%  (impute from deal_type string where NaN)
  4. deal_value >= USD 1 million (= 1,000 in thousands-USD units)
  5. deal_status = Completed or Completed Assumed
  6. tar/acq SIC3 & country code all available

Extra internal constraint (not in paper table):
    Text peer pool >= 10 prior text-equipped deals

Implementation note:
All core criteria above are binding on the final sample.
Rolling peer pool counts are computed on the full input dataset before any filtering.
deal_year uses completed_d_yr with announced_d_yr fallback for Completed Assumed deals.

IMPORTANT — deal_num is NOT unique:
  One M&A deal can have multiple target firms, so multiple rows share deal_num.
  Use _row_id (unique per row) for 1:1 merges to avoid cartesian expansion.

IMPORTANT — Rolling counts (peer pools):
  Counts are calculated on the FULL unfiltered dataset to reflect the true historical
  deal population available at each year, not only the final sample.


Author: CC  Date: 2026-08-03
"""

import os
import re
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = r"D:\MA"
MERGED = os.path.join(BASE, "data", "merged")
QR     = os.path.join(BASE, "quality_reports")

IN_FILE  = os.path.join(MERGED, "04b_deal_firm_country.csv")
OUT_FILE = os.path.join(MERGED, "05_deal_firm_filtered.csv")
LOG_FILE = os.path.join(MERGED, "05_filter_log.txt")

os.makedirs(MERGED, exist_ok=True)
os.makedirs(QR, exist_ok=True)

# ── Logging helpers ────────────────────────────────────────────────────────
log_lines = []

def log(msg=""):
    """Print and buffer for diagnostic output."""
    print(msg)
    log_lines.append(str(msg))

# Filter summary: list of (label_str, n_dropped, n_remaining)
filter_summary = []

def apply_filter(df, keep_mask, label):
    """Apply boolean keep_mask; log and record dropped/remaining counts."""
    n_before = len(df)
    df_out   = df[keep_mask].copy()
    n_after  = len(df_out)
    n_drop   = n_before - n_after
    log(f"  {label}: dropped {n_drop:,} -> {n_after:,} remaining")
    filter_summary.append((label, n_drop, n_after))
    return df_out


# ════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("STEP 1 — Load 04b_deal_firm_country.csv")
log("=" * 70)

# Force Orbis ID columns to str to prevent numeric coercion of ID strings
# (consistent with earlier pipeline scripts 03, 04, 04b).
ORBIS_DTYPE = {c: str for c in [
    "tar_orbis_id_num", "acq_orbis_id_num", "ven_orbis_id_num",
    "tar_orbis_id_num_ovw", "acq_orbis_id_num_ovw",
    "tar_orbis_id_num_fin", "acq_orbis_id_num_fin",
    "tar_orbis_id_num_cf", "tar_orbis_id_num_leg", "acq_orbis_id_num_leg",
]}
df = pd.read_csv(IN_FILE, encoding="utf-8-sig", low_memory=False,
                 dtype=ORBIS_DTYPE)
INPUT_ROWS = len(df)
log(f"Rows: {INPUT_ROWS:,}  |  Columns: {df.shape[1]}")

# _row_id is assigned in Script 04b and carried through 04b_deal_firm_country.csv.
# Verify it is present, unique, and complete before any merges.
assert "_row_id" in df.columns, "_row_id column missing — re-run Script 04b first"
assert df["_row_id"].nunique() == len(df), "_row_id not unique in input file"
assert df["_row_id"].notna().all(), "_row_id has missing values"
log(f"_row_id check PASSED: 0 to {df['_row_id'].max()} (all unique, no NaN)")

log(f"deal_num: {df['deal_num'].nunique():,} unique / {len(df):,} total rows "
    f"(duplicates = multi-target deals)")

log("\ndeal_status distribution:")
for val, cnt in df["deal_status"].value_counts(dropna=False).items():
    log(f"  {str(val):<45s}: {cnt:>6,}")

log("\ndeal_type: non-null starts with Acquisition: "
    f"{df['deal_type'].dropna().str.startswith('Acquisition').sum():,}  "
    f"(NaN: {df['deal_type'].isna().sum():,})")


# ════════════════════════════════════════════════════════════════════════════
# 2. Pre-filter variable preparation  (computed on FULL 58,691-row dataset)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 2 — Pre-filter variable preparation  [full dataset, no filtering]")
log("=" * 70)

# ── 2.0  deal_year: completed_d_yr with announced_d_yr fallback ────────────
# Why: 18,694 Completed Assumed deals have completed_d_yr = NaN.
# deal_year ensures they participate in rolling peer pool counts.
# Mirrors the M1 fix applied in script 04b.
df["deal_year"] = pd.to_numeric(df["completed_d_yr"], errors="coerce").astype("Int64")
_fallback = df["deal_year"].isna() & df["announced_d_yr"].notna()
df.loc[_fallback, "deal_year"] = pd.to_numeric(
    df.loc[_fallback, "announced_d_yr"], errors="coerce"
).astype("Int64")
n_completed_yr = (~_fallback & df["deal_year"].notna()).sum()
n_fallback_yr  = _fallback.sum()
n_still_nan    = df["deal_year"].isna().sum()
log(f"deal_year: {df['deal_year'].notna().sum():,} non-null total")
log(f"  From completed_d_yr   : {n_completed_yr:,}")
log(f"  From announced_d_yr   : {n_fallback_yr:,}  (fallback for Completed Assumed)")
log(f"  Still NaN (no year)   : {n_still_nan:,}  "
    f"(excluded from rolling counts — no timestamp available)")

# ── 2.1  tar_overview_wordcount ────────────────────────────────────────────
df["tar_overview_wordcount"] = (
    df["tar_overview"].str.split().str.len().fillna(0).astype(int)
)
n_ov20 = (df["tar_overview_wordcount"] >= 20).sum()
log(f"\ntar_overview_wordcount >= 20: {n_ov20:,} rows  ({n_ov20/INPUT_ROWS*100:.1f}%)")

# ── 2.2  SIC codes ─────────────────────────────────────────────────────────
df["tar_sic4"] = pd.to_numeric(df["tar_primary_sic_code"], errors="coerce")
df["tar_sic3"] = (df["tar_sic4"] // 10).astype("Int64")
df["tar_sic2"] = (df["tar_sic4"] // 100).astype("Int64")
df["acq_sic3"] = (
    pd.to_numeric(df["acq_primary_sic_code"], errors="coerce") // 10
).astype("Int64")
log(f"\ntar_sic4 non-null: {df['tar_sic4'].notna().sum():,}  "
    f"tar_sic3 non-null: {df['tar_sic3'].notna().sum():,}  "
    f"acq_sic3 non-null: {df['acq_sic3'].notna().sum():,}")

# ── 2.3  SIC rolling peer counts (Filter 7) ───────────────────────────────
# For each row i in year t: count rows in the same SIC-3 group with
# deal_year <= t. cumcount() on a deal_year-sorted DataFrame gives
# 0-indexed rank within the SIC-3 group = number of prior deals = peer pool.
#
# Join key: _row_id (not deal_num) — deal_num is not unique (multi-target
# deals share one deal_num), and merging on a non-unique key creates a
# cartesian product that multiplies rows.
log("\nComputing SIC-3 and SIC-2 rolling peer counts (on full dataset)...")

df_sic = (
    df[["_row_id", "deal_year", "tar_sic3", "tar_sic2"]]
    .dropna(subset=["deal_year", "tar_sic3"])
    .copy()
)

sic3_yr = (
    df_sic.groupby(["tar_sic3", "deal_year"]).size()
    .reset_index(name="n")
    .sort_values(["tar_sic3", "deal_year"])
)
sic3_yr["sic3_rolling_count"] = (
    sic3_yr.groupby("tar_sic3")["n"].cumsum() - 1
)

sic2_yr = (
    df_sic.dropna(subset=["tar_sic2"])
    .groupby(["tar_sic2", "deal_year"]).size()
    .reset_index(name="n")
    .sort_values(["tar_sic2", "deal_year"])
)
sic2_yr["sic2_rolling_count"] = (
    sic2_yr.groupby("tar_sic2")["n"].cumsum() - 1
)

df_sic = df_sic.merge(
    sic3_yr[["tar_sic3", "deal_year", "sic3_rolling_count"]],
    on=["tar_sic3", "deal_year"], how="left"
)
df_sic = df_sic.merge(
    sic2_yr[["tar_sic2", "deal_year", "sic2_rolling_count"]],
    on=["tar_sic2", "deal_year"], how="left"
)

# 1:1 merge via _row_id — row count must stay at INPUT_ROWS
df = df.merge(
    df_sic[["_row_id", "sic3_rolling_count", "sic2_rolling_count"]],
    on="_row_id", how="left"
)
assert len(df) == INPUT_ROWS, \
    f"SIC merge changed row count: {INPUT_ROWS} -> {len(df)}"

n_sic_null = df["sic3_rolling_count"].isna().sum()
log(f"  sic3_rolling_count: {df['sic3_rolling_count'].notna().sum():,} non-null "
    f"({n_sic_null:,} NaN = no deal_year or no tar_sic3)")
log(f"  Deals with sic3_rolling_count >= 3  : "
    f"{(df['sic3_rolling_count'] >= 3).sum():,}")
log(f"  Deals with sic2_rolling_count >= 3  : "
    f"{(df['sic2_rolling_count'] >= 3).sum():,}")
log(f"  Rows passing Filter 7 (SIC3>=3 OR SIC2>=3): "
    f"{((df['sic3_rolling_count'] >= 3) | (df['sic2_rolling_count'] >= 3)).sum():,}")

# sic_benchmark_level: which SIC level will be used for SIC Benchmark (script 07)
df["sic_benchmark_level"] = np.where(
    df["sic3_rolling_count"] >= 3, 3,
    np.where(df["sic2_rolling_count"] >= 3, 2, np.nan)
)

# ── 2.4  Text peer pool rolling count (Filter 8) ──────────────────────────
# For each row i in year t: count all text-equipped rows with deal_year <= t.
# Cross-year cumulative pool. Row index in a deal_year-sorted DataFrame
# equals number of prior rows = prior text-deal count for that row.
#
# Uses deal_year (not completed_d_yr) so Completed Assumed rows with text
# are included in the pool — consistent with Doc2Vec training on full corpus.
# Join key: _row_id (not deal_num) — same reason as above.
log("\nComputing text peer pool rolling count (on full dataset)...")

df_text = (
    df.loc[df["tar_overview"].notna() & df["deal_year"].notna(),
           ["_row_id", "deal_year"]]
    .sort_values(["deal_year", "_row_id"])
    .reset_index(drop=True)
    .copy()
)
# Row index 0 = first text-equipped row (0 prior peers); index k = k prior text rows
df_text["text_pool_rolling_count"] = df_text.index

df = df.merge(
    df_text[["_row_id", "text_pool_rolling_count"]],
    on="_row_id", how="left"
)
assert len(df) == INPUT_ROWS, \
    f"Text merge changed row count: {INPUT_ROWS} -> {len(df)}"

n_text_null = df["text_pool_rolling_count"].isna().sum()
log(f"  text_pool_rolling_count non-null: {df['text_pool_rolling_count'].notna().sum():,} "
    f"({n_text_null:,} NaN = no tar_overview or no deal_year)")
log(f"  Rows with text_pool_rolling_count >= 10: "
    f"{(df['text_pool_rolling_count'] >= 10).sum():,}")

# ════════════════════════════════════════════════════════════════════════════
# 3. Sequential filters [core 6 in specified order; additional filters follow]
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 3 — Sequential filters (aligned with empirical_design sample table)")
log("=" * 70)

log(f"\nInput: {INPUT_ROWS:,} rows")
filter_summary.append(("Input", 0, INPUT_ROWS))

# --------------------------
# [Pre-definition: equity extraction function]
# --------------------------
def infer_stake_from_deal_type(deal_type_str):
    """Extract max percentage from deal_type string; return NaN if none found."""
    if pd.isna(deal_type_str):
        return np.nan
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", str(deal_type_str))
    if not pcts:
        return np.nan
    return max(float(p) for p in pcts)


# ── Filter 1: sample period 2000–2024 ─────────────────────────────────────
keep1 = (df["deal_year"] >= 2000) & (df["deal_year"] <= 2024)
df = apply_filter(df, keep1,
    "Filter 1 (sample period 2000–2024)")

# ── Filter 2: deal_type starts with "Acquisition" ───────────────────────────
keep2 = df["deal_type"].notna() & df["deal_type"].str.startswith("Acquisition")
df = apply_filter(df, keep2,
    "Filter 2 (deal_type starts with Acquisition)")

# ── Filter 3: stake_acq_pct >= 50% (impute from deal_type where NaN) ──────
mask_nan_stake     = df["stake_acq_pct"].isna()
n_nan_stake_before = mask_nan_stake.sum()
df.loc[mask_nan_stake, "stake_acq_pct_imputed"] = (
    df.loc[mask_nan_stake, "deal_type"].apply(infer_stake_from_deal_type)
)
stake_effective = df["stake_acq_pct"].fillna(
    df.get("stake_acq_pct_imputed", pd.Series(np.nan, index=df.index))
)
keep3 = stake_effective >= 50
df = apply_filter(df, keep3,
    "Filter 3 (stake_acq_pct >= 50%, impute from deal_type if missing)")

# ── Filter 4: deal_value >= USD 1 million ─────────────────────────────────
keep4 = df["deal_value"] >= 1000
df = apply_filter(df, keep4,
    "Filter 4 (deal_value >= USD 1M)")

# ── Filter 5: deal_status Completed / Completed Assumed ─────260906 removed──────────────
# ★ Never applied: main sample must retain non-completed observations, otherwise completion probability equation has no variation.
#   Unknown outcomes (Announced / Pending / Postponed) keep original values, never filled as 0 (failed).
#keep5 = df["deal_status"].isin(["Completed", "Completed Assumed"])
#df = apply_filter(df, keep5,
    #"Filter 5 (deal_status = Completed/Completed Assumed)")

# ── Filter 7: tar/acq SIC3 & country code both available ───────────────────
keep7 = (df["tar_sic3"].notna() & df["acq_sic3"].notna() &
         df["tar_country_code"].notna() & df["acq_country_code"].notna())
df = apply_filter(df, keep7,
    "Filter 7 (tar/acq SIC3 & country code all available)")

# ══ Below are overall definitions at the [research design] level ══
# Order aligns with Table 1: first derive analysis file (19,745), then apply listing status and purity constraints.
# Filters are commutative; reordering does not change final N, only Table 1 stepwise presentation.

# ── Filter 8: acq_listed = 1   (Table 1 step 10) ──────────────────────────
if "acq_listed" in df.columns and df["acq_listed"].notna().any():
    keep8 = pd.to_numeric(df["acq_listed"], errors="coerce") == 1
    df = apply_filter(df, keep8, "Filter 8 (acq_listed = 1)")
else:
    log("  [skip] Filter 8: no acq_listed column in 04b (or all null)")

# ── Filter 9: tar_listed = 0   (Table 1 step 11) ──────────────────────────
if "tar_listed" in df.columns and df["tar_listed"].notna().any():
    keep9 = pd.to_numeric(df["tar_listed"], errors="coerce") == 0
    df = apply_filter(df, keep9, "Filter 9 (tar_listed = 0)")
else:
    log("  [skip] Filter 9: no tar_listed column in 04b (or all null)")

# -- Filter 10: drop public-offer structures (Table 1 step 12) --
# Public offers can only target listed targets; drop to avoid residual targets that "delisted after deal"
PUB_STRUCT = ["public takeover", "scheme of arrangement", "recommended bid",
              "hostile bid", "contested bid", "tender", "takeover"]
_struct_col = next((c for c in ["deal_struct_all", "deal_struct"]
                    if c in df.columns), None)
if _struct_col is not None:
    _s = df[_struct_col].astype(str).str.strip().str.lower()
    df = apply_filter(df, ~_s.apply(lambda x: any(p in x for p in PUB_STRUCT)),
        "Filter 10 (drop public-offer structures)")
else:
    log("  [skip] Filter 10: no deal_struct column in 04b")
    
# ════════════════════════════════════════════════════════════════════════════
# 4. Post-filter variable construction
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 4 — Post-filter variable construction")
log("=" * 70)

# 4.1 cross-industry: 1 = cross-industry (target and acquirer in different SIC-3)
# tar_sic3 and acq_sic3 are nullable Int64; comparing two NA values returns
# <NA>, which cannot be cast to int directly. Use np.where with explicit
# validity mask to handle NAs safely.
both_valid = df["tar_sic3"].notna() & df["acq_sic3"].notna()
df["cross_industry"] = np.where(
    both_valid,
    (df["tar_sic3"].astype(float) != df["acq_sic3"].astype(float)).astype(int),
    np.nan
)
n_cross = int((df["cross_industry"] == 1).sum())
n_same  = int((df["cross_industry"] == 0).sum())
log(f"cross_industry: same-industry=0: {n_same:,}  "
    f"cross_industry=1: {n_cross:,}  NaN: {df['cross_industry'].isna().sum():,}")

# 4.2 ln_deal_value: natural log of deal value
df["ln_deal_value"] = np.log(df["deal_value"])
log(f"ln_deal_value: mean={df['ln_deal_value'].mean():.3f}  "
    f"sd={df['ln_deal_value'].std():.3f}  "
    f"min={df['ln_deal_value'].min():.2f}  max={df['ln_deal_value'].max():.2f}")

# 4.3 cross_border: 1 = acquirer and target in different countries
both_cc = df["tar_country_code"].notna() & df["acq_country_code"].notna()
df["cross_border"] = np.where(
    both_cc,
    (df["tar_country_code"] != df["acq_country_code"]).astype(int),
    np.nan
)
n_dom = int((df["cross_border"] == 0).sum())
n_cb  = int((df["cross_border"] == 1).sum())
log(f"cross_border: domestic=0: {n_dom:,}  "
    f"cross_border=1: {n_cb:,}  NaN: {df['cross_border'].isna().sum():,}")

# 4.4 cash: 1 if deal_pay_method contains "cash" (case-insensitive)
#df["cash"] = (
   # df["deal_pay_method"].str.lower().str.contains("cash", na=False).astype(int)
#)
#n_noncash = int((df["cash"] == 0).sum())
#n_cash    = int((df["cash"] == 1).sum())
#log(f"cash: non-cash/unknown=0: {n_noncash:,}  includes-cash=1: {n_cash:,}")

# ══ 4.4 Payment method dummies (rewritten 2026-09-07) ══════════════════════════════════
# Source: 01a Module C, pipe-separated set aggregated [before] deal_num dedup.
# Background: Zephyr one payment method per row; mixed payment (Cash|Shares) spans multiple rows;
#       old keep-first dedup silently dropped the second row. Empirical share-exchange rate was underestimated
#       13.4% -> 18.6% (share-exchange info for 2,238 deals was dropped).
#
# Design: first merge by category (Cash and Cash Reserves both cash, not mixed),
#       then determine pure/mixed. If judged by value count directly, 3,244 Cash|Cash Reserves
#       would be misclassified as mixed payment, severely underestimating pure cash group.
PAY_SRC = "deal_pay_method_all"
if PAY_SRC not in df.columns:
    raise ValueError(
        f"Missing {PAY_SRC}. Please rerun 01a_clean_deal_modules.py -- this column is generated in Module C "
        f"before dedup, aggregated by deal_num, to fix mixed payment methods dropped by keep-first "
        f"dedup.")

# Value -> category (covers all 15 values observed in 01a)
PAY_CLASS_MAP = {
    # Cash
    "Cash":               "cash",
    "Cash Reserves":      "cash",
    # Equity -- key group triggering securities review
    "Shares":             "shares",
    "Third party shares": "shares",
    # Debt
    "Liabilities":        "debt",
    "Converted Debt":     "debt",
    "Bonds":              "debt",
    # Other / deferred
    "Deferred payment":   "other",
    "Earn-out":           "other",
    "Business assets":    "other",
    "Dividend":           "other",
    "Services":           "other",
    "Other":              "other",
    "Cash assumed":       "other",   # Assume target cash, semantically ambiguous, conservatively other
}

_unmapped = set()

def _to_classes(s):
    """'Cash|Shares' -> {'cash','shares'}; unmapped values go to _unmapped for reporting."""
    if pd.isna(s) or not str(s).strip():
        return frozenset()
    out = set()
    for v in (x.strip() for x in str(s).split("|")):
        if not v:
            continue
        c = PAY_CLASS_MAP.get(v)
        if c is None:
            _unmapped.add(v)
        else:
            out.add(c)
    return frozenset(out)

_ps = df[PAY_SRC].apply(_to_classes)

if _unmapped:
    log(f"\n  WARNING: unmapped payment method values {len(_unmapped)}: {sorted(_unmapped)}")
    log(f"     Please add to PAY_CLASS_MAP and rerun, otherwise these deals fall into wrong category")

# -- Non-exclusive: contains any category --
df["pay_has_cash"]   = _ps.apply(lambda c: int("cash"   in c))
df["pay_has_shares"] = _ps.apply(lambda c: int("shares" in c))
df["pay_has_debt"]   = _ps.apply(lambda c: int("debt"   in c))
df["pay_has_other"]  = _ps.apply(lambda c: int("other"  in c))
df["pay_n_class"]    = _ps.apply(len)          # 1=pure, >1=mixed, 0=no record

# -- Mutually exclusive complete set: 6 dummies sum to 1 --
df["pay_pure_cash"]   = _ps.apply(lambda c: int(c == {"cash"}))
df["pay_pure_shares"] = _ps.apply(lambda c: int(c == {"shares"}))
df["pay_pure_debt"]   = _ps.apply(lambda c: int(c == {"debt"}))
df["pay_pure_other"]  = _ps.apply(lambda c: int(c == {"other"}))
df["pay_mix"]         = _ps.apply(lambda c: int(len(c) > 1))
df["pay_unknown"]     = _ps.apply(lambda c: int(len(c) == 0))

# -- Mixed payment breakdown (coexists with pay_mix, can be used together) --
df["pay_mix_cash_shares"] = _ps.apply(lambda c: int(c == {"cash", "shares"}))
df["pay_mix_with_shares"] = _ps.apply(lambda c: int(len(c) > 1 and "shares" in c))
df["pay_mix_with_cash"]   = _ps.apply(lambda c: int(len(c) > 1 and "cash"   in c))
df["pay_mix_with_debt"]   = _ps.apply(lambda c: int(len(c) > 1 and "debt"   in c))

# -- Mutually exclusive category labels (C(pay_class) convenient in regression, base group = cash) --
df["pay_class"] = np.select(
    [df["pay_unknown"] == 1,
     df["pay_pure_cash"] == 1,
     df["pay_pure_shares"] == 1,
     df["pay_pure_debt"] == 1,
     df["pay_pure_other"] == 1],
    ["unknown", "cash", "shares", "debt", "other"],
    default="mix"
)

# -- Check: must be complete set and mutually exclusive --
_chk = df[["pay_pure_cash", "pay_pure_shares", "pay_pure_debt",
           "pay_pure_other", "pay_mix", "pay_unknown"]].sum(axis=1)
if not (_chk == 1).all():
    raise ValueError(f"Payment method dummies do not form complete set ({int((_chk != 1).sum())} abnormal rows)")

log("\n" + "=" * 66)
log("Payment method distribution (category level, merge first then pure/mixed)")
log("=" * 66)
log(f"  Sample N={len(df):,}")
log("")

# -- Safe column access: some derived variables not yet generated at stage 05 (e.g. tar_china derived in 08b) --
#    Note: pd.to_numeric(None, errors="coerce") does not error, returns nan scalar,
#    scalar has no .fillna, so must check column existence first.
def _num_or_zero(name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)

_has_sec = "reg_securities" in df.columns
# At stage 05 tar_china not yet derived, fall back to computing from acq/tar country code
if "tar_china" in df.columns:
    _cn = _num_or_zero("tar_china")
elif "tar_country_code" in df.columns:
    _cn = df["tar_country_code"].astype(str).str.upper().eq("CN").astype(float)
    log("  [note] tar_china not derived, China% computed from tar_country_code=='CN'")
else:
    _cn = pd.Series(0.0, index=df.index)
    log("  [note] no tar_china / tar_country_code, China% column = 0")

_rs = _num_or_zero("reg_securities")

log(f"  {'Category':<10s}{'N':>9s}{'Pct':>8s}"
    + (f"{'SecReview%':>11s}" if _has_sec else "")
    + f"{'China%':>9s}")
for _k in ["cash", "shares", "debt", "other", "mix", "unknown"]:
    _m = (df["pay_class"] == _k)
    if _m.sum() == 0:
        continue
    _line = f"  {_k:<10s}{int(_m.sum()):>9,}{_m.mean():>7.1%}"
    if _has_sec:
        _line += f"{_rs[_m].mean():>10.1%}"
    _line += f"{_cn[_m].mean():>8.1%}"
    log(_line)

log("")
log(f"  Share exchange (shares class, incl. mixed): {df['pay_has_shares'].mean():>6.1%}"
    f"   <- old dedup method was only 13.4%")
log(f"  Cash                      : {df['pay_has_cash'].mean():>6.1%}")
log(f"  Debt assumption           : {df['pay_has_debt'].mean():>6.1%}")
log(f"  Mixed payment             : {df['pay_mix'].mean():>6.1%}")
log(f"  of which cash+shares      : {df['pay_mix_cash_shares'].mean():>6.1%}")
log(f"  No payment record         : {df['pay_unknown'].mean():>6.1%}"
    f"   <- needs discussion in main text")
log("=" * 66)
# ══ 4.4 Payment method dummies (rewritten 2026-09-07) end══════════════════════════════════

# tar_overview_wordcount already computed in §2.1; retained in output.
# _row_id retained in output (useful for debugging; drop before Stata if preferred).

# ════════════════════════════════════════════════════════════════════════════
# 5. Save output dataset
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 5 — Save output")
log("=" * 70)

# Columns added vs input:
#   _row_id, deal_year, tar_overview_wordcount, tar_sic4, tar_sic3, tar_sic2,
#   acq_sic3, sic3_rolling_count, sic2_rolling_count, sic_benchmark_level,
#   text_pool_rolling_count, stake_acq_pct_imputed,
#   cross-industry, ln_deal_value, cross-border, cash
# Retained from input:
#   _year_source — for Stata robustness checks that exclude fallback-year observations

df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
log(f"Saved -> {OUT_FILE}")
log(f"Size  : {os.path.getsize(OUT_FILE)/1024/1024:.1f} MB")
log(f"Rows  : {len(df):,}  |  Columns: {df.shape[1]}")

n_unique_deals = df["deal_num"].nunique()
log(f"Unique deal_num: {n_unique_deals:,}")


# ════════════════════════════════════════════════════════════════════════════
# 6. Filter log summary
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("FILTER LOG SUMMARY")
log("=" * 70)

FILTER_DISPLAY = {
    "Filter 1 (sample period 2000–2024)":
        "Filter 1  (sample period 2000-2024)",
    "Filter 2 (deal_type starts with Acquisition)":
        "Filter 2  (deal_type starts with Acquisition)",
    "Filter 3 (stake_acq_pct >= 50%, impute from deal_type if missing)":
        "Filter 3  (stake_acq_pct >= 50%, impute if missing)",
    "Filter 4 (deal_value >= USD 1M)":
        "Filter 4  (deal_value >= USD 1M)",
    "Filter 5 (deal_status = Completed/Completed Assumed)":
        "Filter 5  (deal_status = Completed/Completed Assumed)",
    "Filter 7 (tar/acq SIC3 & country code all available)":
        "Filter 7  (tar/acq SIC3 & country code all available)",
    "Filter 8 (acq_listed = 1)":
        "Filter 8  (acq_listed = 1)",
    "Filter 9 (tar_listed = 0)":
        "Filter 9  (tar_listed = 0)",
    "Filter 10 (drop public-offer structures)":
        "Filter 10 (drop public-offer structures)",
}
    
log_text_lines = [
    "=== SAMPLE FILTER LOG ===",
    f"Input  : 04b_deal_firm_country.csv  {INPUT_ROWS:,} rows",
    "",
]
for label, n_drop, n_remaining in filter_summary:
    if label == "Input":
        continue
    display = FILTER_DISPLAY.get(label, label)
    log_text_lines.append(
        f"{display:<52s}: dropped {n_drop:>6,}  -> {n_remaining:>6,} remaining"
    )
log_text_lines += [
    "",
    f"Final sample : {len(df):,} rows  |  {n_unique_deals:,} unique deals",
]

log_text = "\n".join(log_text_lines)
log("\n" + log_text)

with open(LOG_FILE, "w", encoding="utf-8") as fh:
    fh.write(log_text + "\n")
log(f"Filter log saved -> {LOG_FILE}")

# ════════════════════════════════════════════════════════════════════════════
# 7. Validation checks
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("STEP 7 — Validation checks")
log("=" * 70)

# Row count monotone decreasing across filters
remainders = [n_remaining for _, _, n_remaining in filter_summary]
monotone   = all(remainders[i] >= remainders[i+1]
                 for i in range(len(remainders)-1))
log(f"Monotone decreasing filter counts: {'PASS' if monotone else 'FAIL'}")
if not monotone:
    for i, (lbl, nd, nr) in enumerate(filter_summary):
        if i > 0 and nr > filter_summary[i-1][2]:
            log(f"  WARNING: {lbl} shows increase "
                f"({filter_summary[i-1][2]:,} -> {nr:,})")

# Final sample in expected range
EXPECTED_LOW, EXPECTED_HIGH = 10_000, 25_000
in_range = EXPECTED_LOW <= len(df) <= EXPECTED_HIGH
log(f"Final sample in expected range [{EXPECTED_LOW:,}–{EXPECTED_HIGH:,}]: "
    f"{'PASS' if in_range else 'WARNING — outside expected range'} "
    f"({len(df):,} rows)")

# Output file non-empty
out_size = os.path.getsize(OUT_FILE)
log(f"Output file size: {out_size:,} bytes "
    f"({'OK' if out_size > 0 else 'EMPTY — ERROR'})")

log("\nScript 05 complete.")

