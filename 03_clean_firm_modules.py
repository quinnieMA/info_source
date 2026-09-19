"""
03_clean_firm_modules.py
========================
Clean SIX MA_firm sub-modules and save to data/cleaned/.

Sources:
  Module 03pre   — Pre-deal snapshot  : raw/MA_firm/financial/acquisition_financial_*_cleaned.csv
  Module 03post  — Post-deal snapshot : same raw files as 03pre (re-read; see note below)
  Module 03com_fin — Company time-series: raw/MA_firm/financial/acquisition_company_financial_*_cleaned.csv
              (file _8 is an exact duplicate of _7 — excluded by filename filter)
  Module 03legal — Legal status       : raw/MA_firm/legal/acquisition_legal_*_cleaned.csv
  Module 03list  — Listed-firm status : raw/MA_firm/listedstatus/list_*.csv
  Module 03adv   — Advisor counts     : raw/MA_firm/advisor/advisor*_cleaned.csv

  extract incorporated year added 2026-07-27

Processing principle (Modules 03pre, 03post, 03com_fin, 03legal — via dedup_by_deal()):
  - Read each batch CSV in file order — NO sort before ffill
  - ffill deal_num within each batch BEFORE stacking
  - Replace n.a., n.s., -, NA with NaN
  - Strip whitespace from string columns
  - Deduplicate via a 7-step pipeline (see the dedup_by_deal() docstring):
      count entities → drop ven_* → DROP MULTI-TARGET DEALS →
      ffill all columns → drop rows with incomplete acq identity triple →
      dedup on [deal_num, tar_bvd_id_num, acq_bvd_id_num] → merge stats back
  - Numeric financial columns coerced to float; entity ID columns are
    whitelisted and never coerced (they may contain letters/hyphens)

  FINAL GRAIN for 03pre/03post/03com_fin/03legal: one row per
      (deal_num, tar_bvd_id_num, acq_bvd_id_num)
  NOT one row per deal_num — a deal with N acquirers yields N rows.

  Module 03list is the exception: it dedups on deal_num alone (keep first),
  because the listed-status file carries one target + one acquirer per deal.
  Module 03adv aggregates advisor detail rows to one row per deal_num.

⚠️ MULTI-TARGET DEALS ARE DROPPED HERE (Step 3 of dedup_by_deal):
  Any deal with ≥2 distinct targets is discarded entirely, because the
  target↔acquirer pairing is ambiguous.
  This is ASYMMETRIC with Module 01b (01b_deal_overview.py), which RETAINS
  multi-target deals and therefore has 58,963 rows vs 55,586 unique
  deal_nums. Downstream merges in 04a collapse to one row per deal_num;
  if you ever change this step, re-verify that 04a still produces unique
  deal_nums.

No financial RATIO variables are constructed here.
All ratio variables (EBITDA margin, leverage, ROA, revenue growth) are
computed downstream (Stata / 04a-08b pipeline).
Exception: tar_incorp_d_year / acq_incorp_d_year ARE derived here by parsing
tar_incorp_d / acq_incorp_d (added 2026-07-27).

Outputs (data/cleaned/):
  03_firm_financial_predeal.csv      Module 03pre
  03b_firm_financial_postdeal.csv    Module 03post   (added 2026-08-03)
  03_firm_financial_company.csv      Module 03com_fin
  03_firm_legal.csv                  Module 03legal
  03b_listed_status.csv              Module 03list
  03b_firm_advisor_count.csv         Module 03adv

Coverage diagnostics (data/cleaned/), one per module except I:
  _cov_module03_predeal.csv, _cov_module03_postdeal.csv,
  _cov_module03_company.csv, _cov_module03_legal.csv,
  _cov_moduleJ_advisor.csv
  Each lists every column with non-null count, numeric-parseable count and
  coverage %, plus a tar-vs-acq comparison that flags columns present for
  the target but entirely empty for the acquirer.

Console log: all print() output is tee'd to
  data/merged/03_clean_firm_modules_log.txt   (overwritten each run)

Author: Zhaohua Li  Date: 2026-04-12
adjusted: Qing  Date: 2026-08-03
Revised 2026-09-06: docstring corrected — six modules (was "three"), dedup
  grain is (deal_num, tar, acq) not deal_num, multi-target deals are dropped
  (asymmetric with 01b), 03list/03adv sources and outputs added, inc-year
  derivation and log redirection documented.
"""

import os
import glob
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = r"D:\MA"
RAW_FIRM  = os.path.join(BASE, "raw", "MA_firm")
CLEANED   = os.path.join(BASE, "data", "cleaned")

os.makedirs(CLEANED, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# Log redirect: all prints go to console + file (added 2026-09-06)
# No changes to any print needed; same diagnostic log directory as 02/04a/08b
# ════════════════════════════════════════════════════════════════════════════
import sys, atexit
from datetime import datetime

class _Tee(object):
    """Write output to multiple streams simultaneously (console + log file)"""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

# Same diagnostic log directory as other scripts (data/merged)
# To put logs next to output CSVs, change "merged" below to "cleaned"
LOG_DIR = os.path.join(BASE, "data", "merged")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "03_clean_firm_modules_log.txt")

_log_fh      = open(LOG_PATH, "w", encoding="utf-8")   # "w" overwrites each run; see note below
_orig_stdout = sys.stdout
sys.stdout   = _Tee(_orig_stdout, _log_fh)

def _close_log():
    try:
        _log_fh.flush(); _log_fh.close()
    except Exception:
        pass
atexit.register(_close_log)          # Ensure file is flushed even on abnormal exit

print("=" * 70)
print("03_clean_firm_modules.py — LOG")
print(f"started : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"log file: {LOG_PATH}")
print("=" * 70)

MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]


def read_and_ffill(filepath, encoding="utf-8-sig"):
    """Read a Zephyr CSV in file order and immediately forward-fill deal_num."""
    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
    df.columns = df.columns.str.strip()
    df["deal_num"] = df["deal_num"].ffill()
    return df


#def clean_missing(df):
    """Replace Zephyr placeholder strings with NaN; strip whitespace."""
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].replace(MISSING_VALS, np.nan).str.strip()
    return df

def clean_missing(df):
    """Replace Zephyr placeholder strings with NaN across all object columns."""
    # pandas 3.x compatibility: use "object" only, as pandas 3.x does not support "str"
    try:
        # Try the original author's approach (pandas 2.x)
        cols = df.select_dtypes(include=["object", "str"]).columns
    except TypeError:
        # pandas 3.x: use "object" only
        cols = df.select_dtypes(include=["object"]).columns
    
    for col in cols:
        # Safely process string columns
        df[col] = df[col].astype('object')
        df[col] = df[col].replace(MISSING_VALS, np.nan)
        # Only strip non-null values
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
            # Convert empty strings to NaN
            df.loc[mask, col] = df.loc[mask, col].replace('', np.nan)
    return df

# ════════════════════════════════════════════════════════════════════════════
# Generic coverage report (added 2026-09-06)
#   Iterate all df columns, group by entity role, flag low/empty coverage, compare tar vs acq same-name metrics
# ════════════════════════════════════════════════════════════════════════════

def _role_of(col):
    c = col.lower()
    if "tar_" in c: return "target"
    if "acq_" in c: return "acquirer"
    if "ven_" in c: return "vendor"
    return "deal/other"

def coverage_report(df, label, low=50.0, out_csv=None):
    """
    Iterate all df columns and output coverage.
      low     : flag with warning if below this percentage
      out_csv : if path given, save detail table (all columns, not just low-coverage)
    """
    n = len(df)
    print("\n" + "=" * 78)
    print(f"COVERAGE REPORT — {label}   rows={n:,}  cols={len(df.columns)}")
    print("=" * 78)

    rows = []
    for col in df.columns:
        s = df[col]
        c = col.lower()
        is_id = (col == "deal_num") or any(p in c for p in ("name", "bvd_id", "orbis_id"))
        n_raw = int(s.notna().sum())
        if is_id:
            n_num, kind = n_raw, "id"
        else:
            n_num = int(pd.to_numeric(s, errors="coerce").notna().sum())
            kind = "num" if pd.api.types.is_numeric_dtype(s) else "str"
        pct = n_raw / n * 100 if n else 0.0
        rows.append(dict(column=col, role=_role_of(col), kind=kind,
                         n_nonnull=n_raw, n_numeric=n_num, pct=round(pct, 2)))
    rep = pd.DataFrame(rows)

    # -- Summary: by entity role --
    print("\n  [Summary by entity role]")
    for role, g in rep.groupby("role"):
        sub = g[g["kind"] != "id"]
        if len(sub) == 0: continue
        n_empty = int((sub["n_nonnull"] == 0).sum())
        flag = f"   EMPTY columns: {n_empty}" if n_empty else ""
        print(f"    {role:<12s} vars {len(sub):>3d} | mean {sub['pct'].mean():>5.1f}%"
              f" | median {sub['pct'].median():>5.1f}%{flag}")

    # -- Comparison: tar vs acq same-name metrics (key: spot if acquirer side is entirely missing) --
    pairs = []
    for t in [c for c in df.columns if "tar_" in c.lower()]:
        a = t.replace("tar_", "acq_")
        if a in df.columns and a != t:
            tp = rep.loc[rep.column == t, "pct"].iloc[0]
            ap = rep.loc[rep.column == a, "pct"].iloc[0]
            pairs.append((t.replace("tar_", "*_"), tp, ap))
    if pairs:
        print("\n  [Compare] same-name metrics tar vs acq   (delta = acq - tar)")
        print(f"    {'metric':<46s}{'tar%':>7s}{'acq%':>7s}{'diff pp':>8s}")
        for name, tp, ap in sorted(pairs, key=lambda x: x[2] - x[1]):
            d = ap - tp
            flag = "  ACQ ALL EMPTY" if (ap == 0 and tp > 0) else ("  WARN" if d < -20 else "")
            print(f"    {name:<46s}{tp:>7.1f}{ap:>7.1f}{d:>+8.1f}{flag}")

    # -- Numeric conversion loss warning (string columns that fail to_numeric) --
    loss = rep[(rep["kind"] != "id") & (rep["n_numeric"] < rep["n_nonnull"] * 0.99)]
    if len(loss):
        print("\n  WARNING [numeric loss] non-numeric chars present, some become NaN after to_numeric:")
        for _, r in loss.iterrows():
            print(f"    {r['column']:<46s} non-null {r['n_nonnull']:>6,} -> numeric {r['n_numeric']:>6,}")

    # -- Detail: low coverage / all empty --
    detail = rep[(rep["pct"] < low) | (rep["n_nonnull"] == 0)].sort_values("pct")
    print(f"\n  [Detail] coverage below {low}% or all empty ({len(detail)} / {len(rep)} columns)")
    if len(detail) == 0:
        print(f"    OK: no columns below {low}% coverage")
    else:
        for _, r in detail.iterrows():
            tag = "ALL EMPTY" if r["n_nonnull"] == 0 else "LOW COVERAGE"
            print(f"    {r['column']:<48s} {r['n_nonnull']:>7,}/{n:<7,} ({r['pct']:>5.1f}%) {tag}")

    if out_csv:
        rep.sort_values("pct").to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n  Detail (all columns) saved -> {out_csv}")
    print("=" * 78 + "\n")
    return rep


def audit_keep_list(df, keep_list, label):
    """Audit KEEP list: catch ghost columns (listed but absent) and dropped columns (present but unlisted)"""
    ghost   = [c for c in keep_list if c not in df.columns]
    dropped = [c for c in df.columns if c not in keep_list]
    print(f"\n[KEEP list audit] {label}")
    print(f"  In list but absent from data (typo?): {ghost if ghost else 'none'}")
    print(f"  In data but dropped by list ({len(dropped)} cols): {dropped if dropped else 'none'}")
    return ghost, dropped

def stack_batches(files, label):
    """Read, ffill, and stack all batch files."""
    batches = []
    for fp in files:
        df_b = read_and_ffill(fp)
        n_raw = len(df_b)
        df_b = clean_missing(df_b)
        batches.append(df_b)
        print(f"  {os.path.basename(fp)}: {n_raw} rows read")
    df = pd.concat(batches, ignore_index=True)
    print(f"After stacking {len(files)} batches ({label}): {len(df):,} rows")
    return df


def dedup_by_deal(df, label):
    """
    Deduplicate Zephyr firm-module data to (deal_num, tar_bvd_id_num, acq_bvd_id_num) grain.

    ============================================================================
    PROCESSING PIPELINE (Qing's logic, revised 2026-08-15):
    ----------------------------------------------------------------------------
    Step 1. COUNT — per-deal entity stats on raw data (n_target, n_acquirer,
            n_vendor, raw row count), BEFORE any filtering.
    Step 2. DROP all vendor columns (ven_* discarded entirely).
    Step 3. DROP multi-target deals (n_target_entity > 1).
            Rationale: M:N target-acquirer pairing is ambiguous; only single-
            target deals retained, leaving clean 1-target : N-acquirer structure.
            *** NO DEDUPLICATION IS PERFORMED BEFORE THIS POINT ***
    Step 4. FORWARD-FILL (ffill) ALL columns WITHIN each deal_num group.
            Zephyr cascade rows split acquirer identity across multiple rows:
            one row may carry acq_bvd_id_num but empty acq_name, another row
            may carry acq_name but empty acq_orbis_id_num. ffill propagates
            the first non-empty value of EVERY column downward, so a single
            row can assemble the complete acquirer identity triple
            (acq_name + acq_bvd_id_num + acq_orbis_id_num) from fragments
            scattered across preceding rows.
            Both string and numeric columns are ffilled uniformly.
    Step 5. DROP rows where ANY of the three acquirer identifier columns is
            missing: acq_name OR acq_bvd_id_num OR acq_orbis_id_num.
            All three must be non-null to uniquely identify an acquirer entity
            ("three stamps required"). After ffill, most rows will have all
            three assembled; rows still missing any one are genuinely invalid
            acquirer observations and are discarded.
    Step 6. DEDUPLICATE on triple key [deal_num, tar_bvd_id_num, acq_bvd_id_num],
            keep first. Collapses residual duplicate (tar, acq) pairs per deal.
    Step 7. MERGE entity_stats back onto the deduplicated frame.
    ----------------------------------------------------------------------------
    FINAL GRAIN: one row per (deal_num, tar_bvd_id_num, acq_bvd_id_num).
    Every retained row has complete acq_name + acq_bvd_id_num + acq_orbis_id_num.
    ============================================================================
    """
    n0 = len(df)
    df_raw = df[df["deal_num"].notna()].copy()

    # ── Step 1: per-deal entity census (raw, pre-filter) ──
    entity_stats = df_raw.groupby("deal_num").agg(
        raw_entity_total_rows = ("deal_num", "count"),
        n_target_entity       = ("tar_bvd_id_num", lambda x: x.dropna().nunique()),
        n_acquirer_entity     = ("acq_bvd_id_num", lambda x: x.dropna().nunique()),
        n_vendor_entity       = ("ven_bvd_id_num", lambda x: x.dropna().nunique())
    ).reset_index()

    global_unique_tar = df_raw["tar_bvd_id_num"].dropna().nunique()
    global_unique_acq = df_raw["acq_bvd_id_num"].dropna().nunique()
    print(f"\n[{label}] GLOBAL UNIQUE ENTITY CENSUS (raw, pre-filter):")
    print(f"  Total distinct target firms  (tar_bvd_id_num): {global_unique_tar:,}")
    print(f"  Total distinct acquirer firms (acq_bvd_id_num): {global_unique_acq:,}")
    print(f"  Total distinct deals (raw)                   : {df_raw['deal_num'].nunique():,}")

    # ── Step 2: DROP all vendor columns ──
    ven_cols = [c for c in df_raw.columns if c.startswith("ven_")]
    df_work = df_raw.drop(columns=ven_cols).copy()
    print(f"\n[{label}] Step 2 — Dropped {len(ven_cols)} vendor columns")

    # ── Step 3: DROP multi-target deals (n_target_entity > 1) ──
    multi_target_deals = set(entity_stats[entity_stats["n_target_entity"] > 1]["deal_num"])
    n_multi_target_deals = len(multi_target_deals)
    n_rows_multi_target = df_work["deal_num"].isin(multi_target_deals).sum()

    df_work = df_work[~df_work["deal_num"].isin(multi_target_deals)].copy()
    print(f"\n[{label}] Step 3 — Multi-target filter:")
    print(f"  Deals with ≥2 targets dropped : {n_multi_target_deals:,}")
    print(f"  Rows removed                  : {n_rows_multi_target:,}")
    print(f"  Remaining deals (single-target): {df_work['deal_num'].nunique():,}")
    print(f"  Remaining rows                : {len(df_work):,}")

    # ── Step 4: FORWARD-FILL all columns within deal_num group ──
    # CRITICAL: ffill BEFORE dropping empty-acq rows. Zephyr splits acquirer
    # identity (name / bvd_id / orbis_id) across cascade rows; ffill assembles
    # the full triple onto each row before we judge validity.
    df_work["deal_num"] = pd.to_numeric(df_work["deal_num"], errors="coerce").astype("Int64")
    fill_cols = [c for c in df_work.columns if c != "deal_num"]
    for col in fill_cols:
        df_work[col] = df_work.groupby("deal_num")[col].ffill()
    print(f"\n[{label}] Step 4 — Forward-fill (ffill) all {len(fill_cols)} columns "
          f"within deal_num groups: done")

    # Diagnostics: how many rows have complete acq triple after ffill?
    acq_triple_cols = ["acq_name", "acq_bvd_id_num", "acq_orbis_id_num"]
    acq_triple_present = [c for c in acq_triple_cols if c in df_work.columns]
    complete_acq_mask = df_work[acq_triple_present].notna().all(axis=1)
    print(f"  Rows with complete acq triple (name+bvd+orbis) after ffill: "
          f"{complete_acq_mask.sum():,} / {len(df_work):,} "
          f"({complete_acq_mask.mean()*100:.1f}%)")

    # ── Step 5: DROP rows where ANY acquirer identifier is missing ──
    # All three stamps required to uniquely identify an acquirer entity.
    n_before_drop = len(df_work)
    df_work = df_work[complete_acq_mask].copy()
    print(f"\n[{label}] Step 5 — Drop rows with incomplete acq identity triple "
          f"(any of {acq_triple_present} missing):")
    print(f"  Rows removed (incomplete acquirer) : {n_before_drop - len(df_work):,}")
    print(f"  Remaining rows                     : {len(df_work):,}")

    # ── Step 6: DEDUPLICATE on triple key ──
    dedup_key = ["deal_num", "tar_bvd_id_num", "acq_bvd_id_num"]
    # After ffill + Step 5, tar_bvd_id_num should also be non-null on every row;
    # guard with dropna in case a deal's first row itself lacked tar ID.
    df_valid = df_work.dropna(subset=dedup_key)
    n_dropped_tar = len(df_work) - len(df_valid)
    if n_dropped_tar > 0:
        print(f"  Additional rows dropped (missing tar_bvd_id_num): {n_dropped_tar:,}")

    n_before_dedup = len(df_valid)
    df_dedup = df_valid.drop_duplicates(subset=dedup_key, keep="first")
    print(f"\n[{label}] Step 6 — Dedup on key {dedup_key}:")
    print(f"  Rows before dedup : {n_before_dedup:,}")
    print(f"  Rows after dedup  : {len(df_dedup):,}")
    print(f"  Duplicate rows removed: {n_before_dedup - len(df_dedup):,}")

    # ── Step 7: merge entity stats back ──
    df_dedup = df_dedup.merge(entity_stats, on="deal_num", how="left")

    # ── Final summary ──
    print(f"\n[{label}] FINAL SUMMARY (single-target, complete-acq-triple retained):")
    print(f"  Unique deals          : {df_dedup['deal_num'].nunique():,}")
    print(f"  Unique (deal,tar,acq) : {len(df_dedup):,}")
    print(f"  Avg acquirers per deal: {df_dedup.groupby('deal_num')['acq_bvd_id_num'].nunique().mean():.2f}")
    print(f"  Deals with ≥2 acquirers: {(df_dedup['n_acquirer_entity'] >= 2).sum():,}")
    # Verify no missing acq fields in output
    for c in acq_triple_present:
        n_miss = df_dedup[c].isna().sum()
        print(f"  {c:<20s} missing in output: {n_miss:,}")

    if len(df_dedup) < n0:
        print(f"  Total rows: raw {n0:,} → final {len(df_dedup):,} "
              f"({(1 - len(df_dedup)/n0)*100:.1f}% removed)")

    return df_dedup

# F1 FIXER R1: verify first firm row per deal matches first deal_master row
# Wrapped in try/except — deal_master may not exist on first pipeline run (chicken-and-egg)
def verify_f1_firm_ordering(df, label):
    """
    Cross-check that the first firm row per deal_num matches the first deal_master row
    for the same deal (i.e. the dedup-kept row is the primary target, not a secondary).
    Uses tar_bvd_id_num for comparison; falls back to tar_name.
    Only runs if deal_master.csv already exists.

    NOTE 2026-08-15: After multi-target deals are dropped in dedup_by_deal,
    all output rows are single-target deals. This check remains meaningful for
    verifying that the retained target BvD ID matches deal_master's first row.
    """
    DEAL_MASTER_PATH = os.path.join(
        #r"D:\MA",BASE,
        "data", "merged", "02_deal_master.csv"
    )
    try:
        df_dm = pd.read_csv(DEAL_MASTER_PATH,
                            usecols=["deal_num", "tar_bvd_id_num", "tar_name"],
                            encoding="utf-8-sig", low_memory=False)
        df_dm["deal_num"] = pd.to_numeric(df_dm["deal_num"], errors="coerce").astype("Int64")
    except FileNotFoundError:
        print(f"  SKIP F1 verification ({label}): deal_master.csv not yet created")
        return
    except Exception as e:
        print(f"  SKIP F1 verification ({label}): could not load deal_master — {e}")
        return

    # First deal_master row per deal (deal_master may still have multi-target rows)
    dm_first = (df_dm.drop_duplicates(subset="deal_num", keep="first")
                [["deal_num", "tar_bvd_id_num", "tar_name"]])

    # Merge with our single-target output
    merged = df[["deal_num", "tar_bvd_id_num", "tar_name"]].merge(
        dm_first, on="deal_num", suffixes=("_firm", "_dm"))
    Y = len(merged)
    if Y == 0:
        print(f"  F1 ({label}): no overlapping deals found to compare")
        return

    use_bvd = merged["tar_bvd_id_num_firm"].notna() & merged["tar_bvd_id_num_dm"].notna()
    match_bvd = use_bvd & (merged["tar_bvd_id_num_firm"] == merged["tar_bvd_id_num_dm"])
    use_name = ~use_bvd & merged["tar_name_firm"].notna() & merged["tar_name_dm"].notna()
    match_name = use_name & (merged["tar_name_firm"].str.strip() == merged["tar_name_dm"].str.strip())
    X = (match_bvd | match_name).sum()

    print(f"\nF1 VERIFICATION ({label}): single-target firm rows vs deal_master first rows")
    print(f"  Deals compared     : {Y:,}")
    print(f"  tar_bvd_id matches : {X:,}/{Y:,} ({X/Y*100:.1f}%)")
    if Y - X > 0:
        print(f"  WARNING: {Y - X} deals where target ID does NOT match deal_master")
        mismatch_deals = merged[~(match_bvd | match_name)]["deal_num"].head(5).tolist()
        print(f"  Example mismatch deal_nums: {mismatch_deals}")
    else:
        print(f"  All target IDs match deal_master first row.")
        

# ════════════════════════════════════════════════════════════════════════════
# Module 03pre — Pre-deal Snapshot
#   Source : raw/MA_firm/financial/acquisition_financial_1-4  (75 cols each)
#   Output : data/cleaned/firm_financial_predeal.csv
#   Unit   : deal_num (one row per deal, first target row)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODULE 03pre — Pre-deal Snapshot (acquisition_financial_*)")
print("=" * 60)

fin_dir   = os.path.join(RAW_FIRM, "financial")
fin_files = sorted(glob.glob(os.path.join(fin_dir, "acquisition_financial_*_cleaned.csv")))
print(f"Found {len(fin_files)} batch files")

df_fin = stack_batches(fin_files, "Module F raw")

# Drop row-index column if present
if "unnamed__0" in df_fin.columns:
    df_fin = df_fin.drop(columns=["unnamed__0"])

df_fin = dedup_by_deal(df_fin, "Module F")
verify_f1_firm_ordering(df_fin, "Module F")

# ── Select columns for analysis (empirical_design.md §4.7) ─────────────────
FIN_KEEP = [
    "deal_num",
    # Entity identifiers
    "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    "ven_name", "ven_bvd_id_num",
    # New entity counts (generated by dedup function)
    "n_target_entity",
    "n_acquirer_entity",
    "n_vendor_entity",
    "raw_entity_total_rows",
    # Target pre-deal financials (control variables)
    "pre_deal_tar_rev_rev_last_avail_yr",           # Revenue (EBITDA margin denom)
    "pre_deal_tar_ebitda_last_avail_yr",             # EBITDA (margin num)
    "pre_deal_tar_ebit_last_avail_yr",               # EBIT
    "pre_deal_tar_pbt_last_avail_yr",                # Pre-tax profit
    "pre_deal_tar_pat_last_avail_yr",                # Profit after tax (ROA num)
    "pre_deal_tar_np_last_avail_yr",                 # Net profit
    "pre_deal_tar_ta_last_avail_yr",                 # Total assets (size + leverage denom)
    "pre_deal_tar_na_last_avail_yr",                 # Net assets
    "pre_deal_tar_eq_last_avail_yr",                 # Equity (leverage)
    "pre_deal_tar_current_liabilities_last_avail_yr",
    "pre_deal_tar_cap",
    # Acquirer pre-deal financials (relative size, acquirer ROA)
    "pre_deal_acq_rev_rev_last_avail_yr",
    "pre_deal_acq_ebitda_last_avail_yr",
    "pre_deal_acq_ebit_last_avail_yr",       # New acq EBIT
    "pre_deal_acq_pbt_last_avail_yr",
    "pre_deal_acq_pat_last_avail_yr",
    "pre_deal_acq_np_last_avail_yr",         # New acq Net profit
    "pre_deal_acq_ta_last_avail_yr",         # Acquirer total assets
    "pre_deal_acq_na_last_avail_yr",         # New acq Net assets
    "pre_deal_acq_eq_last_avail_yr",
    "pre_deal_acq_current_liabilities_last_avail_yr", # New acq current liabilities
    "pre_deal_acq_cap",                      # New acq cap
    # Vendor pre-deal financials (for completeness)
    "pre_deal_ven_rev_rev_last_avail_yr",
    "pre_deal_ven_ta_last_avail_yr",
]
audit_keep_list(df_fin, FIN_KEEP, "Module F")  
FIN_KEEP_PRESENT = [c for c in FIN_KEEP if c in df_fin.columns]
df_fin = df_fin[FIN_KEEP_PRESENT].copy()

# Convert financial columns to numeric
FIN_ID_COLS = {"deal_num", "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
               "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
               "ven_name", "ven_bvd_id_num"}
for col in FIN_KEEP_PRESENT:
    if col not in FIN_ID_COLS:
        df_fin[col] = pd.to_numeric(df_fin[col], errors="coerce")

# Diagnostics
print(f"\nKey variable coverage:")
check_pre_vars = [
    "pre_deal_tar_ta_last_avail_yr",
    "pre_deal_tar_rev_rev_last_avail_yr",
    "pre_deal_tar_ebitda_last_avail_yr",
    "pre_deal_tar_eq_last_avail_yr",
    "pre_deal_tar_pat_last_avail_yr",
    # Full set of acquirer key metrics
    "pre_deal_acq_ta_last_avail_yr",
    "pre_deal_acq_rev_rev_last_avail_yr",
    "pre_deal_acq_ebitda_last_avail_yr",
    "pre_deal_acq_ebit_last_avail_yr",
    "pre_deal_acq_eq_last_avail_yr",
    "pre_deal_acq_pat_last_avail_yr"
]
#for v in check_pre_vars:
    #if v in df_fin.columns:
        #n = df_fin[v].notna().sum()
        #print(f"  {v:<45s}: {n:,} ({n/len(df_fin)*100:.1f}%)")
coverage_report(df_fin, "Module 03pre — pre-deal",
                out_csv=os.path.join(CLEANED, "_cov_module03_predeal.csv"))

out_fin = os.path.join(CLEANED, "03_firm_financial_predeal.csv")
df_fin.to_csv(out_fin, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {out_fin}  ({os.path.getsize(out_fin)//1024:,} KB)")

# ════════════════════════════════════════════════════════════════════════════
# Module 03post — Post-deal Snapshot
#   Source : raw/MA_firm/financial/acquisition_financial_1-4  (same raw files)
#   Output : data/cleaned/03b_firm_financial_postdeal.csv
#   Unit   : deal_num (one row per deal, first target row)
#   Note   : Re-reads the same raw files (Module F already saved to disk)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE 03post — Post-deal Snapshot (acquisition_financial_*)")
print("=" * 60)

# Reuse df_fin from Module F (already stacked + ffilled + deduped)
# If you want to run Fb independently, uncomment the 3 lines below:
fin_dir   = os.path.join(RAW_FIRM, "financial")
fin_files = sorted(glob.glob(os.path.join(fin_dir, "acquisition_financial_*_cleaned.csv")))
df_fin = stack_batches(fin_files, "Module Fb raw")
if "unnamed__0" in df_fin.columns:
    df_fin = df_fin.drop(columns=["unnamed__0"])
    
df_fin = dedup_by_deal(df_fin, "Module Fb")

print(f"Module Fb stacked: {len(df_fin):,} rows")

# ── Select post-deal columns only ──────────────────────────────────────────
FIN_POST_KEEP = [
    "deal_num",
    # Entity identifiers
    "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    "ven_name", "ven_bvd_id_num", "ven_orbis_id_num",
    # Target post-deal financials (1st available year after deal)
    "post_deal_tar_rev_rev_1st_avail_yr",
    "post_deal_tar_ebitda_1st_avail_yr",
    "post_deal_tar_ebit_1st_avail_yr",
    "post_deal_tar_pbt_1st_avail_yr",
    "post_deal_tar_pat_1st_avail_yr",
    "post_deal_tar_np_1st_avail_yr",
    "post_deal_tar_ta_1st_avail_yr",
    "post_deal_tar_na_1st_avail_yr",
    "post_deal_tar_current_liabilities_1st_avail_yr",
    "post_deal_tar_shareholder_funds_1st_avail_yr",
    "post_deal_tar_cap",
    # Acquirer post-deal financials
    "post_deal_acq_rev_rev_1st_avail_yr",
    "post_deal_acq_ebitda_1st_avail_yr",
    "post_deal_acq_ebit_1st_avail_yr",
    "post_deal_acq_pbt_1st_avail_yr",
    "post_deal_acq_pat_1st_avail_yr",
    "post_deal_acq_np_1st_avail_yr",
    "post_deal_acq_ta_1st_avail_yr",
    "post_deal_acq_na_1st_avail_yr",
    "post_deal_acq_shareholder_funds_1st_avail_yr",
    "post_deal_acq_cap",
    # Vendor post-deal financials
    "post_deal_ven_rev_rev_1st_avail_yr",
    "post_deal_ven_ebitda_1st_avail_yr",
    "post_deal_ven_ebit_1st_avail_yr",
    "post_deal_ven_pbt_1st_avail_yr",
    "post_deal_ven_pat_1st_avail_yr",
    "post_deal_ven_np_1st_avail_yr",
    "post_deal_ven_ta_1st_avail_yr",
    "post_deal_ven_na_1st_avail_yr",
    "post_deal_ven_shareholder_funds_1st_avail_yr",
    "post_deal_ven_cap",
]
audit_keep_list(df_fin, FIN_POST_KEEP, "Module Fb") 
FIN_POST_KEEP_PRESENT = [c for c in FIN_POST_KEEP if c in df_fin.columns]
df_fin_post = df_fin[FIN_POST_KEEP_PRESENT].copy()

# Convert financial columns to numeric
FIN_POST_ID_COLS = {"deal_num", "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
                    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
                    "ven_name", "ven_bvd_id_num", "ven_orbis_id_num"}
for col in FIN_POST_KEEP_PRESENT:
    if col not in FIN_POST_ID_COLS:
        df_fin_post[col] = pd.to_numeric(df_fin_post[col], errors="coerce")

# Diagnostics
print(f"\nKey variable coverage (post-deal):")
check_post_vars = [
    "post_deal_tar_rev_rev_1st_avail_yr",
    "post_deal_tar_ebitda_1st_avail_yr",
    "post_deal_tar_ta_1st_avail_yr",
    "post_deal_tar_shareholder_funds_1st_avail_yr",
    "post_deal_acq_rev_rev_1st_avail_yr",
    "post_deal_acq_ta_1st_avail_yr",
    "post_deal_acq_ebitda_1st_avail_yr",
    "post_deal_acq_shareholder_funds_1st_avail_yr"
]

#for v in check_post_vars:
    #if v in df_fin_post.columns:
        #n = df_fin_post[v].notna().sum()
        #total = len(df_fin_post)
        #print(f"  {v:<45s}: {n:,} ({n/total*100:.1f}%)")       
# Replaced original check_post_vars loop:
coverage_report(df_fin_post, "Module 03post — post-deal",
                out_csv=os.path.join(CLEANED, "_cov_module03_postdeal.csv"))
        
out_fin_post = os.path.join(CLEANED, "03b_firm_financial_postdeal.csv")
df_fin_post.to_csv(out_fin_post, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {out_fin_post}  ({os.path.getsize(out_fin_post)//1024:,} KB)")
if os.environ.get("DEBUG_03"):
    print("df_fin all original column names:", df_fin.columns.tolist())
# ════════════════════════════════════════════════════════════════════════════
# Module 03com_fin — Company Financial Time Series
#   Source : raw/MA_firm/financial/acquisition_company_financial_1-7
#            (file _8 is exact duplicate of _7 — skipped)
#   Output : data/cleaned/firm_financial_company.csv
#   Unit   : deal_num (one row per deal)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE 03com_fin — Company Financials (acquisition_company_financial_1-7)")
print("=" * 60)

# Files 1-7 only; _8 is confirmed duplicate of _7
compfin_files = sorted(glob.glob(
    os.path.join(fin_dir, "acquisition_company_financial_*_cleaned.csv")
))
# Exclude file 8
compfin_files = [f for f in compfin_files
                 if not os.path.basename(f).startswith("acquisition_company_financial_8")]
print(f"Using {len(compfin_files)} batch files (file _8 excluded — exact duplicate of _7)")

df_cf = stack_batches(compfin_files, "Module 03com_fin raw")

# Drop unnamed row-index column; drop structural __1 duplicates
# (yr__1, yr__2 are VALID lagged-year columns — keep them)
drop_cols = []
if "unnamed__0" in df_cf.columns:
    drop_cols.append("unnamed__0")
for col in list(df_cf.columns):
    if col.endswith("__1") and "yr__" not in col:
        base = col[:-3]
        if base in df_cf.columns:
            drop_cols.append(col)
if drop_cols:
    df_cf = df_cf.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} unnamed/structural duplicate columns")

# M2 FIXER R1: verify __1 cols that contain "yr__" (currently kept) vs their base column
print(f"\nM2 VERIFICATION — checking yr__ __1 columns kept as lagged-year cols:")
for _col in list(df_cf.columns):
    if _col.endswith("__1") and "yr__" in _col:
        _base = _col[:-3]
        if _base in df_cf.columns:
            _pct_identical = (df_cf[_col] == df_cf[_base]).mean()
            _n_both_notna = (df_cf[_col].notna() & df_cf[_base].notna()).sum()
            print(f"  M2 CHECK: {_col} vs {_base}: "
                  f"{_pct_identical*100:.1f}% identical ({_n_both_notna:,} both non-null)")

df_cf = dedup_by_deal(df_cf, "Module 03com_fin")
verify_f1_firm_ordering(df_cf, "Module 03com_fin")

# Select target financial columns only (plan §2.2)
CF_KEEP = [
    "deal_num",
    "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    # Revenue (current + prior year for Revenue Growth)
    "tar_rev_rev_last_avail_yr",
    "tar_rev_rev_yr__1",
    "tar_rev_rev_yr__2",
    # EBITDA and EBIT
    "tar_ebitda_last_avail_yr",
    "tar_ebit_last_avail_yr",
    # Total assets and equity
    "tar_ta_last_avail_yr",
    "tar_ta_yr__1",
    "tar_na_last_avail_yr",
    "tar_eq_last_avail_yr",
    # Employment (headcount size proxy)
    "tar_emp_last_avail_yr",
    # Enterprise value (reference)
    "tar_ev_last_avail_yr",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    # Revenue (current + prior year for Revenue Growth)
    "acq_rev_rev_last_avail_yr",
    "acq_rev_rev_yr__1",
    "acq_rev_rev_yr__2",
    # EBITDA and EBIT
    "acq_ebitda_last_avail_yr",
    "acq_ebit_last_avail_yr",
    # Total assets and equity
    "acq_ta_last_avail_yr",
    "acq_ta_yr__1",
    "acq_na_last_avail_yr",
    "acq_eq_last_avail_yr",
    # Employment (headcount size proxy)
    "acq_emp_last_avail_yr",
    # Enterprise value (reference)
    "acq_ev_last_avail_yr"
]
audit_keep_list(df_cf, CF_KEEP, "Module 03com_fin")   
CF_KEEP_PRESENT = [c for c in CF_KEEP if c in df_cf.columns]
df_cf = df_cf[CF_KEEP_PRESENT].copy()
coverage_report(df_cf, "Module 03com_fin — company",
                out_csv=os.path.join(CLEANED, "_cov_module03_company.csv"))
# ==============================================
# CAUTION: Batch numeric coercion for financial variables ONLY
# Entity identifier columns (tar/acq name, BvD ID, Orbis ID) MUST be excluded here.
# BvD/Orbis IDs may contain letters/hyphens; firm names are plain text.
# If these ID columns are included in pd.to_numeric, non-numeric values will be coerced to NaN,
# which destroys the acquirer/target entity identifiers populated & validated in dedup_by_deal().
# CF_ID_COLS acts as the whitelist: columns in this set SKIP numeric conversion and stay as raw strings.
# Bug history: Module G originally missed acq_name/acq_bvd_id_num/acq_orbis_id_num in CF_ID_COLS,
# causing valid acquirer identifiers to turn into NaN after deduplication.
# ==============================================
CF_ID_COLS = {"deal_num", "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
              "acq_name", "acq_bvd_id_num", "acq_orbis_id_num"}
for col in CF_KEEP_PRESENT:
    if col not in CF_ID_COLS:
        df_cf[col] = pd.to_numeric(df_cf[col], errors="coerce")

print(f"\nKey variable coverage:")
for v in ["tar_rev_rev_last_avail_yr", "tar_rev_rev_yr__1",
          "tar_ta_last_avail_yr", "tar_emp_last_avail_yr"]:
    if v in df_cf.columns:
        n = df_cf[v].notna().sum()
        print(f"  {v:<45s}: {n:,} ({n/len(df_cf)*100:.1f}%)")

out_cf = os.path.join(CLEANED, "03_firm_financial_company.csv")
df_cf.to_csv(out_cf, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {out_cf}  ({os.path.getsize(out_cf)//1024:,} KB)")


# ════════════════════════════════════════════════════════════════════════════
# Module 03legal — Legal Status
#   Source : raw/MA_firm/legal/acquisition_legal_1-2  (43 cols each)
#   Output : data/cleaned/firm_legal.csv
#   Unit   : deal_num (one row per deal)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE 03legal — Legal Status (acquisition_legal_*)")
print("=" * 60)

leg_dir   = os.path.join(RAW_FIRM, "legal")
leg_files = sorted(glob.glob(os.path.join(leg_dir, "acquisition_legal_*_cleaned.csv")))
print(f"Found {len(leg_files)} batch files")

df_leg = stack_batches(leg_files, "Module H raw")
df_leg = dedup_by_deal(df_leg, "Module H")
verify_f1_firm_ordering(df_leg, "Module H")

# Select all tar/acq/ven legal columns (plan §2.3)
LEG_KEEP = [
    "deal_num",
    # Entity identifiers
    "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    "ven_name", "ven_bvd_id_num",
    # Target legal
    "tar_status", "tar_ent_type",
    "tar_incorp_d", "tar_incorp_d_year",
    "tar_bvd_indep", "tar_acct_types", "tar_filing_type",
    "tar_last_acct_d", "tar_last_acct_d_year", "tar_acct_pub",
    # Acquirer legal
    "acq_status", "acq_ent_type", "acq_legal_form",
    "acq_incorp_d", "acq_incorp_d_year",
    "acq_bvd_indep", "acq_acct_types", "acq_last_acct_d_year",
    # Vendor legal
    #"ven_status", "ven_ent_type", "ven_legal_form",
    #"ven_incorp_d", "ven_incorp_d_year",
    #"ven_bvd_indep",
]
audit_keep_list(df_leg, LEG_KEEP, "Module 03legal")  
LEG_KEEP_PRESENT = [c for c in LEG_KEEP if c in df_leg.columns]
df_leg = df_leg[LEG_KEEP_PRESENT].copy()


# Fix: generic date parsing to extract incorporation year, handles all mixed formats
def extract_inc_year(s):
    if pd.isna(s) or str(s).strip() == "":
        return pd.NA
    s_clean = str(s).strip()
    if s_clean.isdigit() and len(s_clean) == 4:
        return int(s_clean)
    try:
        dt = pd.to_datetime(s_clean, errors="raise")
        return dt.year
    except:
        return pd.NA

# Cover incorporation year for all three entity types, fill missing values
df_leg["tar_incorp_d_year"] = df_leg["tar_incorp_d"].apply(extract_inc_year)
df_leg["acq_incorp_d_year"] = df_leg["acq_incorp_d"].apply(extract_inc_year)
#df_leg["ven_incorp_d_year"] = df_leg["ven_incorp_d"].apply(extract_inc_year)

#================== end new fix code ===========================================


# m5 FIXER R1: cast year columns to Int64 (not float) to prevent float residue
for yr_col in [c for c in LEG_KEEP_PRESENT if c.endswith("_year")]:
    df_leg[yr_col] = pd.to_numeric(df_leg[yr_col], errors="coerce").astype("Int64")

print(f"\nKey variable coverage:")
#for v in ["tar_status", "tar_incorp_d_year", "tar_bvd_indep"]:
    #if v in df_leg.columns:
       #n = df_leg[v].notna().sum()
        #   print(f"  {v:<45s}: {n:,} ({n/len(df_leg)*100:.1f}%)")


# Replaced original 3-line loop (note: after extract_inc_year, before out_leg):
coverage_report(df_leg, "Module 03legal — legal",
                out_csv=os.path.join(CLEANED, "_cov_module03_legal.csv"))

out_leg = os.path.join(CLEANED, "03_firm_legal.csv")
df_leg.to_csv(out_leg, index=False, encoding="utf-8-sig")
print(f"\nSaved -> {out_leg}  ({os.path.getsize(out_leg)//1024:,} KB)")



# ════════════════════════════════════════════════════════════════════════════
# Module 03list — Listed-firm Status
#   Source : raw/MA_firm/listedstatus/list_1.csv + list_2.csv  (2 batches)
#   Output : data/cleaned/03b_listed_status.csv
#   Unit   : deal_num (one row per deal)
#
#   Coding rule: listed = 1 iff BOTH exchange AND ticker are non-missing.
#   Two dummies constructed: tar_listed (target) and acq_listed (acquiror).
#
#   Column naming: raw files use "Deal Number" (capital, space) and
#   "Target/Acquiror stock exchange(s) listed" etc. — normalised below.
#   Cascade rows (additional vendor entities) have blank Deal Number and
#   blank exchange/ticker columns; ffill restores deal_num, dedup keeps
#   the first row per deal (which carries the main target+acquiror data).
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE I — Listed-firm Status (listedstatus/list_1-2)")
print("=" * 60)

ls_dir   = os.path.join(RAW_FIRM, "listedstatus")
ls_files = sorted(glob.glob(os.path.join(ls_dir, "list_*.csv")))
print(f"Found {len(ls_files)} listed-status batch files")

# Column rename map: raw column names (after stripping whitespace/newlines) → snake_case
LS_COL_RENAME = {
    "Deal Number":                          "deal_num",
    "Target stock exchange(s) listed":      "tar_exchange",
    "Target ticker symbol":                 "tar_ticker",
    "Target name":                          "tar_name",
    "Target BvD ID number":                 "tar_bvd_id_num",
    "Target Orbis ID number":               "tar_orbis_id_num",
    "Acquiror stock exchange(s) listed":    "acq_exchange",
    "Acquiror ticker symbol":               "acq_ticker",
    "Acquiror name":                        "acq_name",
    "Acquiror BvD ID number":               "acq_bvd_id_num",
    "Acquiror Orbis ID number":             "acq_orbis_id_num",
}

batches_ls = []
for fp in ls_files:
    batch_name = os.path.basename(fp)
    df_b = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
    # Normalise column names: strip whitespace and embedded newlines
    df_b.columns = [c.strip().replace("\n", " ").replace("\r", "") for c in df_b.columns]
    df_b = df_b.rename(columns=LS_COL_RENAME)
    n_raw = len(df_b)
    # ffill deal_num (cascade rows have blank Deal Number)
    df_b["deal_num"] = df_b["deal_num"].ffill()
    df_b = clean_missing(df_b)
    batches_ls.append(df_b)
    print(f"  {batch_name}: {n_raw} rows read, deal_num ffilled")

df_ls = pd.concat(batches_ls, ignore_index=True)
print(f"\nAfter stacking: {len(df_ls):,} rows")

# Drop rows with missing deal_num
n_before = len(df_ls)
df_ls = df_ls[df_ls["deal_num"].notna()].copy()
if n_before > len(df_ls):
    print(f"Dropped {n_before - len(df_ls):,} rows with NaN deal_num after ffill")

# Convert deal_num to Int64
df_ls["deal_num"] = pd.to_numeric(df_ls["deal_num"], errors="coerce").astype("Int64")
df_ls = df_ls[df_ls["deal_num"].notna()].copy()

# Dedup by deal_num: keep first row (main target+acquiror data is on the first Zephyr row;
# cascade rows add extra vendor entities with blank exchange/ticker columns)
n_before_dedup = len(df_ls)
df_ls = df_ls.drop_duplicates(subset=["deal_num"], keep="first")
print(f"After dedup by deal_num: {len(df_ls):,} rows "
      f"(dropped {n_before_dedup - len(df_ls):,} cascade/duplicate rows)")

# ── Construct listed dummies ─────────────────────────────────────────────────
# listed = 1 iff BOTH exchange AND ticker are non-missing (Zephyr provides both
# when the firm is exchange-listed; missing either means unlisted or unknown).
for role, exch_col, tick_col, dummy_col in [
    ("target",   "tar_exchange", "tar_ticker", "tar_listed"),
    ("acquiror", "acq_exchange", "acq_ticker", "acq_listed"),
]:
    has_exch = df_ls[exch_col].notna() if exch_col in df_ls.columns else pd.Series(False, index=df_ls.index)
    has_tick = df_ls[tick_col].notna() if tick_col in df_ls.columns else pd.Series(False, index=df_ls.index)
    df_ls[dummy_col] = (has_exch & has_tick).astype(int)
    n_listed = df_ls[dummy_col].sum()
    print(f"  {dummy_col}: {n_listed:,} listed ({n_listed/len(df_ls)*100:.1f}%)")

# ── Select output columns ────────────────────────────────────────────────────
LS_KEEP = ["deal_num", "tar_listed", "acq_listed",
           "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
           "acq_name", "acq_bvd_id_num", "acq_orbis_id_num"]   # IDs kept for QC / diagnostics
LS_KEEP_PRESENT = [c for c in LS_KEEP if c in df_ls.columns]
df_ls = df_ls[LS_KEEP_PRESENT].copy()

print(f"\nModule 03list output: {len(df_ls):,} rows | {df_ls['deal_num'].nunique():,} unique deals")

out_ls = os.path.join(CLEANED, "03b_listed_status.csv")
df_ls.to_csv(out_ls, index=False, encoding="utf-8-sig")
print(f"Saved -> {out_ls}  ({os.path.getsize(out_ls)//1024:,} KB)")

# ════════════════════════════════════════════════════
# Module 03adv — Advisor COUNT version (aligned with info-source module)
#   Source : raw/MA_firm/advisor/  (advisor1~5_cleaned.csv 5 batches)
#   Output : data/cleaned/03b_firm_advisor_count.csv
#   Unit   : deal_num, columns = count how many times each advisor-type appears
# ---------------- Module 03adv inline notes ----------------
# Module 03adv: M&A advisor count cleaning module
# Input: raw/MA_firm/advisor 5 raw advisor detail CSVs
# Output: data/cleaned/03b_firm_advisor_count.csv
# Core logic:
# 1. Read batches, unify ID suffix tar_bvd_id -> tar_bvd_id_num, align global merge key
# 2. Deduplicate at detail level by deal + all advisor names, remove duplicate advisor records
# 3. Mark non-empty advisor name as 1, group by deal and sum to get num_* counts
# 4. Generate has_tar_advisor / has_acq_advisor 0-1 dummies (any advisor present?)
# 5. Attach only BVD/Orbis numeric IDs; drop tar_name/acq_name from output
#    to avoid _x/_y name conflicts in later multi-table merges
# Output has no company names; links to master table via deal_num + four ID columns
# --------------------------------------------------------
# ════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE 03adv — Firm Advisor Count")
print("=" * 60)

adv_dir = os.path.join(BASE, "raw", "MA_firm", "advisor")
adv_files = sorted(glob.glob(os.path.join(adv_dir, "advisor*_cleaned.csv")))
print(f"Found {len(adv_files)} advisor batch files")

# Raw CSV full column list
ADV_COLS_RAW = [
    "deal_num",
    "tar_name", "acq_name",
    "tar_bvd_id", "tar_orbis_id",
    "acq_bvd_id", "acq_orbis_id",
    # Target advisors
    "tar_adv_acc_name", "tar_adv_asset_fin_name", "tar_adv_broker_name",
    "tar_adv_debt_name", "tar_adv_equity_name", "tar_adv_fa_name",
    "tar_adv_pr_name", "tar_adv_insurer_name", "tar_adv_law_name",
    "tar_adv_lead_name", "tar_adv_mezz_name", "tar_adv_other_name",
    "tar_adv_underwriter_name", "tar_adv_vcpe_name",
    # Acquirer advisors
    "acq_adv_acc_name", "acq_adv_asset_fin_name", "acq_adv_broker_name",
    "acq_adv_debt_name", "acq_adv_equity_name", "acq_adv_fa_name",
    "acq_adv_pr_name", "acq_adv_insurer_name", "acq_adv_law_name",
    "acq_adv_lead_name", "acq_adv_mezz_name", "acq_adv_other_name",
    "acq_adv_underwriter_name", "acq_adv_vcpe_name",
]

# ID rename map: original no _num suffix -> add _num to match global merge key
RENAME_MAP = {
    "tar_bvd_id":   "tar_bvd_id_num",
    "tar_orbis_id": "tar_orbis_id_num",
    "acq_bvd_id":   "acq_bvd_id_num",
    "acq_orbis_id": "acq_orbis_id_num",
}
rename_cols_list = list(RENAME_MAP.keys())
print(f"\n[Rename rules] original id columns: {rename_cols_list}, new names: {list(RENAME_MAP.values())}")

batches_adv = []
for fp in adv_files:
    fname_short = os.path.basename(fp)
    df_b = read_and_ffill(fp)
    # Strip header whitespace
    df_b.columns = df_b.columns.str.strip()
    df_b = clean_missing(df_b)
    # Keep only preset valid columns
    keep_adv = [c for c in ADV_COLS_RAW if c in df_b.columns]
    df_b = df_b[keep_adv].copy()
    # Match renameable ID columns
    hit_rename = [k for k in RENAME_MAP if k in df_b.columns]
    print(f"\n[{fname_short}] loaded, renameable id columns: {len(hit_rename)} | list: {hit_rename}")
    # Execute ID renaming
    df_b = df_b.rename(columns=RENAME_MAP)
    batches_adv.append(df_b)

# Merge all batch details
df_adv = pd.concat(batches_adv, ignore_index=True)
# Remove duplicate columns as fallback
df_adv = df_adv.loc[:, ~df_adv.columns.duplicated(keep="first")]

print(f"\n[Merged df all column list]")
print(df_adv.columns.tolist())

# Verify all 4 IDs required for global merge are present
target_id_cols = list(RENAME_MAP.values())
exist_id_cols = [c for c in target_id_cols if c in df_adv.columns]
missing_id_cols = [c for c in target_id_cols if c not in df_adv.columns]
print(f"\n[Key ID column check]")
print(f"Expected 4 merge ID fields: {target_id_cols}")
print(f"Present and read: {exist_id_cols} ({len(exist_id_cols)} total)")
print(f"Missing fields: {missing_id_cols} ({len(missing_id_cols)} total)")
if len(missing_id_cols) > 0:
    print("WARNING: some bvd/orbis id columns missing, later merge will lose data!")

# Drop invalid rows with null deal_num
df_adv = df_adv[df_adv["deal_num"].notna()].copy()
# Fix: pass only single-column deal_num to pd.to_numeric
df_adv["deal_num"] = pd.to_numeric(df_adv["deal_num"], errors="coerce").astype("Int64")

# Detail-level dedup: drop only exact duplicates (same deal + all advisor names)
name_cols = [c for c in df_adv.columns if c.endswith("_name")]
dup_subset = ["deal_num"] + name_cols
df_adv = df_adv.drop_duplicates(subset=dup_subset, keep="first")

# Generate 0/1 dummies for each advisor type (non-empty=1, empty=0)
advisor_raw_cols = [c for c in df_adv.columns if c.endswith("_name") and c not in ["tar_name", "acq_name"]]
for col in advisor_raw_cols:
    out_col = "num_" + col.replace("_name", "")
    df_adv[out_col] = (~df_adv[col].isna()).astype(int)

count_adv_cols = [c for c in df_adv.columns if c.startswith("num_")]

# Group by deal_num and sum to get per-deal advisor counts
gb = df_adv.groupby("deal_num")
df_adv_count = gb[count_adv_cols].sum().reset_index()

# ====================== Core optimization: drop transform fill, use unique ID mapping table ======================
export_keys = [
    "tar_bvd_id_num", "tar_orbis_id_num",
    "acq_bvd_id_num", "acq_orbis_id_num"
]
export_keys = [c for c in export_keys if c in df_adv.columns]
print(f"\n[Entity key fields to attach to output csv] {export_keys}")

# Keep only one set of unique IDs per deal, no batch fill duplication
id_mapping = df_adv[["deal_num"] + export_keys].drop_duplicates(subset="deal_num", keep="first")
df_adv_count = df_adv_count.merge(id_mapping, on="deal_num", how="left")
# ===========================================================================================

total_unique_deals = len(df_adv_count)

# Generate target/acquirer advisor presence dummies
tar_num_cols = [c for c in count_adv_cols if c.startswith("num_tar_")]
acq_num_cols = [c for c in count_adv_cols if c.startswith("num_acq_")]
df_adv_count["has_tar_advisor"] = (df_adv_count[tar_num_cols].sum(axis=1) > 0).astype(int)
df_adv_count["has_acq_advisor"] = (df_adv_count[acq_num_cols].sum(axis=1) > 0).astype(int)

# Coverage statistics
tar_adv_deal_cnt = int(df_adv_count["has_tar_advisor"].sum())
acq_adv_deal_cnt = int(df_adv_count["has_acq_advisor"].sum())
tar_adv_pct = tar_adv_deal_cnt / total_unique_deals * 100
acq_adv_pct = acq_adv_deal_cnt / total_unique_deals * 100

#print(f"\nAdvisor coverage across unique deals:")
#print(f"  Target-side any advisor: {tar_adv_deal_cnt:,} deals ({tar_adv_pct:.1f}%)")
#print(f"  Acquirer-side any advisor: {acq_adv_deal_cnt:,} deals ({acq_adv_pct:.1f}%)")
# Before out_path_adv, replaced original 3-line coverage print:
coverage_report(df_adv_count, "Module 03adv — advisor count",
                out_csv=os.path.join(CLEANED, "_cov_moduleJ_advisor.csv"))

# Drop tar_name/acq_name to avoid _x/_y duplicate column conflicts in later merges
drop_name = ["tar_name", "acq_name"]
df_adv_count = df_adv_count.drop(columns=[c for c in drop_name if c in df_adv_count.columns], errors="ignore")

print(f"\nModule 03adv output: {len(df_adv_count):,} rows | {total_unique_deals:,} unique deals")
print(f"\n[Output csv final column list] {df_adv_count.columns.tolist()}")



# Save cleaned advisor count table
out_path_adv = os.path.join(CLEANED, "03b_firm_advisor_count.csv")
df_adv_count.to_csv(out_path_adv, index=False, encoding="utf-8-sig")
file_size_kb = os.path.getsize(out_path_adv) // 1024
print(f"Saved -> {out_path_adv}  ({file_size_kb:,} KB)")
print("=" * 60)
# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Script 03 Complete")
print("=" * 60)
for label, path in [
    ("03_firm_financial_predeal.csv",  out_fin),
    ("03b_firm_financial_postdeal.csv", out_fin_post),   # <- new line
    ("03_firm_financial_company.csv",  out_cf),
    ("03_firm_legal.csv",              out_leg),
    ("03b_listed_status.csv",          out_ls),
    ("03b_firm_advisor_count.csv",     out_path_adv),
]:
    size_kb = os.path.getsize(path) // 1024
    print(f"  {label:<30s} {size_kb:>6,} KB  |  "
          f"{pd.read_csv(path, usecols=['deal_num']).shape[0]:,} rows")
    
# === Wrap up: restore console output ===
sys.stdout = _orig_stdout
_log_fh.flush()
print(f"\nLog saved -> {LOG_PATH}")
