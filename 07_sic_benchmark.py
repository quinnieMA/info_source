"""
07_sic_benchmark.py
Date: 2026-04-14
Purpose: Compute SIC-based valuation benchmarks and BenchmarkDiv.
         - SIC benchmark: median multiple of prior deals in same SIC-3 (or SIC-2)
           group within rolling window (deal_year <= t). Baseline tries SIC-3 first
           and falls back to SIC-2 only if the SIC-3 peer count is insufficient;
           pass --sic-mode sic2only to always group by SIC-2 (robustness variant,
           SIC-3 never attempted).
         - BenchmarkDiv = |ln(text_bench_m) - ln(sic_bench_m)| -- primary IV for H1.
         - Raw values only; winsorisation is done in Stata (plan Q5).

Inputs:
    data/merged/07_deal_firm_benchmark{suffix}.csv  -- output of Script 06 (text benchmarks added)
    data/merged/04b_deal_firm_country.csv           -- full corpus (SIC peer pool)

Outputs:
    data/merged/07_deal_firm_benchmark{suffix}.csv  -- overwritten with SIC + BenchmarkDiv columns
    data/merged/07_sic_benchmark_log{suffix}.txt    -- diagnostics and coverage stats
    {suffix} defaults to "" (baseline pipeline run); pass --suffix to read/write a
    parallel file for a robustness variant, matching the --suffix used in Script 06.

New columns added:
    sic_bench_rev, sic_bench_ebitda, sic_bench_ebit
    sic_peer_count_rev, sic_peer_count_ebitda, sic_peer_count_ebit  (diagnostic: N peers)
    sic_peer_ids  (pipe-separated _row_id of all SIC peers in rolling window; for Script 08 FinSimGap)
    benchdiv_rev, benchdiv_ebitda, benchdiv_ebit
    benchdiv_mean  (mean across available multiples -- robustness only)

CLI args (robustness variants; all optional, defaults reproduce baseline behaviour):
    --min-peers INT              minimum peers with valid multiple for non-NaN benchmark (default 3)
    --sic-mode {fallback,sic2only}  fallback = SIC-3 then SIC-2 (default/baseline);
                                     sic2only = always SIC-2, SIC-3 never attempted
    --suffix STR                 suffix identifying the input/output benchmark file variant

Update (2026-04-15): Added sic_peer_ids column for H4 FinSimGap computation.
"""

import os
import sys
import argparse
import logging
import random

import numpy as np
import pandas as pd

# ── CLI args (robustness variants) — defaults reproduce baseline behaviour ────
_parser = argparse.ArgumentParser()
_parser.add_argument("--min-peers", type=int, default=3,
                      help="minimum peers with valid multiple for non-NaN benchmark")
_parser.add_argument("--sic-mode", choices=["fallback", "sic2only"], default="fallback",
                      help="fallback = try SIC-3 then SIC-2 (baseline); "
                           "sic2only = always group by SIC-2, skip SIC-3")
_parser.add_argument("--suffix", type=str, default="",
                      help="suffix identifying the input/output benchmark file variant, e.g. _K5")
_args = _parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = r"D:\MA"
MERGED = os.path.join(ROOT, "data", "merged")
LOG_PATH = os.path.join(MERGED, f"07_sic_benchmark_log{_args.suffix}.txt")

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("07_sic_benchmark")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(fh)
log.addHandler(ch)

log.info("=== Script 07: SIC Benchmark and BenchmarkDiv ===")
if _args.suffix or _args.sic_mode != "fallback" or _args.min_peers != 3:
    log.info(f"  Robustness variant run: min_peers={_args.min_peers}  "
             f"sic_mode={_args.sic_mode}  suffix={_args.suffix}")

MIN_PEERS = _args.min_peers  # minimum peers with valid multiple for non-NaN benchmark
SIC2_ONLY = (_args.sic_mode == "sic2only")

# ── Load Data ─────────────────────────────────────────────────────────────────
log.info("Loading data ...")
bench_path = os.path.join(MERGED, f"07_deal_firm_benchmark{_args.suffix}.csv")
full_path = os.path.join(MERGED, "04b_deal_firm_country.csv")

df = pd.read_csv(bench_path, low_memory=False)
df_full = pd.read_csv(full_path, low_memory=False)

df["_row_id"] = df["_row_id"].astype(int)
df_full["_row_id"] = df_full["_row_id"].astype(int)
# FIXER R1: m7 — assert _row_id uniqueness in input file before any processing
assert df["_row_id"].is_unique, "FATAL: _row_id not unique in 07_deal_firm_benchmark.csv"

log.info(f"  07_deal_firm_benchmark.csv:  {len(df):,} rows (focal deals + text benchmarks)")
log.info(f"  04b_deal_firm_country.csv:   {len(df_full):,} rows (full corpus)")

# Verify expected text benchmark columns are present
for col in ["text_bench_rev", "text_bench_ebitda", "text_bench_ebit",
            "sic_benchmark_level", "tar_sic3", "tar_sic2", "deal_year"]:
    assert col in df.columns, (
        f"Column '{col}' missing from 07_deal_firm_benchmark.csv -- re-run Script 06"
    )
log.info("  Column presence check: PASS")

# ── Derive deal_year and SIC codes for full corpus ────────────────────────────
log.info("Deriving deal_year and SIC codes for full corpus ...")

# deal_year: completed_d_yr with announced_d_yr fallback (same logic as Script 05/06)
df_full["deal_year"] = df_full["completed_d_yr"].where(
    df_full["completed_d_yr"].notna(), df_full["announced_d_yr"]
)

# SIC codes: derived from tar_primary_sic_code (4-digit float -> int)
df_full["tar_sic4"] = pd.to_numeric(
    df_full["tar_primary_sic_code"], errors="coerce"
)
df_full["tar_sic3"] = np.floor(df_full["tar_sic4"] / 10).where(
    df_full["tar_sic4"].notna() & (df_full["tar_sic4"] >= 100) & (df_full["tar_sic4"] <= 9999)
).astype("Int64")
df_full["tar_sic2"] = np.floor(df_full["tar_sic4"] / 100).where(
    df_full["tar_sic4"].notna() & (df_full["tar_sic4"] >= 100) & (df_full["tar_sic4"] <= 9999)
).astype("Int64")

n_sic3_valid = df_full["tar_sic3"].notna().sum()
n_year_valid = df_full["deal_year"].notna().sum()
log.info(f"  Full corpus: deal_year notna={n_year_valid:,}, tar_sic3 notna={n_sic3_valid:,}")

# ── Build SIC Group Lookup Dicts ─────────────────────────────────────────────
# For efficiency: group full corpus by SIC-3 and SIC-2 so each focal deal
# only scans its relevant peer group (not all 58,691 rows).
log.info("Building SIC group lookup dicts ...")

# Only keep rows that can appear in a peer pool: must have deal_year
df_pool = df_full[df_full["deal_year"].notna()].copy()


def build_sic_lookup(df_src, sic_col):
    """
    Returns dict: {sic_code (int) -> dict of numpy arrays}.
    Arrays: row_ids, deal_years, mul_rev, mul_ebitda, mul_ebit.
    """
    lookup = {}
    for sic_val, grp in df_src.groupby(sic_col, dropna=True):
        sic_key = int(sic_val)
        lookup[sic_key] = {
            "row_ids":    grp["_row_id"].values.astype(int),
            "deal_years": grp["deal_year"].values.astype(float),
            "mul_rev":    grp["pre_rev_mul_ly"].values.astype(float),
            "mul_ebitda": grp["pre_ebitda_mul_ly"].values.astype(float),
            "mul_ebit":   grp["pre_ebit_mul_ly"].values.astype(float),
        }
    return lookup


sic3_lookup = build_sic_lookup(df_pool, "tar_sic3")
sic2_lookup = build_sic_lookup(df_pool, "tar_sic2")

log.info(f"  SIC-3 groups: {len(sic3_lookup):,}")
log.info(f"  SIC-2 groups: {len(sic2_lookup):,}")

# ── SIC Benchmark Computation ────────────────────────────────────────────────
log.info("Computing SIC benchmarks (rolling window, no look-ahead) ...")

# Initialize result columns
for col in [
    "sic_bench_rev", "sic_bench_ebitda", "sic_bench_ebit",
    "sic_peer_count_rev", "sic_peer_count_ebitda", "sic_peer_count_ebit",
]:
    df[col] = np.nan
# sic_peer_ids: pipe-separated _row_id list of ALL SIC peers in rolling window.
# Added 2026-04-15 for FinSimGap computation in Script 08 (H4).
df["sic_peer_ids"] = pd.NA
df["n_sic_peers_total"] = pd.NA    # total SIC group size (all members, not just valid-multiple subset)

# FIXER R1: M3 — per-multiple fallback level tracking columns.
# sic_level_used_{m} records whether SIC-3 (=3), SIC-2 (=2), or no benchmark (=NaN) was used
# for each (deal, multiple) pair.  The pre-assigned sic_benchmark_level is kept for
# documentation/logging purposes but is NOT used to gate the fallback decision.
for m_name in ["rev", "ebitda", "ebit"]:
    df[f"sic_level_used_{m_name}"] = np.nan

n_processed = 0
n_sic_level2_used = 0   # deals where at least one multiple used SIC-2
n_sic_nan = 0           # deals where sic lookup returned no group at SIC-3 level

for i, row in df.iterrows():
    focal_year = row["deal_year"]
    focal_rid  = int(row["_row_id"])

    if pd.isna(focal_year):
        n_sic_nan += 1
        continue

    # Resolve focal SIC codes (may be NaN)
    focal_sic3 = row["tar_sic3"]
    focal_sic2 = row["tar_sic2"]

    # Determine the base SIC-3 peer group (for sic_peer_ids — used by FinSimGap in Script 08).
    # sic_peer_ids uses SIC-3 peers, falling back to SIC-2 only if SIC-3 is unavailable
    # (fallback mode) — or always SIC-2 in --sic-mode sic2only. Set once per deal.
    sic_peer_ids_set = False
    if not SIC2_ONLY and pd.notna(focal_sic3):
        sic3_int = int(focal_sic3)
        if sic3_int in sic3_lookup:
            grp3 = sic3_lookup[sic3_int]
            pool_mask3 = (grp3["deal_years"] <= focal_year) & (grp3["row_ids"] != focal_rid)
            if pool_mask3.any():
                pool_ids3 = grp3["row_ids"][pool_mask3]
                df.at[i, "sic_peer_ids"] = "|".join(str(x) for x in pool_ids3)
                df.at[i, "n_sic_peers_total"] = len(pool_ids3)
                sic_peer_ids_set = True

    if not sic_peer_ids_set and pd.notna(focal_sic2):
        sic2_int = int(focal_sic2)
        if sic2_int in sic2_lookup:
            grp2 = sic2_lookup[sic2_int]
            pool_mask2 = (grp2["deal_years"] <= focal_year) & (grp2["row_ids"] != focal_rid)
            if pool_mask2.any():
                pool_ids2 = grp2["row_ids"][pool_mask2]
                df.at[i, "sic_peer_ids"] = "|".join(str(x) for x in pool_ids2)
                df.at[i, "n_sic_peers_total"] = len(pool_ids2)

    # FIXER R1: M3 — per-multiple fallback: try SIC-3 first; if < MIN_PEERS valid
    # multiples for this specific m, try SIC-2; if still < MIN_PEERS, leave as NaN.
    deal_has_any = False
    deal_used_level2 = False

    for m_name, mul_key in [
        ("rev",    "mul_rev"),
        ("ebitda", "mul_ebitda"),
        ("ebit",   "mul_ebit"),
    ]:
        bench_val   = np.nan
        n_valid_val = 0
        level_used  = np.nan

        # Try SIC-3 (skipped entirely in sic2only robustness mode)
        if not SIC2_ONLY and pd.notna(focal_sic3):
            sic3_int = int(focal_sic3)
            if sic3_int in sic3_lookup:
                grp3 = sic3_lookup[sic3_int]
                pm3  = (grp3["deal_years"] <= focal_year) & (grp3["row_ids"] != focal_rid)
                valid_mul3 = grp3[mul_key][pm3]
                valid_mul3 = valid_mul3[valid_mul3 > 0]
                if len(valid_mul3) >= MIN_PEERS:
                    bench_val   = float(np.median(valid_mul3))
                    n_valid_val = len(valid_mul3)
                    level_used  = 3

        # If SIC-3 insufficient, try SIC-2 fallback
        if np.isnan(bench_val) and pd.notna(focal_sic2):
            sic2_int = int(focal_sic2)
            if sic2_int in sic2_lookup:
                grp2 = sic2_lookup[sic2_int]
                pm2  = (grp2["deal_years"] <= focal_year) & (grp2["row_ids"] != focal_rid)
                valid_mul2 = grp2[mul_key][pm2]
                valid_mul2 = valid_mul2[valid_mul2 > 0]
                if len(valid_mul2) >= MIN_PEERS:
                    bench_val   = float(np.median(valid_mul2))
                    n_valid_val = len(valid_mul2)
                    level_used  = 2
                    deal_used_level2 = True

        df.at[i, f"sic_bench_{m_name}"]      = bench_val
        df.at[i, f"sic_peer_count_{m_name}"] = n_valid_val if n_valid_val > 0 else np.nan
        df.at[i, f"sic_level_used_{m_name}"] = level_used

        if not np.isnan(bench_val):
            deal_has_any = True

    if not deal_has_any:
        n_sic_nan += 1
    else:
        n_processed += 1
        if deal_used_level2:
            n_sic_level2_used += 1

    if n_processed % 2000 == 0:
        log.info(f"  Processed {n_processed:,}/{len(df):,} focal deals ...")

log.info(
    f"SIC benchmark complete: {n_processed:,} focal deals with at least one benchmark, "
    f"{n_sic_nan} with no valid benchmark for any multiple, "
    f"{n_sic_level2_used} used SIC-2 fallback for at least one multiple"
)

# ── Coverage Statistics ───────────────────────────────────────────────────────
log.info("=== SIC Benchmark Coverage ===")
n_total = len(df)
for m in ["rev", "ebitda", "ebit"]:
    n_valid = df[f"sic_bench_{m}"].notna().sum()
    pc = df[f"sic_peer_count_{m}"].dropna()
    log.info(
        f"  sic_bench_{m}: {n_valid:,}/{n_total:,} ({100.0 * n_valid / n_total:.1f}%) | "
        f"peer_count: mean={pc.mean():.1f}, median={pc.median():.0f}, "
        f"p25={pc.quantile(0.25):.0f}, p75={pc.quantile(0.75):.0f}"
    )

# ── BenchmarkDiv Computation ──────────────────────────────────────────────────
log.info("Computing BenchmarkDiv = |ln(text_bench) - ln(sic_bench)| ...")

for m in ["rev", "ebitda", "ebit"]:
    tb = df[f"text_bench_{m}"]
    sb = df[f"sic_bench_{m}"]
    both_valid = tb.notna() & sb.notna() & (tb > 0) & (sb > 0)
    df[f"benchdiv_{m}"] = np.where(
        both_valid,
        np.abs(np.log(tb) - np.log(sb)),
        np.nan,
    )

# Composite measure (robustness): mean across available multiples
benchdiv_cols = ["benchdiv_rev", "benchdiv_ebitda", "benchdiv_ebit"]
df["benchdiv_mean"] = df[benchdiv_cols].mean(axis=1, skipna=True)
# Set to NaN if no multiple available
df.loc[df[benchdiv_cols].isna().all(axis=1), "benchdiv_mean"] = np.nan

# ── BenchmarkDiv Statistics ───────────────────────────────────────────────────
log.info("=== BenchmarkDiv Statistics ===")
log.info("  (Raw values; winsorisation at 99th pct is done in Stata)")
for m in ["rev", "ebitda", "ebit", "mean"]:
    col = f"benchdiv_{m}"
    s = df[col].dropna()
    n = len(s)
    if n == 0:
        log.info(f"  benchdiv_{m}: N=0")
        continue
    log.info(
        f"  benchdiv_{m}: N={n:,}, mean={s.mean():.4f}, sd={s.std():.4f}, "
        f"p10={s.quantile(0.10):.4f}, p50={s.quantile(0.50):.4f}, "
        f"p90={s.quantile(0.90):.4f}"
    )

# Correlation matrix across the three multiples
log.info("=== BenchmarkDiv Correlations ===")
bd_cols_3 = ["benchdiv_rev", "benchdiv_ebitda", "benchdiv_ebit"]
corr = df[bd_cols_3].corr()
for col_a in bd_cols_3:
    for col_b in bd_cols_3:
        if col_a < col_b:
            log.info(f"  corr({col_a}, {col_b}) = {corr.loc[col_a, col_b]:.4f}")

# Expected: corr > 0.3 (the three BenchmarkDiv measures should be positively correlated)
for col_a, col_b in [
    ("benchdiv_rev", "benchdiv_ebitda"),
    ("benchdiv_rev", "benchdiv_ebit"),
    ("benchdiv_ebitda", "benchdiv_ebit"),
]:
    corr_val = corr.loc[col_a, col_b]
    if corr_val < 0.3:
        log.warning(
            f"  LOW CORRELATION: {col_a} vs {col_b} = {corr_val:.4f} (expected > 0.3)"
        )
    else:
        log.info(f"  Correlation check {col_a} vs {col_b}: OK ({corr_val:.4f} > 0.3)")

# ── Spot-check: 5 focal deals ─────────────────────────────────────────────────
log.info("=== Spot-check: 5 focal deals with SIC benchmark ===")
has_sic = df["sic_bench_rev"].notna() | df["sic_bench_ebitda"].notna()
spot = df[has_sic].sample(5, random_state=SEED)
for _, row in spot.iterrows():
    # m1 FIXER R1: use the per-run sic_level_used_ebitda (computed above from the
    # actual --sic-mode used this run) instead of the upstream sic_benchmark_level
    # (fixed baseline SIC-3-first logic from Script 05) — sic_benchmark_level would
    # print the wrong level label under --sic-mode sic2only. Fall back to
    # sic_benchmark_level only if sic_level_used_ebitda is NaN (e.g. no EBITDA
    # benchmark for this row) so the diagnostic still prints something sensible.
    _lvl_used = row.get('sic_level_used_ebitda', np.nan)
    sic_lvl = int(_lvl_used) if pd.notna(_lvl_used) else int(row['sic_benchmark_level'])
    sic_val = row[f'tar_sic{sic_lvl}']
    log.info(
        f"  {row.get('tar_name','?')} | yr={row['deal_year']:.0f} | "
        f"sic{sic_lvl}={sic_val}"
    )
    log.info(
        f"    sic_bench:  rev={row['sic_bench_rev']:.3f}, "
        f"ebitda={row['sic_bench_ebitda']:.3f}, ebit={row['sic_bench_ebit']:.3f}"
    )
    log.info(
        f"    text_bench: rev={row['text_bench_rev']:.3f}, "
        f"ebitda={row['text_bench_ebitda']:.3f}, ebit={row['text_bench_ebit']:.3f}"
    )
    log.info(
        f"    benchdiv:   rev={row['benchdiv_rev']:.3f}, "
        f"ebitda={row['benchdiv_ebitda']:.3f}, ebit={row['benchdiv_ebit']:.3f}"
    )

# ── Row Count Check ───────────────────────────────────────────────────────────
n_filtered_orig = len(
    pd.read_csv(
        os.path.join(MERGED, "05_deal_firm_filtered.csv"),
        usecols=["_row_id"],
        low_memory=False,
    )
)
assert len(df) == n_filtered_orig, (
    f"Row count mismatch: benchmark={len(df)}, filtered={n_filtered_orig}"
)
log.info(f"Row count assertion: PASS ({len(df):,} rows)")

# ── Save Output ───────────────────────────────────────────────────────────────
# FIXER R1: m4 — Atomic write: write to .tmp file then os.replace to avoid partial writes
out_path = os.path.join(MERGED, f"07_deal_firm_benchmark{_args.suffix}.csv")
tmp_path = str(out_path) + ".tmp"
df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
os.replace(tmp_path, str(out_path))
log.info(f"  Saved atomically: {out_path}  ({len(df):,} rows)")

# Final column list
new_cols = [
    "sic_bench_rev", "sic_bench_ebitda", "sic_bench_ebit",
    "sic_peer_count_rev", "sic_peer_count_ebitda", "sic_peer_count_ebit",
    "sic_peer_ids", "n_sic_peers_total",
    "benchdiv_rev", "benchdiv_ebitda", "benchdiv_ebit", "benchdiv_mean",
    # FIXER R1: M3 — per-multiple fallback level tracking
    "sic_level_used_rev", "sic_level_used_ebitda", "sic_level_used_ebit",
]
log.info(f"New columns added: {new_cols}")

log.info("=== Script 07 complete ===")
logging.shutdown()
