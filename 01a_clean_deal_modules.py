"""
01a_clean_deal_modules.py
========================
Batch clean raw Zephyr M&A source files into standardized module CSV files stored under data/cleaned.
All raw data resides in subfolders of raw/MA_deal, processed sequentially by module.

Core processing rules for every batch file:
1. Preserve original row order strictly; forward-fill deal_num per file BEFORE concatenation.
   Zephyr attaches supplementary cascade entity rows right after parent deal records. Sorting breaks
   cascade row grouping and invalidates forward-fill recovery logic.
2. Standardize missing value representations (Zephyr placeholder strings → uniform NaN).
3. Normalize Orbis ID format to consistent 9-digit zero-padded string for cross-table matching.
4. Recover acquirer SIC codes from within the same deal_num cascade block if unambiguous.
5. Deduplication logic varies by module observation unit:
   - Module A (Industry): Unique key = (deal_num, tar_key), tar_key = tar_bvd_id_num / tar_name fallback
   - Module B/C/D (Multiples / Structure / Value): Unique key = deal_num (single record per transaction)
   - Module F (Info Source): Aggregate source category counts to deal-level from multi-line raw records

Module Output Mapping:
Module A (Industry & Firm Text) → data/cleaned/01_deal_sic_industry.csv
    Unit: Deal × Target (supports multi-target transactions; retains tar_overview Doc2Vec text)
Module B (Valuation Multiples) → data/cleaned/01_deal_multiples.csv
    Unit: Single row per deal, pre/post transaction multiples
Module C (Deal Structure & Dates) → data/cleaned/01_deal_structure_date.csv
    Unit: Single row per deal, transaction status & time horizon variables
- 2026-09-07: Module C — generalize categorical aggregation to all multi-value
  fields (pay_method / struct / fin / type) BEFORE dedup. deal_struct has
  6,435 multi-value deals (vs 1,735 for pay_method) — dedup was dropping ~1/3
  of structure tags.
  
Module D (Transaction Value & Stake) → data/cleaned/01_deal_value.csv
    Unit: Single row per deal, equity/EV consideration & acquired ownership share
Module F (Information Source Categories) → data/cleaned/01_deal_info_source_count.csv
    Unit: Single row per deal, dummy count variables for each disclosure channel

Revision Log:
- 2026-07-27: Rewrite clean_missing() for cross-Pandas version compatibility
Author: Q  Date: 2026-08-05
"""

import os
import glob
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = r"D:\MA"
RAW_DEAL  = os.path.join(BASE, "raw", "MA_deal")
CLEANED   = os.path.join(BASE, "data", "cleaned")
os.makedirs(CLEANED, exist_ok=True)

# Placeholder strings Zephyr uses for missing values
MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]

def read_and_ffill(filepath, encoding="utf-8-sig"):
    """
    Read a Zephyr CSV in file order and immediately forward-fill deal_num.
    Never sort before ffill — cascade rows have blank unnamed_0 too.
    """
    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
    # Normalise the BOM-prefixed unnamed column (appears as 'unnamed__0' or 'unnamed_0')
    df.columns = df.columns.str.strip()
    # forward-fill deal_num in place (file order = Zephyr order)
    df["deal_num"] = df["deal_num"].ffill()
    return df

#def clean_missing(df):
    """Replace Zephyr placeholder strings with NaN across all object columns."""
    # m1 FIXER R1: added .str.strip() to match script 03 behaviour
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].replace(MISSING_VALS, np.nan).str.strip()
    return df
def clean_missing(df):
    """Replace Zephyr placeholder strings with NaN across all object columns."""
    # 兼容 pandas 3.x: 只使用 "object"，因为 pandas 3.x 不支持 "str"
    try:
        # 尝试原作者的写法（pandas 2.x）
        cols = df.select_dtypes(include=["object", "str"]).columns
    except TypeError:
        # pandas 3.x: 只使用 "object"
        cols = df.select_dtypes(include=["object"]).columns
    
    for col in cols:
        # 安全地处理字符串列
        df[col] = df[col].astype('object')
        df[col] = df[col].replace(MISSING_VALS, np.nan)
        # 只对非空值进行 strip
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
            # 空字符串转 NaN
            df.loc[mask, col] = df.loc[mask, col].replace('', np.nan)
    return df


def normalize_orbis_id(series):
    """
    Normalise Orbis ID to 9-digit zero-padded string.
    pandas reads purely-numeric columns as float (e.g. 6533168.0);
    this strips the trailing '.0' and zero-pads to match the format
    used in the firm-module files (e.g. '006533168').
    """
    def fix(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s in ("nan", "None", ""):
            return np.nan
        return s.zfill(9)
    return series.apply(fix)


# ════════════════════════════════════════════════════════════════════════════
# Module A — Industry + Text
#   Source : raw/MA_deal/industry/  (5 batches, ~94,322 raw rows)
#   Output : data/cleaned/01_deal_sic_industry.csv
#   Unit   : (deal_num, tar_key) where tar_key = tar_bvd_id_num or tar_name
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODULE A — Industry + Text")
print("=" * 60)

industry_dir = os.path.join(RAW_DEAL, "industry")
industry_files = sorted(glob.glob(os.path.join(industry_dir, "acquisition_industry_*_cleaned.csv")))
print(f"Found {len(industry_files)} industry batch files")

# Columns to retain in the final cleaned file
INDUSTRY_KEEP = [
    "deal_num",
    # Target identifiers
    "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    # Target text (for Doc2Vec)
    "tar_overview", "tar_trade_descr_en", "tar_busi_descr",
    # Acquirer and vendor overview text (for Doc2Vec similarity measures)
    "acq_overview", "ven_overview",
    # Target SIC and NAICS
    "tar_primary_sic_code", "tar_sic_codes",
    "tar_primary_naics_code",
    # Acquirer identifiers + SIC (for CrossInd)
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    "acq_primary_sic_code", "acq_sic_codes",
    # Vendor identifiers (for completeness)
    "ven_name", "ven_bvd_id_num",
    "ven_primary_sic_code",
]

batches_ind = []
for fp in industry_files:
    batch_name = os.path.basename(fp)
    df_b = read_and_ffill(fp)
    n_raw = len(df_b)
    # Clean missing strings
    df_b = clean_missing(df_b)
    # Keep only columns present in INDUSTRY_KEEP
    keep_present = [c for c in INDUSTRY_KEEP if c in df_b.columns]
    df_b = df_b[keep_present]
    batches_ind.append(df_b)
    print(f"  {batch_name}: {n_raw} rows read, deal_num ffilled")

df_ind = pd.concat(batches_ind, ignore_index=True)
print(f"\nAfter stacking all batches: {len(df_ind):,} rows")

# Normalise Orbis IDs to 9-digit zero-padded strings
for _oc in ["tar_orbis_id_num", "acq_orbis_id_num"]:
    if _oc in df_ind.columns:
        df_ind[_oc] = normalize_orbis_id(df_ind[_oc])

# ── Recover acq_primary_sic_code from cascade rows (same deal_num block) ────
# Zephyr splits target and acquirer info across different rows within one
# deal_num (unlike the overview file, where cascade rows are fully blank).
# The row kept below (tar_name notna) often has acq_primary_sic_code=NaN
# while a sibling cascade row (tar_name NaN) for the same deal_num carries it.
# Only fill when the whole deal_num block has EXACTLY ONE distinct non-null
# acq_primary_sic_code value — never guess between competing bidders/targets.
_nunique_sic = df_ind.groupby("deal_num")["acq_primary_sic_code"].transform(
    lambda s: s.dropna().nunique()
)
_unique_val = df_ind.groupby("deal_num")["acq_primary_sic_code"].transform(
    lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else np.nan
)
_fillable = df_ind["acq_primary_sic_code"].isna() & (_nunique_sic == 1)
_is_target_row = df_ind["tar_name"].notna()
n_recovered_kept = int((_fillable & _is_target_row).sum())
n_ambiguous_kept = int((df_ind["acq_primary_sic_code"].isna() & (_nunique_sic >= 2) & _is_target_row).sum())
# M2 FIXER R1: count target rows still missing acq_primary_sic_code where the
# whole deal_num block has ZERO non-null candidates anywhere (truly unrecoverable —
# not ambiguous, just no information exists in the block).
n_unrecoverable_kept = int((df_ind["acq_primary_sic_code"].isna() & (_nunique_sic == 0) & _is_target_row).sum())
df_ind.loc[_fillable, "acq_primary_sic_code"] = _unique_val[_fillable]
print(f"\nCascade-row acq_primary_sic_code recovery (target rows only, tar_name notna): "
      f"{n_recovered_kept:,} rows filled (unique candidate in deal_num block)")
print(f"  Ambiguous (2+ distinct candidates, left NaN): {n_ambiguous_kept:,}")
print(f"  Unrecoverable (0 candidates anywhere in deal_num block, left NaN): {n_unrecoverable_kept:,}")

# ── Filter: keep only rows with target information ──────────────────────────
# Cascade vendor/acquirer-only rows have tar_name=NaN.
# Some valid target rows have tar_bvd_id_num=NaN (no BvD ID) but tar_name filled.
n_before_filter = len(df_ind)
df_ind = df_ind[df_ind["tar_name"].notna()].copy()
print(f"After drop rows with tar_name=NaN: {len(df_ind):,} rows "
      f"(dropped {n_before_filter - len(df_ind):,} non-target cascade rows)")

# ── Dedup by (deal_num, tar_key) ────────────────────────────────────────────
# tar_key = tar_bvd_id_num when available, else tar_name (fallback)
# This correctly handles multi-target deals where some targets lack BvD IDs.
df_ind["_tar_key"] = df_ind["tar_bvd_id_num"].fillna(df_ind["tar_name"])
n_before_dedup = len(df_ind)
df_ind = df_ind.drop_duplicates(subset=["deal_num", "_tar_key"], keep="first")
df_ind = df_ind.drop(columns=["_tar_key"])
print(f"After dedup by (deal_num, tar_key): {len(df_ind):,} rows "
      f"(dropped {n_before_dedup - len(df_ind):,} duplicate rows)")

# ── Convert deal_num to integer where possible ──────────────────────────────
df_ind["deal_num"] = pd.to_numeric(df_ind["deal_num"], errors="coerce")
df_ind = df_ind[df_ind["deal_num"].notna()]  # drop any remaining NaN deal_nums
df_ind["deal_num"] = df_ind["deal_num"].astype("Int64")  # nullable integer

print(f"\nModule A output: {len(df_ind):,} rows | "
      f"{df_ind['deal_num'].nunique():,} unique deals")
print(f"  tar_bvd_id_num missing: {df_ind['tar_bvd_id_num'].isna().sum():,} "
      f"({df_ind['tar_bvd_id_num'].isna().mean()*100:.1f}%)")
print(f"  tar_primary_sic_code missing: {df_ind['tar_primary_sic_code'].isna().sum():,} "
      f"({df_ind['tar_primary_sic_code'].isna().mean()*100:.1f}%)")
print(f"  tar_overview missing: {df_ind['tar_overview'].isna().sum():,} "
      f"({df_ind['tar_overview'].isna().mean()*100:.1f}%)")

out_path_ind = os.path.join(CLEANED, "01_deal_sic_industry.csv")
df_ind.to_csv(out_path_ind, index=False, encoding="utf-8-sig")
print(f"\nSaved → {out_path_ind}")


# ════════════════════════════════════════════════════════════════════════════
# Module B — Multiples
#   Source : raw/MA_deal/multiple/  (2 batches, ~58,975 raw rows)
#   Output : data/cleaned/01_deal_multiples.csv
#   Unit   : deal_num (one row per deal)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE B — Multiples")
print("=" * 60)

mul_dir = os.path.join(RAW_DEAL, "multiple")
mul_files = sorted(glob.glob(os.path.join(mul_dir, "acquisition_multiple_*_cleaned.csv")))
print(f"Found {len(mul_files)} multiple batch files")

batches_mul = []
for fp in mul_files:
    batch_name = os.path.basename(fp)
    df_b = read_and_ffill(fp)
    n_raw = len(df_b)
    df_b = clean_missing(df_b)
    batches_mul.append(df_b)
    print(f"  {batch_name}: {n_raw} rows read")

df_mul = pd.concat(batches_mul, ignore_index=True)
print(f"\nAfter stacking: {len(df_mul):,} rows")

# Drop rows with missing deal_num (should be rare/none after ffill, but guard)
n_before = len(df_mul)
df_mul = df_mul[df_mul["deal_num"].notna()].copy()
if n_before > len(df_mul):
    print(f"Dropped {n_before - len(df_mul):,} rows with NaN deal_num after ffill")

# F2 FIXER R1: verify duplicate rows in Module B are genuinely identical
_dup_mask_b = df_mul.duplicated(subset=["deal_num"], keep=False)
if _dup_mask_b.sum() > 0:
    _dup_df_b = df_mul[_dup_mask_b]
    _key_cols_b = [c for c in ["pre_rev_mul_ly", "pre_ebitda_mul_ly"] if c in _dup_df_b.columns]
    if _key_cols_b:
        _inconsistent_b = _dup_df_b.groupby("deal_num")[_key_cols_b].nunique(dropna=True)
        _n_inconsistent_b = (_inconsistent_b > 1).any(axis=1).sum()
        print(f"\nF2 CHECK Module B: {_dup_mask_b.sum()} duplicate rows across "
              f"{_dup_df_b['deal_num'].nunique()} deals")
        print(f"  Deals with DIFFERENT key values in duplicate rows: {_n_inconsistent_b}")
        if _n_inconsistent_b > 0:
            print(f"  WARNING: {_n_inconsistent_b} deals have inconsistent duplicate rows "
                  f"— 'keep first' may lose real data")

# Dedup by deal_num — keep first (multiples are deal-level attributes)
n_before_dedup = len(df_mul)
df_mul = df_mul.drop_duplicates(subset=["deal_num"], keep="first")
print(f"After dedup by deal_num: {len(df_mul):,} rows "
      f"(dropped {n_before_dedup - len(df_mul):,} duplicate rows)")

# Convert deal_num to Int64
df_mul["deal_num"] = pd.to_numeric(df_mul["deal_num"], errors="coerce").astype("Int64")

# Convert multiple columns from string to numeric
MUL_COLS = [c for c in df_mul.columns if c.endswith("_mul_ly") or c.endswith("_mul_fy")]
for col in MUL_COLS:
    df_mul[col] = pd.to_numeric(df_mul[col], errors="coerce")

print(f"\nModule B output: {len(df_mul):,} rows | "
      f"{df_mul['deal_num'].nunique():,} unique deals")
print(f"  pre_rev_mul_ly missing: {df_mul['pre_rev_mul_ly'].isna().sum():,} "
      f"({df_mul['pre_rev_mul_ly'].isna().mean()*100:.1f}%)")
print(f"  pre_ebitda_mul_ly missing: {df_mul['pre_ebitda_mul_ly'].isna().sum():,} "
      f"({df_mul['pre_ebitda_mul_ly'].isna().mean()*100:.1f}%)")
print(f"  pre_ebit_mul_ly missing: {df_mul['pre_ebit_mul_ly'].isna().sum():,} "
      f"({df_mul['pre_ebit_mul_ly'].isna().mean()*100:.1f}%)")

# === 新增post系列缺失率打印 ===================================================
print(f"  post_rev_mul_fy missing: {df_mul['post_rev_mul_fy'].isna().sum():,} "
      f"({df_mul['post_rev_mul_fy'].isna().mean()*100:.1f}%)")
print(f"  post_ebitda_mul_fy missing: {df_mul['post_ebitda_mul_fy'].isna().sum():,} "
      f"({df_mul['post_ebitda_mul_fy'].isna().mean()*100:.1f}%)")
print(f"  post_ebit_mul_fy missing: {df_mul['post_ebit_mul_fy'].isna().sum():,} "
      f"({df_mul['post_ebit_mul_fy'].isna().mean()*100:.1f}%)")
# Drop the row-index column (unnamed_0) from output
if "unnamed_0" in df_mul.columns:
    df_mul = df_mul.drop(columns=["unnamed_0"])

out_path_mul = os.path.join(CLEANED, "01_deal_multiples.csv")
df_mul.to_csv(out_path_mul, index=False, encoding="utf-8-sig")
print(f"\nSaved → {out_path_mul}")


# ════════════════════════════════════════════════════════════════════════════
# Module C — Structure & Dates
#   Source : raw/MA_deal/structure_date/  (2 batches)
#   Output : data/cleaned/01_deal_structure_date.csv
#   Unit   : deal_num (one row per deal)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE C — Structure & Dates")
print("=" * 60)

sd_dir = os.path.join(RAW_DEAL, "structure_date")
sd_files = sorted(glob.glob(os.path.join(sd_dir, "acquisition_structure_date_*_cleaned.csv")))
print(f"Found {len(sd_files)} structure_date batch files")

batches_sd = []
for fp in sd_files:
    batch_name = os.path.basename(fp)
    df_b = read_and_ffill(fp)
    n_raw = len(df_b)
    df_b = clean_missing(df_b)
    batches_sd.append(df_b)
    print(f"  {batch_name}: {n_raw} rows read")

df_sd = pd.concat(batches_sd, ignore_index=True)
print(f"\nAfter stacking: {len(df_sd):,} rows")

# Drop rows with missing deal_num
n_before = len(df_sd)
df_sd = df_sd[df_sd["deal_num"].notna()].copy()
if n_before > len(df_sd):
    print(f"Dropped {n_before - len(df_sd):,} rows with NaN deal_num")

# ══ CATEGORICAL FIELD AGGREGATION (2026-09-08) ═══════════════════════════
# Zephyr 对多值分类字段采用「一值一行」存储。Module C 按 deal_num keep-first
# 去重会静默丢弃第二行起的取值。
#
# 实测损失：
#   deal_pay_method : 多值 deal 1,735 | 换股占比被低估 13.4% → 18.6%
#   deal_struct     : 多值 deal 6,435 | 每 deal 均值 1.49 个取值 ← 更严重
#
# 故所有分类字段统一在 dedup 之前聚合为竖线分隔集合。
# 新增字段：只需在下面列表里加一个字符串，不要再写特化代码。
CATEGORICAL_AGG = [
    "deal_pay_method",
    "deal_struct",
    "deal_fin",
    "deal_type",
]

# ⚠️ 后处理/特殊情境取值 —— 聚合后仅标记，禁止进入任何回归
STRUCT_POSTTREAT = {
    "Public takeover - Unsuccessful",
    "Public takeover - Withdrawn",
}
STRUCT_DISTRESS = {
    "Receivership", "Insolvency", "Administration",
    "Nationalisation", "Distressed Debt",
}

_agg_frames = []
print("\n" + "=" * 70)
print("CATEGORICAL AGG — 分类字段去重前聚合")
print("=" * 70)

for _src in CATEGORICAL_AGG:
    if _src not in df_sd.columns:
        print(f"  {_src:<20s} 跳过（不在数据中）")
        continue

    _sub = df_sd.dropna(subset=[_src])
    if _sub.empty:
        print(f"  {_src:<20s} 跳过（全缺失）")
        continue

    _out = f"{_src}_all"
    _cnt = f"{_src}_n"
    _g = (_sub.groupby("deal_num")[_src]
          .apply(lambda x: "|".join(sorted(set(x.astype(str)))))
          .rename(_out).to_frame())
    _g[_cnt] = _g[_out].str.split("|").apply(len)
    _g = _g.reset_index()

    _n_deal = len(_g)
    _n_multi = int((_g[_cnt] > 1).sum())
    print(f"\n  【{_src}】{_n_deal:,} deals 有值 | "
          f"多值 deal {_n_multi:,} ({_n_multi/max(_n_deal,1):.1%}) | "
          f"每 deal 均值 {_g[_cnt].mean():.2f}")

    _agg_frames.append(_g)

    # ── deal_pay_method：保留换股低估诊断 ──
    if _src == "deal_pay_method":
        _first = (_sub.drop_duplicates("deal_num", keep="first")
                  .set_index("deal_num")[_src].astype(str))
        _cmp = _g.set_index("deal_num")[[_out]].join(
            _first.rename("_first"), how="left")
        _has = _cmp[_out].str.contains("Shares", na=False)
        _lost = int((_has & (_cmp["_first"] != "Shares")).sum())
        print(f"     ⚠️ {_lost:,} 笔含换股但去重首行非 Shares → 信息被丢弃")
        print(f"        去重口径 {_cmp['_first'].eq('Shares').mean():.1%}"
              f"  →  聚合口径 {_has.mean():.1%}")

    # ── deal_struct：输出全取值域（供分类映射用）──
    if _src == "deal_struct":
        _vals = _sub[_src].astype(str).value_counts()
        print(f"\n     [取值域] {len(_vals)} 个取值，全部列出：")
        for _v, _n in _vals.items():
            _flag = ""
            if _v in STRUCT_POSTTREAT:
                _flag = "  ⚠️后处理-禁入回归"
            elif _v in STRUCT_DISTRESS:
                _flag = "  ⚠️破产/接管-建议剔样本"
            print(f"       {_v:<44s} {_n:>7,}{_flag}")

        print(f"\n     [聚合后组合] Top 15：")
        for _v, _n in _g[_out].value_counts().head(15).items():
            print(f"       {str(_v):<60s} {_n:>7,}")

        # 后处理 / 困境类规模
        for _lab, _set in [("后处理", STRUCT_POSTTREAT),
                           ("困境情境", STRUCT_DISTRESS)]:
            _m = _g[_out].apply(
                lambda s: bool({x.strip() for x in str(s).split("|")} & _set))
            print(f"     含【{_lab}】标记的 deal: {int(_m.sum()):,} "
                  f"({_m.mean():.1%})")

# ── 合并所有聚合结果 ──
if _agg_frames:
    _agg_all = _agg_frames[0]
    for _f in _agg_frames[1:]:
        _agg_all = _agg_all.merge(_f, on="deal_num", how="outer")
    _agg_all["deal_num"] = pd.to_numeric(
        _agg_all["deal_num"], errors="coerce").astype("Int64")
    print(f"\n  聚合表就绪：{len(_agg_all):,} deals | "
          f"{len(_agg_all.columns)-1} 个新列")
else:
    _agg_all = None
    print("\n  ⚠️ 没有任何字段被聚合")
# ══ CATEGORICAL AGGREGATION end ═══════════════════════════════════════════
   
# F2 FIXER R1: verify duplicate rows in Module C are genuinely identical
_dup_mask_c = df_sd.duplicated(subset=["deal_num"], keep=False)
if _dup_mask_c.sum() > 0:
    _dup_df_c = df_sd[_dup_mask_c]
    _key_cols_c = [c for c in ["deal_status", "completed_d_yr"] if c in _dup_df_c.columns]
    if _key_cols_c:
        _inconsistent_c = _dup_df_c.groupby("deal_num")[_key_cols_c].nunique(dropna=True)
        _n_inconsistent_c = (_inconsistent_c > 1).any(axis=1).sum()
        print(f"\nF2 CHECK Module C: {_dup_mask_c.sum()} duplicate rows across "
              f"{_dup_df_c['deal_num'].nunique()} deals")
        print(f"  Deals with DIFFERENT key values in duplicate rows: {_n_inconsistent_c}")
        if _n_inconsistent_c > 0:
            print(f"  WARNING: {_n_inconsistent_c} deals have inconsistent duplicate rows "
                  f"— 'keep first' may lose real data")

# Dedup by deal_num — keep first (deal status and dates are deal-level)
n_before_dedup = len(df_sd)
df_sd = df_sd.drop_duplicates(subset=["deal_num"], keep="first")
print(f"After dedup by deal_num: {len(df_sd):,} rows "
      f"(dropped {n_before_dedup - len(df_sd):,} duplicate rows)")

# Convert deal_num to Int64
#df_sd["deal_num"] = pd.to_numeric(df_sd["deal_num"], errors="coerce").astype("Int64")

# Select key columns for analysis
#SD_KEEP = [
   # "deal_num",
    #"deal_type", "deal_status", "deal_struct", "deal_fin", "deal_pay_method",
   # "announced_d", "completed_d", "withdrawn_d",
   # "announced_d_yr", "completed_d_yr", "withdrawn_d_yr",
    #"assumed_comp_d",   # needed by Script 08 DaysToCompletion (fallback end date)
#]
# Convert deal_num to Int64

df_sd["deal_num"] = pd.to_numeric(df_sd["deal_num"], errors="coerce").astype("Int64")

# ── 把去重前聚合的分类字段 merge 回来 end 260907──
if _agg_all is not None:
    df_sd = df_sd.merge(_agg_all, on="deal_num", how="left")
    _mp = int(df_sd["deal_pay_method_all"].isna().sum())
    print(f"  CATEGORICAL AGG merged: {len(df_sd)-_mp:,}/{len(df_sd):,} deals 有支付方式 | "
          f"缺失 {_mp:,} ({_mp/len(df_sd):.1%})")
    for _c in [c for c in df_sd.columns if c.endswith("_all")]:
        print(f"    {_c:<28s} 覆盖 {df_sd[_c].notna().mean():.1%}")
# ── 把去重前聚合的分类字段 merge 回来 end 260907──
        
        
# Select key columns for analysis
SD_KEEP = [
    "deal_num",
    "deal_type", "deal_status", "deal_struct", "deal_fin", "deal_pay_method",
    # ── 去重前聚合的分类字段（2026-09-08）──
    "deal_pay_method_all", "deal_pay_method_n",
    "deal_struct_all",     "deal_struct_n",
    "deal_fin_all",        "deal_fin_n",
    "deal_type_all",       "deal_type_n",
    "announced_d", "completed_d", "withdrawn_d",
    "announced_d_yr", "completed_d_yr", "withdrawn_d_yr",
    "assumed_comp_d",
]
SD_KEEP_PRESENT = [c for c in SD_KEEP if c in df_sd.columns]
df_sd = df_sd[SD_KEEP_PRESENT].copy()

# m5 FIXER R1: cast year columns to Int64 (not float) to prevent float residue
for yr_col in ["announced_d_yr", "completed_d_yr", "withdrawn_d_yr"]:
    if yr_col in df_sd.columns:
        df_sd[yr_col] = pd.to_numeric(df_sd[yr_col], errors="coerce").astype("Int64")

if "announced_d_yr" in df_sd.columns and "completed_d_yr" in df_sd.columns:
    df_sd["deal_duration_yr"] = df_sd["completed_d_yr"] - df_sd["announced_d_yr"]

# M6 FIXER R1: validate deal_duration_yr after computation
if "deal_duration_yr" in df_sd.columns:
    _dur = df_sd["deal_duration_yr"].dropna()
    _n_negative = (_dur < 0).sum()
    _n_extreme = (_dur > 10).sum()
    print(f"\nM6 deal_duration_yr validation:")
    print(f"  Non-missing: {len(_dur):,} / {len(df_sd):,}")
    print(f"  Negative values (completed < announced): {_n_negative}")
    print(f"  Extreme values (>10 years): {_n_extreme}")
    if len(_dur) > 0:
        print(f"  Distribution: min={_dur.min():.0f} p5={_dur.quantile(.05):.0f} "
              f"median={_dur.median():.0f} p95={_dur.quantile(.95):.0f} max={_dur.max():.0f}")
    if _n_negative > 0:
        print(f"  WARNING: {_n_negative} deals have negative duration — likely data entry error")
        _neg_cols = [c for c in ["deal_num","announced_d_yr","completed_d_yr","deal_duration_yr"]
                     if c in df_sd.columns]
        print(df_sd[df_sd["deal_duration_yr"] < 0][_neg_cols].head(5).to_string(index=False))

print(f"\nModule C output: {len(df_sd):,} rows | "
      f"{df_sd['deal_num'].nunique():,} unique deals")
print(f"  deal_status missing: {df_sd['deal_status'].isna().sum():,}")
print(f"  completed_d_yr missing: {df_sd['completed_d_yr'].isna().sum():,} "
      f"({df_sd['completed_d_yr'].isna().mean()*100:.1f}%)")

if "unnamed_0" in df_sd.columns:
    df_sd = df_sd.drop(columns=["unnamed_0"])

out_path_sd = os.path.join(CLEANED, "01_deal_structure_date.csv")
df_sd.to_csv(out_path_sd, index=False, encoding="utf-8-sig")
print(f"\nSaved → {out_path_sd}")


# ════════════════════════════════════════════════════════════════════════════
# Module D — Value
#   Source : raw/MA_deal/value/  (2 batches, ~58,975 raw rows)
#   Output : data/cleaned/01_deal_value.csv
#   Unit   : deal_num (one row per deal)
#   Note   : deal_value filter (≥ $1M) applied in Step 5 (sample selection),
#             not here. We keep all rows including those with missing deal_value.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE D — Value")
print("=" * 60)

val_dir = os.path.join(RAW_DEAL, "value")
val_files = sorted(glob.glob(os.path.join(val_dir, "acquisition_value_*_cleaned.csv")))
print(f"Found {len(val_files)} value batch files")

batches_val = []
for fp in val_files:
    batch_name = os.path.basename(fp)
    df_b = read_and_ffill(fp)
    n_raw = len(df_b)
    df_b = clean_missing(df_b)
    batches_val.append(df_b)
    print(f"  {batch_name}: {n_raw} rows read")

df_val = pd.concat(batches_val, ignore_index=True)
print(f"\nAfter stacking: {len(df_val):,} rows")

# Drop rows with missing deal_num
n_before = len(df_val)
df_val = df_val[df_val["deal_num"].notna()].copy()
if n_before > len(df_val):
    print(f"Dropped {n_before - len(df_val):,} rows with NaN deal_num")

# F2 FIXER R1: verify duplicate rows in Module D are genuinely identical
_dup_mask_d = df_val.duplicated(subset=["deal_num"], keep=False)
if _dup_mask_d.sum() > 0:
    _dup_df_d = df_val[_dup_mask_d]
    _key_cols_d = [c for c in ["deal_value", "stake_acq_pct"] if c in _dup_df_d.columns]
    if _key_cols_d:
        _inconsistent_d = _dup_df_d.groupby("deal_num")[_key_cols_d].nunique(dropna=True)
        _n_inconsistent_d = (_inconsistent_d > 1).any(axis=1).sum()
        print(f"\nF2 CHECK Module D: {_dup_mask_d.sum()} duplicate rows across "
              f"{_dup_df_d['deal_num'].nunique()} deals")
        print(f"  Deals with DIFFERENT key values in duplicate rows: {_n_inconsistent_d}")
        if _n_inconsistent_d > 0:
            print(f"  WARNING: {_n_inconsistent_d} deals have inconsistent duplicate rows "
                  f"— 'keep first' may lose real data")
            # Print 5 sample rows to understand the structure of inconsistency
            _incon_deals = _inconsistent_d[(_inconsistent_d > 1).any(axis=1)].index[:3]
            _sample_cols = ["deal_num"] + _key_cols_d + (["currency"] if "currency" in _dup_df_d.columns else [])
            print(f"  Sample inconsistent deals (showing up to 3):")
            print(_dup_df_d[_dup_df_d["deal_num"].isin(_incon_deals)][_sample_cols].to_string(index=False))

# Dedup by deal_num — keep first
n_before_dedup = len(df_val)
df_val = df_val.drop_duplicates(subset=["deal_num"], keep="first")
print(f"After dedup by deal_num: {len(df_val):,} rows "
      f"(dropped {n_before_dedup - len(df_val):,} duplicate rows)")

# Convert deal_num to Int64
df_val["deal_num"] = pd.to_numeric(df_val["deal_num"], errors="coerce").astype("Int64")

# Select key value columns
VAL_KEEP = [
    "deal_num",
    "deal_value",                        # USD deal value (th USD)
    "deal_enterprise_value",             # EV (th USD)
    "deal_equity_value",                 # Equity value (th USD)
    "deal_modelled_enterprise_value",    # Modelled EV
    "stake_acq_pct",                     # Acquired stake % (for ≥50% filter)
    "stake_final_pct",                   # Final stake %
    "currency",
]
VAL_KEEP_PRESENT = [c for c in VAL_KEEP if c in df_val.columns]
df_val = df_val[VAL_KEEP_PRESENT].copy()

# Convert numeric columns
NUM_COLS_VAL = [c for c in VAL_KEEP_PRESENT if c not in ("deal_num", "currency")]
for col in NUM_COLS_VAL:
    df_val[col] = pd.to_numeric(df_val[col], errors="coerce")

print(f"\nModule D output: {len(df_val):,} rows | "
      f"{df_val['deal_num'].nunique():,} unique deals")
print(f"  deal_value missing: {df_val['deal_value'].isna().sum():,} "
      f"({df_val['deal_value'].isna().mean()*100:.1f}%)")
print(f"  deal_enterprise_value missing: {df_val['deal_enterprise_value'].isna().sum():,} "
      f"({df_val['deal_enterprise_value'].isna().mean()*100:.1f}%)")
print(f"  stake_acq_pct missing: {df_val['stake_acq_pct'].isna().sum():,} "
      f"({df_val['stake_acq_pct'].isna().mean()*100:.1f}%)")

# m3 FIXER R1: warn if deal_enterprise_value coverage is too low for regressions
if "deal_enterprise_value" in df_val.columns:
    _ev_coverage = df_val["deal_enterprise_value"].notna().mean() * 100
    if _ev_coverage < 20:
        print(f"  WARNING: deal_enterprise_value coverage = {_ev_coverage:.1f}% "
              f"— essentially unusable in regressions")
        print(f"  EV-based multiples (EV/EBITDA) will require a separate data source "
              f"or restricted sample")

if "unnamed_0" in df_val.columns:
    df_val = df_val.drop(columns=["unnamed_0"])

out_path_val = os.path.join(CLEANED, "01_deal_value.csv")
df_val.to_csv(out_path_val, index=False, encoding="utf-8-sig")
print(f"\nSaved → {out_path_val}")

# ════════════════════════════════════════════════════════════════════════════
# Module E — Deal Info Source Category (Deal-level 信息来源分类)
#   Source : raw/MA_deal/info_source/ 单文件 deal_info_source_categories.csv
#   Output : data/cleaned/01_deal_info_source.csv
#   Unit   : 每条信息来源一条记录 (一个deal_num对应多行不同source)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODULE F — Deal Info Source Category")
print("=" * 60)

info_src_dir = os.path.join(BASE, "raw", "MA_deal", "info_source")
info_src_fp = os.path.join(info_src_dir, "deal_info_source_categories.csv")
print(f"Reading single source file: {os.path.basename(info_src_fp)}")

# 1 先原生读取，不再提前调用read_and_ffill（原始列是dealnumber，无deal_num）
df_src = pd.read_csv(info_src_fp, encoding="utf-8-sig", low_memory=False)
df_src.columns = df_src.columns.str.strip()
n_raw = len(df_src)
print(f"Raw total rows: {n_raw:,}")

# 2 【关键修复】先重命名字段，把dealnumber改为deal_num
df_src = df_src.rename(columns={
    "dealnumber": "deal_num",
    "categoryofsource": "source_category",
    "sourcedocumentation": "source_document",
    "targetname": "tar_name",
    "targetbvdidnumber": "tar_bvd_id_num",
    "targetorbisidnumber": "tar_orbis_id_num",
    "acquirorname": "acq_name",
    "acquirorbisidnumber": "acq_orbis_id_num",
    "acquirorbvdidnumber": "acq_bvd_id_num",
    "vendorname": "ven_name",
    "vendorbvdidnumber": "ven_bvdidnumber",
    "index": "src_index"
})

# 3 现在才有deal_num，执行前向填充处理拆分行
df_src["deal_num"] = df_src["deal_num"].ffill()

# 清洗缺失占位符
df_src = clean_missing(df_src)

# var1纯索引无意义，直接删除
drop_cols = ["var1"]
df_src = df_src.drop(columns=drop_cols, errors="ignore")

# 过滤deal_num为空无效行
n_before_filter = len(df_src)
df_src = df_src[df_src["deal_num"].notna()].copy()
print(f"Dropped rows with empty deal_num: {n_before_filter - len(df_src):,}")

# deal_num统一转为可空整数
df_src["deal_num"] = pd.to_numeric(df_src["deal_num"], errors="coerce").astype("Int64")

# ---------------------- 清洗 source_category：逐行剥离括号后缀 ----------------------
import re

def extract_main_cat(text):
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    m = re.match(r"(.*?)\s*\(", s)
    if m:
        return m.group(1).strip()
    return s

# 逐行处理原始每行，不做任何文本合并
df_src["source_main_cat"] = df_src["source_category"].apply(extract_main_cat)

main_source_list = [
    "Stock Exchange",
    "Website",
    "Company Press Release",
    "Electronic Publication",
    "Advisor Submission",
    "Miscellaneous"
]

# ========== 1. 行级 dummy（仅用于中间计算，不单独输出） ==========
for src in main_source_list:
    col = "num_" + src.replace(" ", "_")
    df_src[col] = (df_src["source_main_cat"] == src).astype(int)

df_src["num_source_other"] = (~df_src["source_main_cat"].isin(main_source_list)).astype(int)

dummy_cols = [c for c in df_src.columns if c.startswith("num_")]

# ========== 2. 按 deal_num 求和：一个 deal 该来源出现几次，值就是几 ==========
deal_source_df = df_src.groupby("deal_num")[dummy_cols].sum().reset_index()

# 输出唯一一张 deal 级表（值为计数：0,1,2,3,4...）
out_path_src = os.path.join(CLEANED, "01_deal_info_source_count.csv")
deal_source_df.to_csv(out_path_src, index=False, encoding="utf-8-sig")
print(f"\nSaved deal-level source count table → {out_path_src}")

# 校验：每个 deal 的来源种类数分布（非 0 的列数）
deal_source_df["n_source_types"] = (deal_source_df[dummy_cols] > 0).sum(axis=1)
print("\n==== Deal-level: number of distinct source types per deal ====")
print(deal_source_df["n_source_types"].value_counts().sort_index())

# ========== 主类别统计（基于明细行） ==========
total_source_records = len(df_src)
src_main_stat = df_src["source_main_cat"].value_counts().reset_index()
src_main_stat.columns = ["source_main_cat", "count"]
src_main_stat["pct"] = (src_main_stat["count"] / total_source_records) * 100

print(f"\n==== Cleaned Main Source Category Statistics (Total: {total_source_records:,}) ====")
for _, row in src_main_stat.iterrows():
    cat = row["source_main_cat"]
    cnt = row["count"]
    pct = row["pct"]
    print(f"{cat:<26} {cnt:>8}    {pct:.8f}")
# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Script 01 Complete")
print("=" * 60)
for label, path in [
    ("01_deal_sic_industry.csv",   out_path_ind),
    ("01_deal_multiples.csv",      out_path_mul),
    ("01_deal_structure_date.csv", out_path_sd),
    ("01_deal_value.csv",          out_path_val),
    ("01_deal_info_source_count.csv",    out_path_src),
]:
    size_kb = os.path.getsize(path) / 1024
    print(f"  {label:<30s} {size_kb:>8.0f} KB")

print("\nAll cleaned files saved to data/cleaned/")
