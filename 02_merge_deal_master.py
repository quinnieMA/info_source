"""
02_merge_deal_master.py
=======================
Merge all standardized cleaned module CSV files into one unified master dataset
at deal-target observation level.

Input Cleaned Modules (data/cleaned/):
1. Base table: 01_deal_sic_industry.csv (Unit: deal_num + tar_key, core firm & SIC identifiers)
2. Deal-level auxiliary tables left-joined & broadcast to every target row:
   - 01_deal_multiples.csv: valuation multiples
   - 01_deal_structure_date.csv: transaction status & time variables
   - 01_deal_value.csv: consideration value & acquired ownership share
   - 01b_deal_overview.csv: country tags, STRUCTURED regulatory metadata & brief deal text
   - 01_deal_info_source_count.csv: disclosure source count dummy indicators
   - 01c_comments_features.csv: Loughran-McDonald sentiment & negotiation friction metrics

──────────────────────────────────────────────────────────────────────────────
⚠️ TWO DISTINCT REGULATORY MEASURES — DO NOT CONFUSE (2026-09-06)
──────────────────────────────────────────────────────────────────────────────
  (A) STRUCTURED — reg_* from MODULE 01b.reg (01b_deal_overview.csv)  ← PRIMARY
      reg_antitrust, reg_securities, reg_state_assets, reg_foreign_invest,
      reg_financial, reg_defense_tech, reg_body_count, reg_country_count,
      reg_cross_national, reg_common_law, reg_civil_law, reg_mixed_legal,
      regulatory_bodies, regulatory_countries

      Derived from `regulatory_body_name` — the NAME OF THE REGULATORY
      AUTHORITY recorded in the raw overview files. NOT from deal comments.
      Built by exhaustively enumerating all 443 distinct authority names,
      then resolving each via a three-tier scheme in 01b:
        Tier 1  REG_EXACT_MAP   explicit body-name → category (highest priority)
        Tier 2  REG_EXCLUDE     exchanges / courts → no category
        Tier 3  keyword         substring fallback (short acronyms need
                                whole-word match)
      reg_* dummies are deal-level ORs across ALL bodies on that deal and are
      NOT mutually exclusive (a combined supervisor sets several).

  (B) TEXT-DERIVED — from (01c_comments_features.csv)  ← SECONDARY
      reg_event_count, num_unique_reg

      Counted from comment TEXT via keyword scanning. Coverage is low and
      partially conflates mention-frequency with review intensity.

  Empirical work uses (A). (B) may serve as a robustness / alternative
  measure only. Any docstring or table note claiming reg_* comes from
  comments is WRONG — see 08b docstring, corrected 2026-09-06.


Output Files:
    data/merged/02_deal_master.csv        Full integrated master panel
    data/merged/02_deal_master_diagnostics.txt QC log with coverage & cross-source consistency statistics

Processing Notes:
  - No sample screening filters implemented here. Sample restrictions (year range, stake threshold, deal value floor) are applied in the subsequent empirical design script Step 5.
  - Duplicate identifier fields (tar_name, acq_name, deal_status, deal_value) exist across separate raw modules for cross-source validation. Duplicated columns are suffixed with _ovw/_mul/_sd/_val and compared in diagnostic logs to quantify data inconsistency rates.
  - 01c_comments_features only retains numerically derived sentiment & timeline features; raw unstructured long text columns are discarded before merging to control output file size.
  - All left joins use the base industry table as anchor; transactions without matching target SIC records are permanently dropped.
  - Country codes carried in from MODULE 01b.overview_country are RAW and UNSCREENED. They are
    not the analysis-sample composition (China is heavily over-represented
    before sample restrictions). Do not quote MODULE 01b.overview_country's country
    distribution as the sample composition — see table1 Panel E instead.

──────────────────────────────────────────────────────────────────────────────
REVISED 2026-09-18 (vendor identity columns)
──────────────────────────────────────────────────────────────────────────────
  [1] ★ STEP 5c 的筛列前缀新增 "ven_has_", 并单独纳入 ven_disclosure
      旧版只认 ven_type / ven_ner / ven_n_names / ven_is_ 四个前缀,
      导致 01d v4 新增的两列被静默丢弃:
        ven_has_mbo      (MBO / 管理层参与标记, overlapping, 2,175 笔)
        ven_disclosure   (披露维度: named / generic / undisclosed / no_record)
      这两列永远进不了 02, 也就进不了 08b 的 dta。
  [2] 合并后新增诊断: 打印 ven_disclosure 与 ven_has_mbo 的分布,
      便于一眼确认新列已带入。
  [3] 原文件有两个同名 "STEP 5c" (vendor identity 与 source dummy),
      诊断日志无法区分。后者标题改为 "STEP 5c-src"。

  注: ven_is_* dummy 由 08b 补丁 B 生成 (不在 01d), 此处的 "ven_is_"
      前缀保留仅为兼容, 当前匹配不到任何列, 无害。

Author: Q Date: 2026-08-08
Revised 2026-09-06: documented the two distinct regulatory measures and the
    MODULE 01b.overview_country composite merge key.
Revised 2026-09-18: vendor-identity column whitelist + new-column diagnostics.
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = r"D:\MA"
CLEANED = os.path.join(BASE, "data", "cleaned")
MERGED  = os.path.join(BASE, "data", "merged")
os.makedirs(MERGED, exist_ok=True)

diag_lines = []   # collect diagnostics for the text report

def log(msg=""):
    print(msg)
    diag_lines.append(msg)

# Orbis IDs are zero-padded 9-digit strings; force str dtype to prevent
# pandas from reading them as float (which would strip leading zeros).
ORBIS_DTYPE = {
    "tar_orbis_id_num": str,
    "acq_orbis_id_num": str,
    "ven_orbis_id_num": str,
}

# ════════════════════════════════════════════════════════════════════════════
# 1. Load base dataset
# ════════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("STEP 1 — Load base dataset (01_deal_sic_industry)")
log("=" * 60)

df_base = pd.read_csv(os.path.join(CLEANED, "01_deal_sic_industry.csv"),
                      encoding="utf-8-sig", low_memory=False, dtype=ORBIS_DTYPE)
# Ensure deal_num is Int64
df_base["deal_num"] = pd.to_numeric(df_base["deal_num"], errors="coerce").astype("Int64")
log(f"Base rows         : {len(df_base):,}")
log(f"Unique deal_num   : {df_base['deal_num'].nunique():,}")
log(f"Unique tar_bvd_id : {df_base['tar_bvd_id_num'].nunique():,}")


# ════════════════════════════════════════════════════════════════════════════
# 2. Merge multiples
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 2 — Merge deal_multiples (deal-level → broadcast to all targets)")
log("=" * 60)

df_mul = pd.read_csv(os.path.join(CLEANED, "01_deal_multiples.csv"),
                     encoding="utf-8-sig", low_memory=False)
df_mul["deal_num"] = pd.to_numeric(df_mul["deal_num"], errors="coerce").astype("Int64")
log(f"Multiples rows    : {len(df_mul):,} | unique deals: {df_mul['deal_num'].nunique():,}")

# F2/M4 FIXER R1: deals in Module B not in Module A base (dropped by left join)
# Use int() cast to avoid pandas Int64 vs numpy int64 set-comparison mismatch
_a_set = set(df_base["deal_num"].dropna().astype(int).tolist())
_missing_from_a_b = set(df_mul["deal_num"].dropna().astype(int).tolist()) - _a_set
log(f"F2/M4: {len(_missing_from_a_b)} deal_nums in Module B not in Module A "
    f"(will be dropped by left join on A)")
if 0 < len(_missing_from_a_b) <= 20:
    log(f"  deal_nums: {sorted(_missing_from_a_b)}")

df = df_base.merge(df_mul, on="deal_num", how="left", suffixes=("", "_mul"))
n_matched = df["pre_rev_mul_ly"].notna().sum()
log(f"After merge: {len(df):,} rows | deals with ≥1 multiple: {n_matched:,} "
    f"({n_matched/len(df)*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════
# 3. Merge structure & dates
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 3 — Merge deal_structure_date")
log("=" * 60)

df_sd = pd.read_csv(os.path.join(CLEANED, "01_deal_structure_date.csv"),
                    encoding="utf-8-sig", low_memory=False)
df_sd["deal_num"] = pd.to_numeric(df_sd["deal_num"], errors="coerce").astype("Int64")
# m5: re-cast year columns to Int64 after CSV round-trip (CSV read restores float)
for _yr_col in ["announced_d_yr", "completed_d_yr", "withdrawn_d_yr"]:
    if _yr_col in df_sd.columns:
        df_sd[_yr_col] = pd.to_numeric(df_sd[_yr_col], errors="coerce").astype("Int64")
log(f"Structure rows    : {len(df_sd):,} | unique deals: {df_sd['deal_num'].nunique():,}")

# F2/M4 FIXER R1: deals in Module C not in Module A base
_missing_from_a_c = set(df_sd["deal_num"].dropna().astype(int).tolist()) - _a_set
log(f"F2/M4: {len(_missing_from_a_c)} deal_nums in Module C not in Module A "
    f"(will be dropped by left join on A)")
if 0 < len(_missing_from_a_c) <= 20:
    log(f"  deal_nums: {sorted(_missing_from_a_c)}")

df = df.merge(df_sd, on="deal_num", how="left", suffixes=("", "_sd"))
n_status = df["deal_status"].notna().sum()
log(f"After merge: {len(df):,} rows | rows with deal_status: {n_status:,}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Merge value
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 4 — Merge deal_value")
log("=" * 60)

df_val = pd.read_csv(os.path.join(CLEANED, "01_deal_value.csv"),
                     encoding="utf-8-sig", low_memory=False)
df_val["deal_num"] = pd.to_numeric(df_val["deal_num"], errors="coerce").astype("Int64")
log(f"Value rows        : {len(df_val):,} | unique deals: {df_val['deal_num'].nunique():,}")

# F2/M4 FIXER R1: deals in Module D not in Module A base
_missing_from_a_d = set(df_val["deal_num"].dropna().astype(int).tolist()) - _a_set
log(f"F2/M4: {len(_missing_from_a_d)} deal_nums in Module D not in Module A "
    f"(will be dropped by left join on A)")
if 0 < len(_missing_from_a_d) <= 20:
    log(f"  deal_nums: {sorted(_missing_from_a_d)}")

df = df.merge(df_val, on="deal_num", how="left", suffixes=("", "_val"))
n_ev = df["deal_enterprise_value"].notna().sum()
log(f"After merge: {len(df):,} rows | rows with deal EV: {n_ev:,}")


# ════════════════════════════════════════════════════════════════════════════
# 5. Merge overview + country data (Module E)
#    Adds: tar_country_code, acq_country_code, deal_headline, tar_busi_descr
#    Also provides: deal_status_ovw, deal_value_ovw  (for QC against master)
#    Merge key: (deal_num, tar_key) where tar_key = tar_bvd_id_num or tar_name
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5 — Merge deal_overview_country (Module E)")
log("=" * 60)

df_ovw = pd.read_csv(os.path.join(CLEANED, "01b_deal_overview.csv"),
                     encoding="utf-8-sig", low_memory=False, dtype=ORBIS_DTYPE)
df_ovw["deal_num"] = pd.to_numeric(df_ovw["deal_num"], errors="coerce").astype("Int64")
log(f"Overview rows     : {len(df_ovw):,} | unique deals: {df_ovw['deal_num'].nunique():,}")

# F2/M4 FIXER R1: deals in Module E not in Module A base
_missing_from_a_e = set(df_ovw["deal_num"].dropna().astype(int).tolist()) - _a_set
log(f"F2/M4: {len(_missing_from_a_e)} deal_nums in Module E not in Module A "
    f"(will be dropped by left join on A)")
if 0 < len(_missing_from_a_e) <= 20:
    log(f"  deal_nums: {sorted(_missing_from_a_e)}")

# Rename overview identifiers to avoid collision (we QC-compare them after merge)
rename_ovw = {}
for col in ["deal_status", "tar_name", "acq_name",
            "tar_bvd_id_num", "tar_orbis_id_num",
            "acq_bvd_id_num", "acq_orbis_id_num",
            "ven_bvd_id_num",
            "tar_busi_descr", "ven_name"]:
    if col in df_ovw.columns:
        rename_ovw[col] = col + "_ovw"
df_ovw = df_ovw.rename(columns=rename_ovw)

# Build composite merge key in both datasets
df["_tar_key"]     = df["tar_bvd_id_num"].fillna(df["tar_name"])
df_ovw["_tar_key"] = df_ovw.get("tar_bvd_id_num_ovw", pd.Series(dtype=str)).fillna(
                         df_ovw.get("tar_name_ovw", pd.Series(dtype=str)))

df = df.merge(df_ovw, on=["deal_num", "_tar_key"], how="left")
df = df.drop(columns=["_tar_key"])

n_country = df["tar_country_code"].notna().sum() if "tar_country_code" in df.columns else 0
log(f"After merge: {len(df):,} rows | rows with tar_country_code: {n_country:,} "
    f"({n_country/len(df)*100:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# 5c. Merge vendor / seller identity (01a Module G) —— deal-level, left join
#     粒度说明: 01_deal_vendor_type.csv 是 deal 级; 基表是 deal × target 级。
#     多目标交易会把同一 vendor 属性广播到该 deal 的每个 target 行 —— 符合预期。
#     03_clean_firm_modules 里的 "DROP all ven_*" 只作用于 firm 模块内部,
#     与本步骤注入的 deal 级 vendor 列无关, 不受影响。
#
#     ⚠️ 2026-09-18 修订: 白名单新增 "ven_has_" 前缀与 ven_disclosure。
#        01d v4 新增 ven_has_mbo / ven_disclosure 两列, 旧白名单只认四个前缀
#        (ven_type / ven_ner / ven_n_names / ven_is_), 这两列会被静默丢弃。
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5c — Merge 01_deal_vendor_type.csv (seller identity, 01a Module G)")
log("=" * 60)

_VENDOR_FP = os.path.join(CLEANED, "01_deal_vendor_type.csv")
if os.path.exists(_VENDOR_FP):
    _dfv = pd.read_csv(_VENDOR_FP, encoding="utf-8-sig", low_memory=False)
    _dfv["deal_num"] = pd.to_numeric(_dfv["deal_num"], errors="coerce").astype("Int64")
    _dfv = _dfv.drop_duplicates(subset=["deal_num"], keep="first")
    log(f"Vendor rows: {len(_dfv):,} | unique deals: {_dfv['deal_num'].nunique():,}")

    # ── 白名单: 只带入分类相关列 (不引入 ven_name_all 等长文本, 控制文件体积) ──
    #    v4 修订: 新增 "ven_has_" 前缀 + 显式纳入 ven_disclosure
    _vcols = ["deal_num"] + [
        c for c in _dfv.columns
        if c.startswith(("ven_type", "ven_ner", "ven_n_names",
                         "ven_is_", "ven_has_"))
        or c == "ven_disclosure"
    ]
    _vcols = [c for c in _vcols if c in _dfv.columns]
    log(f"  带入列 ({len(_vcols)}): {_vcols}")

    # 防冲突: 基表若已有同名列, 先丢弃再用 01a 最新版
    _dup = [c for c in _vcols if c != "deal_num" and c in df.columns]
    if _dup:
        log(f"  [warn] 基表已存在同名列, 将被覆盖: {_dup}")
        df = df.drop(columns=_dup)

    _before = len(df)
    df = df.merge(_dfv[_vcols], on="deal_num", how="left")
    if len(df) != _before:
        raise AssertionError(
            f"vendor merge 改变了行数: {_before:,} -> {len(df):,} (应为 deal 级左连接, 不增行)"
        )

    if "ven_type_primary" in df.columns:
        _n = df["ven_type_primary"].notna().sum()
        log(f"\n  ven_type_primary 非空: {_n:,} / {len(df):,} ({_n/len(df)*100:.1f}%)")
        log("  分布:")
        for _k, _c in df["ven_type_primary"].value_counts().head(12).items():
            log(f"    {str(_k):<30s} {_c:>7,}")
    else:
        log("  [warn] ven_type_primary 未出现在合并结果中, 检查 01a Module G 是否已运行")

    # ── v4 新增诊断: 确认新列已带入 ──
    if "ven_disclosure" in df.columns:
        log("\n  ven_disclosure 分布:")
        for _k, _c in df["ven_disclosure"].value_counts(dropna=False).items():
            log(f"    {str(_k):<30s} {_c:>7,}")
    else:
        log("  [warn] ven_disclosure 未带入 —— 检查 01d 是否已重跑")

    if "ven_has_mbo" in df.columns:
        _m = pd.to_numeric(df["ven_has_mbo"], errors="coerce")
        log(f"\n  ven_has_mbo = 1: {int(_m.fillna(0).sum()):,} "
            f"({_m.fillna(0).mean()*100:.1f}%)")
    else:
        log("  [warn] ven_has_mbo 未带入 —— 检查 01d 是否已重跑")
else:
    log(f"  [skip] 未找到 {_VENDOR_FP} —— 请先运行 01a (Module G)")

# ════════════════════════════════════════════════════════════════════════════
# 5b. Duplicate variable QC check
#    Compare deal_status and deal_value from overview vs master (structure_date
#    and value files). These appear in multiple source files by design.
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5b — Duplicate variable QC check")
log("=" * 60)

def qc_compare(df, col_a, col_b, label_a, label_b):
    """Compare two columns, report mismatch rate and sample rows."""
    mask = df[col_a].notna() & df[col_b].notna()
    n_comparable = mask.sum()
    if n_comparable == 0:
        log(f"  {label_a} vs {label_b}: no overlapping non-null values")
        return
    match = (df.loc[mask, col_a].astype(str).str.strip() ==
             df.loc[mask, col_b].astype(str).str.strip())
    n_mismatch = (~match).sum()
    pct = n_mismatch / n_comparable * 100
    log(f"  {label_a:<25s} vs {label_b:<25s}: "
        f"{n_mismatch:,} / {n_comparable:,} mismatches ({pct:.2f}%)")
    if n_mismatch > 0 and n_mismatch <= 200:
        sample = df.loc[mask & ~match, ["deal_num", col_a, col_b]].head(5)
        log(f"    Sample:\n{sample.to_string(index=False)}")

# deal_status: structure_date vs overview
if "deal_status" in df.columns and "deal_status_ovw" in df.columns:
    qc_compare(df, "deal_status", "deal_status_ovw", "deal_status(sd)", "deal_status(ovw)")

# deal_value: value file vs overview
if "deal_value" in df.columns and "deal_value_ovw" in df.columns:
    # Numeric comparison with 1% tolerance
    mask = df["deal_value"].notna() & df["deal_value_ovw"].notna()
    n_comp = mask.sum()
    if n_comp > 0:
        ratio = df.loc[mask, "deal_value"] / df.loc[mask, "deal_value_ovw"]
        n_mismatch = ((ratio < 0.99) | (ratio > 1.01)).sum()
        pct = n_mismatch / n_comp * 100
        log(f"  deal_value(val)           vs deal_value(ovw)          : "
            f"{n_mismatch:,} / {n_comp:,} differ by >1% ({pct:.2f}%)")

# tar_name: industry vs overview
if "tar_name" in df.columns and "tar_name_ovw" in df.columns:
    qc_compare(df, "tar_name", "tar_name_ovw", "tar_name(ind)", "tar_name(ovw)")

# acq_name: industry vs overview
if "acq_name" in df.columns and "acq_name_ovw" in df.columns:
    qc_compare(df, "acq_name", "acq_name_ovw", "acq_name(ind)", "acq_name(ovw)")

# M5 FIXER R1: for tar_name mismatches, check if tar_bvd_id_num is also NaN
if "tar_name" in df.columns and "tar_name_ovw" in df.columns:
    _mask_m5 = df["tar_name"].notna() & df["tar_name_ovw"].notna()
    _name_mismatch = _mask_m5 & (
        df["tar_name"].astype(str).str.strip() != df["tar_name_ovw"].astype(str).str.strip()
    )
    if _name_mismatch.sum() > 0:
        _mismatch_df = df[_name_mismatch].copy()
        if "tar_bvd_id_num" in _mismatch_df.columns:
            _n_bvd_null = _mismatch_df["tar_bvd_id_num"].isna().sum()
            log(f"  M5: Of {_name_mismatch.sum()} name-mismatch deals, "
                f"{_n_bvd_null} also have NaN tar_bvd_id_num")
            log(f"  -> These {_n_bvd_null} deals used tar_name as merge key "
                f"and may have failed to match Module E")
            if _n_bvd_null > 0:
                _show_cols = [c for c in ["deal_num", "tar_name", "tar_name_ovw",
                                          "tar_country_code"] if c in _mismatch_df.columns]
                log(_mismatch_df[_mismatch_df["tar_bvd_id_num"].isna()][_show_cols]
                    .to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# 5c-src. Merge deal‑level source count dummy (Module E info‑source)
#   (原文件名里也叫 STEP 5c, 与上面的 vendor identity 重名;
#    2026-09-18 改标题为 5c-src, 避免诊断日志混淆)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5c-src — Merge deal source count dummy (01_deal_source_dummy)")
log("=" * 60)

df_src_dummy = pd.read_csv(os.path.join(CLEANED, "01_deal_info_source_count.csv"),
                           encoding="utf-8-sig", low_memory=False)
df_src_dummy["deal_num"] = pd.to_numeric(df_src_dummy["deal_num"], errors="coerce").astype("Int64")

log(f"Source dummy rows : {len(df_src_dummy):,} | unique deals: {df_src_dummy['deal_num'].nunique():,}")

# F2/M4 FIXER：统计master里面有，但是source_dummy没有覆盖到的deal
_deal_master_set = set(df["deal_num"].dropna().astype(int).tolist())
_deal_src_set = set(df_src_dummy["deal_num"].dropna().astype(int).tolist())
_no_source_deals = _deal_master_set - _deal_src_set
log(f"F2/M4: {len(_no_source_deals)} deal_nums in master without source‑info record (will fill NA)")
if 0 < len(_no_source_deals) <=20:
    log(f"  sample deal_num: {sorted(_no_source_deals)}")

# left join，按deal_num广播到所有target行
df = df.merge(df_src_dummy, on="deal_num", how="left")

# 没有来源记录的deal全部填充0（没有该类来源）
src_dummy_cols = [c for c in df_src_dummy.columns if c.startswith("num_")]
#df[src_dummy_cols] = df[src_dummy_cols].fillna(0)
df["has_source_record"] = df[src_dummy_cols].notna().any(axis=1).astype("Int64")
# 保留 NaN，不 fillna(0)

# QC：统计各个来源非零样本数量
log("\nSource‑count variable QC (non‑zero count):")
for col in src_dummy_cols:
    non_zero = (df[col] > 0).sum()
    pct_nonzero = non_zero / len(df) *100
    log(f"  {col:<35s}: {non_zero:>7,} non‑zero ({pct_nonzero:5.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# 5d. Merge deal comments text features (Module F)
# One-to-one match on deal_num, drop raw long text columns to shrink file size
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5d — Merge deal comments text features (comments_features)")
log("=" * 60)

df_cmt = pd.read_csv(os.path.join(CLEANED, "01c_comments_features.csv"),
                     encoding="utf-8-sig", low_memory=False)
df_cmt["deal_num"] = pd.to_numeric(df_cmt["deal_num"], errors="coerce").astype("Int64")

# F2/M4统计无匹配交易
_deal_master_set = set(df["deal_num"].dropna().astype(int).tolist())
_deal_cmt_set = set(df_cmt["deal_num"].dropna().astype(int))
_no_cmt_deals = _deal_master_set - _deal_cmt_set
log(f"F2/M4: {len(_no_cmt_deals)} deal_nums in master without comment record (fill NA)")
if 0 < len(_no_cmt_deals) <= 20:
    log(f"  sample deal_num: {sorted(_no_cmt_deals)}")

# 删除原始超长文本字段（comments/editorial/edit_date 全部丢弃，不进主数据集）
_drop_cmt_cols = ["comments", "editorial", "edit_date", "reg_entity_list", "dt_first", "dt_last"]
_existing_drop = [c for c in _drop_cmt_cols if c in df_cmt.columns]
if len(_existing_drop) > 0:
    df_cmt = df_cmt.drop(columns=_existing_drop)
    log(f"Dropped raw heavy text columns before merge: {_existing_drop}")

# 一对一左连接
df = df.merge(df_cmt, on="deal_num", how="left")

# 全量LM+交易特征覆盖率诊断
log("\nComments-feature non-null coverage:")
CMT_KEY_VARS = [
    # 时序 / 里程碑
    "total_timeline_days", "has_rumour", "has_target_reject",
    "has_unconditional_offer", "has_deal_complete",
    # 价格 / 竞标
    "price_event_num", "max_premium_pct", "avg_premium_pct", "rival_bidder_num",
    # 条款 / 救济 / 融资
    "has_goshop", "has_reg_remedy", "has_debt_assumption",
    # 监管（文本口径，覆盖率低；主口径用 01b 的 reg_*）
    "reg_event_count", "num_unique_reg",
    # 文本长度（回归控制用）
    "comment_wordcount", "log_comment_wordcount", "comment_char_length",
    # LM 计数
    "lm_pos_count", "lm_neg_count", "lm_uncertain_count", "lm_litigious_count",
    "lm_strongmodal_count", "lm_weakmodal_count", "lm_constrain_count",
    # LM 二值 presence（主口径）
    "has_lm_pos", "has_lm_neg", "has_lm_uncertain", "has_lm_litigious",
    "has_lm_strongmodal", "has_lm_weakmodal", "has_lm_constrain",
]
for v in CMT_KEY_VARS:
    if v in df.columns:
        non_null = df[v].notna().sum()
        pct = non_null / len(df) * 100
        log(f"  {v:<30s}: {non_null:>7,} ({pct:5.1f}%)")
    else:
        log(f"  {v:<30s}: missing column")

# ════════════════════════════════════════════════════════════════════════════
# 6. Diagnostic report
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 6 — Diagnostic report")
log("=" * 60)

log(f"\nFINAL MASTER DATASET")
log(f"  Total rows           : {len(df):,}")
log(f"  Unique deal_num      : {df['deal_num'].nunique():,}")
log(f"  Unique tar_bvd_id_num: {df['tar_bvd_id_num'].nunique():,}")

# M4 FIXER R1: document total attrition from Module A base
log(f"\nM4 ATTRITION SUMMARY:")
log(f"  Module A (base) unique deals: {df_base['deal_num'].nunique():,}")
log(f"  Module B unique deals not in A: {len(_missing_from_a_b):,} (dropped by left join on A)")
log(f"  Module C unique deals not in A: {len(_missing_from_a_c):,} (dropped by left join on A)")
log(f"  Module D unique deals not in A: {len(_missing_from_a_d):,} (dropped by left join on A)")
log(f"  Module E unique deals not in A: {len(_missing_from_a_e):,} (dropped by left join on A)")
log(f"  These deals have multiples/dates/values but no SIC code (not in Module A)")

log("\nKey variable missing rates:")
KEY_VARS = [
    "tar_overview", "tar_trade_descr_en",
    "tar_primary_sic_code", "acq_primary_sic_code",
    "pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly",
    "deal_status", "completed_d_yr", "announced_d_yr",
    "deal_value", "deal_enterprise_value", "stake_acq_pct",
    "tar_country_code", "acq_country_code",
    "deal_type",                            # M3 FIX R2: needed for Step 5 filter
    "type_of_deal_opportunity",
]
for v in KEY_VARS:
    if v in df.columns:
        n_miss = df[v].isna().sum()
        pct = n_miss / len(df) * 100
        log(f"  {v:<35s}: {n_miss:>7,} missing ({pct:5.1f}%)")
    else:
        log(f"  {v:<35s}: [column not present]")

if "deal_status" in df.columns:
    log("\ndeal_status distribution:")
    status_counts = df["deal_status"].value_counts(dropna=False)
    for val, cnt in status_counts.items():
        log(f"  {str(val):<40s}: {cnt:,}")

if "completed_d_yr" in df.columns:
    log("\ncompleted_d_yr distribution (top 15 years):")
    yr_counts = df["completed_d_yr"].value_counts(dropna=False).sort_index()
    for yr, cnt in yr_counts.items():
        log(f"  {str(yr):<10s}: {cnt:,}")

# Cross-industry preview (tar_sic3 ≠ acq_sic3)
if "tar_primary_sic_code" in df.columns and "acq_primary_sic_code" in df.columns:
    log("\nCross-industry indicator preview:")
    df["tar_sic3"] = (df["tar_primary_sic_code"] // 10).astype("Int64")
    df["acq_sic3"] = (df["acq_primary_sic_code"] // 10).astype("Int64")
    mask_both = df["tar_sic3"].notna() & df["acq_sic3"].notna()
    n_both = mask_both.sum()
    n_cross = (df.loc[mask_both, "tar_sic3"] != df.loc[mask_both, "acq_sic3"]).sum()
    log(f"  Deals with both tar+acq SIC3: {n_both:,}")
    log(f"  CrossInd=1 (tar_sic3 ≠ acq_sic3): {n_cross:,} ({n_cross/n_both*100:.1f}%)")
    log(f"  CrossInd=0 (same-industry)         : {n_both-n_cross:,} ({(n_both-n_cross)/n_both*100:.1f}%)")

# QC: count non-null for key derived features
log("\nComments-feature QC (non-null count):")
CMT_KEY_VARS = [ #此处有重复定义，不影响结果，待处理
    "total_timeline_days", "has_rumour", "has_target_reject",
    "has_phase2_investigation", "num_unique_reg", "has_goshop",
    "deal_complexity_score",
    "lm_pos_density", "lm_neg_density", "lm_uncertain_density",
    "lm_litigious_density",
    "lm_strongmodal_density", "lm_weakmodal_density", "lm_constrain_density",
    "lm_net_sentiment",
    "comment_wordcount",
]
for v in CMT_KEY_VARS:
    if v in df.columns:
        non_null = df[v].notna().sum()
        pct = non_null / len(df) * 100
        log(f"  {v:<30s}: {non_null:>7,} non-null ({pct:5.1f}%)")
    else:
        log(f"  {v:<30s}: [column not present]")

# ════════════════════════════════════════════════════════════════════════════
# 7. Save outputs
# ════════════════════════════════════════════════════════════════════════════
# Drop the temporary SIC3 columns before saving (re-computed in later scripts)
if "tar_country_code" in df.columns:
    log("\nTop 15 target countries (deal × target rows):")
    ctry = df["tar_country_code"].value_counts(dropna=False).head(15)
    for c, n in ctry.items():
        log(f"  {str(c):<6s}: {n:,}")

if "tar_sic3" in df.columns:
    df = df.drop(columns=["tar_sic3", "acq_sic3"])

out_master = os.path.join(MERGED, "02_deal_master.csv")
df.to_csv(out_master, index=False, encoding="utf-8-sig")
log(f"\nSaved → {out_master}")
log(f"File size: {os.path.getsize(out_master)/1024/1024:.1f} MB")

diag_path = os.path.join(MERGED, "02_deal_master_diagnostics.txt")
with open(diag_path, "w", encoding="utf-8") as f:
    f.write("\n".join(diag_lines))
log(f"\nDiagnostics saved → {diag_path}")
log("\nScript 02 complete.")
