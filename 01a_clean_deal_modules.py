# -*- coding: utf-8 -*-
r"""
01a_clean_deal_modules.py
========================
2026-09-17:
  [0] 2026-09-18: Vendor identity (Module vendorNER, G-0..G-5) is SUPERSEDED
      by 01d_vendor_identity.py (v4: NER cache + graceful no-Java fallback).
      Module kept here for run_all compatibility; 01d re-writes the same
      output files (01_deal_vendor_type.csv, 01_deal_vendor_type_profile.csv).
  [1] Entity identification switched to Stanford CoreNLP (mirrors the user's
      existing pipeline)
      - JAVA: C:\Program Files\Java\jdk1.8.0_202\bin
      - CoreNLP: D:/OneDrive/NLP/stanford-corenlp-4.5.7
      - import / start failure => raise; do NOT silently fall back to
        hand-written rules
  [2] Discriminator words live in an EXTERNAL CSV: data/vendor_lexicon.csv
      - default version generated on first run; hand-editable, git-versioned
      - classification priority lives in code (reproducible); the words live
        in the CSV (auditable)
  [3] Placeholders become explicit classes (no more "other"):
      SHAREHOLDER -> shareholders_undisclosed (unnamed shareholders, worst
                      information environment)
      RECEIVER    -> insolvency (bankruptcy receivership)
      MANAGEMENT  -> management
  [4] Unclassifiable names => UNMAPPED; a frequency list is written out,
      never silently accepted
"""
import os
import glob
import re
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
BASE      = r"D:\MA"
RAW_DEAL  = os.path.join(BASE, "raw", "MA_deal")
CLEANED   = os.path.join(BASE, "data", "cleaned")
os.makedirs(CLEANED, exist_ok=True)
MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]
def read_and_ffill(filepath, encoding="utf-8-sig"):
    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
    df.columns = df.columns.str.strip()
    df["deal_num"] = df["deal_num"].ffill()
    return df
def clean_missing(df):
    try:
        cols = df.select_dtypes(include=["object", "str"]).columns
    except TypeError:
        cols = df.select_dtypes(include=["object"]).columns
    for col in cols:
        df[col] = df[col].astype("object")
        df[col] = df[col].replace(MISSING_VALS, np.nan)
        mask = df[col].notna()
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
            df.loc[mask, col] = df.loc[mask, col].replace("", np.nan)
    return df
def normalize_orbis_id(series):
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
# ══ Module 01a.import ═══════════════════════════════════════════════════════
print("=" * 70); print("MODULE 01a.import — Industry + Text"); print("=" * 70)
industry_dir = os.path.join(RAW_DEAL, "industry")
industry_files = sorted(glob.glob(os.path.join(industry_dir, "acquisition_industry_*_cleaned.csv")))
print(f"Found {len(industry_files)} industry batch files")
INDUSTRY_KEEP = [
    "deal_num", "tar_name", "tar_bvd_id_num", "tar_orbis_id_num",
    "tar_overview", "tar_trade_descr_en", "tar_busi_descr",
    "acq_overview", "ven_overview",
    "tar_primary_sic_code", "tar_sic_codes", "tar_primary_naics_code",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num",
    "acq_primary_sic_code", "acq_sic_codes",
    "ven_name", "ven_bvd_id_num", "ven_primary_sic_code",
]
batches_ind = []
for fp in industry_files:
    df_b = read_and_ffill(fp); n_raw = len(df_b)
    df_b = clean_missing(df_b)
    df_b = df_b[[c for c in INDUSTRY_KEEP if c in df_b.columns]]
    batches_ind.append(df_b)
    print(f"  {os.path.basename(fp)}: {n_raw} rows read")
df_ind = pd.concat(batches_ind, ignore_index=True)
print(f"\nAfter stacking: {len(df_ind):,} rows")
for _oc in ["tar_orbis_id_num", "acq_orbis_id_num"]:
    if _oc in df_ind.columns:
        df_ind[_oc] = normalize_orbis_id(df_ind[_oc])
# ══ Module 01a.vendorname ═══════════════════════════════════════════════════
# ── VENDOR AGG (must run BEFORE the tar_name filter) ────────────────────────────────────────────
VENDOR_AGG = ["ven_name", "ven_bvd_id_num", "ven_primary_sic_code"]
print("\n" + "=" * 70)
print("VENDOR AGG — vendor-name aggregation before dedup")
print("=" * 70)
_ven_frames = []
for _vc in VENDOR_AGG:
    if _vc not in df_ind.columns:
        print(f"  {_vc:<24s} skipped"); continue
    _s = df_ind.dropna(subset=[_vc])
    if _s.empty:
        print(f"  {_vc:<24s} all missing"); continue
    _g = (_s.groupby("deal_num")[_vc]
          .apply(lambda x: "|".join(sorted(set(
              str(v).strip() for v in x
              if str(v).strip() and str(v).strip().lower() != "nan"))))
          .rename(_vc + "_all").to_frame())
    _g[_vc + "_n"] = _g[_vc + "_all"].apply(
        lambda s: len([x for x in str(s).split("|") if x]) if pd.notna(s) else 0)
    _g = _g.reset_index()
    _n_multi = int((_g[_vc + "_n"] > 1).sum())
    print(f"  {_vc:<24s} {len(_g):,} deals | multi {_n_multi:,} "
          f"| mean {_g[_vc+'_n'].mean():.2f}")
    _ven_frames.append(_g)
_ven_agg = None
if _ven_frames:
    _ven_agg = _ven_frames[0]
    for _f in _ven_frames[1:]:
        _ven_agg = _ven_agg.merge(_f, on="deal_num", how="outer")
    _ven_agg["deal_num"] = pd.to_numeric(_ven_agg["deal_num"], errors="coerce").astype("Int64")
    _out_ven = os.path.join(CLEANED, "01_deal_vendor_name.csv")
    _ven_agg.to_csv(_out_ven, index=False, encoding="utf-8-sig")
    print(f"  Saved → {_out_ven} ({len(_ven_agg):,} deals)")
# ══ Module 01a.SIC ══════════════════════════════════════════════════════════
# ── cascade SIC recovery ──────────────────────────────────────────
_nunique_sic = df_ind.groupby("deal_num")["acq_primary_sic_code"].transform(
    lambda s: s.dropna().nunique())
_unique_val = df_ind.groupby("deal_num")["acq_primary_sic_code"].transform(
    lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else np.nan)
df_ind.loc[df_ind["acq_primary_sic_code"].isna() & (_nunique_sic == 1),
           "acq_primary_sic_code"] = _unique_val[
    df_ind["acq_primary_sic_code"].isna() & (_nunique_sic == 1)]
n_before = len(df_ind)
df_ind = df_ind[df_ind["tar_name"].notna()].copy()
print(f"After drop tar_name=NaN: {len(df_ind):,} (dropped {n_before-len(df_ind):,})")
df_ind["_tar_key"] = df_ind["tar_bvd_id_num"].fillna(df_ind["tar_name"])
df_ind = df_ind.drop_duplicates(subset=["deal_num", "_tar_key"], keep="first")
df_ind = df_ind.drop(columns=["_tar_key"])
df_ind["deal_num"] = pd.to_numeric(df_ind["deal_num"], errors="coerce")
df_ind = df_ind[df_ind["deal_num"].notna()]
df_ind["deal_num"] = df_ind["deal_num"].astype("Int64")
out_path_ind = os.path.join(CLEANED, "01_deal_sic_industry.csv")
df_ind.to_csv(out_path_ind, index=False, encoding="utf-8-sig")
print(f"Saved → {out_path_ind}")
# ══ Module 01a.multiple ═════════════════════════════════════════════════════
print("\n" + "=" * 70); print("01a.multiple — Multiples"); print("=" * 70)
mul_files = sorted(glob.glob(os.path.join(RAW_DEAL, "multiple",
                                          "acquisition_multiple_*_cleaned.csv")))
batches = [clean_missing(read_and_ffill(fp)) for fp in mul_files]
df_mul = pd.concat(batches, ignore_index=True)
df_mul = df_mul[df_mul["deal_num"].notna()]
df_mul = df_mul.drop_duplicates(subset=["deal_num"], keep="first")
df_mul["deal_num"] = pd.to_numeric(df_mul["deal_num"], errors="coerce").astype("Int64")
for col in [c for c in df_mul.columns if c.endswith(("_mul_ly", "_mul_fy"))]:
    df_mul[col] = pd.to_numeric(df_mul[col], errors="coerce")
if "unnamed_0" in df_mul.columns:
    df_mul = df_mul.drop(columns=["unnamed_0"])
out_path_mul = os.path.join(CLEANED, "01_deal_multiples.csv")
df_mul.to_csv(out_path_mul, index=False, encoding="utf-8-sig")
print(f"Module 01a.multiple: {len(df_mul):,} rows → {out_path_mul}")
# ══ Module 01a.structure_date ═══════════════════════════════════════════════
print("\n" + "=" * 70); print("01a.structure_date — Structure & Dates"); print("=" * 70)
sd_files = sorted(glob.glob(os.path.join(RAW_DEAL, "structure_date",
                                         "acquisition_structure_date_*_cleaned.csv")))
batches = [clean_missing(read_and_ffill(fp)) for fp in sd_files]
df_sd = pd.concat(batches, ignore_index=True)
df_sd = df_sd[df_sd["deal_num"].notna()].copy()
CATEGORICAL_AGG = ["deal_pay_method", "deal_struct", "deal_fin", "deal_type"]
print("\nCATEGORICAL AGG")
_agg_frames = []
for _src in CATEGORICAL_AGG:
    if _src not in df_sd.columns:
        continue
    _sub = df_sd.dropna(subset=[_src])
    if _sub.empty:
        continue
    _g = (_sub.groupby("deal_num")[_src]
          .apply(lambda x: "|".join(sorted(set(x.astype(str)))))
          .rename(_src + "_all").to_frame())
    _g[_src + "_n"] = _g[_src + "_all"].str.split("|").apply(len)
    _g = _g.reset_index()
    print(f"  [{_src}] {len(_g):,} deals | multi {int((_g[_src+'_n']>1).sum()):,}")
    _agg_frames.append(_g)
_agg_all = None
if _agg_frames:
    _agg_all = _agg_frames[0]
    for _f in _agg_frames[1:]:
        _agg_all = _agg_all.merge(_f, on="deal_num", how="outer")
    _agg_all["deal_num"] = pd.to_numeric(_agg_all["deal_num"], errors="coerce").astype("Int64")
df_sd = df_sd.drop_duplicates(subset=["deal_num"], keep="first")
df_sd["deal_num"] = pd.to_numeric(df_sd["deal_num"], errors="coerce").astype("Int64")
if _agg_all is not None:
    df_sd = df_sd.merge(_agg_all, on="deal_num", how="left")
SD_KEEP = ["deal_num", "deal_type", "deal_status", "deal_struct", "deal_fin",
           "deal_pay_method", "deal_pay_method_all", "deal_pay_method_n",
           "deal_struct_all", "deal_struct_n", "deal_fin_all", "deal_fin_n",
           "deal_type_all", "deal_type_n",
           "announced_d", "completed_d", "withdrawn_d",
           "announced_d_yr", "completed_d_yr", "withdrawn_d_yr", "assumed_comp_d"]

# ── Rumor date (JFE 2021 dialogue) ─────────────────────────────────────
# BvD column names may use British "rumour_*" or American "rumor_*"; probe both.
RUM_D_COL  = next((c for c in ["rumour_d", "rumor_d"] if c in df_sd.columns), None)
RUM_YR_COL = next((c for c in ["rumour_d_yr", "rumor_d_yr"] if c in df_sd.columns), None)
if RUM_D_COL:
    print(f"  [rumor] Found rumor date column: {RUM_D_COL}"
          + (f" / {RUM_YR_COL}" if RUM_YR_COL else ""))
    SD_KEEP.append(RUM_D_COL)
else:
    print("  [warn] rumour_d/rumor_d not found in raw structure_date columns")
    print("         Existing columns:", [c for c in df_sd.columns if "rum" in c.lower()])
if RUM_YR_COL and RUM_YR_COL not in SD_KEEP:
    SD_KEEP.append(RUM_YR_COL)

df_sd = df_sd[[c for c in SD_KEEP if c in df_sd.columns]].copy()
for yc in ["announced_d_yr", "completed_d_yr", "withdrawn_d_yr"]:
    if yc in df_sd.columns:
        df_sd[yc] = pd.to_numeric(df_sd[yc], errors="coerce").astype("Int64")
if RUM_YR_COL and RUM_YR_COL in df_sd.columns:
    df_sd[RUM_YR_COL] = pd.to_numeric(df_sd[RUM_YR_COL], errors="coerce").astype("Int64")
if {"announced_d_yr", "completed_d_yr"} <= set(df_sd.columns):
    df_sd["deal_duration_yr"] = df_sd["completed_d_yr"] - df_sd["announced_d_yr"]

# -- has_rumor: rumor date non-empty OR deal_status contains Rumour ------------
_status = df_sd["deal_status"].astype(str).str.lower()
df_sd["has_rumor_status"] = _status.str.contains("rumour|rumor").astype(int)
if RUM_D_COL and RUM_D_COL in df_sd.columns:
    df_sd["has_rumor_date"] = df_sd[RUM_D_COL].notna().astype(int)
    df_sd["has_rumor"] = (df_sd["has_rumor_date"] | df_sd["has_rumor_status"]).astype(int)
else:
    df_sd["has_rumor"] = df_sd["has_rumor_status"]

# -- Rumor lead (relative to announced): year gap + day gap -------------------
if RUM_YR_COL and RUM_YR_COL in df_sd.columns and "announced_d_yr" in df_sd.columns:
    df_sd["rumor_lead_yr"] = df_sd["announced_d_yr"] - df_sd[RUM_YR_COL]
if RUM_D_COL and RUM_D_COL in df_sd.columns and "announced_d" in df_sd.columns:
    try:
        _rum = pd.to_datetime(df_sd[RUM_D_COL], errors="coerce")
        _ann = pd.to_datetime(df_sd["announced_d"], errors="coerce")
        df_sd["rumor_lead_days"] = (_ann - _rum).dt.days
    except Exception as _e:
        print(f"  [warn] rumor_lead_days computation failed: {_e}")

# -- Verification: rumor vs announced overlap ----------------------------------
print("\n  -- Rumor x Announced overlap check --")
_n_total = len(df_sd)
_n_rum_date = int(df_sd["has_rumor_date"].sum()) if "has_rumor_date" in df_sd.columns else 0
_n_rum_status = int(df_sd["has_rumor_status"].sum())
_n_rum_any = int(df_sd["has_rumor"].sum())
_n_ann = int(df_sd["announced_d_yr"].notna().sum()) if "announced_d_yr" in df_sd.columns else 0
print(f"  Total deals:             {_n_total:>8,}")
print(f"  has_rumor (any source):  {_n_rum_any:>8,}  ({_n_rum_any/_n_total:.1%})")
print(f"    of which rumor date non-empty: {_n_rum_date:>8,}  ({_n_rum_date/_n_total:.1%})")
print(f"    of which status contains Rumour:{_n_rum_status:>8,}  ({_n_rum_status/_n_total:.1%})")
print(f"  Has announced year:      {_n_ann:>8,}")
if "has_rumor_date" in df_sd.columns and "announced_d_yr" in df_sd.columns:
    _both = df_sd[(df_sd["has_rumor_date"] == 1) & df_sd["announced_d_yr"].notna()]
    _rum_only = df_sd[(df_sd["has_rumor_date"] == 1) & df_sd["announced_d_yr"].isna()]
    _ann_only = df_sd[(df_sd["has_rumor_date"] == 0) & df_sd["announced_d_yr"].notna()]
    print(f"\n  rumor_date=1 AND announced=1: {len(_both):>6,}  "
          f"(share of rumor_date: {len(_both)/max(_n_rum_date,1):.1%})")
    print(f"  Rumor only, no announced:  {len(_rum_only):>6,}  (mostly failed/withdrawn)")
    print(f"  Announced only, no rumor:  {len(_ann_only):>6,}")
    if "rumor_lead_yr" in df_sd.columns and len(_both) > 0:
        _ld = _both["rumor_lead_yr"].dropna()
        print(f"\n  rumor_lead_yr (rumor->announce, both present only):")
        print(f"    N={len(_ld):,}  mean={_ld.mean():.2f}  median={_ld.median():.0f}"
              f"  p10={_ld.quantile(.1):.0f}  p90={_ld.quantile(.9):.0f}")
        print(f"    Same year (lead=0): {(_ld==0).sum():,} ({(_ld==0).mean():.1%})")
        print(f"    lead>=0 (rumor precedes announce): {(_ld>=0).sum():,} "
              f"({(_ld>=0).mean():.1%})")
        print(f"    lead<0 (announce precedes rumor, anomaly): {(_ld<0).sum():,} "
              f"({(_ld<0).mean():.1%})")
    if "rumor_lead_days" in df_sd.columns and len(_both) > 0:
        _ldd = _both["rumor_lead_days"].dropna()
        if len(_ldd) > 0:
            print(f"\n  rumor_lead_days: mean={_ldd.mean():.1f}  median={_ldd.median():.0f}"
                  f"  p25={_ldd.quantile(.25):.0f}  p75={_ldd.quantile(.75):.0f}")
# rumor deals completion rate (JFE key fact: rumor is a deal breaker)
if "deal_status" in df_sd.columns:
    _st = df_sd["deal_status"].astype(str)
    df_sd["completed_flag"] = _st.str.strip().str.startswith("Completed").astype(int)
    _rum_done = df_sd[df_sd["has_rumor"] == 1]["completed_flag"].mean()
    _non_done = df_sd[df_sd["has_rumor"] == 0]["completed_flag"].mean()
    print(f"\n  Completion rate comparison (JFE benchmark):")
    print(f"    has_rumor=1: {_rum_done:.1%}  (N={int(df_sd['has_rumor'].sum()):,})")
    print(f"    has_rumor=0: {_non_done:.1%}  (N={int((1-df_sd['has_rumor']).sum()):,})")

out_path_sd = os.path.join(CLEANED, "01_deal_structure_date.csv")
df_sd.to_csv(out_path_sd, index=False, encoding="utf-8-sig")
print(f"\nModule 01a.structure_date: {len(df_sd):,} rows → {out_path_sd}")
# ══ Module 01a.value ════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("MODULE 01a.value — Value"); print("=" * 70)
val_files = sorted(glob.glob(os.path.join(RAW_DEAL, "value",
                                          "acquisition_value_*_cleaned.csv")))
batches = [clean_missing(read_and_ffill(fp)) for fp in val_files]
df_val = pd.concat(batches, ignore_index=True)
df_val = df_val[df_val["deal_num"].notna()]
df_val = df_val.drop_duplicates(subset=["deal_num"], keep="first")
df_val["deal_num"] = pd.to_numeric(df_val["deal_num"], errors="coerce").astype("Int64")
VAL_KEEP = ["deal_num", "deal_value", "deal_enterprise_value", "deal_equity_value",
            "deal_modelled_enterprise_value", "stake_acq_pct", "stake_final_pct", "currency"]
df_val = df_val[[c for c in VAL_KEEP if c in df_val.columns]].copy()
for c in [c for c in df_val.columns if c not in ("deal_num", "currency")]:
    df_val[c] = pd.to_numeric(df_val[c], errors="coerce")
out_path_val = os.path.join(CLEANED, "01_deal_value.csv")
df_val.to_csv(out_path_val, index=False, encoding="utf-8-sig")
print(f"Module 01a.value: {len(df_val):,} rows → {out_path_val}")
# ══ Module 01a.infor_source ═════════════════════════════════════════════════
print("\n" + "=" * 70); print("MODULE 01a.infor_source — Info Source"); print("=" * 70)
info_src_fp = os.path.join(BASE, "raw", "MA_deal", "info_source",
                           "deal_info_source_categories.csv")
df_src = pd.read_csv(info_src_fp, encoding="utf-8-sig", low_memory=False)
df_src.columns = df_src.columns.str.strip()
df_src = df_src.rename(columns={
    "dealnumber": "deal_num", "categoryofsource": "source_category",
    "sourcedocumentation": "source_document", "targetname": "tar_name",
    "targetbvdidnumber": "tar_bvd_id_num", "targetorbisidnumber": "tar_orbis_id_num",
    "acquirorname": "acq_name", "acquirorbisidnumber": "acq_orbis_id_num",
    "acquirorbvdidnumber": "acq_bvd_id_num", "vendorname": "ven_name",
    "vendorbvdidnumber": "ven_bvd_id_num", "index": "src_index"})
df_src["deal_num"] = df_src["deal_num"].ffill()
df_src = clean_missing(df_src)
df_src = df_src.drop(columns=["var1"], errors="ignore")
df_src = df_src[df_src["deal_num"].notna()].copy()
df_src["deal_num"] = pd.to_numeric(df_src["deal_num"], errors="coerce").astype("Int64")
def extract_main_cat(text):
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    m = re.match(r"(.*?)\s*\(", s)
    return m.group(1).strip() if m else s
df_src["source_main_cat"] = df_src["source_category"].apply(extract_main_cat)
main_source_list = ["Stock Exchange", "Website", "Company Press Release",
                    "Electronic Publication", "Advisor Submission", "Miscellaneous"]
for src in main_source_list:
    df_src["num_" + src.replace(" ", "_")] = (df_src["source_main_cat"] == src).astype(int)
df_src["num_source_other"] = (~df_src["source_main_cat"].isin(main_source_list)).astype(int)
dummy_cols = [c for c in df_src.columns if c.startswith("num_")]
deal_source_df = df_src.groupby("deal_num")[dummy_cols].sum().reset_index()
out_path_src = os.path.join(CLEANED, "01_deal_info_source_count.csv")
deal_source_df.to_csv(out_path_src, index=False, encoding="utf-8-sig")
print(f"Module 01a.infor_source: {len(deal_source_df):,} deals → {out_path_src}")
# ══ Module 01a.vendorNER ════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Module 01a.vendorNER — Vendor Identity (Stanford CoreNLP NER)")
print("=" * 70)
# NOTE (2026-09-18): This module is SUPERSEDED by 01d_vendor_identity.py
# (v4). Vendor classification now lives in 01d — it reuses the NER cache
# and degrades gracefully without Java, and re-writes the same output
# files (01_deal_vendor_type.csv, 01_deal_vendor_type_profile.csv).
# Module kept here for run_all compatibility; 01d is authoritative.
# ── G-0. Stanford CoreNLP initialization (mirrors user pipeline; raise on failure) ──────────────────────────
STANFORD_CORENLP_PATH = r'D:\MA\stanford-corenlp-4.5.7'
JAVA_BIN_PATH = r"C:\Program Files\Java\jdk1.8.0_202\bin"
os.environ['PATH'] = JAVA_BIN_PATH + os.pathsep + os.environ.get('PATH', '')
_nlp = None
try:
    from stanfordcorenlp import StanfordCoreNLP
    print(f"  CoreNLP path: {STANFORD_CORENLP_PATH}")
    print(f"  Java bin    : {JAVA_BIN_PATH}")
    _nlp = StanfordCoreNLP(STANFORD_CORENLP_PATH, lang='en')
    print("  ✅ StanfordCoreNLP started (NER available)")
except Exception as _e:
    print("\n" + "!" * 70)
    print(f"  ✗ Stanford CoreNLP failed to start: {type(_e).__name__}: {_e}")
    print("  Module 01a.vendorNER aborted — will not silently fall back to hand-written rules.")
    print("  Please verify: (1) stanfordcorenlp installed (2) Java 8 path correct")
    print("           (3) stanford-corenlp-4.5.7 directory exists")
    print("!" * 70)
    raise
# ── G-1. External lexicon (generated on first run; hand-editable) ─────────────────────────────────────────────────
LEXICON_PATH = os.path.join(BASE, "data", "vendor_lexicon.csv")
DEFAULT_LEXICON = [
    # ── placeholders / role markers (highest priority; these are NOT company names) ──
    ("placeholder_shareholders", r"^share\s?holders?$"),
    ("placeholder_shareholders", r"^holders?$"),
    ("placeholder_shareholders", r"^shareholders?\s*\(.*\)$"),
    ("placeholder_insolvency",   r"^receivers?$"),
    ("placeholder_insolvency",   r"\breceiver\b"),
    ("placeholder_insolvency",   r"\badministrator\b"),
    ("placeholder_insolvency",   r"\bliquidators?\b"),
    ("placeholder_insolvency",   r"\btrustee\b"),
    ("placeholder_management",   r"^management$"),
    ("placeholder_management",   r"\bmanagement buy[- ]?out\b"),
    ("placeholder_management",   r"\bmbo\b"),
    # ── undisclosed ──
    ("undisclosed", r"undisclosed"),
    ("undisclosed", r"not disclosed"),
    ("undisclosed", r"\bunknown\b"),
    ("undisclosed", r"未披露"),
    # ── PE / VC ──
    ("pe_vc", r"capital partners?"),
    ("pe_vc", r"private equity"),
    ("pe_vc", r"venture capital"),
    ("pe_vc", r"\bventures?\b"),
    ("pe_vc", r"\bbuyout\b"),
    ("pe_vc", r"equity partners?"),
    ("pe_vc", r"\bKKR\b"), ("pe_vc", r"\bBain Capital\b"),
    ("pe_vc", r"\bCarlyle\b"), ("pe_vc", r"\bCVC\b"), ("pe_vc", r"\bEQT\b"),
    ("pe_vc", r"\bTPG\b"), ("pe_vc", r"\bPermira\b"), ("pe_vc", r"\bBlackstone\b"),
    ("pe_vc", r"\bApollo\b"), ("pe_vc", r"\bAdvent\b"), ("pe_vc", r"\bWarburg\b"),
    ("pe_vc", r"\bPincus\b"),
    ("pe_vc", r"股权投资基金"), ("pe_vc", r"创业投资"),
    ("pe_vc", r"创投"), ("pe_vc", r"投资基金"), ("pe_vc", r"私募"),
    ("pe_vc", r"股权投资"),
    # ── government ──
    ("government", r"\bgovernment\b"), ("government", r"\bministry\b"),
    ("government", r"\bmunicipal\w*\b"), ("government", r"state[- ]owned"),
    ("government", r"\bfederal\b"), ("government", r"\bcity of\b"),
    ("government", r"\bprovince\b"), ("government", r"\bSASAC\b"),
    ("government", r"政府"), ("government", r"国有"),
    ("government", r"国资"), ("government", r"财政"),
    ("government", r"国资委"), ("government", r"人民政府"),
    # ── financial institutions ──
    ("financial_inst", r"\bbank\b"), ("financial_inst", r"\bbancorp\b"),
    ("financial_inst", r"\binsurance\b"), ("financial_inst", r"\bassurance\b"),
    ("financial_inst", r"\bsecurities\b"), ("financial_inst", r"\btrust\b"),
    ("financial_inst", r"asset management"),
    ("financial_inst", r"银行"), ("financial_inst", r"保险"),
    ("financial_inst", r"证券"), ("financial_inst", r"信托"),
    ("financial_inst", r"资产管理"),
    # ── family / natural persons ──
    ("family", r"\bfamily\b"), ("family", r"family[- ]owned"),
    ("family", r"family trust"), ("family", r"家族"),
    # ── corporate suffixes (fallback only when NER finds no entity) ──
    ("corporate_suffix", r"\bInc\.?\b"), ("corporate_suffix", r"\bLtd\.?\b"),
    ("corporate_suffix", r"\bLLC\b"), ("corporate_suffix", r"\bGmbH\b"),
    ("corporate_suffix", r"\bAG\b"), ("corporate_suffix", r"\bS\.?A\.?\b"),
    ("corporate_suffix", r"\bSAS\b"), ("corporate_suffix", r"\bN\.?V\.?\b"),
    ("corporate_suffix", r"\bB\.?V\.?\b"), ("corporate_suffix", r"\bPLC\b"),
    ("corporate_suffix", r"\bGroup\b"), ("corporate_suffix", r"\bHoldings?\b"),
    ("corporate_suffix", r"\bIndustries\b"), ("corporate_suffix", r"\bCorporation\b"),
    ("corporate_suffix", r"\bCompany\b"),
    ("corporate_suffix", r"公司"), ("corporate_suffix", r"集团"),
    ("corporate_suffix", r"控股"), ("corporate_suffix", r"股份"),
    ("corporate_suffix", r"有限"),
]
if not os.path.exists(LEXICON_PATH):
    pd.DataFrame(DEFAULT_LEXICON, columns=["type", "pattern"]).to_csv(
        LEXICON_PATH, index=False, encoding="utf-8-sig")
    print(f"  Generated default lexicon → {LEXICON_PATH}")
    print("  ⚠ First run: review and extend rules from your sample (CSV is editable)")
_lex = pd.read_csv(LEXICON_PATH, encoding="utf-8-sig")
_lex = _lex.dropna(subset=["type", "pattern"])
print(f"  Loaded lexicon: {LEXICON_PATH} ({len(_lex)} rules)")
# Classification priority lives in code (reproducible); words live in the CSV (auditable)
PRIORITY = [
    "placeholder_shareholders",   # SHAREHOLDER and similar role placeholders
    "placeholder_insolvency",     # RECEIVER - bankruptcy receivership
    "placeholder_management",
    "undisclosed",
    "pe_vc",
    "government",
    "financial_inst",
    "family",
    "corporate_suffix",           # fallback only when NER fails
]
_LEX_DICT = {}
for _t, _p in zip(_lex["type"], _lex["pattern"]):
    _LEX_DICT.setdefault(_t, []).append(str(_p))
def lex_hit(s):
    """Return the first lexicon type matching by priority; None if no hit."""
    for _t in PRIORITY:
        for _p in _LEX_DICT.get(_t, []):
            try:
                if re.search(_p, s, flags=re.IGNORECASE):
                    return _t
            except re.error:
                continue
    return None
# ── G-2. Run CoreNLP NER on unique vendor names (deduped; cached) ─────────────────────────────────────────
ven_fp = os.path.join(CLEANED, "01_deal_vendor_name.csv")
df_ven = pd.read_csv(ven_fp, low_memory=False)
df_ven["deal_num"] = pd.to_numeric(df_ven["deal_num"], errors="coerce").astype("Int64")
print(f"\n  Vendor table: {len(df_ven):,} deals")
# Collect all unique vendor names
_unique = set()
for _s in df_ven["ven_name_all"].dropna():
    for _x in str(_s).split("|"):
        _x = _x.strip()
        if _x and _x.lower() != "nan":
            _unique.add(_x)
_unique = sorted(_unique)
print(f"  Unique vendor names: {len(_unique):,} → running CoreNLP NER one by one")
_ner_cache = {}
for _i, _nm in enumerate(_unique):
    try:
        _pairs = _nlp.ner(_nm)
        _tags = [t for _, t in _pairs if t != 'O']
        if any(t == 'PERSON' for t in _tags):
            _et = 'PERSON'
        elif any(t == 'ORGANIZATION' for t in _tags):
            _et = 'ORGANIZATION'
        elif _tags:
            _et = _tags[0]
        else:
            _et = 'NONE'
    except Exception:
        _et = 'ERROR'
    _ner_cache[_nm] = _et
    if (_i + 1) % 3000 == 0:
        print(f"      NER {_i+1:,}/{len(_unique):,}")
_nlp.close()
print("  ✅ NER complete, CoreNLP connection closed")
def ner_of(nm):
    return _ner_cache.get(nm, 'NONE')
# ── G-3. Combined classification ───────────────────────────────────────────────────────────────
def classify_one(s):
    """Single vendor name -> (type, ner_type)"""
    s = str(s).strip()
    if not s or s.lower() == 'nan':
        return 'missing', 'NONE'
    _nt = ner_of(s)
    # 1) lexicon (incl. placeholders; highest priority)
    _lt = lex_hit(s)
    if _lt:
        _alias = {
            'placeholder_shareholders': 'shareholders_undisclosed',
            'placeholder_insolvency':   'insolvency',
            'placeholder_management':   'management',
            'corporate_suffix':         'corporate',
            'family':                   'family_person',
        }
        return _alias.get(_lt, _lt), _nt
    # 2) NER native capabilities
    if _nt == 'PERSON':
        return 'family_person', _nt
    if _nt == 'ORGANIZATION':
        return 'corporate', _nt
    # 3) no determination - never silent
    return 'UNMAPPED', _nt
def classify_deal(name_all):
    if pd.isna(name_all):
        return 'missing', '', 'NONE'
    _names = [x.strip() for x in str(name_all).split("|")
              if x.strip() and x.strip().lower() != 'nan']
    if not _names:
        return 'missing', '', 'NONE'
    _types, _ners = [], []
    for _nm in _names:
        _t, _nt = classify_one(_nm)
        _types.append(_t)
        _ners.append(_nt)
    # Primary type: PRIORITY order; UNMAPPED / missing last
    _rank = {t: i for i, t in enumerate(PRIORITY)}
    _rank.update({'UNMAPPED': 99, 'missing': 100})
    _alias_rank = {
        'shareholders_undisclosed': _rank['placeholder_shareholders'],
        'insolvency': _rank['placeholder_insolvency'],
        'management': _rank['placeholder_management'],
        'corporate': _rank['corporate_suffix'],
        'family_person': _rank['family'],
    }
    _prim = min(_types, key=lambda t: _alias_rank.get(t, _rank.get(t, 99)))
    # NER aggregation
    if any(n == 'PERSON' for n in _ners):
        _nt_agg = 'PERSON'
    elif any(n == 'ORGANIZATION' for n in _ners):
        _nt_agg = 'ORGANIZATION'
    elif _ners:
        _nt_agg = _ners[0]
    else:
        _nt_agg = 'NONE'
    return _prim, "|".join(sorted(set(_types))), _nt_agg
_res = df_ven["ven_name_all"].apply(classify_deal)
df_ven["ven_type_primary"] = [r[0] for r in _res]
df_ven["ven_type_all"]     = [r[1] for r in _res]
df_ven["ven_ner_primary"]  = [r[2] for r in _res]
print("\n  Vendor-type distribution (CoreNLP NER + external lexicon)")
for k, n in df_ven["ven_type_primary"].value_counts().items():
    print(f"    {k:<26s} {n:>8,}  ({n/len(df_ven):.1%})")
out_ven_type = os.path.join(CLEANED, "01_deal_vendor_type.csv")
df_ven.to_csv(out_ven_type, index=False, encoding="utf-8-sig")
print(f"  Saved → {out_ven_type}")
# ── G-4. UNMAPPED diagnostics (never silent; prompts rule updates) ─────────────────────────────────────────────
_unm = df_ven[df_ven["ven_type_primary"] == 'UNMAPPED']
print(f"\n  ⚠ UNMAPPED: {len(_unm):,} deals ({len(_unm)/len(df_ven):.1%})")
if len(_unm):
    _cnt = {}
    for _s in _unm["ven_name_all"].dropna():
        for _x in str(_s).split("|"):
            _x = _x.strip()
            if _x and _x.lower() != 'nan':
                _cnt[_x] = _cnt.get(_x, 0) + 1
    _top = sorted(_cnt.items(), key=lambda x: -x[1])[:100]
    _dfu = pd.DataFrame(_top, columns=["vendor_name", "freq"])
    _fp = os.path.join(CLEANED, "01_vendor_unmapped.csv")
    _dfu.to_csv(_fp, index=False, encoding="utf-8-sig")
    print(f"  top 20 unmapped names (full list → 01_vendor_unmapped.csv):")
    for _nm, _f in _top[:20]:
        print(f"      {_nm[:50]:<50s} {_f:>6,}")
    print(f"\n  → Add rules to {LEXICON_PATH} and re-run; do not accept UNMAPPED as the final result")
# ── G-5. Vendor type × information-environment profile ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G-5 Vendor type × information-environment profile")
print("=" * 70)
df_prof = df_ven.merge(deal_source_df, on="deal_num", how="left")
df_prof = df_prof.merge(df_val[["deal_num", "deal_value"]], on="deal_num", how="left")
_08b = os.path.join(BASE, "data", "08b_deal_firm_analysis.dta")
if os.path.exists(_08b):
    try:
        _d8 = pd.read_stata(_08b, columns=["deal_num", "ln_days",
                                           "ln_mul_rev", "num_Advisor_Submission"])
        _d8["deal_num"] = pd.to_numeric(_d8["deal_num"], errors="coerce").astype("Int64")
        _d8["has_adv"] = (pd.to_numeric(_d8["num_Advisor_Submission"],
                                        errors="coerce").fillna(0) > 0).astype(int)
        df_prof = df_prof.merge(
            _d8[["deal_num", "ln_days", "ln_mul_rev", "has_adv"]],
            on="deal_num", how="left")
        print("  Merged 08b")
    except Exception as _e:
        print(f"  08b merge failed: {type(_e).__name__}")
CH_COLS = [c for c in ["num_Advisor_Submission", "num_Stock_Exchange",
                       "num_Company_Press_Release", "num_Website",
                       "num_Electronic_Publication", "num_Miscellaneous"]
           if c in df_prof.columns]
EXTRA = [c for c in ["ln_days", "ln_mul_rev", "has_adv"] if c in df_prof.columns]
print("\n  %-26s %7s %8s %8s %8s %10s" %
      ("Vendor type", "N", "advsub", "exch", "press", "value median"))
for t, g in df_prof.groupby("ven_type_primary"):
    if len(g) < 20:
        continue
    a = g["num_Advisor_Submission"].mean() if "num_Advisor_Submission" in g else np.nan
    e = g["num_Stock_Exchange"].mean() if "num_Stock_Exchange" in g else np.nan
    p = g["num_Company_Press_Release"].mean() if "num_Company_Press_Release" in g else np.nan
    dv = g["deal_value"].median() if "deal_value" in g else np.nan
    print("  %-26s %7s %8.3f %8.3f %8.3f %10.0f" %
          (t, f"{len(g):,}", a, e, p, dv))
if EXTRA:
    print("\n  %-26s %7s %9s %11s %8s" %
          ("Vendor type", "N", "ln_days", "ln_mul_rev", "has_adv"))
    for t, g in df_prof.groupby("ven_type_primary"):
        if len(g) < 20:
            continue
        row = [t, f"{len(g):,}"]
        for c in ["ln_days", "ln_mul_rev", "has_adv"]:
            row.append("%.3f" % pd.to_numeric(g[c], errors="coerce").mean()
                       if c in g.columns else "n/a")
        print("  %-26s %7s %9s %11s %8s" % tuple(row))
_rows = []
for t, g in df_prof.groupby("ven_type_primary"):
    _r = {"ven_type": t, "N": len(g)}
    for c in CH_COLS + EXTRA:
        if c in g.columns:
            _r[c + "_mean"] = pd.to_numeric(g[c], errors="coerce").mean()
    if "deal_value" in g.columns:
        _r["deal_value_median"] = pd.to_numeric(g["deal_value"], errors="coerce").median()
    _rows.append(_r)
pd.DataFrame(_rows).to_csv(
    os.path.join(CLEANED, "01_deal_vendor_type_profile.csv"),
    index=False, encoding="utf-8-sig")
print(f"\n  Saved → 01_deal_vendor_type_profile.csv")
# ══ SUMMARY ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for label, path in [
    ("01_deal_sic_industry.csv", out_path_ind),
    ("01_deal_multiples.csv", out_path_mul),
    ("01_deal_structure_date.csv", out_path_sd),
    ("01_deal_value.csv", out_path_val),
    ("01_deal_info_source_count.csv", out_path_src),
    ("01_deal_vendor_name.csv", os.path.join(CLEANED, "01_deal_vendor_name.csv")),
    ("01_deal_vendor_type.csv", out_ven_type),
    ("01_deal_vendor_type_profile.csv",
     os.path.join(CLEANED, "01_deal_vendor_type_profile.csv")),
]:
    if os.path.exists(path):
        print(f"  {label:<34s} {os.path.getsize(path)/1024:>8.0f} KB")
#（注：内容由AI生成）
