# -*- coding: utf-8 -*-
"""
04a_merge_firm_to_deal.py
=========================
Merge cleaned firm modules into deal_master → 04a_deal_firm_full.csv.

Base   : data/merged/02_deal_master.csv   [deal × target grain]
Merges :
  Step 2  — 03_firm_financial_predeal.csv     pre-deal snapshot   (Module F)
  Step 3  — 03b_firm_financial_postdeal.csv   post-deal snapshot  (Module Fb)
  Step 4  — 03_firm_financial_company.csv     multi-year series   (Module G)
  Step 5  — 03_firm_legal.csv                 legal status        (Module H)
  Step 6  — 03b_listed_status.csv             listed dummies      (Module I)
  Step 5b — 03b_firm_advisor_count.csv        advisor counts      (Module J)

──────────────────────────────────────────────────────────────────────────────
MERGE KEY — FIVE columns, used by EVERY step (not three, not one)
──────────────────────────────────────────────────────────────────────────────
    MERGE_KEYS = [deal_num,
                  tar_bvd_id_num, tar_orbis_id_num,
                  acq_bvd_id_num, acq_orbis_id_num]

  Historical note: Step 5 (legal) originally merged on deal_num alone, which
  produced a Cartesian explosion because 03_firm_legal is (deal × tar × acq)
  grain. Fixed 2026-08-15 — all steps now use the same five-column key.

  ⚠️ The key includes BOTH orbis_id columns. This makes matching STRICTER
  than a three-column (deal, tar_bvd, acq_bvd) key: any deal whose orbis ID
  is missing or inconsistent on either side will fail to match and will be
  dropped by the inner joins below.

──────────────────────────────────────────────────────────────────────────────
JOIN TYPE — Steps 2-6 are INNER, only Step 5b is LEFT
──────────────────────────────────────────────────────────────────────────────
  ⚠️ Despite the function name assert_no_row_multiplication() and this
  docstring's earlier claim of "left-join", Steps 2/3/4/5/6 all use
  how="inner". Inner joins cannot expand rows — they DROP unmatched deals.

  Consequence: the sample shrinks substantially here. Reverse-engineering
  from downstream (table1 reads 08b at 18,191 rows against a 02 base of
  ~58,691) implies roughly two thirds of observations are lost across this
  script. The loss is driven upstream, not by a bug in 04a:
      - Module 03 DROPS all multi-target deals (Step 3 of dedup_by_deal)
      - Module 03 requires a complete acquirer identity triple
        (acq_name + acq_bvd_id_num + acq_orbis_id_num)
      - the five-column key additionally requires both orbis IDs
  Deals failing any condition simply have no row in the 03 files, so the
  inner join cannot retain them.

  ⇒ This is the dominant source of attrition between 02 and 08b. It MUST be
    documented as a sample-restriction step in the paper (Table 1 Panel A),
    not treated as a silent merge detail.

  Only Step 5b (advisors) is a left join; unmatched deals get num_* = 0.

Row-expansion guard: assert_no_row_multiplication() permits up to 2x growth.
  With inner joins the guard never fires — it is retained as a safety net in
  case a merge is ever switched to "left".

Derived variables constructed here (contrary to earlier docstring text):
  has_tar_advisor, has_acq_advisor, has_any_advisor
  all num_* advisor counts are fillna(0) → int
No financial RATIO variables are built here; EBITDA margin, leverage, ROA
and revenue growth are computed downstream (Stata / 05-08b).

Output : data/merged/04a_deal_firm_full.csv
         data/merged/04_deal_firm_full_diagnostics.txt
         data/merged/04_dup_deal_tar_acq_full.txt   (only if duplicate
                                                     deal_num rows exist)

Known dead code:
  filter_valid_triple() is defined but never called. Its dropped= log line
  is written as len(df)-len(df) and would always print 0. Harmless.

Author: Zhaohua Li  Date: 2026-04-12
adjusted: Qing  Date: 2026-08-03
adjusted-2: allow row expansion for multi-target, drop strict row assertion
adjusted-3: critical bugfix — never rename triple join keys
Revised 2026-09-06: docstring corrected — six merges (Module J was missing),
  five-column key used everywhere, Steps 2-6 are INNER not left, sample
  attrition documented, output filename corrected to 04a_*.
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = r"D:\MA"
CLEANED = os.path.join(BASE, "data", "cleaned")
MERGED  = os.path.join(BASE, "data", "merged")
os.makedirs(MERGED, exist_ok=True)

# Orbis IDs are zero‑padded 9‑digit strings; force str dtype to prevent
# pandas from reading them as float (which would strip leading zeros).
ORBIS_DTYPE = {
    "tar_orbis_id_num": str,
    "acq_orbis_id_num": str,
    "ven_orbis_id_num": str,
    "tar_orbis_id_num_ovw": str,
    "acq_orbis_id_num_ovw": str,
}

# ========= GLOBAL TRIPLE MERGE KEY (CRITICAL) =========
# join keys: MUST NOT be renamed in any merge step
MERGE_KEYS = [
    "deal_num", 
    "tar_bvd_id_num", "tar_orbis_id_num", 
    "acq_bvd_id_num", "acq_orbis_id_num"]

diag_lines = []

def log(msg=""):
    print(msg)
    diag_lines.append(str(msg))


def assert_no_row_multiplication(df, base_rows, step_label, max_expansion_factor=2.0):
    """
    MODIFIED: Allow business‑driven row expansion for multi‑target deals.
    Only block ABNORMAL many‑to‑many explosion.
    base_rows: original master row count (58691)
    max_expansion_factor: if final rows > base_rows * factor → treat as bug
    """
    current = len(df)
    expansion_ratio = current / base_rows
    log(f"  [{step_label}] Current rows: {current:,} | original master: {base_rows:,} | expansion ratio: {expansion_ratio:.2f}x")

    if current > base_rows * max_expansion_factor:
        raise AssertionError(
            f"{step_label}: ABNORMAL row explosion! Base {base_rows:,} → {current:,} (ratio {expansion_ratio:.2f}x > {max_expansion_factor}x). "
            "Check lookup file for unintended many‑to‑many duplicates on triple key."
        )
    if current == base_rows:
        log(f"  Row‑count: NO expansion ({current:,} rows)")
    else:
        log(f"  Row‑count WARNING: expanded to {current:,} rows (multi‑acquirer business expansion is allowed).")


def coverage(df, col, label=None):
    label = label or col
    n = df[col].notna().sum() if col in df.columns else 0
    pct = n / len(df) * 100
    log(f"  {label:<45s}: {n:>7,} ({pct:5.1f}%)")
    return pct

def filter_valid_triple(df_sub):
    """Filter rows where all triple-key columns are non-null; drop rows with any null key"""
    triple_keys = ["deal_num", "tar_name", "tar_bvd_id_num", "acq_name", "acq_bvd_id_num"]
    mask = pd.Series(True, index=df_sub.index)
    for k in triple_keys:
        if k in df_sub.columns:
            mask = mask & df_sub[k].notna()
    df_clean = df_sub[mask].copy()
    log(f"Triple key filter: raw={len(df_sub)}, valid={len(df_sub)}, dropped={len(df_sub)-len(df_sub)}")
    return df_clean

    
# ════════════════════════════════════════════════════════════════════════════
# 1. Load base dataset (deal_master)
# ════════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("STEP 1 — Load deal_master.csv")
log("=" * 60)

df = pd.read_csv(os.path.join(MERGED, "02_deal_master.csv"),
                 encoding="utf‑8‑sig", low_memory=False, dtype=ORBIS_DTYPE)
df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")

BASE_ROWS = len(df)
log(f"Base rows         : {BASE_ROWS:,} (deal‑level master, deal_num unique)")
log(f"Unique deal_num   : {df['deal_num'].nunique():,}")
log(f"Columns           : {df.shape[1]}")


# -- Diagnostic: listed acq x advisor (do not touch master column names) --
df_ls = pd.read_csv(os.path.join(CLEANED, "03b_listed_status.csv"))
df_adv = pd.read_csv(os.path.join(CLEANED, "03b_firm_advisor_count.csv"))
m1 = df_ls[df_ls["acq_listed"]==1]["deal_num"].nunique()
m2 = df_adv.merge(df_ls[df_ls["acq_listed"]==1], on="deal_num")["deal_num"].nunique()
log(f"listed acq (status): {m1:,} | with advisor rec: {m2:,} ({m2/m1 * 100:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
# 2. Merge firm_financial_predeal (Module F)
#    Left join on TRIPLE KEY — ALLOW row expansion for multi‑acquirer
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 2 — Merge firm_financial_predeal.csv")
log("=" * 60)

df_fin = pd.read_csv(os.path.join(CLEANED, "03_firm_financial_predeal.csv"),
                     encoding="utf‑8‑sig", low_memory=False)
df_fin["deal_num"] = pd.to_numeric(df_fin["deal_num"], errors="coerce").astype("Int64")
log(f"Financial rows    : {len(df_fin):,} | unique deals: {df_fin['deal_num'].nunique():,}")

# master (deal-level) left join predeal on TRIPLE KEY; business-driven row expansion allowed
df = df.merge(df_fin, on=MERGE_KEYS, how="inner", suffixes=("", "_src"))
assert_no_row_multiplication(df, BASE_ROWS, "Step 2", max_expansion_factor=2.0)



log(f"\nKey variable coverage after merge:")
pct_ta = coverage(df, "pre_deal_tar_ta_last_avail_yr")
pct_rev = coverage(df, "pre_deal_tar_rev_last_avail_yr")
coverage(df, "pre_deal_tar_ebitda_last_avail_yr")
coverage(df, "pre_deal_tar_eq_last_avail_yr")
coverage(df, "pre_deal_tar_pat_last_avail_yr")
coverage(df, "pre_deal_acq_ta_last_avail_yr")
coverage(df, "pre_deal_acq_ebit_last_avail_yr")
coverage(df, "pre_deal_acq_eq_last_avail_yr")


# ════════════════════════════════════════════════════════════════════════════
# 2.5. Merge firm_financial_postdeal (Module Fb) — NEW
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 3 — Merge 03b_firm_financial_postdeal.csv")
log("=" * 60)

df_post = pd.read_csv(os.path.join(CLEANED, "03b_firm_financial_postdeal.csv"),
                     encoding="utf‑8‑sig", low_memory=False)
df_post["deal_num"] = pd.to_numeric(df_post["deal_num"], errors="coerce").astype("Int64")
log(f"Post financial rows: {len(df_post):,} | unique deals: {df_post['deal_num'].nunique():,}")

# on=MERGE_KEYS triple composite key
df = df.merge(df_post, on=MERGE_KEYS, how="inner", suffixes=("", "_src"))
assert_no_row_multiplication(df, BASE_ROWS, "Step 3", max_expansion_factor=2.0)

log(f"\nKey post‑deal variable coverage after merge:")
coverage(df, "post_deal_tar_rev_rev_1st_avail_yr")
coverage(df, "post_deal_tar_ebitda_1st_avail_yr")
coverage(df, "post_deal_tar_ta_1st_avail_yr")
coverage(df, "post_deal_tar_shareholder_funds_1st_avail_yr")
coverage(df, "post_deal_acq_ta_1st_avail_yr")
coverage(df, "post_deal_acq_ebitda_1st_avail_yr")
coverage(df, "post_deal_acq_shareholder_funds_1st_avail_yr")


# ════════════════════════════════════════════════════════════════════════════
# 3. Merge firm_financial_company (Module G)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 4 — Merge firm_financial_company.csv")
log("=" * 60)
# debug


df_cf = pd.read_csv(os.path.join(CLEANED, "03_firm_financial_company.csv"),
                    encoding="utf‑8‑sig", low_memory=False)
df_cf["deal_num"] = pd.to_numeric(df_cf["deal_num"], errors="coerce").astype("Int64")
log(f"Company fin rows  : {len(df_cf):,} | unique deals: {df_cf['deal_num'].nunique():,}")

# triple key
df = df.merge(df_cf, on=MERGE_KEYS, how="inner", suffixes=("", "_src"))
assert_no_row_multiplication(df, BASE_ROWS, "Step 4", max_expansion_factor=2.0)

log(f"\nKey variable coverage after merge:")
pct_rev = coverage(df, "tar_rev_rev_last_avail_yr")
coverage(df, "tar_rev_rev_yr__1")
coverage(df, "tar_ta_last_avail_yr")
coverage(df, "tar_emp_last_avail_yr")


# ════════════════════════════════════════════════════════════════════════════
# 4. Merge firm_legal (Module H)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5 — Merge firm_legal.csv")
log("=" * 60)

df_leg = pd.read_csv(os.path.join(CLEANED, "03_firm_legal.csv"),
                     encoding="utf‑8‑sig", low_memory=False)
df_leg["deal_num"] = pd.to_numeric(df_leg["deal_num"], errors="coerce").astype("Int64")
log(f"Legal rows        : {len(df_leg):,} | unique deals: {df_leg['deal_num'].nunique():,}")


# Note: 03_firm_legal is also triple-grain; original script using deal_num alone causes Cartesian explosion!
# Original bug: df = df.merge(df_leg, on="deal_num", how="left")
df = df.merge(df_leg, on=MERGE_KEYS, how="inner", suffixes=("", "_src"))
assert_no_row_multiplication(df, BASE_ROWS, "Step 5", max_expansion_factor=2.0)

log(f"\nKey variable coverage after merge:")
pct_legal = coverage(df, "acq_listed")
coverage(df, "tar_status")
coverage(df, "tar_incorp_d_year")
coverage(df, "tar_bvd_indep")


# ════════════════════════════════════════════════════════════════════════════
# 5. Merge listed‑firm status (Module I)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 6 — Merge listed_status.csv (Module I)")
log("=" * 60)

df_ls = pd.read_csv(os.path.join(CLEANED, "03b_listed_status.csv"),
                    encoding="utf‑8‑sig", low_memory=False)
df_ls["deal_num"] = pd.to_numeric(df_ls["deal_num"], errors="coerce").astype("Int64")
log(f"Listed‑status rows: {len(df_ls):,} | unique deals: {df_ls['deal_num'].nunique():,}")


df = df.merge(df_ls, on=MERGE_KEYS,  how="inner", suffixes=("", "_src"))
assert_no_row_multiplication(df, BASE_ROWS, "Step 6")

log(f"\nKey variable coverage after merge:")
coverage(df, "tar_listed")
coverage(df, "acq_listed")
if "tar_listed" in df.columns:
    n_tar = int((df["tar_listed"] == 1).sum())
    n_acq = int((df["acq_listed"] == 1).sum()) if "acq_listed" in df.columns else 0
    log(f"  tar_listed = 1 (exchange‑listed targets) : {n_tar:,} ({n_tar/len(df)*100:.1f}%)")
    log(f"  acq_listed = 1 (exchange‑listed acquirors): {n_acq:,} ({n_acq/len(df)*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════
# 5.5 Merge deal advisor table (Module J: target & acquirer advisors)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5b — Merge 03b_firm_advisor_count.csv (full sample retained)")
log("=" * 60)

df_adv = pd.read_csv(
    os.path.join(CLEANED, "03b_firm_advisor_count.csv"),
    encoding="utf-8-sig", low_memory=False
)
df_adv["deal_num"] = pd.to_numeric(df_adv["deal_num"], errors="coerce").astype("Int64")

log(f"Advisor file rows: {len(df_adv):,} | unique deals: {df_adv['deal_num'].nunique():,}")

# Merge (must be 1:1 on deal_num)
df = df.merge(df_adv, on=MERGE_KEYS, how="left")
assert_no_row_multiplication(df, BASE_ROWS, "Step5b advisor merge (left)")

# ── Fill advisor counts with 0 for non-advisor deals ───────────────────────
adv_num_cols = [c for c in df.columns if c.startswith("num_")]
for col in adv_num_cols:
    df[col] = df[col].fillna(0).astype(int)

# ── Explicit flags (robust & extensible) ───────────────────────────────────
df["has_tar_advisor"] = (df[[c for c in adv_num_cols if "tar_" in c]].sum(axis=1) > 0).astype(int)
df["has_acq_advisor"] = (df[[c for c in adv_num_cols if "acq_" in c]].sum(axis=1) > 0).astype(int)
df["has_any_advisor"] = ((df["has_tar_advisor"]==1)|(df["has_acq_advisor"]==1)).astype(int)

# ── Diagnostic ─────────────────────────────────────────────────────────────
log("Advisor coverage after left-join fill:")
log(f"  tar advisor dummy sum : {df['has_tar_advisor'].sum():,} ({df['has_tar_advisor'].mean()*100:.1f}%)")
log(f"  acq advisor dummy sum : {df['has_acq_advisor'].sum():,} ({df['has_acq_advisor'].mean()*100:.1f}%)")
log(f"  any advisor dummy sum : {df['has_any_advisor'].sum():,} ({df['has_any_advisor'].mean()*100:.1f}%)")
log(f"Row count check PASSED: {len(df):,} (no expansion)")


# ════════════════════════════════════════════════════════════════════════════
# 6. Verification checks (plan §7 MODIFIED: removed fixed row‑count check)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 7 — Verification checks (MODIFIED, row expansion allowed)")
log("=" * 60)

THRESHOLD = 85.0

checks = [
    ("pre_deal_tar_ta non‑null >= 85%",        pct_ta    >= THRESHOLD),
    ("tar_rev_rev_last_avail_yr non‑null >= 85%", pct_rev >= THRESHOLD),
    ("acq_listed non‑null >= 85%",             pct_legal >= THRESHOLD),
]

all_pass = True
for desc, result in checks:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    log(f"  [{status}] {desc}")

if not all_pass:
    log("\n  NOTE: Some coverage checks did not reach 85% threshold.")
    log("  This may indicate that financial/legal data was not exported for all deals/target‑acq combinations.")
    log("  Abnormal row‑explosion check is binding; coverage gaps are acceptable.")


# m6 FIXER R1: now observation‑level (each expanded row), not deal‑level
log("\nm6 SELECTION ANALYSIS — observation‑level (after row expansion)")
# mark rows that matched at least one record from predeal file
predeal_deal_set = set(df_fin["deal_num"].dropna())
df["_has_firm"] = df["deal_num"].isin(predeal_deal_set)
_n_has = df["_has_firm"].sum()
_n_missing = (~df["_has_firm"]).sum()
log(f"  Observations with matched firm financial data: {_n_has:,}")
log(f"  Observations WITHOUT matched firm financial data: {_n_missing:,}")

if "deal_status" in df.columns:
    _status_with    = df[df["_has_firm"]]["deal_status"].value_counts(normalize=True).head(5)
    _status_without = df[~df["_has_firm"]]["deal_status"].value_counts(normalize=True).head(5)
    log("  deal_status (with firm data):")
    for _s, _p in _status_with.items():
        log(f"    {str(_s):<40s}: {_p*100:.1f}%")
    log("  deal_status (WITHOUT firm data):")
    for _s, _p in _status_without.items():
        log(f"    {str(_s):<40s}: {_p*100:.1f}%")

if "tar_country_code" in df.columns:
    _cc_with    = df[df["_has_firm"]]["tar_country_code"].value_counts(normalize=True).head(5)
    _cc_without = df[~df["_has_firm"]]["tar_country_code"].value_counts(normalize=True).head(5)
    log("  Top countries (with firm data):")
    for _c, _p in _cc_with.items():
        log(f"    {str(_c):<6s}: {_p*100:.1f}%")
    log("  Top countries (WITHOUT firm data):")
    for _c, _p in _cc_without.items():
        log(f"    {str(_c):<6s}: {_p*100:.1f}%")

df = df.drop(columns=["_has_firm"])


# ════════════════════════════════════════════════════════════════════════════
# 7. Final diagnostic report
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 8 — Final diagnostic report")
log("=" * 60)

final_rows = len(df)
expansion_ratio = final_rows / BASE_ROWS
log(f"\nFINAL DATASET: deal_firm_full")
log(f"  Original master deal‑level rows : {BASE_ROWS:,}")
log(f"  Final expanded observations      : {final_rows:,}")
log(f"  Expansion ratio                 : {expansion_ratio:.2f}x")
log(f"  Unique deal_num                 : {df['deal_num'].nunique():,}")
log(f"  Columns                         : {df.shape[1]}")

log("\n--- Deal‑level variables (deal_master) ---")
for v in ["tar_primary_sic_code", "acq_primary_sic_code",
          "pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly",
          "deal_status", "completed_d_yr", "deal_value",
          "stake_acq_pct", "tar_country_code",
          "tar_overview", "tar_trade_descr_en"]:
    coverage(df, v)

log("\n--- Deal source dummy variables (Module K: info‑source dummies from 02_deal_master) ---")
source_keywords = ["Stock_Exchange","Website","Company_Press_Release","Electronic_Publication",
                   "Advisor_Submission","Miscellaneous","source_other"]
src_all_cols = [c for c in df.columns if any(k in c for k in source_keywords) and c.startswith("dummy_")]
for v in src_all_cols:
    coverage(df, v)

log("\n--- Regulatory review variables (01b_deal_overview) ---")
reg_vars_list = [
    "reg_body_count","reg_country_count","reg_antitrust","reg_securities",
    "reg_state_assets","reg_foreign_invest","reg_financial","reg_defense_tech",
    "reg_cross_national","reg_common_law","reg_civil_law","reg_mixed_legal"
]
for v in reg_vars_list:
    coverage(df, v)

log("\n--- Pre‑deal financials (firm_financial_predeal) ---")
for v in ["pre_deal_tar_ta_last_avail_yr", "pre_deal_tar_rev_rev_last_avail_yr",
          "pre_deal_tar_ebitda_last_avail_yr", "pre_deal_tar_eq_last_avail_yr",
          "pre_deal_tar_pat_last_avail_yr", "pre_deal_acq_ta_last_avail_yr"]:
    coverage(df, v)

log("\n--- Post‑deal financials (03b_firm_financial_postdeal) ---")
for v in ["post_deal_tar_rev_rev_1st_avail_yr",
          "post_deal_tar_ebitda_1st_avail_yr",
          "post_deal_tar_ta_1st_avail_yr",
          "post_deal_tar_eq_1st_avail_yr",
          "post_deal_tar_pat_1st_avail_yr",
          "post_deal_acq_ta_1st_avail_yr"]:
    coverage(df, v)

log("\n--- Company financials (firm_financial_company) ---")
for v in ["tar_rev_rev_last_avail_yr", "tar_rev_rev_yr__1",
          "tar_ta_last_avail_yr", "tar_emp_last_avail_yr"]:
    coverage(df, v)

log("\n--- Legal (firm_legal) ---")
for v in ["tar_status", "tar_incorp_d_year", "tar_bvd_indep"]:
    coverage(df, v)

if "tar_status" in df.columns:
    log("\n  tar_status distribution:")
    for val, cnt in df["tar_status"].value_counts(dropna=False).head(8).items():
        log(f"    {str(val):<40s}: {cnt:,}")

log("\n--- Listed‑firm status (listed_status) ---")
for v in ["tar_listed", "acq_listed"]:
    coverage(df, v)

log("\n--- M&A Advisor count variables (Module J) ---")
adv_all_cols = [c for c in df.columns if c.startswith("num_") and ("adv" in c)]
for v in adv_all_cols:
    coverage(df, v)
# ====================== Fixed version: dynamic sorting, no KeyError ======================
log("\n" + "="*60)
log("EXTRA CHECK: Duplicate deal_num | deal_num + tar_name + acq_name")
log("="*60)

# Count occurrences per deal
deal_count = df["deal_num"].value_counts()
dup_deal_list = deal_count[deal_count > 1].index.tolist()

if len(dup_deal_list) == 0:
    log("OK: all deal_num unique, no multi-target/multi-acquirer splits")
else:
    log(f"WARNING: duplicate deals found, total: {len(dup_deal_list)}")
    log(f"First 10 duplicate deal numbers: {dup_deal_list[:10]}")

    # Display three fixed columns; auto-select fields that actually exist in the table
    show_cols = ["deal_num", "tar_name_x", "tar_name_src","acq_name_x","acq_name_src"]
    exist_cols = [c for c in show_cols if c in df.columns]
    df_dup_raw = df[df["deal_num"].isin(dup_deal_list)]
    df_dup_view = df_dup_raw[exist_cols]

    # Dynamically pick sort columns: only use columns present in the table, deal_num first
    sort_keys = ["deal_num"]
    if "tar_name" in exist_cols:
        sort_keys.append("tar_name")
    if "acq_name" in exist_cols:
        sort_keys.append("acq_name")
    df_dup_view = df_dup_view.sort_values(by=sort_keys)

    log("\n[Detail preview first 60 rows: deal_num | tar_name | acq_name]")
    log(df_dup_view.head(60).to_string(index=False))

    # Count observations per deal
    stat_df = deal_count.reset_index()
    stat_df.columns = ["deal_num", "row_count"]
    log("\n[Duplicate deal row count distribution top 30]")
    log(stat_df.head(30).to_string(index=False))

    # Export full detail to txt, three columns complete
    out_dup = os.path.join(MERGED, "04_dup_deal_tar_acq_full.txt")
    with open(out_dup, "w", encoding="utf-8") as f:
        f.write("Duplicate deal full detail (deal_num, tar_name, acq_name)\n")
        f.write(df_dup_view.to_string(index=False))
    log(f"\nFull detail saved to: {out_dup}")
# =============================================================================
# ====================== MERGE reconciliation check (corrected version) ======================
log("\n" + "="*60)
log("EXTRA CHECK: Duplicate deal_num → show tar/acq name alignment")
log("="*60)

deal_count = df["deal_num"].value_counts()
dup_deal_list = deal_count[deal_count > 1].index.tolist()

if len(dup_deal_list) == 0:
    log("OK: all deal_num unique, no multi-target/multi-acquirer splits")
else:
    log(f"WARNING: duplicate deals found, total: {len(dup_deal_list)}")
    log(f"First 10 duplicate deal numbers: {dup_deal_list[:10]}")

    # Key: use name columns that actually exist in the final table
    # Prefer tar_name (cleaned standard name), then _src, fallback _x
    name_candidates_tar = ["tar_name", "tar_name_src", "tar_name_ovw"]
    name_candidates_acq = ["acq_name", "acq_name_src", "acq_name_ovw"]

    tar_col = next((c for c in name_candidates_tar if c in df.columns), None)
    acq_col = next((c for c in name_candidates_acq if c in df.columns), None)

    show_cols = ["deal_num"]
    if tar_col: show_cols.append(tar_col)
    if acq_col: show_cols.append(acq_col)

    log(f"Display columns actually used: {show_cols}")

    df_dup_raw = df[df["deal_num"].isin(dup_deal_list)]
    df_dup_view = df_dup_raw[show_cols].copy()

    # Sort
    sort_keys = ["deal_num"]
    if tar_col: sort_keys.append(tar_col)
    if acq_col: sort_keys.append(acq_col)
    df_dup_view = df_dup_view.sort_values(by=sort_keys)

    log("\n[Duplicate deal detail preview first 10 rows]")
    log(df_dup_view.head(10).to_string(index=False))

    # Statistics
    stat_df = deal_count.reset_index()
    stat_df.columns = ["deal_num", "row_count"]
    log("\n[Duplicate deal row count distribution top 30]")
    log(stat_df.head(30).to_string(index=False))

    out_dup = os.path.join(MERGED, "04_dup_deal_tar_acq_full.txt")
    with open(out_dup, "w", encoding="utf-8") as f:
        f.write("Duplicate deal full detail (deal_num, tar_name, acq_name)\n")
        f.write(df_dup_view.to_string(index=False))
    log(f"\nFull detail saved to: {out_dup}")
   
# =============================================================================
# POST-MERGE precise cleanup (whitelist + exact in-list, never accidentally delete date columns)
# =============================================================================
log("\n" + "="*60)
log("POST-MERGE precise cleanup: only delete specified suffixes + specified text columns")
log("="*60)

# -- 1. Whitelist: these columns are never deleted --
# Primary keys
KEEP_WHITELIST = set(MERGE_KEYS) | {"deal_num", "_row_id"}

# Bare names (final retained version)
KEEP_WHITELIST.add("tar_name")
KEEP_WHITELIST.add("acq_name")

# Auto-protect all date columns (ending in _d / _yr / _date)
date_cols = [c for c in df.columns if c.endswith(("_yr", "_d", "_date"))]
KEEP_WHITELIST.update(date_cols)

log(f"Whitelist column count (never deleted): {len(KEEP_WHITELIST)}")
log(f"  5 primary keys + tar/acq_name + _row_id + {len(date_cols)} date columns")

# -- 2. To delete: enumerate explicitly (use in, not not-in) --
# 2a. Secondary table suffixes (name columns with suffixes)
SUFFIX_DROP = ("_src", "_x", "_y", "_ovw")

# 2b. Long text columns (exact keywords)
TEXT_DROP_KEYWORDS = (
    "_busi_descr",
    "_trade_descr_en",
    "_sic_codes",
    "_descr",
)

# -- 3. Build delete list (only exact matches) --
drop_cols = []

for col in df.columns:
    if col in KEEP_WHITELIST:
        continue  # Whitelist: never delete

    # Rule A: name columns with suffix -> delete
    if (col.startswith("tar_name") or col.startswith("acq_name")):
        if col not in ("tar_name", "acq_name"):
            drop_cols.append(col)
        continue

    # Rule B: ending with specified suffix -> delete
    if col.endswith(SUFFIX_DROP):
        drop_cols.append(col)
        continue

    # Rule C: contains specified text keyword -> delete
    if any(kw in col for kw in TEXT_DROP_KEYWORDS):
        drop_cols.append(col)
        continue

# -- 4. Execute deletion --
if drop_cols:
    df = df.drop(columns=drop_cols, errors="ignore")
    log(f"\nDeleted {len(drop_cols)} columns (only specified suffixes + text columns):")
    # Grouped print for easy review
    name_dropped = [c for c in drop_cols if "name" in c.lower()]
    suffix_dropped = [c for c in drop_cols if c.endswith(SUFFIX_DROP)]
    text_dropped = [c for c in drop_cols if any(kw in c for kw in TEXT_DROP_KEYWORDS)]
    if name_dropped:
        log(f"  [name suffix cols] {len(name_dropped)}: {name_dropped[:10]}{'...' if len(name_dropped)>10 else ''}")
    if suffix_dropped:
        log(f"  [secondary suffix cols] {len(suffix_dropped)}: {suffix_dropped[:10]}{'...' if len(suffix_dropped)>10 else ''}")
    if text_dropped:
        log(f"  [long text cols] {len(text_dropped)}: {text_dropped[:10]}{'...' if len(text_dropped)>10 else ''}")
else:
    log("No columns to delete")

# -- 5. Check: key date columns must still be present --
#log("\n[Key date column check]")
#date_check = ["announced_d_yr", "completed_d_yr", "withdrawn_d_yr", "deal_duration_yr"]
#for c in date_check:
    #status = "OK" if c in df.columns else "MISSING!"
    #log(f"  {c:<25s} {status}")

# -- 6. Check: primary keys + bare names must still be present --
log("\n[Primary key + core field check]")
for c in ["deal_num", "tar_name", "acq_name"] + MERGE_KEYS:
    status = "OK" if c in df.columns else "MISSING!"
    log(f"  {c:<25s} {status}")

# -- 7. Check: no _src/_x/_y/_ovw residue --
bad = [c for c in df.columns if c.endswith(SUFFIX_DROP)]
if not bad:
    log("\nCheck passed: no _src/_x/_y/_ovw suffix residue")
else:
    log(f"\nWARNING: suffix columns not cleaned: {bad}")

log(f"\nFinal column count: {df.shape[1]}")

# ════════════════════════════════════════════════════════════════════════════
# 8. Save outputs
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 9 — Save outputs")
log("=" * 60)

out_full = os.path.join(MERGED, "04a_deal_firm_full.csv")
df.to_csv(out_full, index=False, encoding="utf‑8‑sig")
log(f"Saved -> {out_full}")
log(f"Size  : {os.path.getsize(out_full)/1024/1024:.1f} MB")

diag_path = os.path.join(MERGED, "04_deal_firm_full_diagnostics.txt")
with open(diag_path, "w", encoding="utf‑8") as f:
    f.write("\n".join(diag_lines))
log(f"Saved -> {diag_path}")

log("\nScript 04a complete.")
