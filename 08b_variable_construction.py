# -*- coding: utf-8 -*-
"""
08b_variable_construction.py
Date: 2026-04-15
Last Updated: 2026-09-07
Purpose: Construct all regression-ready firm, transaction, textual sentiment and
regulatory variables; export the analysis dataset to Stata .dta format.

NOTE ON "panel": the output is a cross-section of DEAL observations (one row per
deal × target-acquirer pair), NOT a balanced panel — there is no time dimension.

Pipeline Inputs:
    1. data/merged/07_deal_firm_benchmark.csv
       Core focal deal panel merged with Doc2Vec text similarity & SIC peer benchmark identifiers
    2. data/merged/04b_deal_firm_country.csv
       Full global firm corpus for peer financial characteristic calculation

    Derived inputs already merged upstream (no raw file read here):
       - Regulatory metadata          <- 01b_deal_overview.csv (Script 01 Module E)
       - Disclosure source dummies    <- 01_deal_info_source_count.csv (via 02)
       - LM comment sentiment metrics <- 01c_comments_features.csv (Script 01c)

    ⚠️ MODULE LETTERS COLLIDE ACROSS SCRIPTS:
       "Module F" means the COMMENTS file in Script 01/02, but the PRE-DEAL
       FINANCIAL snapshot in Script 03. Always cite the FILENAME, never the
       letter.

Core Output:
    data/merged/08b_deal_firm_analysis.dta
    Stata-compatible analysis dataset containing all constructed control,
    independent, heterogeneity & friction variables

──────────────────────────────────────────────────────────────────────────────
VARIABLE CONSTRUCTION CATALOGUE
──────────────────────────────────────────────────────────────────────────────

1. Valuation Multiples (Raw & Log Transformed)
    ln_mul_{rev,ebitda,ebit}          Log pre-transaction target valuation multiples
    ln_post_mul_{rev,ebitda,ebit}     Log post-transaction operating multiples
    ln_text_{rev,ebitda,ebit}         Log Doc2Vec text peer median multiples
    ln_sic_{rev,ebitda,ebit}          Log SIC industry peer median multiples

2. Target & Acquirer Fundamental Ratios (Pre / Post Deal, Winsorised 1%/99%)
    tar_ebitda_margin, tar_leverage, tar_rev_growth, tar_roa
    acq_ebitda_margin, acq_leverage, acq_roa
    tar_post_ebitda_margin, tar_post_roa, tar_post_leverage
    acq_post_ebitda_margin, acq_post_roa, acq_post_leverage
    ln_tar_size, ln_acq_size, ln_tar_post_size, ln_acq_post_size
    ln_rel_size: Log acquirer / target total asset relative scale
    NOTE: ln_rel_size = ln_acq_size − ln_tar_size. It is EXACTLY collinear with
          the two level terms — never include all three in one regression.

3. Firm Age Controls (2026-07-27 Addition)
    target_age: Deal year minus target incorporation year
    acquirer_age: Deal year minus acquirer incorporation year
    Negative ages (incorporation after deal year) set to NaN.

4. Listing Status Dummies
    tar_listed, acq_listed (1 = exchange AND ticker both present in Zephyr)

5. Transaction Time Horizon — see the DURATION CLEANING note below
    DaysToCompletion: Calendar days from formal announcement to close
    ln_days: Natural log of completion duration

──────────────────────────────────────────────────────────────────────────────
⚠️ DURATION CLEANING (STEP 6) — MATERIAL FOR REPLICATION, READ BEFORE USING ln_days
──────────────────────────────────────────────────────────────────────────────
  Two classes of Zephyr MECHANICAL placeholder dates are removed before
  DaysToCompletion is computed. Both are database artefacts, not real durations:

  (a) THE 730-DAY CLUSTER. When completed_d is missing, Zephyr sets
      assumed_comp_d = announced_d + 730 as an extrapolation placeholder. Any
      row where completed_d is null AND (assumed_comp_d − announced_d) falls in
      [700, 760] has assumed_comp_date set to NaT. Without this rule the
      duration variable is dominated by a spurious spike at exactly 730 days.

  (b) ZERO-DURATION FILLS. When announced_d is missing, Zephyr copies
      completed_d into it, yielding DaysToCompletion = 0. Rows where
      completed_date == announced_date have completed_date set to NaT.

  end_date = completed_date, falling back to assumed_comp_date.
  DaysToCompletion <= 0 is then set to NaN; ln_days = log(DaysToCompletion).

  ⇒ The count of rows removed by each rule is printed in the log. Report these
    numbers when documenting the duration variable in the paper.

──────────────────────────────────────────────────────────────────────────────
6. Peer Benchmark Aggregates (Table 6 Descriptive Regressions)
    text_peer_*_median: Median financial metrics of text-similarity peer group
    sic_peer_*_median:  Median financial metrics of same SIC3 industry peer group
    Additional peer medians: *_ta_median, *_rev_median, *_ebitda_margin_median,
        *_leverage_median, *_rev_growth_median, *_roa_median,
        *_ev_ebitda_median, *_ev_rev_median, *_ev_ebit_median,
        *_mktcap_median, *_ev_median, *_mb_median
    Focal-side Table 6 columns: tar_ta_for_t6, tar_rev_for_t6,
        tar_mktcap_for_t6, tar_ev_for_t6, tar_mb_for_t6

    ⚠️ MIN_PEERS IS NOT ENFORCED. MIN_PEERS = 3 is defined as a module
      parameter, but compute_peer_metrics() never references it — it only
      checks `if n > 0`. Peer medians are therefore computed even for a single
      peer. The "minimum 3 peers" rule stated in earlier versions of this
      docstring was aspirational, not implemented. If you want it, add
      `if len(peer_ids) < MIN_PEERS: return result` after the peer_ids parse.

7. FinSimGap Distance Components (built in STEP 7, previously undocumented)
    dist_comp:  Standardised Euclidean distance, focal to centroid of peers
                having all three dimensions non-NaN (D2 formula)
    dist_ebitda / dist_lev / dist_grow: Mean absolute per-dimension gap
    All sub-distances and the composite use the SAME peer subset
    (all three z-dims non-NaN). Using different subsets per dimension
    previously made the composite incoherent.
    Peer ratios are pre-filtered (denominator >= 100k) then winsorised 1/99
    BEFORE standardisation — without this, near-zero denominators produce
    ratios above 1e6 and the z-scores collapse to noise.

8. Cross-Industry Heterogeneity Proxies
    cross_industry_alt: Alternative cross-industry indicator (peer overlap < 50%)
    cross_industry: Baseline SIC3 mismatch dummy (retained for comparison)
    peer_overlap_frac: Fraction of TEXT peers that also appear in the SIC peer
        set. ASYMMETRIC by design — denominator is len(text peers), not the
        union. Document this definition in any table note.

9. SIC Affiliation & Similarity (Table 7)
    text_same_sic{1,2,3,4}_frac: Share of text peers matching target SIC depth
    sic_same_sic{1,2,3,4}_frac:  Share of SIC peers matching target SIC depth
        (sic_same_sic3_frac is ~1 by construction; kept for verification)
    text_sim_same_sic{lv}_mean / text_sim_diff_sic{lv}_mean, lv in 1..4:
        Mean Doc2Vec similarity to same-SIC vs different-SIC text peers
    sic_peer_sim_mean: Mean cosine similarity to SIC peers (needs the Doc2Vec
        model at data/models/doc2vec_full_corpus.model; NaN if load fails)

10. Legal Origin Institutional Controls (La Porta et al. 1998)
    ⚠️ VARIABLE NAMES ARE ROLE-SUFFIXED. The bare names legal_origin and
       common_law are COMMENTED OUT in the code. What is actually produced:
         legal_origin_tar / legal_origin_acq   (English / French / German /
                                                Scandinavian)
         common_law_tar   / common_law_acq     (1 = English common law)
       Unmapped country codes yield NaN and are listed in the log.
    national_class: 1 = target country uses NAICS (US, CA, MX)
    sic_coverage_rate: Country-level valid SIC3 share, computed on the FULL
        corpus (04b), not the analysis sample — the sample has SIC3 by
        construction, so computing it there would give 100% and no variation.

    ⚠️ SIC-1 is a NON-STANDARD shorthand: first digit of the 3-digit SIC code
       (0-9). Standard SIC Divisions use letter ranges. Footnote this.

11. Classification Distance (H3 continuous moderator)
    classification_distance_tar / classification_distance_acq
    Ordinal 0-4 distance from the US SIC mapping system (0 = NAICS-native,
    4 = China CSRC). Used as the running variable in table2/table3.
    ⚠️ NOT a continuous "information environment" index: diagnostics show the
       securities-review share is NON-MONOTONIC across levels
       (0% / 4.0% / 1.4% / 3.1% / 45.6%). Treat as a categorical regime
       indicator; do not interpret as a transparency gradient.

──────────────────────────────────────────────────────────────────────────────
12. REGULATORY VARIABLES — TWO MEASURES, PRIMARY vs SECONDARY
──────────────────────────────────────────────────────────────────────────────
  (A) PRIMARY — structured reg_* from 01b_deal_overview.csv
      reg_body_count, reg_country_count
      reg_antitrust, reg_securities, reg_state_assets, reg_foreign_invest,
      reg_financial, reg_defense_tech
      reg_cross_national, reg_common_law, reg_civil_law, reg_mixed_legal
      Built from the `regulatory_body_name` field by enumerating all 443
      distinct authority names and resolving each through a three-tier scheme
      (exact map > exclude list > keyword fallback). Deal-level ORs across all
      bodies on a deal; categories are NOT mutually exclusive.
      reg_common_law / reg_civil_law / reg_mixed_legal rely on exact country-
      name matching and are INTERMEDIATE — 04b overrides them with the
      dedicated legal-origin dataset. Do not use for final inference.

  (B) SECONDARY — text-derived, from 01c_comments_features.csv
      num_unique_reg, reg_event_count, has_phase2_investigation
      Counted by regex-scanning comment text. Coverage ~1.84% vs ~12.8% for
      the structured measure, and the keyword list covers only Western
      antitrust agencies with NO securities regulators — deals reviewed via
      the securities route score 0 BY CONSTRUCTION.
      ⇒ Robustness / alternative measure only. Never substitute for (A).

13. M&A Advisor Count Aggregates (Module J = 03b_firm_advisor_count.csv)
    tar_total_advisor, acq_total_advisor: Sum of advisor type dummies
    num_tar_adv_* / num_acq_adv_* : per-type counts

14. Information Disclosure Source Metrics (01_deal_info_source_count.csv)
    total_info_source: Sum of per-channel disclosure dummies
    Uses min_count=1 so a deal with no source record stays NaN rather than 0.

15. Deal Comment Features (01c_comments_features.csv)
    — merged here, not rebuilt; see 01c for construction detail.
    Text volume control: comment_wordcount, log_comment_wordcount,
        comment_char_length, sentence_event_count
    LM word counts: lm_*_count (7 categories)
    LM presence dummies (main specification): has_lm_* (7 categories,
        NaN = no comment)
    Comment availability: has_comment
    Timeline / negotiation flags: total_timeline_days, has_rumour,
        has_target_reject, has_unconditional_offer, has_deal_complete,
        has_goshop, has_reg_remedy, has_debt_assumption
    Premium / bidding: price_event_num, max_premium_pct, avg_premium_pct,
        rival_bidder_num

    NOT emitted by 01c (do not expect these columns):
      lm_*_density          EMIT_LM_DENSITY = False. Median comment ~88 words
                            and expected LM hits < 1 per category, so
                            density = count/88 is not interpretable.
      lm_net_sentiment      Never computed. LM (2011) positive vs negative word
                            lists differ ~6.6x (354 vs 2355), so a net score is
                            systematically negative; the authors advise against.
      deal_complexity_score EMIT_COMPLEXITY = False, deprecated.

PROCESSING RULES
    1. Ratio & size variables winsorised at 1% / 99%.
       Winsorised set = the 18 ratio/size columns in STEP 4
                      + tar_total_advisor, acq_total_advisor
                      + total_info_source
                      + ind_text_diversity -> ind_text_diversity_w
       The PEER CORPUS ratios get an extra pre-filter (denominator >= 100k)
       before winsorising and standardising — see item 7.
    2. Log transforms applied only to strictly positive values; else NaN.
    3. Categorical institutional dummies stored as nullable Int64 for Stata.
    4. Raw comment text was discarded upstream in 02; only numeric features
       are carried through, to keep file size manageable.
    5. Missing peer identifiers yield NaN benchmarks (never imputed).

Author: Q  Date: 2026-04-15
Updated: 2026-08-08: add LM sentiment & negotiation feature suite from 01c
Revised: 2026-09-06: docstring reconciled with code — duration cleaning
         documented, legal_origin / classification_distance role suffixes
         corrected, FinSimGap components added, MIN_PEERS gap flagged,
         primary-vs-secondary regulatory measures separated, "balanced panel"
         wording corrected.
"""

import os
import sys
import logging
import warnings
import glob# 20260727==========================================================

import numpy as np
import pandas as pd
import pyreadstat
from gensim.models.doc2vec import Doc2Vec

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)

# ── Reproducibility ────────────────────────────────────────────────────────────
import random
random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = r"D:\MA"
MERGED = os.path.join(ROOT, "data", "merged")
RAW    = os.path.join(ROOT, "raw", "MA_deal", "structure_date")# 20260727======
LOG_PATH = os.path.join(MERGED, "08b_variable_construction_log.txt")

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger("08b_var_construction")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
# 防止 %runfile 重复运行导致 handler 累积（每行打印 N 遍的根因）
log.handlers.clear()
log.propagate = False          # 同时阻断向 root logger 二次传播

fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(fh)
log.addHandler(ch)

log.info("=== Script 08: Variable Construction ===")

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_PEERS = 3          # minimum peers for valid benchmark (consistent with earlier scripts)
WINSOR_LO = 0.01       # lower tail for winsorisation (1%)
WINSOR_HI = 0.99       # upper tail for winsorisation (99%)

# ── Helper: winsorise ──────────────────────────────────────────────────────────
def winsorise(series, lo=None, hi=None):
    """Winsorise a pandas Series at the given quantile bounds."""
    s = series.copy()
    if lo is not None:
        lower = s.quantile(lo)
        s = s.clip(lower=lower)
    if hi is not None:
        upper = s.quantile(hi)
        s = s.clip(upper=upper)
    return s

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 1 — Load data")
log.info("=" * 70)

df = pd.read_csv(os.path.join(MERGED, "07_deal_firm_benchmark.csv"), low_memory=False)
df["_row_id"] = df["_row_id"].astype(int)
log.info(f"  07_deal_firm_benchmark.csv: {len(df):,} rows, {len(df.columns)} cols")

# Corpus for peer financial lookups (FinSimGap + Table 6/7)
PEER_FIN_COLS = [
    "_row_id", "tar_primary_sic_code", "tar_country_code",
    "pre_deal_tar_ebitda_last_avail_yr", "pre_deal_tar_rev_rev_last_avail_yr",
    "pre_deal_tar_ta_last_avail_yr",     "pre_deal_tar_eq_last_avail_yr",
    "pre_deal_tar_pat_last_avail_yr",
    "tar_rev_rev_last_avail_yr",         "tar_rev_rev_yr__1",
    # New for Table 6: valuation multiples, Market Cap, EV
    "pre_ebitda_mul_ly", "pre_rev_mul_ly", "pre_ebit_mul_ly",
    "post_rev_mul_fy", "post_ebitda_mul_fy", "post_ebit_mul_fy",
    "deal_equity_value", "deal_modelled_enterprise_value",
]
corpus = pd.read_csv(os.path.join(MERGED, "04b_deal_firm_country.csv"),
                     usecols=PEER_FIN_COLS, low_memory=False)
corpus["_row_id"] = corpus["_row_id"].astype(int)
log.info(f"  04b_deal_firm_country.csv: {len(corpus):,} rows loaded (peer pool)")

# Derive SIC codes for corpus (needed for Table 7 peer SIC affiliation)
corpus["tar_sic4_float"] = pd.to_numeric(corpus["tar_primary_sic_code"], errors="coerce")
# Use float (np.nan) throughout — avoids pd.NA type conflicts in peer lookups
valid_sic = corpus["tar_sic4_float"].between(100, 9999, inclusive="both")
corpus["tar_sic4"] = np.where(valid_sic, np.floor(corpus["tar_sic4_float"]), np.nan)
corpus["tar_sic3"] = np.where(valid_sic, np.floor(corpus["tar_sic4_float"] / 10),   np.nan)
corpus["tar_sic2"] = np.where(valid_sic, np.floor(corpus["tar_sic4_float"] / 100),  np.nan)
corpus["tar_sic1"] = np.where(valid_sic, np.floor(corpus["tar_sic4_float"] / 1000), np.nan)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOG MULTIPLES  (horse-race regressions)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 3 — Log multiples")
log.info("=" * 70)

for src_col, new_col in [
    ("pre_rev_mul_ly",    "ln_mul_rev"),
    ("pre_ebitda_mul_ly", "ln_mul_ebitda"),
    ("pre_ebit_mul_ly",   "ln_mul_ebit"),
    ("post_rev_mul_fy",    "ln_post_mul_rev"),
    ("post_ebitda_mul_fy", "ln_post_mul_ebitda"),
    ("post_ebit_mul_fy",   "ln_post_mul_ebit"),
    ("text_bench_rev",    "ln_text_rev"),
    ("text_bench_ebitda", "ln_text_ebitda"),
    ("text_bench_ebit",   "ln_text_ebit"),
    ("sic_bench_rev",     "ln_sic_rev"),
    ("sic_bench_ebitda",  "ln_sic_ebitda"),
    ("sic_bench_ebit",    "ln_sic_ebit"),
]:
    vals = pd.to_numeric(df[src_col], errors="coerce")
    df[new_col] = np.where(vals > 0, np.log(vals), np.nan)
    log.info(f"  {new_col}: N={df[new_col].notna().sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TARGET & ACQUIRER FIRM FINANCIAL CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 4 — Target & Acquirer firm financial controls")
log.info("=" * 70)

def safe_ratio(num, den, min_den=None):
    """Compute num/den; return NaN when den ≤ 0 (or < min_den if provided)."""
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    threshold = min_den if min_den is not None else 0
    return np.where((d.notna()) & (d > threshold), n / d, np.nan)

# --------------------------
# Pre-deal Target Ratios
# --------------------------
# EBITDA margin: EBITDA / Revenue
df["tar_ebitda_margin"] = safe_ratio(
    df["pre_deal_tar_ebitda_last_avail_yr"],
    df["pre_deal_tar_rev_rev_last_avail_yr"]
)

# Target Leverage: (TA - Equity) / TA
ta  = pd.to_numeric(df["pre_deal_tar_ta_last_avail_yr"], errors="coerce")
eq  = pd.to_numeric(df["pre_deal_tar_eq_last_avail_yr"], errors="coerce")
df["tar_leverage"] = safe_ratio(ta - eq, ta)

# Target Revenue growth
df["tar_rev_growth"] = safe_ratio(
    pd.to_numeric(df["tar_rev_rev_last_avail_yr"], errors="coerce") -
    pd.to_numeric(df["tar_rev_rev_yr__1"], errors="coerce"),
    df["tar_rev_rev_yr__1"]
)

# Target ROA
df["tar_roa"] = safe_ratio(
    df["pre_deal_tar_pat_last_avail_yr"],
    df["pre_deal_tar_ta_last_avail_yr"]
)

# Log target size
ta_val = pd.to_numeric(df["pre_deal_tar_ta_last_avail_yr"], errors="coerce")
df["ln_tar_size"] = np.where(ta_val > 0, np.log(ta_val), np.nan)

# --------------------------
# Pre-deal Acquirer Ratios (NEW)
# --------------------------
# Acquirer EBITDA margin
df["acq_ebitda_margin"] = safe_ratio(
    df["pre_deal_acq_ebitda_last_avail_yr"],
    df["pre_deal_acq_rev_rev_last_avail_yr"]
)

# Acquirer Leverage
acq_ta_pre = pd.to_numeric(df["pre_deal_acq_ta_last_avail_yr"], errors="coerce")
acq_eq_pre = pd.to_numeric(df["pre_deal_acq_eq_last_avail_yr"], errors="coerce")
df["acq_leverage"] = safe_ratio(acq_ta_pre - acq_eq_pre, acq_ta_pre)

# Acquirer ROA (原有保留)
df["acq_roa"] = safe_ratio(
    df["pre_deal_acq_pat_last_avail_yr"],
    df["pre_deal_acq_ta_last_avail_yr"]
)

# Log acquirer size
acq_ta_val = pd.to_numeric(df["pre_deal_acq_ta_last_avail_yr"], errors="coerce")
df["ln_acq_size"] = np.where(acq_ta_val > 0, np.log(acq_ta_val), np.nan)

# Relative size: ln(Acquirer TA / Target TA) (原有保留)
acq_ta = pd.to_numeric(df["pre_deal_acq_ta_last_avail_yr"], errors="coerce")
tar_ta = pd.to_numeric(df["pre_deal_tar_ta_last_avail_yr"], errors="coerce")
df["ln_rel_size"] = np.where(
    (acq_ta > 0) & (tar_ta > 0), np.log(acq_ta / tar_ta), np.nan
)

# ==============================================
# Post-deal Target financial ratios
# ==============================================
df["tar_post_ebitda_margin"] = safe_ratio(
    df["post_deal_tar_ebitda_1st_avail_yr"],
    df["post_deal_tar_rev_rev_1st_avail_yr"]
)

df["tar_post_roa"] = safe_ratio(
    df["post_deal_tar_pat_1st_avail_yr"],
    df["post_deal_tar_ta_1st_avail_yr"]
)

tar_post_ta = pd.to_numeric(df["post_deal_tar_ta_1st_avail_yr"], errors="coerce")
df["ln_tar_post_size"] = np.where(tar_post_ta > 0, np.log(tar_post_ta), np.nan)

post_ta = pd.to_numeric(df["post_deal_tar_ta_1st_avail_yr"], errors="coerce")
post_eq = pd.to_numeric(df["post_deal_tar_shareholder_funds_1st_avail_yr"], errors="coerce")
df["tar_post_leverage"] = safe_ratio(post_ta - post_eq, post_ta)

# ==============================================
# Post-deal Acquirer financial ratios (NEW 对称全套)
# ==============================================
# Post Acquirer EBITDA Margin
df["acq_post_ebitda_margin"] = safe_ratio(
    df["post_deal_acq_ebitda_1st_avail_yr"],
    df["post_deal_acq_rev_rev_1st_avail_yr"]
)

# Post Acquirer ROA
df["acq_post_roa"] = safe_ratio(
    df["post_deal_acq_pat_1st_avail_yr"],
    df["post_deal_acq_ta_1st_avail_yr"]
)

# Post Acquirer log total asset size
acq_post_ta = pd.to_numeric(df["post_deal_acq_ta_1st_avail_yr"], errors="coerce")
df["ln_acq_post_size"] = np.where(acq_post_ta > 0, np.log(acq_post_ta), np.nan)

# Post Acquirer leverage
acq_post_ta_df = pd.to_numeric(df["post_deal_acq_ta_1st_avail_yr"], errors="coerce")
acq_post_eq_df = pd.to_numeric(df["post_deal_acq_shareholder_funds_1st_avail_yr"], errors="coerce")
df["acq_post_leverage"] = safe_ratio(acq_post_ta_df - acq_post_eq_df, acq_post_ta_df)

# Winsorise all ratio variables (新增acq pre/post全部加入列表)
win_cols = [
    # Pre Target
    "tar_ebitda_margin", "tar_leverage", "tar_rev_growth", "tar_roa", "ln_tar_size",
    # Pre Acquirer 新增
    "acq_ebitda_margin", "acq_leverage", "acq_roa", "ln_acq_size", "ln_rel_size",
    # Post Target
    "tar_post_ebitda_margin", "tar_post_roa", "ln_tar_post_size", "tar_post_leverage",
    # Post Acquirer 新增
    "acq_post_ebitda_margin", "acq_post_roa", "ln_acq_post_size", "acq_post_leverage"
]
for col in win_cols:
    df[col] = winsorise(df[col], lo=WINSOR_LO, hi=WINSOR_HI)
    n_valid = df[col].notna().sum()
    log.info(f"  {col}: N={n_valid:,}  mean={df[col].mean():.4f}  median={df[col].median():.4f}")
# Winsorise ind_text_diversity (from Script 06) → ind_text_diversity_w
if "ind_text_diversity" in df.columns:
    df["ind_text_diversity_w"] = winsorise(
        pd.to_numeric(df["ind_text_diversity"], errors="coerce"),
        lo=WINSOR_LO, hi=WINSOR_HI
    )
    log.info(f"  ind_text_diversity_w: N={df['ind_text_diversity_w'].notna().sum():,}  "
             f"mean={df['ind_text_diversity_w'].mean():.4f}  "
             f"median={df['ind_text_diversity_w'].median():.4f}")
else:
    raise ValueError("ind_text_diversity missing from pipeline — re-run Script 06 first")

# =========================
# M&A advisor count aggregates (Module J)
# =========================
# select all advisor count columns
tar_adv_cols  = [c for c in df.columns if c.startswith("num_tar_adv_")]
acq_adv_cols  = [c for c in df.columns if c.startswith("num_acq_adv_")]
all_advisor_dummies = tar_adv_cols + acq_adv_cols

df["tar_total_advisor"] = df[tar_adv_cols].sum(axis=1)
df["acq_total_advisor"] = df[acq_adv_cols].sum(axis=1)

log.info(f"  tar_total_advisor: N={df['tar_total_advisor'].notna().sum():,} "
         f"mean={df['tar_total_advisor'].mean():.2f} median={df['tar_total_advisor'].median():.1f}")
log.info(f"  acq_total_advisor: N={df['acq_total_advisor'].notna().sum():,} "
         f"mean={df['acq_total_advisor'].mean():.2f} median={df['acq_total_advisor'].median():.1f}")

# winsor
win_cols.extend(["tar_total_advisor","acq_total_advisor"])
for col in ["tar_total_advisor","acq_total_advisor"]:
    df[col] = winsorise(df[col], lo=WINSOR_LO, hi=WINSOR_HI)
    n_valid = df[col].notna().sum()
    log.info(f"  {col} (winsorised): N={n_valid:,}  mean={df[col].mean():.4f}  median={df[col].median():.4f}")


# =========================
# Info‑source dummy aggregates (Module K, from 02_deal_master)
# =========================
source_keywords = ["Stock_Exchange","Website","Company_Press_Release","Electronic_Publication",
                   "Advisor_Submission","Miscellaneous","source_other"]
src_dummy_cols = [c for c in df.columns if any(k in c for k in source_keywords) and c.startswith("num_")]
all_source_dummies = src_dummy_cols

#df["total_info_source"] = df[src_dummy_cols].sum(axis=1)
df["total_info_source"] = df[src_dummy_cols].sum(axis=1, min_count=1)

log.info(f"  total_info_source: N={df['total_info_source'].notna().sum():,} "
         f"mean={df['total_info_source'].mean():.2f} median={df['total_info_source'].median():.1f}")

win_cols.extend(["total_info_source"])
df["total_info_source"] = winsorise(df["total_info_source"], lo=WINSOR_LO, hi=WINSOR_HI)
n_valid = df["total_info_source"].notna().sum()
log.info(f"  total_info_source (winsorised): N={n_valid:,}  mean={df['total_info_source'].mean():.4f}  median={df['total_info_source'].median():.4f}")    
# ═══════════════════════════════════════════════════════════════════════════════
# 5. DAYS TO COMPLETION  (H5, Table 14)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 6 — DaysToCompletion")
log.info("=" * 70)

# assumed_comp_d is now included in SD_KEEP (Script 01 Module C) and flows
# through the full pipeline into 07_deal_firm_benchmark.csv — no raw import needed.
assert "assumed_comp_d" in df.columns, (
    "assumed_comp_d missing — re-run Scripts 01-07 after adding it to SD_KEEP in Script 01"
)
log.info(f"  assumed_comp_d non-null: {df['assumed_comp_d'].notna().sum():,} / {len(df):,}")

# Convert Excel serial numbers to dates (origin: 1899-12-30)
def excel_serial_to_date(series):
    return pd.to_datetime(
        pd.to_numeric(series, errors="coerce"),
        unit="D", origin="1899-12-30", errors="coerce"
    )

df["announced_date"]    = excel_serial_to_date(df["announced_d"])
df["completed_date"]    = excel_serial_to_date(df["completed_d"])
df["assumed_comp_date"] = excel_serial_to_date(df["assumed_comp_d"])

# ── 剔除 Zephyr 的机械填充日期 ──
# (1) 730 簇：completed_d 缺失时 assumed_comp_d = announced_d + 730（占位外推）
_off  = (pd.to_numeric(df["assumed_comp_d"], errors="coerce")
         - pd.to_numeric(df["announced_d"], errors="coerce"))
_mech = _off.between(700, 760) & df["completed_date"].isna()
log.info(f"  730天机械填充置缺失: {int(_mech.sum()):,} 行")
df.loc[_mech, "assumed_comp_date"] = pd.NaT

# (2) 零时长：announced_d 缺失被填成 completed_d
_zero = (df["completed_date"].notna() & df["announced_date"].notna()
         & (df["completed_date"] == df["announced_date"]))
log.info(f"  零时长占位填充置缺失: {int(_zero.sum()):,} 行")
df.loc[_zero, "completed_date"] = pd.NaT
# ── 剔除 Zephyr 的机械填充日期 end──

# End date: completed_d → assumed_comp_d fallback（用 _date，不是 _d）
df["end_date"] = df["completed_date"].fillna(df["assumed_comp_date"])
log.info(f"  end_date dtype: {df['end_date'].dtype}")


# DaysToCompletion
df["DaysToCompletion"] = (df["end_date"] - df["announced_date"]).dt.days
n_neg = (df["DaysToCompletion"] <= 0).sum()
if n_neg > 0:
    log.warning(f"  {n_neg} deals with DaysToCompletion <= 0 — set to NaN")
df.loc[df["DaysToCompletion"] <= 0, "DaysToCompletion"] = np.nan
df["ln_days"] = np.where(df["DaysToCompletion"].notna(),
                          np.log(df["DaysToCompletion"]), np.nan)

n_days = df["DaysToCompletion"].notna().sum()
log.info(f"  DaysToCompletion non-null: {n_days:,} ({100*n_days/len(df):.1f}%)")
log.info(f"  DaysToCompletion: median={df['DaysToCompletion'].median():.0f} days, "
         f"mean={df['DaysToCompletion'].mean():.0f} days")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.5. YEARS OF BUSINESS
# ═══════════════════════════════════════════════════════════════════════════════

# 生成标的/收购方年龄
df["target_age"] = df["deal_year"] - df["tar_incorp_d_year"]
df["acquirer_age"] = df["deal_year"] - df["acq_incorp_d_year"]
# 注册年份晚于交易年份设为空
df.loc[df["target_age"] < 0, "target_age"] = np.nan
df.loc[df["acquirer_age"] < 0, "acquirer_age"] = np.nan

log.info(f"tar_incorp_d_year 有效样本：{df['tar_incorp_d_year'].notna().sum():,}")
log.info(f"target_age 有效样本：{df['target_age'].notna().sum():,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PEER FINANCIAL LOOKUP — FinSimGap + Table 6/7 aggregates
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 7 — FinSimGap + peer financial aggregates")
log.info("=" * 70)

# --- 7.1 Compute financial characteristics for ALL corpus rows ---
log.info("  Computing financial characteristics for corpus ...")

def safe_div(num, den, positive_den=True):
    """Divide num by den; return NaN when den is zero/negative/NaN."""
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    cond = (d.notna() & (d > 0)) if positive_den else (d.notna() & (d != 0))
    return np.where(cond, n / d, np.nan)

corpus["c_ebitda_margin"] = safe_div(
    corpus["pre_deal_tar_ebitda_last_avail_yr"],
    corpus["pre_deal_tar_rev_rev_last_avail_yr"]
)
corpus_ta = pd.to_numeric(corpus["pre_deal_tar_ta_last_avail_yr"], errors="coerce")
corpus_eq = pd.to_numeric(corpus["pre_deal_tar_eq_last_avail_yr"], errors="coerce")
corpus["c_leverage"] = np.where(corpus_ta > 0, (corpus_ta - corpus_eq) / corpus_ta, np.nan)

rev_ly  = pd.to_numeric(corpus["tar_rev_rev_last_avail_yr"], errors="coerce")
rev_y1  = pd.to_numeric(corpus["tar_rev_rev_yr__1"], errors="coerce")
corpus["c_rev_growth"] = np.where(rev_y1 > 0, (rev_ly - rev_y1) / rev_y1, np.nan)

corpus["c_roa"] = safe_div(
    corpus["pre_deal_tar_pat_last_avail_yr"],
    corpus["pre_deal_tar_ta_last_avail_yr"]
)

log.info(
    f"  c_ebitda_margin notna: {corpus['c_ebitda_margin'].notna().sum():,}  "
    f"c_leverage: {corpus['c_leverage'].notna().sum():,}  "
    f"c_rev_growth: {corpus['c_rev_growth'].notna().sum():,}"
)

# ── P0 修复：比率变量先过滤小分母，再 winsorise，最后才标准化 ──
# 原因：分母接近 0 时比率爆炸（实测 max=1,036,235），
#       sigma 被拉到 15298 → z-score 全被除没 → FinSimGap 变噪音
_MIN_DEN = 100          # 单位千，即 10 万；低于此视为分母不可靠
_rev_den = pd.to_numeric(corpus["pre_deal_tar_rev_rev_last_avail_yr"], errors="coerce")
_ta_den  = pd.to_numeric(corpus["pre_deal_tar_ta_last_avail_yr"], errors="coerce")
_y1_den  = pd.to_numeric(corpus["tar_rev_rev_yr__1"], errors="coerce")

corpus.loc[_rev_den < _MIN_DEN, "c_ebitda_margin"] = np.nan
corpus.loc[_ta_den  < _MIN_DEN, "c_leverage"]      = np.nan
corpus.loc[_y1_den  < _MIN_DEN, "c_rev_growth"]    = np.nan

for _v in ["c_ebitda_margin", "c_leverage", "c_rev_growth"]:
    _lo = corpus[_v].quantile(WINSOR_LO)
    _hi = corpus[_v].quantile(WINSOR_HI)
    corpus[_v] = corpus[_v].clip(_lo, _hi)
    log.info(f"  [P0-fix] {_v}: mu={corpus[_v].mean():.4f}  "
             f"sigma={corpus[_v].std():.4f}  N={corpus[_v].notna().sum():,}")
    
# --- 7.2 Standardise financial variables across full corpus ---
log.info("  Standardising financial variables across full corpus ...")

FIN_VARS = ["c_ebitda_margin", "c_leverage", "c_rev_growth"]
std_params = {}
for v in FIN_VARS:
    mu    = corpus[v].mean()
    sigma = corpus[v].std()
    corpus[f"{v}_z"] = (corpus[v] - mu) / sigma
    std_params[v] = (mu, sigma)
    log.info(f"  {v}: mu={mu:.4f}, sigma={sigma:.4f}")

# --- 7.3 Build corpus lookup dict: {_row_id -> array of values} ---
log.info("  Building corpus lookup dict ...")

# Build index for fast lookup
corpus_idx = corpus.set_index("_row_id")

def get_peer_stats(peer_ids):
    """
    peer_ids: list of int _row_id
    Returns: dict with per-peer arrays for financial vars (NaN-aware)
    """
    peer_data = corpus_idx.reindex(peer_ids)
    return peer_data

# --- 7.4 Compute per-deal peer distances and aggregates ---
log.info("  Computing peer distances for all focal deals ...")
log.info("  (This may take 1-2 minutes ...)")

# Pre-build numpy arrays from corpus for fast lookup
corpus_np = corpus_idx[
    ["c_ebitda_margin_z", "c_leverage_z", "c_rev_growth_z",
     "c_ebitda_margin", "c_leverage", "c_rev_growth", "c_roa",
     "pre_deal_tar_ta_last_avail_yr", "pre_deal_tar_rev_rev_last_avail_yr",
     "tar_sic1", "tar_sic2", "tar_sic3", "tar_sic4",
     # New columns for Table 6 (valuation multiples, Market Cap, EV, M/B)
     "pre_ebitda_mul_ly", "pre_rev_mul_ly", "pre_ebit_mul_ly",
     "deal_equity_value", "deal_modelled_enterprise_value",
     "pre_deal_tar_eq_last_avail_yr"]
].copy()

def compute_peer_metrics(peer_ids_str, focal_z, focal_sic3, focal_sic2, focal_sic1, focal_sic4,
                         peer_sims_str=""):
    """
    Given a pipe-separated string of peer _row_ids, compute:
    - Standardised Euclidean distance (for FinSimGap)
    - Mean financial characteristics (for Table 6)
    - SIC affiliation fractions (for Table 7 Panel A)
    - Per-SIC-level similarity breakdown (for Table 7 Panel B; requires peer_sims_str)
    Returns a flat dict.
    """
    result = {
        "dist_comp": np.nan, "dist_ebitda": np.nan, "dist_lev": np.nan, "dist_grow": np.nan,
        "peer_ta_median": np.nan, "peer_rev_median": np.nan, "peer_ebitda_margin_median": np.nan,
        "peer_leverage_median": np.nan, "peer_rev_growth_median": np.nan, "peer_roa_median": np.nan,
        "same_sic1_frac": np.nan, "same_sic2_frac": np.nan,
        "same_sic3_frac": np.nan, "same_sic4_frac": np.nan,
        # Table 6 additional peer financials
        "peer_ev_ebitda_median": np.nan, "peer_ev_rev_median": np.nan, "peer_ev_ebit_median": np.nan,
        "peer_mktcap_median": np.nan, "peer_ev_median": np.nan, "peer_mb_median": np.nan,
        # Table 7 Panel B: per-SIC-level mean similarity (same SIC vs diff SIC)
        "sim_same_sic1_mean": np.nan, "sim_diff_sic1_mean": np.nan,
        "sim_same_sic2_mean": np.nan, "sim_diff_sic2_mean": np.nan,
        "sim_same_sic3_mean": np.nan, "sim_diff_sic3_mean": np.nan,
        "sim_same_sic4_mean": np.nan, "sim_diff_sic4_mean": np.nan,
    }
    if pd.isna(peer_ids_str) or not str(peer_ids_str).strip():
        return result

    try:
        peer_ids = [int(x) for x in str(peer_ids_str).split("|") if x.strip()]
    except Exception:
        return result

    peers = corpus_np.reindex(peer_ids)

    # --- Financial distances ---
    z_cols = ["c_ebitda_margin_z", "c_leverage_z", "c_rev_growth_z"]
    peer_z = peers[z_cols].values   # shape (K, 3)
    focal_z_arr = np.array(focal_z, dtype=float)  # shape (3,)

    diff = peer_z - focal_z_arr   # (K, 3)
    # FIXER R1: F1 — All sub-distances and composite use only has_all peers (all 3 dims non-NaN).
    # Previously sub-distances used nanmean over ALL peers (different subset from composite) → incoherent.
    has_all = np.all(~np.isnan(peer_z), axis=1)
    if not has_all.any():
        # leave all dist_* as NaN
        pass
    else:
        diff_h = diff[has_all]                          # (n_has, 3)
        result["dist_ebitda"] = float(np.mean(np.abs(diff_h[:, 0])))
        result["dist_lev"]    = float(np.mean(np.abs(diff_h[:, 1])))
        result["dist_grow"]   = float(np.mean(np.abs(diff_h[:, 2])))
        # Composite: Euclidean distance from focal to centroid of has_all peers (D2 formula)
        centroid = np.mean(peer_z[has_all], axis=0)     # (3,)
        result["dist_comp"] = float(np.sqrt(np.sum((focal_z_arr - centroid) ** 2)))

    # --- Median financials (for Table 6; Eaton et al. Table 5 uses medians) ---
    result["peer_ta_median"]            = np.nanmedian(peers["pre_deal_tar_ta_last_avail_yr"].values.astype(float))
    result["peer_rev_median"]           = np.nanmedian(peers["pre_deal_tar_rev_rev_last_avail_yr"].values.astype(float))
    result["peer_ebitda_margin_median"] = np.nanmedian(peers["c_ebitda_margin"].values.astype(float))
    result["peer_leverage_median"]      = np.nanmedian(peers["c_leverage"].values.astype(float))
    result["peer_rev_growth_median"]    = np.nanmedian(peers["c_rev_growth"].values.astype(float))
    result["peer_roa_median"]           = np.nanmedian(peers["c_roa"].values.astype(float))

    # --- New Table 6 peer medians: valuation multiples, Market Cap, EV, M/B ---
    result["peer_ev_ebitda_median"] = np.nanmedian(peers["pre_ebitda_mul_ly"].values.astype(float))
    result["peer_ev_rev_median"]    = np.nanmedian(peers["pre_rev_mul_ly"].values.astype(float))
    result["peer_ev_ebit_median"]   = np.nanmedian(peers["pre_ebit_mul_ly"].values.astype(float))
    result["peer_mktcap_median"]    = np.nanmedian(peers["deal_equity_value"].values.astype(float))
    result["peer_ev_median"]        = np.nanmedian(peers["deal_modelled_enterprise_value"].values.astype(float))
    # M/B per peer = deal_equity_value / pre_deal_tar_eq_last_avail_yr; then take median
    eq_val  = peers["deal_equity_value"].values.astype(float)
    bk_eq   = peers["pre_deal_tar_eq_last_avail_yr"].values.astype(float)
    mb_vals = np.where((bk_eq != 0) & ~np.isnan(bk_eq), eq_val / bk_eq, np.nan)
    result["peer_mb_median"] = np.nanmedian(mb_vals)

    # --- SIC affiliation fractions (for Table 7) ---
    n = len(peer_ids)
    if n > 0:
        sic1 = peers["tar_sic1"].values
        sic2 = peers["tar_sic2"].values
        sic3 = peers["tar_sic3"].values
        sic4 = peers["tar_sic4"].values

        def frac_match(peer_sic, focal_sic):
            if pd.isna(focal_sic) or n == 0:
                return np.nan
            # peer_sic is a numpy float array (np.nan for missing)
            peer_float = pd.to_numeric(pd.Series(peer_sic.tolist()), errors="coerce").values
            return float(np.nansum(peer_float == float(focal_sic))) / n

        result["same_sic1_frac"] = frac_match(sic1, focal_sic1)
        result["same_sic2_frac"] = frac_match(sic2, focal_sic2)
        result["same_sic3_frac"] = frac_match(sic3, focal_sic3)
        result["same_sic4_frac"] = frac_match(sic4, focal_sic4)

    # --- Per-SIC-level similarity breakdown (Table 7 Panel B) ---
    # Only meaningful when peer_sims_str is provided (text peers from Script 06)
    if peer_sims_str and str(peer_sims_str).strip():
        try:
            peer_sims_arr = np.array(
                [float(x) for x in str(peer_sims_str).split("|") if x.strip()],
                dtype=float
            )
        except Exception:
            peer_sims_arr = np.array([])

        if len(peer_sims_arr) == len(peer_ids):
            for lv, focal_sic, sic_col in [
                (1, focal_sic1, "tar_sic1"),
                (2, focal_sic2, "tar_sic2"),
                (3, focal_sic3, "tar_sic3"),
                (4, focal_sic4, "tar_sic4"),
            ]:
                if pd.isna(focal_sic):
                    continue
                peer_sic_vals = pd.to_numeric(
                    pd.Series(peers[sic_col].values.tolist()), errors="coerce"
                ).values
                same_mask = peer_sic_vals == float(focal_sic)
                diff_mask = (~same_mask) & (~np.isnan(peer_sic_vals))

                same_sims = peer_sims_arr[same_mask]
                diff_sims = peer_sims_arr[diff_mask]

                result[f"sim_same_sic{lv}_mean"] = float(np.mean(same_sims)) if len(same_sims) > 0 else np.nan
                result[f"sim_diff_sic{lv}_mean"] = float(np.mean(diff_sims)) if len(diff_sims) > 0 else np.nan

    return result


# Get focal deal's own z-scores from corpus (by _row_id)
# Focal deals are a subset of corpus; look up their z-scores
focal_zscores = corpus_np[["c_ebitda_margin_z", "c_leverage_z", "c_rev_growth_z"]].reindex(
    df["_row_id"].values
)

# Get focal SIC codes (from df — already derived in Script 05)
focal_sic3 = df["tar_sic3"].values
focal_sic2 = df["tar_sic2"].values
# Need sic1 and sic4 for Table 7 — derive from tar_sic3 (3-digit) and tar_sic4 (4-digit)
focal_sic4_vals = pd.to_numeric(df.get("tar_sic4", pd.Series(dtype=float)), errors="coerce").values

# Derive SIC-1 from SIC-3
focal_sic3_num = pd.to_numeric(pd.Series(focal_sic3), errors="coerce").values
# FIXER R1: m3 — SIC-1 = first digit of 3-digit SIC code (0-9). Non-standard shorthand;
# standard SIC Divisions use letter ranges. Document in Table 7 footnote.
focal_sic1 = np.where(~np.isnan(focal_sic3_num), focal_sic3_num // 100, np.nan)

# Containers for results
text_results = []
sic_results  = []

for i, (idx, row) in enumerate(df.iterrows()):
    if i % 2000 == 0:
        log.info(f"  [{i:,}/{len(df):,}] Processing peer distances ...")

    focal_z = [
        focal_zscores.at[row["_row_id"], "c_ebitda_margin_z"],
        focal_zscores.at[row["_row_id"], "c_leverage_z"],
        focal_zscores.at[row["_row_id"], "c_rev_growth_z"],
    ]
    fsic3 = focal_sic3[i]
    fsic2 = focal_sic2[i]
    fsic1 = focal_sic1[i]
    fsic4 = focal_sic4_vals[i] if not pd.isna(focal_sic4_vals[i]) else np.nan

    text_results.append(compute_peer_metrics(
        row["text_peer_ids"], focal_z, fsic3, fsic2, fsic1, fsic4,
        peer_sims_str=str(row.get("text_peer_sim_list", ""))
    ))
    sic_results.append(compute_peer_metrics(
        row["sic_peer_ids"], focal_z, fsic3, fsic2, fsic1, fsic4,
        peer_sims_str=""  # no per-peer similarity scores for SIC benchmark
    ))

log.info("  Peer distance computation complete.")

text_df = pd.DataFrame(text_results, index=df.index)
sic_df  = pd.DataFrame(sic_results,  index=df.index)


# --- 7.6 Peer financial aggregates (Table 6; medians per Eaton et al.) ---
for prefix, res_df in [("text_peer", text_df), ("sic_peer", sic_df)]:
    df[f"{prefix}_ta_median"]            = res_df["peer_ta_median"]
    df[f"{prefix}_rev_median"]           = res_df["peer_rev_median"]
    df[f"{prefix}_ebitda_margin_median"] = res_df["peer_ebitda_margin_median"]
    df[f"{prefix}_leverage_median"]      = res_df["peer_leverage_median"]
    df[f"{prefix}_rev_growth_median"]    = res_df["peer_rev_growth_median"]
    df[f"{prefix}_roa_median"]           = res_df["peer_roa_median"]
    # New: valuation multiples, Market Cap, EV, M/B peer medians
    df[f"{prefix}_ev_ebitda_median"] = res_df["peer_ev_ebitda_median"]
    df[f"{prefix}_ev_rev_median"]    = res_df["peer_ev_rev_median"]
    df[f"{prefix}_ev_ebit_median"]   = res_df["peer_ev_ebit_median"]
    df[f"{prefix}_mktcap_median"]    = res_df["peer_mktcap_median"]
    df[f"{prefix}_ev_median"]        = res_df["peer_ev_median"]
    df[f"{prefix}_mb_median"]        = res_df["peer_mb_median"]

# Also add focal target financials for Table 6 (use already-constructed cols)
df["tar_ta_for_t6"]  = pd.to_numeric(df["pre_deal_tar_ta_last_avail_yr"], errors="coerce")
df["tar_rev_for_t6"] = pd.to_numeric(df["pre_deal_tar_rev_rev_last_avail_yr"], errors="coerce")

# New: focal target Market Cap, EV, and M/B for Table 6
df["tar_mktcap_for_t6"] = pd.to_numeric(df["deal_equity_value"], errors="coerce")
df["tar_ev_for_t6"]     = pd.to_numeric(df["deal_modelled_enterprise_value"], errors="coerce")
_eq_val = pd.to_numeric(df["deal_equity_value"], errors="coerce")
_bk_eq  = pd.to_numeric(df["pre_deal_tar_eq_last_avail_yr"], errors="coerce")
df["tar_mb_for_t6"] = np.where((_bk_eq != 0) & _bk_eq.notna(), _eq_val / _bk_eq, np.nan)

log.info(f"  tar_mktcap_for_t6 coverage: {df['tar_mktcap_for_t6'].notna().sum():,} / {len(df):,}")
log.info(f"  tar_ev_for_t6 coverage:     {df['tar_ev_for_t6'].notna().sum():,} / {len(df):,}")
log.info(f"  tar_mb_for_t6 coverage:     {df['tar_mb_for_t6'].notna().sum():,} / {len(df):,}")

# --- 7.7 SIC affiliation fractions (Table 7) ---
df["text_same_sic1_frac"] = text_df["same_sic1_frac"]
df["text_same_sic2_frac"] = text_df["same_sic2_frac"]
df["text_same_sic3_frac"] = text_df["same_sic3_frac"]
df["text_same_sic4_frac"] = text_df["same_sic4_frac"]
df["sic_same_sic1_frac"]  = sic_df["same_sic1_frac"]
df["sic_same_sic2_frac"]  = sic_df["same_sic2_frac"]
# sic_same_sic3_frac is always ~100% by construction (SIC-3 benchmark level)
df["sic_same_sic3_frac"]  = sic_df["same_sic3_frac"]   # keep for verification
df["sic_same_sic4_frac"]  = sic_df["same_sic4_frac"]

log.info(f"  text_same_sic3_frac: mean={df['text_same_sic3_frac'].mean():.3f} "
         f"(expected < 1 — text peers span industries)")
log.info(f"  sic_same_sic3_frac: mean={df['sic_same_sic3_frac'].mean():.3f} "
         f"(expected ≈ 1 — SIC peers matched at SIC-3)")

# --- 7.7b Per-SIC-level similarity breakdown (Table 7 Panel B) ---
for lv in [1, 2, 3, 4]:
    df[f"text_sim_same_sic{lv}_mean"] = text_df[f"sim_same_sic{lv}_mean"]
    df[f"text_sim_diff_sic{lv}_mean"] = text_df[f"sim_diff_sic{lv}_mean"]
    n_same = df[f"text_sim_same_sic{lv}_mean"].notna().sum()
    n_diff = df[f"text_sim_diff_sic{lv}_mean"].notna().sum()
    log.info(
        f"  SIC-{lv} sim breakdown: "
        f"same_sic mean={df[f'text_sim_same_sic{lv}_mean'].mean():.4f} (N={n_same:,})  "
        f"diff_sic mean={df[f'text_sim_diff_sic{lv}_mean'].mean():.4f} (N={n_diff:,})"
    )

# --- 7.8 SIC peer cosine similarity mean (Table 7) ---
# Load Doc2Vec model once; compute cosine similarity of focal deal vs each SIC peer.
log.info("Computing sic_peer_sim_mean from Doc2Vec model ...")
_model_path = os.path.join(ROOT, "data", "models", "doc2vec_full_corpus.model")
try:
    _d2v = Doc2Vec.load(_model_path)
    _trained_tags = set(_d2v.dv.index_to_key)

    def _cosine(a, b):
        """Cosine similarity between two 1-D arrays."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else np.nan

    def _sic_sim_mean(focal_rid, sic_ids_str):
        if pd.isna(sic_ids_str) or not str(sic_ids_str).strip():
            return np.nan
        if focal_rid not in _trained_tags:
            return np.nan
        focal_vec = _d2v.dv[focal_rid]
        sims = []
        for pid_str in str(sic_ids_str).split("|"):
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid in _trained_tags:
                sims.append(_cosine(focal_vec, _d2v.dv[pid]))
        return float(np.nanmean(sims)) if sims else np.nan

    df["sic_peer_sim_mean"] = [
        _sic_sim_mean(int(row["_row_id"]), row["sic_peer_ids"])
        for _, row in df.iterrows()
    ]
    n_sic_sim = df["sic_peer_sim_mean"].notna().sum()
    log.info(f"  sic_peer_sim_mean: N={n_sic_sim:,}, mean={df['sic_peer_sim_mean'].mean():.4f}")
    del _d2v   # free memory
except Exception as _e:
    log.warning(f"  Doc2Vec model load failed — sic_peer_sim_mean will be NaN: {_e}")
    df["sic_peer_sim_mean"] = np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# 8. ALTERNATIVE cross_industry + PEER OVERLAP  (Table 15e)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 8 — Alternative cross_industry (peer overlap)")
log.info("=" * 70)

def overlap_frac(text_ids_str, sic_ids_str):
    # FIXER R1: m5 — overlap_frac is asymmetric: fraction of TEXT peers also in SIC peer set.
    # Denominator = len(text_set) (up to K=10). Intentionally target-centric:
    # we ask how well the text peer set is covered by the industry peer set.
    # Document definition in Table 7 footnote.
    if pd.isna(text_ids_str) or pd.isna(sic_ids_str):
        return np.nan
    text_set = set(str(text_ids_str).split("|"))
    sic_set  = set(str(sic_ids_str).split("|"))
    if not text_set:
        return np.nan
    return len(text_set & sic_set) / len(text_set)

df["peer_overlap_frac"] = [
    overlap_frac(r["text_peer_ids"], r["sic_peer_ids"])
    for _, r in df.iterrows()
]
df["cross_industry_alt"] = np.where(
    df["peer_overlap_frac"].notna(),
    (df["peer_overlap_frac"] < 0.5).astype("Int64"),
    pd.NA
)

n_alt = df["cross_industry_alt"].sum(skipna=True)  # FIXER R1: m6 — skipna=True for nullable Int64
n_orig = df["cross_industry"].sum(skipna=True)      # FIXER R1: m6 — skipna=True for nullable Int64
log.info(f"  cross_industry (SIC-3 mismatch): {n_orig:,} ({100*n_orig/len(df):.1f}%)")
log.info(f"  cross_industry_alt (overlap<50%): {n_alt:,} ({100*n_alt/len(df):.1f}%)")
log.info(f"  peer_overlap_frac: mean={df['peer_overlap_frac'].mean():.3f}, "
         f"median={df['peer_overlap_frac'].median():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. COUNTRY-LEVEL SIC RELIABILITY PROXIES  (H3)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 9 — Country-level SIC reliability proxies")
log.info("=" * 70)

# --- 9.1 SIC Coverage Rate (computed from FULL corpus, not analysis sample) ---
# Analysis sample has SIC-3 by construction (sample filter 6 requires it → coverage = 100%).
# Must use 04b_deal_firm_country.csv (full corpus, ~58k rows) for meaningful variation.
sic_cov = (
    corpus.groupby("tar_country_code")["tar_sic3"]
    .apply(lambda x: x.notna().mean())
    .rename("sic_coverage_rate")
    .reset_index()
)
df = df.merge(sic_cov, on="tar_country_code", how="left")
log.info(f"  sic_coverage_rate (from full corpus): min={df['sic_coverage_rate'].min():.3f}, "
         f"max={df['sic_coverage_rate'].max():.3f}, "
         f"mean={df['sic_coverage_rate'].mean():.3f}")

# --- 9.2 La Porta 1998 Legal Origin ---
# Source: La Porta et al. (1998, JPE) Appendix Table 1
# Common law (English origin)
LEGAL_ORIGIN = {
    # English common law
    "AU": "English", "CA": "English", "GH": "English", "HK": "English",
    "IN": "English", "IE": "English", "JM": "English", "KE": "English",
    "MY": "English", "NZ": "English", "NG": "English", "PK": "English",
    "SG": "English", "ZA": "English", "TZ": "English", "UG": "English",
    "GB": "English", "US": "English", "ZW": "English", "BD": "English",
    "LK": "English", "ZM": "English",
    # French civil law (incl. most EU, LatAm, MENA)
    "BE": "French", "BR": "French", "CL": "French", "CO": "French",
    "EG": "French", "FR": "French", "GR": "French", "ID": "French",
    "IT": "French", "JO": "French", "LB": "French", "LU": "French",
    "MX": "French", "MA": "French", "NL": "French", "PE": "French",
    "PH": "French", "PT": "French", "RO": "French", "ES": "French",
    "TH": "French", "TN": "French", "TR": "French", "VE": "French",
    "CN": "French", "RU": "French", "UA": "French", "VN": "French",
    "KZ": "French", "AE": "French", "SA": "French", "QA": "French",
    "KW": "French", "BH": "French",
    # German civil law
    "AT": "German", "CZ": "German", "DE": "German", "HU": "German",
    "JP": "German", "KR": "German", "SK": "German", "CH": "German",
    "TW": "German", "PL": "German", "HR": "German", "SI": "German",
    "RS": "German",
    # Scandinavian
    "DK": "Scandinavian", "FI": "Scandinavian", "IS": "Scandinavian",
    "NO": "Scandinavian", "SE": "Scandinavian",
    # Additional countries not in La Porta (1998) original sample — assigned by
    # colonial/legal heritage following Djankov et al. (2002) and Berkowitz et al. (2003)
    # English common law heritage
    "KY": "English",   # Cayman Islands (British territory)
    "IL": "English",   # Israel (Ottoman/British Mandate → common law)
    "BM": "English",   # Bermuda (British territory)
    "MT": "English",   # Malta (British colony → common law)
    "CY": "English",   # Cyprus (British colony → common law)
    "MU": "English",   # Mauritius (British colony; mixed but classified English)
    "PG": "English",   # Papua New Guinea (Australian administered)
    "VG": "English",   # British Virgin Islands
    "MH": "English",   # Marshall Islands (US trust territory)
    "GI": "English",   # Gibraltar (British territory)
    "VC": "English",   # Saint Vincent (British colony)
    # French civil law heritage
    "BG": "French",    # Bulgaria (French-influenced civil law)
    "LT": "French",    # Lithuania (French-origin civil law, post-Soviet)
    "BY": "French",    # Belarus (Soviet civil law, French family)
    "UY": "French",    # Uruguay (Spanish/French civil law)
    "UZ": "French",    # Uzbekistan (Soviet civil law)
    "AR": "French",    # Argentina (Spanish/French civil law)
    "DZ": "French",    # Algeria (French civil law)
    "OM": "French",    # Oman (mixed, classified French)
    "PA": "French",    # Panama (Spanish civil law)
    "CD": "French",    # DR Congo (Belgian/French civil law)
    "MN": "French",    # Mongolia (Soviet civil law, French family)
    # German civil law heritage
    "EE": "German",    # Estonia (German legal tradition)
    "LV": "German",    # Latvia (German legal tradition)
    "ME": "German",    # Montenegro (former Yugoslav, German family)
    "BA": "German",    # Bosnia (former Yugoslav, German family)
    "MK": "German",    # North Macedonia (former Yugoslav, German family)
}
# ═══════════════════════════════════════════════════════════════════════════
# 目标国家的法系（原有）
# ═══════════════════════════════════════════════════════════════════════════
df["legal_origin_tar"] = df["tar_country_code"].map(LEGAL_ORIGIN)
df["common_law_tar"] = (df["legal_origin_tar"] == "English").astype("Int64")
df.loc[df["legal_origin_tar"].isna(), "common_law_tar"] = pd.NA

# ═══════════════════════════════════════════════════════════════════════════
# 收购方国家的法系（新增）
# ═══════════════════════════════════════════════════════════════════════════
df["legal_origin_acq"] = df["acq_country_code"].map(LEGAL_ORIGIN)
df["common_law_acq"] = (df["legal_origin_acq"] == "English").astype("Int64")
df.loc[df["legal_origin_acq"].isna(), "common_law_acq"] = pd.NA

# 保持向后兼容：原有的common_law指向目标国家
#df["legal_origin"] = df["legal_origin_tar"]
#df["common_law"] = df["common_law_tar"]

# 诊断输出
unmapped_tar = df.loc[df["legal_origin_tar"].isna(), "tar_country_code"].value_counts()
unmapped_acq = df.loc[df["legal_origin_acq"].isna(), "acq_country_code"].value_counts()

if len(unmapped_tar) > 0:
    log.warning(f"  Unmapped target countries: {unmapped_tar.to_dict()}")
if len(unmapped_acq) > 0:
    log.warning(f"  Unmapped acquirer countries: {unmapped_acq.to_dict()}")

log.info(f"  common_law_tar = 1: {df['common_law_tar'].sum():,} deals")
log.info(f"  common_law_acq = 1: {df['common_law_acq'].sum():,} deals")

# 对比：跨境交易中两者差异
cross_df = df[df["cross_border"]==1]
if len(cross_df) > 0:
    both_common = ((cross_df["common_law_tar"]==1) & (cross_df["common_law_acq"]==1)).sum()
    tar_only = ((cross_df["common_law_tar"]==1) & (cross_df["common_law_acq"]!=1)).sum()
    acq_only = ((cross_df["common_law_tar"]!=1) & (cross_df["common_law_acq"]==1)).sum()
    neither = ((cross_df["common_law_tar"]!=1) & (cross_df["common_law_acq"]!=1)).sum()
    log.info(f"\n  Cross-border deals legal origin breakdown:")
    log.info(f"    Both Common: {both_common:,}")
    log.info(f"    Target Common Only: {tar_only:,}")
    log.info(f"    Acquirer Common Only: {acq_only:,}")
    log.info(f"    Neither Common: {neither:,}")
    
# --- 9.3 National Classification Dummy ---
# = 1 if company has a NAICS code (North American standard: US, CA, MX)
# = 0 if no NAICS code (company uses NACE, JSIC, KSIC, CSRC, etc.)
# Source: tar_primary_naics_code passed through from Script 01 raw industry files
if "tar_primary_naics_code" not in df.columns:
    raise ValueError("tar_primary_naics_code missing — re-run Scripts 01–07 first")
df["national_class"] = df["tar_primary_naics_code"].notna().astype("Int64")
df.loc[df["tar_primary_naics_code"].isna() & df["tar_country_code"].isna(), "national_class"] = pd.NA
n_naics = df["national_class"].sum()
log.info(f"  national_class = 1 (NAICS-native): {n_naics:,} / {len(df):,} deals "
         f"({100*n_naics/len(df):.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 9b. CLASSIFICATION DISTANCE  (H3 mechanism — continuous moderator)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 9b — Classification distance (SIC cross-system noise proxy)")
log.info("=" * 70)

# Ordinal scale (0-4): distance between target's national classification
# system and US SIC/NAICS.  Higher = more noise when mapping to SIC peers.
CLASSIFICATION_DIST = {
    # 0: NAICS-native (no cross-system mapping)
    "US": 0, "CA": 0, "MX": 0,
    # 1: English common law — Orbis/Zephyr maps directly to SIC
    "GB": 1, "AU": 1, "NZ": 1, "IE": 1, "ZA": 1, "IN": 1,
    "SG": 1, "HK": 1, "MY": 1, "NG": 1, "KE": 1, "GH": 1,
    "IL": 1, "CY": 1, "MT": 1, "BM": 1, "KY": 1, "VG": 1,
    # 2: NACE / European civil law
    "DE": 2, "FR": 2, "IT": 2, "ES": 2, "NL": 2, "BE": 2,
    "AT": 2, "CH": 2, "SE": 2, "NO": 2, "DK": 2, "FI": 2,
    "PT": 2, "GR": 2, "LU": 2, "PL": 2, "CZ": 2, "HU": 2,
    "RO": 2, "SK": 2, "HR": 2, "RS": 2, "BG": 2, "LT": 2,
    "LV": 2, "EE": 2, "SI": 2, "MK": 2, "ME": 2, "BA": 2,
    "RU": 2, "TR": 2, "UA": 2, "BR": 2, "AR": 2, "CL": 2,
    "CO": 2, "PE": 2, "UY": 2, "SA": 2, "AE": 2, "QA": 2,
    "KW": 2, "BH": 2, "EG": 2, "MA": 2, "DZ": 2, "TN": 2,
    "JO": 2, "LB": 2, "KZ": 2, "UZ": 2, "BY": 2, "MN": 2,
    # 3: Asian classification systems (JSIC, KSIC, TSE, etc.)
    "JP": 3, "KR": 3, "TW": 3, "TH": 3, "ID": 3,
    "VN": 3, "PH": 3, "BD": 3, "PK": 3, "LK": 3,
    # 4: China (CSRC/GB — most distant from SIC)
    "CN": 4,
}

# 目标国家的分类距离20260814
df["classification_distance_tar"] = df["tar_country_code"].map(CLASSIFICATION_DIST)

# 收购方国家的分类距离
df["classification_distance_acq"] = df["acq_country_code"].map(CLASSIFICATION_DIST)

# 保留原有的classification_distance（指向目标国家，向后兼容）
df["classification_distance"] = df["classification_distance_tar"]

# 诊断输出
n_tar = df["classification_distance_tar"].notna().sum()
n_acq = df["classification_distance_acq"].notna().sum()
log.info(f"  classification_distance_tar: {n_tar:,}/{len(df):,} obs mapped ({100*n_tar/len(df):.1f}%)")
log.info(f"  classification_distance_acq: {n_acq:,}/{len(df):,} obs mapped ({100*n_acq/len(df):.1f}%)")

# 未映射的目标国家
unmapped_tar = df.loc[df["classification_distance_tar"].isna(), "tar_country_code"].value_counts()
if len(unmapped_tar) > 0:
    log.warning(f"  Unmapped target countries: {unmapped_tar.to_dict()}")

# 未映射的收购方国家
unmapped_acq = df.loc[df["classification_distance_acq"].isna(), "acq_country_code"].value_counts()
if len(unmapped_acq) > 0:
    log.warning(f"  Unmapped acquirer countries: {unmapped_acq.to_dict()}")

# 分布
log.info(f"\n  Target country distribution:\n{df['classification_distance_tar'].value_counts(dropna=False).sort_index()}")
log.info(f"\n  Acquirer country distribution:\n{df['classification_distance_acq'].value_counts(dropna=False).sort_index()}")

# 两种定义的交叉表
log.info(f"\n  Cross-tab (tar × acq):")
cross_tab = pd.crosstab(df['classification_distance_tar'], 
                         df['classification_distance_acq'],
                         margins=True)
log.info(f"\n{cross_tab}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. SAVE TO STATA .dta
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 10 — Save to Stata .dta")
log.info("=" * 70)

COLS_TO_SAVE = [
    # ── Identifiers ────────────────────────────────────────────────────────────
    "deal_num", "_row_id", "deal_year", 
    "tar_sic3", "tar_sic2", "acq_sic3",
    "legal_origin_tar","legal_origin_acq",
    # ── Module E 监管特征变量（01b_deal_overview合并变量） ─────────────────────
    "reg_body_count",
    "reg_country_count",
    "reg_antitrust",
    "reg_securities",
    "reg_state_assets",
    "reg_foreign_invest",
    "reg_financial",
    "reg_defense_tech",
    "reg_cross_national",
    "reg_common_law",
    "reg_civil_law",
    "reg_mixed_legal",
    "regulatory_bodies",
    "regulatory_countries",
    # ── 评论文本特征 Module F (2026-08-08新增) ─────────────────────────────
    "total_timeline_days",
    "has_rumour",
    "has_target_reject",
    "has_phase2_investigation",
    "has_goshop",
    "num_unique_reg",
    "deal_complexity_score",
    "comment_char_length",
    "comment_wordcount",
    "sentence_event_count",
    "lm_pos_count",
    "lm_neg_count",
    "lm_uncertain_count",
    "lm_litigious_count",
    "lm_strongmodal_count",
    "lm_weakmodal_count",
    "lm_constrain_count",
    "lm_pos_density",
    "lm_neg_density",
    "lm_uncertain_density",
    "lm_litigious_density",
    "lm_strongmodal_density",
    "lm_weakmodal_density",
    "lm_constrain_density",
    "lm_net_sentiment",
    # ========================= M&A Advisor dummy Module J 手写全部 =========================
    "num_tar_adv_acc",
    "num_tar_adv_asset_fin",
    "num_tar_adv_broker",
    "num_tar_adv_debt",
    "num_tar_adv_equity",
    "num_tar_adv_fa",
    "num_tar_adv_pr",
    "num_tar_adv_insurer",
    "num_tar_adv_law",
    "num_tar_adv_lead",
    "num_tar_adv_mezz",
    "num_tar_adv_other",
    "num_tar_adv_underwriter",
    "num_tar_adv_vcpe",

    "num_acq_adv_acc",
    "num_acq_adv_asset_fin",
    "num_acq_adv_broker",
    "num_acq_adv_debt",
    "num_acq_adv_equity",
    "num_acq_adv_fa",
    "num_acq_adv_pr",
    "num_acq_adv_insurer",
    "num_acq_adv_law",
    "num_acq_adv_lead",
    "num_acq_adv_mezz",
    "num_acq_adv_other",
    "num_acq_adv_underwriter",
    "num_acq_adv_vcpe",

    "tar_total_advisor",
    "acq_total_advisor",

    # ── Info‑source dummy variables (Module K, from 02_deal_master) ───────────
    "num_Stock_Exchange",
    "num_Website",
    "num_Company_Press_Release",
    "num_Electronic_Publication",
    "num_Advisor_Submission",
    "num_Miscellaneous",
    "num_source_other",
    "total_info_source",

    # ── Raw target pre-deal financials (03 Module F) ─────────────────────────
    "pre_deal_tar_rev_rev_last_avail_yr",
    "pre_deal_tar_ebitda_last_avail_yr",
    "pre_deal_tar_ebit_last_avail_yr",
    "pre_deal_tar_pbt_last_avail_yr",
    "pre_deal_tar_pat_last_avail_yr",
    "pre_deal_tar_np_last_avail_yr",
    "pre_deal_tar_ta_last_avail_yr",
    "pre_deal_tar_na_last_avail_yr",
    "pre_deal_tar_eq_last_avail_yr",
    "pre_deal_tar_current_liabilities_last_avail_yr",
    "pre_deal_tar_cap",

    # ── Raw target post-deal financials (03b Module Fb) ───────────────────────
    "post_deal_tar_rev_rev_1st_avail_yr",
    "post_deal_tar_ebitda_1st_avail_yr",
    "post_deal_tar_ebit_1st_avail_yr",
    "post_deal_tar_pbt_1st_avail_yr",
    "post_deal_tar_pat_1st_avail_yr",
    "post_deal_tar_np_1st_avail_yr",
    "post_deal_tar_ta_1st_avail_yr",
    "post_deal_tar_na_1st_avail_yr",
    #"post_deal_tar_eq_1st_avail_yr",          # KeyError 缺失，注释
    "post_deal_tar_current_liabilities_1st_avail_yr",
    #"post_deal_tar_cap_1st_avail_yr",         # KeyError 缺失，注释

    # ── Raw acquirer pre-deal financials ─────────────────────────────────────
    "pre_deal_acq_rev_rev_last_avail_yr",
    "pre_deal_acq_ebitda_last_avail_yr",
    "pre_deal_acq_pat_last_avail_yr",
    "pre_deal_acq_ta_last_avail_yr",
    "pre_deal_acq_eq_last_avail_yr",
    
    # ── Raw acquirer post-deal financials (03b Module Fb) ───────────────────
    "post_deal_acq_rev_rev_1st_avail_yr",
    "post_deal_acq_ebitda_1st_avail_yr",
    "post_deal_acq_ebit_1st_avail_yr",
    "post_deal_acq_pbt_1st_avail_yr",
    "post_deal_acq_pat_1st_avail_yr",
    "post_deal_acq_np_1st_avail_yr",
    "post_deal_acq_ta_1st_avail_yr",
    "post_deal_acq_na_1st_avail_yr",
    #"post_deal_acq_eq_1st_avail_yr",          # KeyError 缺失，注释
    #"post_deal_acq_current_liabilities_1st_avail_yr", # KeyError 缺失，注释
    #"post_deal_acq_cap_1st_avail_yr",         # KeyError 缺失，注释
    "post_deal_acq_shareholder_funds_1st_avail_yr",

    # ── Raw vendor pre-deal financials（按需保留，不需要可以注释）──────────────
    "pre_deal_ven_rev_rev_last_avail_yr",
    "pre_deal_ven_ta_last_avail_yr",

    # ── Dependent variables (raw multiples) ─────────────────────────────────────
    "pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly",
    "post_rev_mul_fy", "post_ebitda_mul_fy", "post_ebit_mul_fy",

    # ── Log multiples ─────────────────────────────────────────────────────────
    "ln_mul_rev", "ln_mul_ebitda", "ln_mul_ebit",
    "ln_post_mul_rev", "ln_post_mul_ebitda", "ln_post_mul_ebit",
    "ln_text_rev", "ln_text_ebitda", "ln_text_ebit",
    "ln_sic_rev",  "ln_sic_ebitda",  "ln_sic_ebit",

    # ── ind_text_diversity (Revised H2) ──────────────────────────────────────
    "ind_text_diversity", "ind_text_diversity_w",

    # ── DaysToCompletion (H5) ─────────────────────────────────────────────────
    "DaysToCompletion", "ln_days",

    # ── Firm age controls #20260727 ───────────────────────────────────────────
    "tar_incorp_d_year", "acq_incorp_d_year",
    "target_age", "acquirer_age",

    # ── Deal-level controls (pre-existing from Script 05) ─────────────────────
    "cross_industry", "cross_industry_alt", "ln_deal_value", "cross_border",
    #"deal_pay_method",  # cashKeyError缺失，注释掉
    
    # ── 支付方式（2026-09-07 新增，务必进入 Stata 导出）──
    "deal_pay_method",            # 原始单值（保留向后兼容）
    "deal_pay_method_all",        # 01a 去重前聚合的竖线分隔全集 ← 关键
    #"pay_method_count",
    "deal_pay_method_n",          # 改了名
    
    "pay_has_cash", "pay_has_shares", "pay_has_debt", "pay_has_other",
    "pay_n_class",
    "pay_pure_cash", "pay_pure_shares", "pay_pure_debt", "pay_pure_other",
    "pay_mix", "pay_unknown",
    "pay_mix_cash_shares", "pay_mix_with_shares",
    "pay_mix_with_cash", "pay_mix_with_debt",
    "pay_class",                  # 互斥类别标签（字符串，回归用）
    # ── 支付方式（2026-09-07 新增，务必进入 Stata 导出）end──
    # ── structure（2026-09-07 新增，务必进入 Stata 导出）──   
    "deal_struct_all", "deal_struct_n",
    "deal_fin_all",    "deal_fin_n",
    "deal_type_all",   "deal_type_n",
    # ── structure（2026-09-07 新增，务必进入 Stata 导出）end──

    # ── deal characteristics  ───────────────────────────────────
    "deal_value","deal_status",

    # ── Target firm controls ──────────────────────────────────────────────────
    "ln_tar_size", "tar_ebitda_margin",
    "tar_leverage", "tar_rev_growth", "tar_roa",
    "tar_post_ebitda_margin", "tar_post_roa", "ln_tar_post_size", "tar_post_leverage",
    

    # ── Acquirer controls ─────────────────────────────────────────────────────
    "ln_rel_size", "acq_roa",

    # ── Country-level controls & proxies ─────────────────────────────────────
    "tar_country_code","acq_country_code",
    "stock_traded_gdp_tar","stock_traded_gdp_acq", 
    "national_class",
    "common_law_tar","common_law_acq",

    # ── Raw target financials ─────────────────────────────────────
    "tar_ta_for_t6", "tar_rev_for_t6",
    "tar_mktcap_for_t6", "tar_ev_for_t6", "tar_mb_for_t6",

    # ── Listed-firm status (acquiror and target) ──────────────────────────────
    "tar_listed", "acq_listed",

    # ── SIC classification distance ────────────────────────────
    "sic_coverage_rate","acq_tar_similarity",
    "classification_distance_tar","classification_distance_acq" 
]

# ── 兼容处理：剔除已弃用列，补充 01c 新列 ──
_DEPRECATED_COLS = [
    "deal_complexity_score",
    "lm_pos_density", "lm_neg_density", "lm_uncertain_density",
    "lm_litigious_density", "lm_strongmodal_density",
    "lm_weakmodal_density", "lm_constrain_density",
    "lm_net_sentiment",
]

# 新增：comment 可得性哑变量（区分「无 comment」vs「有 comment 但无该类词」）
if "has_lm_litigious" in df.columns:
    df["has_comment"] = df["has_lm_litigious"].notna().astype("Int64")
    log.info(f"  has_comment: {df['has_comment'].sum():,} / {len(df):,} "
             f"({100*df['has_comment'].mean():.1f}%)")

_NEW_TEXT_COLS = [
    # 文本长度控制
    "comment_wordcount", "log_comment_wordcount", "comment_char_length",
    "sentence_event_count",
    # LM 计数
    "lm_pos_count", "lm_neg_count", "lm_uncertain_count", "lm_litigious_count",
    "lm_strongmodal_count", "lm_weakmodal_count", "lm_constrain_count",
    # LM 二值 presence
    "has_lm_pos", "has_lm_neg", "has_lm_uncertain", "has_lm_litigious",
    "has_lm_strongmodal", "has_lm_weakmodal", "has_lm_constrain",
    # 可得性
    "has_comment",
]

COLS_TO_SAVE = [c for c in COLS_TO_SAVE if c not in _DEPRECATED_COLS]
COLS_TO_SAVE = COLS_TO_SAVE + [c for c in _NEW_TEXT_COLS
                               if c in df.columns and c not in COLS_TO_SAVE]
# ── 补齐 STEP 4 新增财务比率（原 COLS_TO_SAVE 硬编码名单未同步）──
_RATIO_COLS = [
    # Pre Target
    "tar_ebitda_margin", "tar_leverage", "tar_rev_growth", "tar_roa", "ln_tar_size",
    # Pre Acquirer（2026 新增，此前被漏掉）
    "acq_ebitda_margin", "acq_leverage", "acq_roa", "ln_acq_size", "ln_rel_size",
    # Post Target
    "tar_post_ebitda_margin", "tar_post_roa", "ln_tar_post_size", "tar_post_leverage",
    # Post Acquirer（2026 新增，此前被漏掉）
    "acq_post_ebitda_margin", "acq_post_roa", "ln_acq_post_size", "acq_post_leverage",
    # 交易结构
    "stake_acq_pct", "stake_final_pct",
]
COLS_TO_SAVE = COLS_TO_SAVE + [c for c in _RATIO_COLS
                               if c in df.columns and c not in COLS_TO_SAVE]

# ── 审计：打印 df 里有但名单外的列，防止下次再漏 ──
_orphan = [c for c in df.columns if c not in COLS_TO_SAVE]
log.info(f"  未纳入导出的列（{len(_orphan)} 个，确认是否故意排除）: {_orphan[:60]}")

# 兜底：任何不在 df 里的列一律剔除，避免再次 KeyError
#_missing_cols = [c for c in COLS_TO_SAVE if c not in df.columns]
#if _missing_cols:
    #log.warning(f"  COLS_TO_SAVE 中以下列不存在，已剔除: {_missing_cols}")
#COLS_TO_SAVE = [c for c in COLS_TO_SAVE if c in df.columns]

log.info(f"  最终导出列数: {len(COLS_TO_SAVE)}")
df_save = df[COLS_TO_SAVE].copy()


# Rename _row_id → row_id (Stata rejects variable names starting with '_')
if "_row_id" in df_save.columns:
    df_save = df_save.rename(columns={"_row_id": "row_id"})

# Convert any Pandas Int64 / boolean dtypes to plain types for pyreadstat
for col in df_save.columns:
    if hasattr(df_save[col], "dtype"):
        if str(df_save[col].dtype) in ("Int64", "Int32", "boolean"):
            df_save[col] = df_save[col].astype("float64")

out_dta = os.path.join(MERGED, "08b_deal_firm_analysis.dta")
# FIXER R1: m8 — value labels for common_law and national_class
value_labels = {
    "common_law_tar": {0: "Civil law", 1: "Common law"},
    "common_law_acq": {0: "Civil law", 1: "Common law"},
    "national_class": {0: "Non-NAICS", 1: "NAICS"},
}
pyreadstat.write_dta(df_save, out_dta, version=15, variable_value_labels=value_labels)
log.info(f"  Saved: {out_dta}")
log.info(f"  Shape: {df_save.shape[0]:,} rows × {df_save.shape[1]} columns")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. VERIFICATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
log.info("=" * 70)
log.info("STEP 11 — Verification summary")
log.info("=" * 70)

checks = {
    "N in expected range [10,000-12,000]": 10000 <= len(df_save) <= 12000,
    "ln_days notna > 5k": df_save["ln_days"].notna().sum() > 5000,
    "common_law_tar has values": df_save["common_law_tar"].notna().sum() > 0,
    "common_law_acq has values": df_save["common_law_acq"].notna().sum() > 0,
    "sic_coverage has at least one obs": df_save["sic_coverage_rate"].notna().any(),
    "reg_body_count exists": "reg_body_count" in df_save.columns,
    "deal_complexity_score exists": "deal_complexity_score" in df_save.columns
}
for check, result in checks.items():
    status = "PASS" if result else "FAIL"
    log.info(f"  [{status}] {check}")

all_pass = all(checks.values())
log.info(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — review above'}")

log.info("\n=== Core Regression Key Statistics ===")
log.info(f"  DaysToCompletion valid obs: {df_save['ln_days'].notna().sum():,}")
log.info(f"  DaysToCompletion valid obs: {df_save['ln_days'].notna().sum():,}")
log.info(f"  Common law target deals: {int(df_save['common_law_tar'].sum()):,}")
log.info(f"  Common law acquirer deals: {int(df_save['common_law_acq'].sum()):,}")

# ---------------- 评论文本特征统计 ----------------
log.info("\n=== Comment Text Feature Key Stats ===")
cmt_stats = ["deal_complexity_score", "lm_net_sentiment", "comment_wordcount", "comment_char_length"]
for v in cmt_stats:
    if v in df_save.columns:
        valid = df_save[v].notna().sum()
        log.info(f"  {v:<35s}: N={valid:,}  median={df_save[v].median():.4f}")

# ---------------- 监管变量统计 ----------------
log.info("\n=== Regulatory Review Variable Stats ===")
reg_dummy_list = ["reg_antitrust","reg_securities","reg_state_assets","reg_foreign_invest","reg_financial","reg_defense_tech","reg_cross_national","reg_common_law","reg_civil_law","reg_mixed_legal"]
for v in reg_dummy_list:
    if v in df_save.columns:
        cnt = int(df_save[v].sum())
        log.info(f"  {v:<35s} = 1 count: {cnt:,}")

if "reg_body_count" in df_save.columns:
    log.info(f"  reg_body_count min/max/mean: {df_save['reg_body_count'].min()} / {df_save['reg_body_count'].max()} / {df_save['reg_body_count'].mean():.3f}")
    log.info(f"  reg_country_count mean: {df_save['reg_country_count'].mean():.3f}")

log.info("\n=== Script 08 complete ===")
logging.shutdown()


# ========== 08b 末尾：导出 ==========
from global_config import MERGED

dta_path = os.path.join(MERGED, "08b_deal_firm_analysis.dta")
df, meta = pyreadstat.read_dta(dta_path)

df.head(10)
print("变量个数：", df.shape[1])
print("样本量：", df.shape[0])

csv_path = os.path.join(MERGED, "08b_temp_export.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
