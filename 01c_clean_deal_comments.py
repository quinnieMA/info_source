# -*- coding: utf-8 -*-
"""
Extract M&A Deal Comment Text Features with Loughran-McDonald Financial Dictionary
Author: CC  Date: 2026-08-08
Revised: 2026-09-06 — docstring aligned with actually-emitted columns;
         regulatory measures demoted to SECONDARY
Input Source: raw/MA_deal/acquisition_comments.csv
Output Target: data/cleaned/01c_comments_features.csv
Diagnostic Log: data/merged/01c_comments_diagnostics.txt

Core Processing Logic:
This script extracts structured numerical and textual sentiment metrics from Zephyr's raw deal comment editorial narratives,
and eliminates bulky unstructured raw text columns before export to control file storage size.

1. Text Sentiment Pipeline
    Load standard Loughran-McDonald (2011) financial word dictionary; build word-bound regular expressions
    for seven textual semantic categories: Positive, Negative, Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining.
    For each category emit a RAW COUNT (lm_*_count) and a BINARY PRESENCE flag (has_lm_*).

    ⚠️ Columns NOT emitted (this docstring previously implied otherwise):
      - lm_*_density       EMIT_LM_DENSITY = False. Median comment is ~88 words and
                           most LM categories hit <1 time, so density = count/88 is
                           not interpretable.
      - lm_net_sentiment   Never computed. LM (2011) positive vs negative word lists
                           differ ~6.6x in size (354 vs 2355), so a net score is
                           systematically negative; the authors advise against it.
      - deal_complexity_score  EMIT_COMPLEXITY = False (deprecated, code retained).
                           See the note above calc_complex() for the four reasons.

2. Deal Timeline & Negotiation Feature Extraction
    Use regular expressions to parse all date strings within comment text; derive earliest/latest event dates
    and total timeline span (days from first reported rumour to latest closing/update event).
    Binary indicator flags for key transaction milestones: rumour emergence, target board rejection,
    unconditional offer, deal completion, Go-shop clause existence, Phase 2 antitrust investigation,
    regulatory remedy/divestment requirements, debt financing arrangements.
    Count metrics: total currency price mentions, unique regulatory bodies, competitive rival bidders.

3. Transaction Complexity Composite Index  → DEPRECATED, not emitted. See (1).

4. Data Clean & Storage Optimization
    Drop raw long-text fields (comments, editorial, Deal rationale, reg_entity_list) before saving
    output CSV to avoid GB-level oversized files.
    Only retain derived numerical features for subsequent master dataset merge in Script 02_merge_deal_master.py.

──────────────────────────────────────────────────────────────────────────────
⚠️ REGULATORY MEASURES HERE ARE SECONDARY — DO NOT USE AS PRIMARY (2026-09-06)
──────────────────────────────────────────────────────────────────────────────
  This module produces a TEXT-DERIVED regulatory measure by regex-scanning the
  comment narrative:
      reg_event_count, num_unique_reg, has_phase2_investigation, has_reg_remedy

  It is NOT the primary regulatory measure. Two hard limitations:

  (a) The keyword list (reg_keywords, 13 entries) covers almost exclusively
      WESTERN ANTITRUST agencies — European Commission, CMA, FTC, DOJ, CADE,
      SAMR, FCC. It contains NO securities/exchange regulators (CSRC, SEC,
      SEBI, SFC, BaFin, ...). Deals reviewed through the securities route
      therefore score 0 BY CONSTRUCTION.

  (b) Coverage is ~1.84% of rows, versus ~12.8% for the STRUCTURED reg_*
      variables built in 01b_deal_overview.py from the `regulatory_body_name`
      field (443 distinct authority names, three-tier resolution). The text
      measure misses roughly 7x as many reviewed deals.

  ⇒ Empirical work uses the 01b structured reg_* variables.
    The 01c text measures may serve only as a robustness / alternative measure.

  (This is exactly the confusion recorded in 08b's docstring, which wrongly
   attributed reg_antitrust / reg_securities to "01c comments". Those come
   from 01b. Corrected 2026-09-06.)

──────────────────────────────────────────────────────────────────────────────
Unit of observation — VERIFY BEFORE RELYING ON IT:
  Historically stated as "single editorial comment record (one row per Zephyr
  news entry per deal)". Script 02 merges this file on deal_num and assumes a
  ONE-TO-ONE match. If any deal carries more than one editorial, that merge
  silently duplicates master rows. The diagnostics file reports both
  `Total rows processed` and `Unique deal_num` — confirm they are EQUAL before
  treating the 02 merge as safe.

Dictionary Reference:
Loughran, T., & McDonald, B. (2011). When are liability risk disclosures informative? Journal of Finance.
"""
import re
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# ====================== Path Configuration ======================
# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = r"D:\MA"
CLEANED = os.path.join(BASE, "data", "cleaned")
MERGED  = os.path.join(BASE, "data", "merged")
os.makedirs(MERGED, exist_ok=True)
comment_path = os.path.join(BASE, "raw", "MA_deal", "acquisition_comments.csv")
lm_dict_path = os.path.join(BASE, "raw", "Loughran-McDonald_MasterDictionary_1993-2021.csv")
#out_path = r"D:\FlashCenter\LQ\raw\01-deals\comments\comments_features.csv"
# Auto-create directories
os.makedirs(CLEANED, exist_ok=True)
os.makedirs(MERGED, exist_ok=True)
# Simple log printing function
def log(msg):
    print(msg)
# ====================== Output Switches ======================
# Design principle: each metric should be simple enough to be described in one sentence, including its unit.
#   In this sample, the median comment length is 88 words (p25=65, p75=137), and most LM categories have
#   expected hit counts below 1 (pos 0.20 / litigious 0.18 / constrain 0.08),
#   in which case count is essentially presence, and density = count/88 is even harder to interpret.
EMIT_LM_DENSITY  = False   # LM density series: off by default (set to True when needed)
EMIT_LM_PRESENCE = True    # LM binary presence: emitted by default
EMIT_COMPLEXITY  = False   # deal_complexity_score: deprecated, code kept
# ====================== 1 Load Loughran-McDonald Financial Dictionary [Fully Rewritten & Fixed] ======================
lm_df = pd.read_csv(lm_dict_path)
lm_df["Word_lower"] = lm_df["Word"].str.lower().str.strip()
# Correct filter: a column value != 0 means the word belongs to that category (non-zero is the year of first inclusion)
pos_words = lm_df[lm_df["Positive"] != 0]["Word_lower"].dropna().tolist()
neg_words = lm_df[lm_df["Negative"] != 0]["Word_lower"].dropna().tolist()
uncert_words = lm_df[lm_df["Uncertainty"] != 0]["Word_lower"].dropna().tolist()
litigious_words = lm_df[lm_df["Litigious"] != 0]["Word_lower"].dropna().tolist()
strong_modal = lm_df[lm_df["Strong_Modal"] != 0]["Word_lower"].dropna().tolist()
weak_modal = lm_df[lm_df["Weak_Modal"] != 0]["Word_lower"].dropna().tolist()
constrain_words = lm_df[lm_df["Constraining"] != 0]["Word_lower"].dropna().tolist()
# Print vocabulary sizes to confirm they are non-empty
# Print vocabulary sizes to confirm they are non-empty
log(f"Positive vocab size: {len(pos_words)}")
log(f"Negative vocab size: {len(neg_words)}")
log(f"Uncertainty vocab size: {len(uncert_words)}")
log(f"Litigious vocab size: {len(litigious_words)}")
log(f"Strong_Modal vocab size: {len(strong_modal)}")
log(f"Weak_Modal vocab size: {len(weak_modal)}")
log(f"Constraining vocab size: {len(constrain_words)}")
def build_word_re(word_list):
    if len(word_list) == 0:
        return re.compile(r"^$")
    word_esc = [re.escape(w) for w in word_list]
    # (?:...) non-capturing group to avoid duplicate-counting bugs in findall
    pat_str = r"\b(?:{})\b".format("|".join(word_esc))
    return re.compile(pat_str, re.IGNORECASE)
pos_pat = build_word_re(pos_words)
neg_pat = build_word_re(neg_words)
uncert_pat = build_word_re(uncert_words)
litigious_pat = build_word_re(litigious_words)
strong_modal_pat = build_word_re(strong_modal)
weak_modal_pat = build_word_re(weak_modal)
constrain_pat = build_word_re(constrain_words)
# ====================== 2 Objective Deal Fixed Regexes (Unchanged) ======================
date_pat = re.compile(r"(\d{2}/\d{2}(?:/\d{2,4})?)")
reg_keywords = [
    "European Commission", "EU Commission", "CMA", "FTC", "DOJ",
    "CADE", "Fiscalia Nacional Economica", "SAMR", "Korea's FTC",
    "Federal Communications Commission", "Hart-Scott-Rodino", "Phase 1", "Phase 2"
]
reg_pat = re.compile("|".join([re.escape(k) for k in reg_keywords]), re.IGNORECASE)
price_pat = re.compile(
    r"(?:EUR|USD|GBP|CHF|SEK|DKK|NOK|JPY|CNY|HKD|SGD|INR|KRW|AUD|CAD|\$|€|£|¥)\s*"
    r"[\d,.]+(?:\s*(?:billion|bn|million|mm|m)(?![a-z]))?",
    re.IGNORECASE
)
premium_pat = re.compile(r"premium of\s*(\d+\.\d+)\s*per cent|premium.*?(\d+\.\d+)", re.IGNORECASE)
rival_pat = re.compile(r"rival|suitor|white knight|bid for|poised to bid", re.IGNORECASE)
rumour_pat = re.compile(r"reported that|rumour|speculation", re.IGNORECASE)
reject_pat = re.compile(r"rejected", re.IGNORECASE)
unconditional_pat = re.compile(r"unconditional", re.IGNORECASE)
complete_pat = re.compile(r"completed|closed the deal", re.IGNORECASE)
go_shop_pat = re.compile(r"go-shop", re.IGNORECASE)
debt_fin_pat = re.compile(r"loan|syndicated debt|assumption of debt", re.IGNORECASE)
remedy_pat = re.compile(r"divest|sell off asset|remedy|concessions", re.IGNORECASE)
# ====================== 3 General Utility Functions ======================
def parse_date(d_str: str):
    """
    Handle European DD/MM/YY and DD/MM/YYYY formats.
    Supports three-part dates such as 13/11/99 and 23/01/2004; two-part dates such as 06/02 are dropped (no year).
    """
    try:
        parts = d_str.split("/")
        day = int(parts[0])
        month = int(parts[1])
        # Only process three-part dates with a year; two-part dates without a year return None
        if len(parts) != 3:
            return None
        
        year_raw = int(parts[-1])
        if len(str(year_raw)) == 4:
            # Four-digit year, use as is
            year = year_raw
        else:
            # Two-digit year handling
            if year_raw <= 30:
                year = 2000 + year_raw
            else:
                year = 1900 + year_raw
        # Strict date validity check, filtering invalid dates such as 31/04 or 30/02
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        
        # Build the date object (validates days-per-month automatically, e.g., Feb 30 raises an error)
        return datetime(year, month, day)
    except Exception:
        # Out-of-range numbers and invalid dates all return None
        return None
    
    
def extract_all_dates(text):
    raw = date_pat.findall(text)
    dt_list = []
    for d in raw:
        dt = parse_date(d)
        if dt is not None:
            dt_list.append(dt)
    unique_dt = sorted(list(set(dt_list)))
    return unique_dt
def count_match(text, pattern):
    return len(pattern.findall(text))
def has_match(text, pattern):
    return 1 if pattern.search(text) else 0
def get_premium_values(text):
    res = premium_pat.findall(text)
    nums = []
    for t1, t2 in res:
        if t1 and t1.strip():
            nums.append(float(t1))
        if t2 and t2.strip():
            nums.append(float(t2))
    return nums
# ====================== 4 Feature Extraction Functions [Core Fix Area] ======================
def extract_features(row):
    raw_text = str(row["comments"]).strip()
    # ========== Fix 1: removed text_lower = raw_text.lower() ==========
    # Regexes already carry re.IGNORECASE; manual lowercasing is not needed and would break word-boundary \b matching
    if len(raw_text) < 10:
        return {}
    word_total = len(raw_text.split())
    dates = extract_all_dates(raw_text)
    dt_min = dates[0] if len(dates) > 0 else None
    dt_max = dates[-1] if len(dates) > 0 else None
    total_days = (dt_max - dt_min).days if (dt_min is not None and dt_max is not None) else None
    # Basic deal facts
    has_rumour = has_match(raw_text, rumour_pat)
    has_reject = has_match(raw_text, reject_pat)
    has_unconditional = has_match(raw_text, unconditional_pat)
    has_complete = has_match(raw_text, complete_pat)
    price_count = count_match(raw_text, price_pat)
    prem_nums = get_premium_values(raw_text)
    max_prem = max(prem_nums) if len(prem_nums) > 0 else None
    prem_mean = sum(prem_nums)/len(prem_nums) if len(prem_nums) > 0 else None
    rival_count = count_match(raw_text, rival_pat)
    has_goshop = has_match(raw_text, go_shop_pat)
    reg_matches = reg_pat.findall(raw_text)
    reg_entities = list(set(reg_matches))
    reg_count = len(reg_matches)
    has_phase2 = 1 if "phase 2" in raw_text.lower() else 0
    has_remedy = has_match(raw_text, remedy_pat)
    has_debt_fin = has_match(raw_text, debt_fin_pat)
    char_len = len(raw_text)
    word_cnt = len(raw_text.split())
    sentence_list = [s.strip() for s in raw_text.split(".") if len(s.strip()) > 5]
    event_tokens = len(sentence_list)
    # ========== Fix 2: always pass raw_text, never the lowercased text ==========
    pos_cnt = count_lm_term(raw_text, pos_pat)
    neg_cnt = count_lm_term(raw_text, neg_pat)
    uncert_cnt = count_lm_term(raw_text, uncert_pat)
    lit_cnt = count_lm_term(raw_text, litigious_pat)
    strong_mod_cnt = count_lm_term(raw_text, strong_modal_pat)
    weak_mod_cnt = count_lm_term(raw_text, weak_modal_pat)
    constrain_cnt = count_lm_term(raw_text, constrain_pat)
    # Tokenization fix, resolving the density > 1 anomaly
    all_tokens = [t for t in re.split(r"\s+", raw_text) if t.strip()]
    word_total = len(all_tokens)
    word_cnt = word_total
    # ===== Binary presence: the main metric for short texts; "whether a category appears" is clear in one sentence =====
    pres = {}
    if EMIT_LM_PRESENCE:
        pres = {
            "has_lm_pos":         int(pos_cnt > 0),
            "has_lm_neg":         int(neg_cnt > 0),
            "has_lm_uncertain":   int(uncert_cnt > 0),
            "has_lm_litigious":   int(lit_cnt > 0),
            "has_lm_strongmodal": int(strong_mod_cnt > 0),
            "has_lm_weakmodal":   int(weak_mod_cnt > 0),
            "has_lm_constrain":   int(constrain_cnt > 0),
        }
    # ===== Density series (off by default) =====
    # Note 1: comments in this sample are short, so density units are not intuitive; the main analysis uses count / presence.
    # Note 2: lm_net_sentiment was not adopted — the LM (2011) positive and negative word lists differ 6.6x in size
    #      (Positive 354 vs Negative 2355), making the net value systematically negative;
    #      the original paper explicitly advises against constructing a net tone measure and requires separate reporting.
    dens = {}
    if EMIT_LM_DENSITY:
        dens = {
            "lm_pos_density":         pos_cnt / word_total if word_total > 0 else 0,
            "lm_neg_density":         neg_cnt / word_total if word_total > 0 else 0,
            "lm_uncertain_density":   uncert_cnt / word_total if word_total > 0 else 0,
            "lm_litigious_density":   lit_cnt / word_total if word_total > 0 else 0,
            "lm_strongmodal_density": strong_mod_cnt / word_total if word_total > 0 else 0,
            "lm_weakmodal_density":   weak_mod_cnt / word_total if word_total > 0 else 0,
            "lm_constrain_density":   constrain_cnt / word_total if word_total > 0 else 0,
        }
    feat = {
        # Timeline
        "dt_first": dt_min,
        "dt_last": dt_max,
        "total_timeline_days": total_days,
        "has_rumour": has_rumour,
        "has_target_reject": has_reject,
        "has_unconditional_offer": has_unconditional,
        "has_deal_complete": has_complete,
        # Premium
        "price_event_num": price_count,
        "max_premium_pct": max_prem,
        "avg_premium_pct": prem_mean,
        # Competitive bidding
        "rival_bidder_num": rival_count,
        "has_goshop": has_goshop,
        # Regulatory (text-regex measure; coverage is only 1.84% — use the structured reg_* variables from 01b as the primary measure)
        "reg_event_count": reg_count,
        "has_phase2_investigation": has_phase2,
        "has_reg_remedy": has_remedy,
        "reg_entity_list": reg_entities,
        "num_unique_reg": len(reg_entities),
        # Financing
        "has_debt_assumption": has_debt_fin,
        # Text basics
        "comment_char_length": char_len,
        "comment_wordcount": word_cnt,
        "sentence_event_count": event_tokens,
        # LM counts (primary measure: how many times each category appears)
        "lm_pos_count": pos_cnt,
        "lm_neg_count": neg_cnt,
        "lm_uncertain_count": uncert_cnt,
        "lm_litigious_count": lit_cnt,
        "lm_strongmodal_count": strong_mod_cnt,
        "lm_weakmodal_count": weak_mod_cnt,
        "lm_constrain_count": constrain_cnt,
        **pres,
        **dens,
    }
    return feat
# Rewrite the counting function to use finditer for precise counting, avoiding findall pitfalls
def count_lm_term(text: str, pat: re.Pattern) -> int:
    cnt = 0
    for _ in pat.finditer(text):
        cnt +=1
    return cnt
# ====================== 5 Deal Complexity Function ======================
# WARNING: Deprecated (EMIT_COMPLEXITY = False); code retained in case the text source is later swapped and the metric is restarted.
# Deprecation rationale (diagnosed 2026-09, 55,583 rows):
#   1. Single-component dominance: price_event_num>=2 alone contributes about 74% of the mean and about 2/3 of the variance;
#      despite the name "composite index", it effectively measures "how many times Zephyr wrote a price in EUR/USD/GBP".
#   2. Institutional bias: reg_keywords only covers Western antitrust agencies (EC/CMA/FTC/DOJ/SAMR...),
#      with no securities regulators (CSRC/SEC/SEBI/SFC/BaFin...),
#      so deals reviewed via the securities route (84% are Chinese targets) systematically score 0.
#      01c regex hit rate is 1.84% vs 12.8% for the structured 01b records, missing about 7x.
#   3. Dead code: round(lm_uncertain_density*10) triggers on only 0.05% of rows (28/55,583),
#      has_phase2_investigation 0.04% and has_goshop 0.01%, all pure noise.
#   4. Zero inflation: 86.08% of observations are 0, p50=p75=0, making OLS coefficients hard to interpret.
# Alternative: use the structured reg_* binary variables from 01b directly for regulatory friction; simpler and country-neutral.
def calc_complex(row):
    score = 0
    if row["rival_bidder_num"] > 0:
        score += 1
    if row["has_goshop"] == 1:
        score += 1
    if row["reg_event_count"] >= 1:
        score += row["num_unique_reg"]
    if row["has_phase2_investigation"] == 1:
        score += 3
    if row["has_reg_remedy"] == 1:
        score += 2
    if row["price_event_num"] >= 2:
        score += 2
    if row["has_debt_assumption"] == 1:
        score += 1
    # Handle NaN; missing values are treated as 0
    unc_val = row["lm_uncertain_density"] if pd.notna(row["lm_uncertain_density"]) else 0
    score += round(unc_val * 10)
    return score
# ====================== 6 Main Execution Flow ======================
import csv, sys
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))   # Zephyr long text may hit the CSV field size limit
df = pd.read_csv(
    comment_path,
    encoding="utf-8-sig",
    usecols=["Deal Number", "Date of editorial", "Deal comments"],
    engine="c",                 # The C engine handles newlines inside quotes correctly and is an order of magnitude faster
    on_bad_lines="warn",        # Changed from skip to warn; stop silently dropping rows
    dtype={"Deal Number": "str", "Deal comments": "str"},
)
df.rename(columns={
    "Deal Number": "deal_num",
    "Date of editorial": "edit_date",
    "Deal comments": "comments"
}, inplace=True)
_dn = pd.to_numeric(df["deal_num"], errors="coerce")
_bad = _dn.isna() | (_dn % 1 != 0) | (_dn < 1e4) | (_dn >= 1e13)
log(f"[check] 读取 {len(df):,} 行 | deal_num 非法 {int(_bad.sum()):,} 行 "
    f"| unique {int(_dn.nunique()):,}")
if _bad.any():
    log(f"[check] 非法样例: {df.loc[_bad, 'deal_num'].head(5).tolist()}")
    df = df.loc[~_bad].copy()
    
print("开始抽取Deal comments特征（LM金融词典版）...")
feat_result = df.apply(lambda row: extract_features(row), axis=1)
feat_df = pd.DataFrame(feat_result.tolist())
# Fix: must concat with axis=1 to join the original table and the feature table side by side
df_full = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
# ========== Cleanup: short-text rows (extract_features returns an empty dict → all features are NaN) ==========
if "comment_wordcount" in df_full.columns:
    _n0 = len(df_full)
    df_full = df_full[df_full["comment_wordcount"].notna()].copy()
    df_full = df_full.reset_index(drop=True)
    log(f"[clean] 删除空特征行: {_n0 - len(df_full):,} 行 | 剩余 {len(df_full):,} 行")
else:
    log("[clean] 警告：未找到 comment_wordcount 列，跳过清理")
    
# Calculate the composite complexity score (deprecated; see the note above calc_complex)
if EMIT_COMPLEXITY:
    df_full["deal_complexity_score"] = df_full.apply(calc_complex, axis=1)
    log("[info] deal_complexity_score 已生成")
else:
    log("[info] deal_complexity_score 已跳过 (EMIT_COMPLEXITY=False)")
df_full["log_comment_wordcount"] = np.log1p(df_full["comment_wordcount"])
# ========== Added: drop raw long-text fields before export ==========
drop_raw_text = [
    "editorial",
    "Deal rationale",
    "comments", 
    "reg_entity_list"
]
drop_cols = [c for c in drop_raw_text if c in df_full.columns]
if len(drop_cols) > 0:
    df_full = df_full.drop(columns=drop_cols)
    log(f"Dropped raw heavy text columns before export: {drop_cols}")
    
# Standardized output file path
out_comments = os.path.join(CLEANED, "01c_comments_features.csv")
df_full.to_csv(out_comments, index=False, encoding="utf-8-sig")
log(f"\nSaved → {out_comments}")
log(f"File size: {os.path.getsize(out_comments)/1024/1024:.1f} MB")
# Simple diagnostic log
diag_lines = [
    f"Total rows processed: {len(df_full)}",
    f"Unique deal_num: {df_full['deal_num'].nunique()}",
    f"Valid non-empty comments count: {(df_full['comment_wordcount']>10).sum()}",
    f"comment_wordcount: p25={df_full['comment_wordcount'].quantile(.25):.0f} "
    f"p50={df_full['comment_wordcount'].median():.0f} "
    f"p75={df_full['comment_wordcount'].quantile(.75):.0f}",
    "",
    "LM presence rates:",
]
for _c in [c for c in df_full.columns if c.startswith("has_lm_")]:
    diag_lines.append(f"  {_c}: {df_full[_c].mean():.2%}")
diag_lines.append("")
diag_lines.append("LM mean counts:")
for _c in [c for c in df_full.columns
           if c.startswith("lm_") and c.endswith("_count")]:
    diag_lines.append(f"  {_c}: {df_full[_c].mean():.3f}")
if EMIT_COMPLEXITY and "deal_complexity_score" in df_full.columns:
    diag_lines.append(f"Mean deal_complexity: {df_full['deal_complexity_score'].mean():.2f}")
diag_path = os.path.join(MERGED, "01c_comments_diagnostics.txt")
with open(diag_path, "w", encoding="utf-8") as f:
    f.write("\n".join(diag_lines))
log(f"\nDiagnostics saved → {diag_path}")
log("\nScript 01c complete.")
