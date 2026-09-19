# -*- coding: utf-8 -*-
"""
==========================
Deal Overview & Regulatory Feature Construction

Extracted variables:
1. Core entity identifiers & geographic tags
   - tar_country_code, acq_country_code (new field absent in base deal master)
   - tar_name, tar_bvd_id_num, acq_name, acq_bvd_id_num (merge keys for cross-file QC)
2. Deal descriptive & structural text
   - deal_headline: short transaction summary text
3. Duplicate deal metrics for cross-source validation
   - deal_status, deal_value (duplicate copies for master dataset consistency check)
4. Extended regulatory feature suite (primary expansion of this module)
   Raw source columns: regulatory_body_name, regulatory_body_country
   Processing rule: aggregate regulatory metrics across all raw records BEFORE row deduplication
   to preserve accurate multi-agency count statistics.

Regulatory variable taxonomy:
1. Count metrics
   reg_body_count: total distinct regulatory authorities per deal
   reg_country_count: total distinct jurisdictions with oversight
2. Binary dummy indicators by regulator type
   reg_antitrust, reg_securities, reg_state_assets, reg_foreign_invest,
   reg_financial, reg_defense_tech
3. Institutional legal attributes
   reg_cross_national: 1 if supranational regulator (EU, etc.) involved
     — judged on BOTH the country field (regulatory_countries) AND the body
       names themselves (REG_TRANSNATIONAL_BODIES), because the raw country
       field is frequently blank or non-standard for EU-level authorities.
   reg_common_law / reg_civil_law / reg_mixed_legal: legal origin classification
     — derived from regulatory_countries via exact country-name matching.
       Coverage is partial and these are treated as INTERMEDIATE variables;
       downstream scripts (04b) override them with the dedicated legal-origin
       dataset. Do not use for final inference.
4. Concatenated text identifiers
   regulatory_bodies, regulatory_countries: pipe-separated string of all matched bodies/countries
   for subsequent textual heterogeneity analysis

──────────────────────────────────────────────────────────────────────────────
REGULATORY BODY CLASSIFICATION — THREE-TIER RESOLUTION (2026-09-06)
──────────────────────────────────────────────────────────────────────────────
The regulator taxonomy is built by exhaustively enumerating every distinct
value of `regulatory_body_name` in the raw overview files — NOT by guessing
keyword lists. As of 2026-09-06 the raw files contain 443 distinct body names
covering 6,623 deals with at least one recorded regulator.

Resolution order (highest priority first), implemented in
classify_regulatory_category():

  Tier 1 — REG_EXACT_MAP   Explicit body-name fragment → category set.
                           Highest priority; returns immediately on match.
                           Covers major regulators that keyword matching
                           systematically missed (MOFCOM, NDRC, SAFE,
                           European Commission, CONSOB, CNMV, ...) and
                           overrides wrong keyword hits.
  Tier 2 — REG_EXCLUDE     Self-regulatory organisations and courts that do
                           NOT constitute administrative merger approval —
                           excluded from all categories. Examples: stock
                           exchanges acting in listing capacity (Bursa/
                           Malaysia Securities Exchange), Takeover
                           Regulation Panel, National Company Law Tribunal,
                           and courts/tribunals (Federal Court of Australia,
                           Ontario Superior Court, NCLT).
  Tier 3 — Keyword         Substring fallback. Short acronyms (<=4 chars:
           fallback        sec, amb, nma, cade, doj, fsc, fca, fsa, amf,
                           cma, sebi, sfc, csrc, cbirc, ...) require a WHOLE-
                           WORD regex match; longer phrases use plain
                           substring matching.

Why the tiers exist — two defects found in the original keyword-only approach:

  (a) Substring over-matching. The token "sec" matched any body name
      containing the letters s-e-c (e.g. "Malaysia Securities Exchange"),
      and "amb"/"nma" matched unrelated bodies (National Bank of Cambodia,
      Swiss FINMA). All such hits are now either removed or overridden by
      Tier 1. Whole-word matching alone reduced 'sec' hits to zero.
  (b) severe under-coverage. 261 of 443 body names (2,572 of 6,623 deals,
      ~39%) matched no keyword at all — including China's Ministry of
      Commerce (429 deals), the European Commission (295), NDRC (113) and
      SAFE (58). Tier 1 recovers these; residual unclassified bodies fell
      from 2,572 to ~976 deals, and the remainder are almost entirely
      names deliberately placed in REG_EXCLUDE.

Deliberate scope decisions:
  - China Banking and Insurance Regulatory Commission (CBIRC/CBRC) is coded
    `financial`, NOT `securities`. Securities regulation in China is the
    CSRC. A deal may still carry reg_securities=1 if a genuine securities
    regulator (e.g. CSRC) also appears on the same deal — reg_* dummies are
    deal-level ORs across all bodies attached to that deal.
  - Multi-label bodies are permitted: a single regulator may set several
    dummies (e.g. a combined financial supervisor sets both financial and
    securities). The dummies are therefore NOT mutually exclusive.
  - Ministry of Commerce (PRC) is coded `foreign_invest`. NOTE: between 2008
    and 2018 China's anti-monopoly review of concentrations was conducted by
    MOFCOM's Anti-Monopoly Bureau; it moved to SAMR in 2018. For pre-2018
    deals this code understates antitrust involvement.

Maintenance: when regulatory coverage looks wrong, re-enumerate the distinct
body names first (do not add keywords blindly). The audit script writes
  data/cleaned/_reg_body_full_list.csv   (all names + counts + categories)
  data/cleaned/_reg_body_manual_map.csv  (blank manual_cat column template)

Output file: data/cleaned/01b_deal_overview.csv
Observation unit: (deal_num, tar_key) consistent deduplication logic with Module A

Processing notes:
- Columns suffixed with __1 are exact duplicate copies and dropped at initial load.
- Blank cascade rows with all missing values exist in raw overview source, unlike Module A industry file
  which retains partial firm metadata on cascade lines.
- Regulatory aggregation executed prior to deduplication to avoid undercounting multi-authority deals.
- Multi-target deals are RETAINED here (unlike Module 03, which drops them),
  so the output row count (58,963) exceeds the unique deal count (55,586) by
  3,377 rows. Downstream merges in 04a collapse to one row per deal_num.
- The target-country distribution printed at the end of this module describes
  the RAW, UNSCREENED overview universe. It is NOT the analysis sample —
  China is heavily over-represented before the sample restrictions
  (unlisted target x listed acquirer, deal value >= USD 1m, 2000-2024,
  completed-date availability). See table1 Panel E for the analysis sample
  composition.

Author: CC  Date: 2026-08-08
Revised: regulatory classification rebuilt on full body-name enumeration,
         three-tier resolution, 2026-09-06
"""
import os
import glob
import pandas as pd
import numpy as np
import re

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = r"D:\MA"
RAW_OVW  = os.path.join(BASE, "raw", "MA_deal", "overview")
CLEANED  = os.path.join(BASE, "data", "cleaned")
os.makedirs(CLEANED, exist_ok=True)

MISSING_VALS = ["-", "n.a.", "n.s.", "NA", "N/A", "nan", ""]

def read_and_ffill(filepath, encoding="utf-8-sig"):
    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
    df.columns = df.columns.str.strip()
    df["deal_num"] = df["deal_num"].ffill()
    return df

def clean_missing(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace(MISSING_VALS, np.nan).str.strip()
    return pd.DataFrame(df)

# Orbis ID single-value processing function
def fix_orbis_val(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s in ("nan", "None", ""):
        return np.nan
    return s.zfill(9)

# Accepts a single-column Series, converts element-wise
def normalize_orbis_id(ser):
    return ser.apply(fix_orbis_val)

# Regulatory classification constants unchanged
REG_CATEGORY_KEYWORDS = {
    "antitrust": [
        "competition", "antitrust","anti-trust","anti-monopoly", "antimonopoly",
        "monopoly", "kartell", "konkurr", "cade", "fair trade",
        "bundeskartellamt", "ftc", "department of justice", "doj",
        "dg competition", "wettbewerb", "concurrencia", "concurrence",
        "concorrenza", "mededingingsautoriteit", "competition commission",
        "competition and markets authority", "cma", "accc",
        "state administration for market regulation", "samr",
        "anti-monopoly bureau", "amb", "federal antimonopoly service",
        "federal trade commission", "autorité de la concurrence",
        "autorita garante della concorrenza", "office of competition",
        "competition bureau", "jersey competition regulatory authority",
        "monopolkommission", "direccion general de defensa de la competencia",
        "nederlandse mededingingsautoriteit", "nma",
        "direction de la concurrence", "autoriteit financiele markten",
    ],
    "securities": [
        "securities commission", "securities and exchange", "sec",
        "china securities regulatory commission", "csrc",
        "financial supervisory commission", "fsc",
        "securities and futures", "sfc",
        "autorité des marchés financiers", "amf",
        "financial services authority", "fsa",
        "financial conduct authority", "fca",
        "bafin", "bundesanstalt für finanzdienstleistungsaufsicht",
        "commission de surveillance du secteur financier",
        #"guernsey financial services commission",
        #"jersey financial services commission",
        "financial services board",
        "securities and exchange board", "sebi",
        #"financial regulatory authority",
        "monetary authority",
        #"china banking and insurance regulatory commission", "cbrc",
        #"cbirc",
    ],
    "state_assets": [
        "state-owned assets supervision", "sasac",
        "assets supervision and administration commission",
        "state-owned",
    ],
    "foreign_invest": [
        "foreign investment", "foreign investments review",
        "investment commission", "firb",
        "foreign investment committee",
        "committee on foreign investment", "cfius",
        "investment review board",
    ],
    "financial": [
        "banking", "insurance", "financial services commission",
        "financial supervision", "financial regulatory",
        "china banking", "banking and insurance",
        "central bank", "reserve bank",
    ],
    "defense_tech": [
        "defence", "defense", "national security",
        "state administration of science",
        "science, technology and industry for national defence",
        "ministry of defence", "department of defense",
        "national defence",
    ],
}

COMMON_LAW_COUNTRIES = {
    "United States", "United Kingdom", "Australia", "Canada",
    "New Zealand", "Singapore", "Hong Kong", "India", "Malaysia",
    "South Africa", "Ireland", "Israel", "Pakistan", "Bangladesh",
    "Sri Lanka", "Kenya", "Nigeria", "Ghana", "Jamaica",
    "Trinidad and Tobago", "Bahamas", "Barbados", "Mauritius",
    "Cyprus", "Malta", "Guernsey", "Jersey", "Isle of Man",
    "Cayman Islands", "British Virgin Islands", "Bermuda",
}

CIVIL_LAW_COUNTRIES = {
    "China", "Germany", "France", "Italy", "Spain", "Netherlands",
    "Belgium", "Austria", "Switzerland", "Sweden", "Norway",
    "Denmark", "Finland", "Japan", "South Korea", "Taiwan",
    "Russia", "Brazil", "Mexico", "Argentina", "Chile",
    "Colombia", "Peru", "Venezuela", "Philippines", "Indonesia",
    "Thailand", "Vietnam", "Turkey", "Poland", "Czech Republic",
    "Hungary", "Romania", "Bulgaria", "Greece", "Portugal",
    "Luxembourg", "Iceland", "Estonia", "Latvia", "Lithuania",
    "Slovakia", "Slovenia", "Croatia", "Serbia", "Ukraine",
    "Egypt", "Morocco", "Algeria", "Tunisia",
}

TRANSNATIONAL_REGULATORS = {"European Union", "EU", "European Commission"}

# ════════════════════════════════════════════════════════════════════════════
# Exact agency name mapping (2026-09-06, based on full enumeration of 443 unique agency names)
# Priority: exact mapping > exclusion list > keyword fallback
# ════════════════════════════════════════════════════════════════════════════

REG_EXCLUDE = {
    "malaysia securities exchange", "securities industry council",
    "takeover regulation panel", "chamber of the amsterdam court of appeal",
    "the federal court of australia", "ontario superior court of justice",
    "supreme court of british columbia", "national company law tribunal",
    "federal court", "high court", "court of appeal", "supreme court",
}

REG_TRANSNATIONAL_BODIES = {
    "european commission", "european council", "european parliament",
    "council of the european union", "european central bank",
    "european banking authority",
    "european securities and markets authority",
    "european insurance and occupational pensions authority",
    "directorate general for competition",
}

REG_EXACT_MAP = {
    # -- China: banking regulation explicitly excluded from securities --
    "china banking and insurance regulatory commission": {"financial"},
    "cbirc":                                      {"financial"},
    "cbrc":                                       {"financial"},
    # -- Comprehensive financial regulator: classified as financial, not securities --
    "financial regulatory authority":             {"financial"},
    "guernsey financial services commission":     {"financial"},
    "jersey financial services commission":       {"financial"},
    # -- China --
    "ministry of commerce":                       {"foreign_invest"},
    "national development and reform commission": {"state_assets"},
    "state administration of foreign exchange":   {"foreign_invest"},
    "industrial & commercial administration bureau": {"state_assets"},
    "ministry of finance":                        {"financial"},
    "national administration of financial regulation": {"financial"},
    "china's state council":                      {"state_assets"},
    "state council":                              {"state_assets"},
    # -- Supranational --
    "european commission":                        {"antitrust"},
    # -- Antitrust --
    "anti-trust authority":                       {"antitrust"},
    "comisión nacional de la competencia":        {"antitrust"},
    "comision nacional de los mercados y la competencia": {"antitrust"},
    "comisión federal de competencia":            {"antitrust"},
    "autoridade de concorrencia":                 {"antitrust"},
    "autoriteit consument en markt":              {"antitrust"},
    "office of fair trading":                     {"antitrust"},
    "gazdasagi versenyhivatal":                   {"antitrust"},
    "konkurentsiamet":                            {"antitrust"},
    "lietuvos respublikos konkurencijos taryba":  {"antitrust"},
    "commerce commission":                        {"antitrust"},
    "illinois commerce commission":               {"antitrust"},
    # -- Securities --
    "consob":                                     {"securities"},
    "comision nacional del mercado de valores":   {"securities"},
    "komisija za hartije od vrjednosti":          {"securities"},
    "federal service for financial markets":      {"securities"},
    "finanstilsynet":                             {"securities"},
    "finansinspektionen":                         {"securities"},
    "autoriteit financiële markten":              {"securities"},
    "croatian agency for the supervision of financial services": {"securities"},
    "financial supervisory authority":            {"securities"},
    "egyptian financial supervisory authority":   {"securities"},
    "capital market authority":                   {"securities"},
    "capital markets authority":                  {"securities"},
    "capital markets board":                      {"securities"},
    "superintendencia financiera de colombia":    {"securities"},
    "australian securities and investments commission": {"securities"},
    "swiss financial market supervisory authority": {"securities"},
    "securities & commodities authority":         {"securities"},
    "securities market agency":                   {"securities"},
    "comissão de valores mobiliários":            {"securities"},
    "comissão do mercado de valores mobiliários": {"securities"},
    "autorité des services et marchés financiers": {"securities"},
    # -- Financial / central bank --
    "bank negara malaysia":                       {"financial"},
    "federal reserve board":                      {"financial"},
    "bank of italy":                              {"financial"},
    "office of the comptroller of the currency":  {"financial"},
    "national bank of cambodia":                  {"financial"},
    "central bank of mozambique":                 {"financial"},
    "nepal rastra bank":                          {"financial"},
    "federal energy regulatory commission":       {"financial"},
    "prudential regulation authority":            {"financial"},
    # -- Foreign investment --
    "overseas investment office":                 {"foreign_invest"},
    "ministry of international trade and industry": {"foreign_invest"},
    "ministry of economic affairs":               {"foreign_invest"},
    # -- Sectoral --
    "agência nacional de energia elétrica":       {"financial"},
    "agência nacional de telecomunicações":       {"defense_tech"},
    "federal communications commission":          {"defense_tech"},
    # -- Third round: fill remaining gaps --
    "kilpailuvirasto":                            {"antitrust"},   # Finnish Competition Authority
    "australian prudential regulatory authority": {"financial"},   # Australian Prudential Regulation Authority
    "australian prudential regulation authority": {"financial"},
    "bureau of internal revenue":                 {"financial"},
    "ministry of domestic trade and consumer affairs": {"financial"},
    "agência nacional de saúde suplementar":      {"financial"},
}

def _norm(s):
    """Normalize agency name: lowercase + collapse whitespace"""
    return " ".join(str(s).lower().split())

def classify_regulatory_category(body_name):
    """
    Exact mapping first -> exclusion list -> keyword fallback
    """
    if pd.isna(body_name):
        return set()
    raw = str(body_name).strip()
    nm  = _norm(raw)

    # (1) Exact mapping (highest priority, direct return)
    for frag, cats in REG_EXACT_MAP.items():
        if _norm(frag) in nm:
            return set(cats)

    # (2) Exclusion list (exchange self-regulatory bodies / courts, not regulatory approvals)
    for bad in REG_EXCLUDE:
        if _norm(bad) in nm:
            return set()

    # (3) Keyword fallback; short abbreviations use word boundaries to avoid false matches
    cats = set()
    for cat, keywords in REG_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            kwl = kw.lower()
            if len(kwl) <= 4:                      # Short abbreviations require whole-word match
                if re.search(rf"\b{re.escape(kwl)}\b", nm):
                    cats.add(cat)
                    break
            elif kwl in nm:
                cats.add(cat)
                break
    return cats

def is_common_law_country(country):
    if pd.isna(country):
        return False
    return str(country).strip() in COMMON_LAW_COUNTRIES

def is_civil_law_country(country):
    if pd.isna(country):
        return False
    return str(country).strip() in CIVIL_LAW_COUNTRIES

def is_transnational(country):
    if pd.isna(country):
        return False
    return str(country).strip() in TRANSNATIONAL_REGULATORS

def build_regulatory_variables(df_full_raw):
    reg_df = df_full_raw[df_full_raw["regulatory_body_name"].notna()].copy()
    if len(reg_df) == 0:
        return pd.DataFrame()
    reg_agg = reg_df.groupby("deal_num").agg(
        regulatory_bodies=("regulatory_body_name", lambda x: "|".join(sorted(set(x.dropna())))),
        regulatory_countries=("regulatory_body_country", lambda x: "|".join(sorted(set(x.dropna())))),
        reg_body_count=("regulatory_body_name", "nunique"),
        reg_country_count=("regulatory_body_country", "nunique"),
    ).reset_index()
    for cat in REG_CATEGORY_KEYWORDS.keys():
        col_name = f"reg_{cat}"
        reg_agg[col_name] = reg_agg["regulatory_bodies"].apply(
            lambda bodies: any(cat in classify_regulatory_category(b) for b in bodies.split("|") if b)
        ).astype(int)
    def _is_transnational_body(bodies_str):
        """Any agency name matching the transnational list counts as cross-national regulation"""
        parts = [p.strip() for p in str(bodies_str).split("|") if p.strip()]
        return any(any(_norm(t) in _norm(p) for t in REG_TRANSNATIONAL_BODIES)
                   for p in parts)
    
    reg_agg["reg_cross_national"] = reg_agg.apply(
        lambda r: int(
            any(is_transnational(c) for c in str(r["regulatory_countries"]).split("|") if c)
            or _is_transnational_body(r["regulatory_bodies"])
        ), axis=1
    ).astype(int)

    reg_agg["reg_common_law"] = reg_agg["regulatory_countries"].apply(
        lambda countries: any(is_common_law_country(c) for c in countries.split("|") if c)
    ).astype(int)
    reg_agg["reg_civil_law"] = reg_agg["regulatory_countries"].apply(
        lambda countries: any(is_civil_law_country(c) for c in countries.split("|") if c)
    ).astype(int)
    reg_agg["reg_mixed_legal"] = ((reg_agg["reg_common_law"] == 1) & (reg_agg["reg_civil_law"] == 1)).astype(int)
    return reg_agg
# ==================== Main program ====================
print("============================================================")
print("MODULE 01b.overview_country — Overview + Country (Fix Series ValueError)")
print("============================================================")

ovw_files = sorted(glob.glob(os.path.join(RAW_OVW, "acquisition_overview_*_cleaned.csv")))
print(f"Found {len(ovw_files)} overview batch files")

OVW_KEEP = [
    "deal_num",
    "tar_name", "tar_bvd_id_num", "tar_orbis_id_num", "tar_country_code",
    "tar_busi_descr",
    "acq_name", "acq_bvd_id_num", "acq_orbis_id_num", "acq_country_code",
    "ven_name", "ven_bvd_id_num", "ven_country_code",
    "deal_status", "deal_value",
    "deal_headline", "type_of_deal_opportunity",
    "regulatory_body_name", "regulatory_body_country",
]

batches = []
for fp in ovw_files:
    batch_name = os.path.basename(fp)
    df_b = read_and_ffill(fp)
    n_raw = len(df_b)
    dup_cols = [c for c in df_b.columns if "__1" in c]
    df_b = df_b.drop(columns=dup_cols)
    df_b = clean_missing(df_b)
    keep_present = [c for c in OVW_KEEP if c in df_b.columns]
    df_b = df_b[keep_present]
    batches.append(df_b)
    print(f"  {batch_name}: {n_raw} rows read, {len(dup_cols)} __1 cols dropped")

# 1. Concatenate all raw data, no deduplication
df_full_raw = pd.concat(batches, ignore_index=True)
print(f"\nFull raw stacked rows (before dedup): {len(df_full_raw):,}")

# 2. [Mandatory] Aggregate regulatory indicators first (count before dedup requirement)
print("\n============================================================")
print("MODULE 01b.reg Building regulatory variables from full raw data (all reg rows reserved)")
print("============================================================")
reg_vars = build_regulatory_variables(df_full_raw)

# 3. Copy raw table, start cleaning logic
df_ovw = df_full_raw.copy()

# [Core fix] Pass single-column Series inside loop, not the full DataFrame
for _oc in ["tar_orbis_id_num", "acq_orbis_id_num"]:
    if _oc in df_ovw.columns:
        df_ovw[_oc] = normalize_orbis_id(df_ovw[_oc])

# Drop rows with no target
n_before_filter = len(df_ovw)
df_ovw = df_ovw[df_ovw["tar_name"].notna()].copy()
print(f"Drop blank tar_name rows: {n_before_filter - len(df_ovw):,} removed")

# Deduplicate by deal_num + target unique key
df_ovw["_tar_key"] = df_ovw["tar_bvd_id_num"].fillna(df_ovw["tar_name"])
n_before_dedup = len(df_ovw)
df_ovw = df_ovw.drop_duplicates(subset=["deal_num", "_tar_key"], keep="first")
df_ovw = df_ovw.drop(columns=["_tar_key"])
print(f"Deduplicate (deal_num+tar_key): {n_before_dedup - len(df_ovw):,} duplicates dropped")

# Convert deal_num to numeric
df_ovw["deal_num"] = pd.to_numeric(df_ovw["deal_num"], errors="coerce")
df_ovw = df_ovw[df_ovw["deal_num"].notna()]
df_ovw["deal_num"] = df_ovw["deal_num"].astype("Int64")

# Rename deal value
if "deal_value" in df_ovw.columns:
    df_ovw["deal_value_ovw"] = pd.to_numeric(df_ovw["deal_value"], errors="coerce")
    df_ovw = df_ovw.drop(columns=["deal_value"])

# 4. Merge regulatory aggregation results into cleaned main table
if len(reg_vars) > 0:
    print(f"\nReg agg stats: total deals with reg info = {len(reg_vars):,}")
    print(f"reg_body_count min/max/mean: {reg_vars['reg_body_count'].min()} / {reg_vars['reg_body_count'].max()} / {reg_vars['reg_body_count'].mean():.2f}")
    print("\nCategory count summary:")
    for cat in REG_CATEGORY_KEYWORDS.keys():
        cnt = reg_vars[f"reg_{cat}"].sum()
        print(f"  reg_{cat} = {cnt:,}")
    df_ovw = df_ovw.merge(reg_vars, on="deal_num", how="left")
    # Fill missing values
    count_cols = ["reg_body_count", "reg_country_count"]
    dummy_cols = [f"reg_{cat}" for cat in REG_CATEGORY_KEYWORDS] + ["reg_cross_national", "reg_common_law", "reg_civil_law", "reg_mixed_legal"]
    for c in count_cols:
        df_ovw[c] = df_ovw.fillna({c:0})[c].astype(int)
    for c in dummy_cols:
        df_ovw[c] = df_ovw.fillna({c:0})[c].astype(int)
    # Fix typo: previously incorrectly filled regulatory_bodies field
    df_ovw["regulatory_bodies"] = df_ovw.fillna({"regulatory_bodies":""})["regulatory_bodies"]
    df_ovw["regulatory_countries"] = df_ovw.fillna({"regulatory_countries":""})["regulatory_countries"]
    print("\nRegulatory variables merge complete, real multi-count enabled")
else:
    print("Warning: zero regulatory records found in raw data")

# Output statistics
total_rows = len(df_ovw)
tar_miss = df_ovw['tar_country_code'].isna().sum()
acq_miss = df_ovw['acq_country_code'].isna().sum()
deal_status_miss = df_ovw['deal_status'].isna().sum()
tar_bvd_miss = df_ovw['tar_bvd_id_num'].isna().sum()
reg_has = df_ovw['reg_body_count'].gt(0).sum()

print(f"\nMODULE 01b.overview_country Final Output: {len(df_ovw):,} rows | unique deal_num: {df_ovw['deal_num'].nunique():,}")
print(f"tar_country missing: {tar_miss:,} ({tar_miss/total_rows*100:.1f}%)")
print(f"acq_country missing: {acq_miss.sum():,} ({acq_miss/total_rows*100:.1f}%)")
print(f"deal_status missing: {deal_status_miss:,} ({deal_status_miss/total_rows*100:.1f}%)")
print(f"tar_bvd_id missing: {tar_bvd_miss.sum():,} ({tar_bvd_miss/total_rows*100:.1f}%)")
print(f"Deals with regulatory review: {reg_has:,} ({reg_has/total_rows*100:.1f}%)")
print(f"Max regulatory bodies per single deal: {df_ovw['reg_body_count'].max()}")

print("\nTop20 target country distribution:")
print("  " + "!" * 3 + "  Below is the country distribution of the RAW overview file (no sample filtering)")
print("      Contains many small deals / non-listed acquirers / deals without completion dates,")
print("      does not represent the final analysis sample. See table1 Panel E for analysis sample composition.")
print(df_ovw["tar_country_code"].value_counts().head(20).to_string())
print(f"Extra rows from multi-target deals: {len(df_ovw) - df_ovw['deal_num'].nunique():,}")

# Supplement: subset where reg_* variables actually apply
if "reg_body_count" in df_ovw.columns:
    _sub = df_ovw[df_ovw["reg_body_count"] > 0]
    print(f"\n[Appendix] Subset with regulatory records N={len(_sub):,} rows "
          f"(unique deal {_sub['deal_num'].nunique():,})")
    print("      reg_* variables are non-zero only within this subset:")
    print(_sub["tar_country_code"].value_counts().head(10).to_string())
    
# Save file
out_path = os.path.join(CLEANED, "01b_deal_overview.csv")
df_ovw.to_csv(out_path, index=False, encoding="utf-8-sig")
size_kb = os.path.getsize(out_path)/1024
print(f"\nSaved cleaned file → {out_path} ({size_kb:.0f} KB)")
#（注：内容由AI生成）
