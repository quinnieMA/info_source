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
   reg_common_law / reg_civil_law / reg_mixed_legal: legal origin classification
4. Concatenated text identifiers
   regulatory_bodies, regulatory_countries: pipe-separated string of all matched bodies/countries
   for subsequent textual heterogeneity analysis

Output file: data/cleaned/01b_deal_overview.csv
Observation unit: (deal_num, tar_key) consistent deduplication logic with Module A

Processing notes:
- Columns suffixed with __1 are exact duplicate copies and dropped at initial load.
- Blank cascade rows with all missing values exist in raw overview source, unlike Module A industry file
  which retains partial firm metadata on cascade lines.
- Regulatory aggregation executed prior to deduplication to avoid undercounting multi-authority deals.

Author: CC  Date: 2026-08-08
"""
import os
import glob
import pandas as pd
import numpy as np

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

# Orbis ID 单值处理函数
def fix_orbis_val(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s in ("nan", "None", ""):
        return np.nan
    return s.zfill(9)

# 仅接收单列Series，逐元素转换
def normalize_orbis_id(ser):
    return ser.apply(fix_orbis_val)

# 监管分类常量不变
REG_CATEGORY_KEYWORDS = {
    "antitrust": [
        "competition", "antitrust", "anti-monopoly", "antimonopoly",
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
        "guernsey financial services commission",
        "jersey financial services commission",
        "financial services board",
        "securities and exchange board", "sebi",
        "financial regulatory authority",
        "monetary authority",
        "china banking and insurance regulatory commission", "cbrc",
        "cbirc",
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

def classify_regulatory_category(body_name):
    if pd.isna(body_name):
        return set()
    name_lower = str(body_name).lower()
    categories = set()
    for cat, keywords in REG_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                categories.add(cat)
                break
    return categories

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
    reg_agg["reg_cross_national"] = reg_agg["regulatory_countries"].apply(
        lambda countries: any(is_transnational(c) for c in countries.split("|") if c)
    ).astype(int)
    reg_agg["reg_common_law"] = reg_agg["regulatory_countries"].apply(
        lambda countries: any(is_common_law_country(c) for c in countries.split("|") if c)
    ).astype(int)
    reg_agg["reg_civil_law"] = reg_agg["regulatory_countries"].apply(
        lambda countries: any(is_civil_law_country(c) for c in countries.split("|") if c)
    ).astype(int)
    reg_agg["reg_mixed_legal"] = ((reg_agg["reg_common_law"] == 1) & (reg_agg["reg_civil_law"] == 1)).astype(int)
    return reg_agg

# ==================== 主程序 ====================
print("============================================================")
print("MODULE E — Overview + Country (Fix Series ValueError)")
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

# 1. 全部原始数据拼接，不做任何去重
df_full_raw = pd.concat(batches, ignore_index=True)
print(f"\nFull raw stacked rows (before dedup): {len(df_full_raw):,}")

# 2. 【强制先聚合监管指标，满足先count再去重要求】
print("\n============================================================")
print("Building regulatory variables from full raw data (all reg rows reserved)")
print("============================================================")
reg_vars = build_regulatory_variables(df_full_raw)

# 3. 复制原始表，开始清洗逻辑
df_ovw = df_full_raw.copy()

# 【核心修复】循环内传入单列Series，不再传整张DataFrame
for _oc in ["tar_orbis_id_num", "acq_orbis_id_num"]:
    if _oc in df_ovw.columns:
        df_ovw[_oc] = normalize_orbis_id(df_ovw[_oc])

# 过滤无标的空行
n_before_filter = len(df_ovw)
df_ovw = df_ovw[df_ovw["tar_name"].notna()].copy()
print(f"Drop blank tar_name rows: {n_before_filter - len(df_ovw):,} removed")

# 按deal_num+标的唯一键去重
df_ovw["_tar_key"] = df_ovw["tar_bvd_id_num"].fillna(df_ovw["tar_name"])
n_before_dedup = len(df_ovw)
df_ovw = df_ovw.drop_duplicates(subset=["deal_num", "_tar_key"], keep="first")
df_ovw = df_ovw.drop(columns=["_tar_key"])
print(f"Deduplicate (deal_num+tar_key): {n_before_dedup - len(df_ovw):,} duplicates dropped")

# deal_num 数值转换
df_ovw["deal_num"] = pd.to_numeric(df_ovw["deal_num"], errors="coerce")
df_ovw = df_ovw[df_ovw["deal_num"].notna()]
df_ovw["deal_num"] = df_ovw["deal_num"].astype("Int64")

# 交易金额重命名
if "deal_value" in df_ovw.columns:
    df_ovw["deal_value_ovw"] = pd.to_numeric(df_ovw["deal_value"], errors="coerce")
    df_ovw = df_ovw.drop(columns=["deal_value"])

# 4. 合并监管聚合结果到清洗后主表
if len(reg_vars) > 0:
    print(f"\nReg agg stats: total deals with reg info = {len(reg_vars):,}")
    print(f"reg_body_count min/max/mean: {reg_vars['reg_body_count'].min()} / {reg_vars['reg_body_count'].max()} / {reg_vars['reg_body_count'].mean():.2f}")
    print("\nCategory count summary:")
    for cat in REG_CATEGORY_KEYWORDS.keys():
        cnt = reg_vars[f"reg_{cat}"].sum()
        print(f"  reg_{cat} = {cnt:,}")
    df_ovw = df_ovw.merge(reg_vars, on="deal_num", how="left")
    # 空值填充
    count_cols = ["reg_body_count", "reg_country_count"]
    dummy_cols = [f"reg_{cat}" for cat in REG_CATEGORY_KEYWORDS] + ["reg_cross_national", "reg_common_law", "reg_civil_law", "reg_mixed_legal"]
    for c in count_cols:
        df_ovw[c] = df_ovw.fillna({c:0})[c].astype(int)
    for c in dummy_cols:
        df_ovw[c] = df_ovw.fillna({c:0})[c].astype(int)
    # 修复笔误：之前错填regulatory_bodies字段
    df_ovw["regulatory_bodies"] = df_ovw.fillna({"regulatory_bodies":""})["regulatory_bodies"]
    df_ovw["regulatory_countries"] = df_ovw.fillna({"regulatory_countries":""})["regulatory_countries"]
    print("\nRegulatory variables merge complete, real multi-count enabled")
else:
    print("Warning: zero regulatory records found in raw data")

# 输出统计
total_rows = len(df_ovw)
tar_miss = df_ovw['tar_country_code'].isna().sum()
acq_miss = df_ovw['acq_country_code'].isna().sum()
deal_status_miss = df_ovw['deal_status'].isna().sum()
tar_bvd_miss = df_ovw['tar_bvd_id_num'].isna().sum()
reg_has = df_ovw['reg_body_count'].gt(0).sum()

print(f"\nModule E Final Output: {len(df_ovw):,} rows | unique deal_num: {df_ovw['deal_num'].nunique():,}")
print(f"tar_country missing: {tar_miss:,} ({tar_miss/total_rows*100:.1f}%)")
print(f"acq_country missing: {acq_miss.sum():,} ({acq_miss/total_rows*100:.1f}%)")
print(f"deal_status missing: {deal_status_miss:,} ({deal_status_miss/total_rows*100:.1f}%)")
print(f"tar_bvd_id missing: {tar_bvd_miss.sum():,} ({tar_bvd_miss/total_rows*100:.1f}%)")
print(f"Deals with regulatory review: {reg_has:,} ({reg_has/total_rows*100:.1f}%)")
print(f"Max regulatory bodies per single deal: {df_ovw['reg_body_count'].max()}")

print("\nTop20 target country distribution:")
print(df_ovw["tar_country_code"].value_counts().head(20).to_string())

# 保存文件
out_path = os.path.join(CLEANED, "01b_deal_overview.csv")
df_ovw.to_csv(out_path, index=False, encoding="utf-8-sig")
size_kb = os.path.getsize(out_path)/1024
print(f"\nSaved cleaned file → {out_path} ({size_kb:.0f} KB)")