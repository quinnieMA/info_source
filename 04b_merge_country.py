"""
04b_merge_country.py
====================
Merge country-level stock market development data into 04_deal_firm_full.csv.

Input  : data/merged/04_deal_firm_full.csv        [58,691 rows]
         raw/Country_level/world_stock_traded_gdp_ratio_1970_2025.csv
                                                  [50 rows × 103 cols, wide format]

Steps:
  1. Reshape country data wide → long: (year, country_code, stock_traded_gdp)
  2. Map file's country names to ISO2 codes used in deal_firm_full.tar_country_code
  3. Left-merge on (tar_country_code, completed_d_yr)
     - completed_d_yr is used (deal completion year = fiscal year for country control)
     - Deals with missing completed_d_yr receive NaN for country variable
  4. Verify row count unchanged; report match rate and country coverage

Output : data/merged/04b_deal_firm_country.csv   [58,691 rows + country variable]
         data/merged/04b_deal_firm_country_diagnostics.txt

Note: Country data covers 1975-2024. Deals completed in 2025 or with missing
      completed_d_yr will have NaN for stock_traded_gdp.

Variable retained:
  stock_traded_gdp  — stock market value traded as % of GDP (World Bank)
  (empirical_design §4.8: country-level control for H3 equation)

Author: Zhaohua Li  Date: 2026-04-12
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = r"D:\MA"
RAW_CTRY  = os.path.join(BASE, "raw", "Country_level",
                         "world_stock_traded_gdp_ratio_1970_2025.csv")
MERGED    = os.path.join(BASE, "data", "merged")

diag_lines = []

def log(msg=""):
    print(msg)
    diag_lines.append(str(msg))


# ── Country name → ISO2 mapping ────────────────────────────────────────────
# Keys match the file's column names (underscores, no _StockTraded_GDP suffix).
# Values are ISO 3166-1 alpha-2 codes used in tar_country_code.
COUNTRY_MAP = {
    "Argentina"            : "AR",
    "Australia"            : "AU",
    "Austria"              : "AT",
    "Bahrain"              : "BH",
    "Bangladesh"           : "BD",
    "Barbados"             : "BB",
    "Belarus"              : "BY",
    "Belgium"              : "BE",
    "Bermuda"              : "BM",
    "Botswana"             : "BW",
    "Brazil"               : "BR",
    "Bulgaria"             : "BG",
    "Canada"               : "CA",
    "Cayman_Islands"       : "KY",
    "Channel_Islands"      : "JE",   # Jersey (Zephyr uses JE for Channel Islands)
    "Chile"                : "CL",
    "China"                : "CN",
    "Colombia"             : "CO",
    "Costa_Rica"           : "CR",
    "Croatia"              : "HR",
    "Cyprus"               : "CY",
    "Czech_Republic"       : "CZ",
    "Denmark"              : "DK",
    "Ecuador"              : "EC",
    "Egypt"                : "EG",
    "Estonia"              : "EE",
    "Finland"              : "FI",
    "France"               : "FR",
    "Germany"              : "DE",
    "Ghana"                : "GH",
    "Greece"               : "GR",
    "Hong_Kong"            : "HK",
    "Hungary"              : "HU",
    "Iceland"              : "IS",
    "India"                : "IN",
    "Indonesia"            : "ID",
    "Iran"                 : "IR",
    "Ireland"              : "IE",
    "Israel"               : "IL",
    "Italy"                : "IT",
    "Jamaica"              : "JM",
    "Japan"                : "JP",
    "Jordan"               : "JO",
    "Kazakhstan"           : "KZ",
    "Kenya"                : "KE",
    "Korea"                : "KR",
    "Kuwait"               : "KW",
    "Latvia"               : "LV",
    "Lebanon"              : "LB",
    "Lithuania"            : "LT",
    "Luxembourg"           : "LU",
    "Malaysia"             : "MY",
    "Malta"                : "MT",
    "Mauritius"            : "MU",
    "Mexico"               : "MX",
    "Montenegro"           : "ME",
    "Morocco"              : "MA",
    "Namibia"              : "NA",
    "Netherlands"          : "NL",
    "New_Zealand"          : "NZ",
    "Nigeria"              : "NG",
    "Norway"               : "NO",
    "Oman"                 : "OM",
    "Pakistan"             : "PK",
    "Panama"               : "PA",
    "Papua_New_Guinea"     : "PG",
    "Paraguay"             : "PY",
    "Peru"                 : "PE",
    "Philippines"          : "PH",
    "Poland"               : "PL",
    "Portugal"             : "PT",
    "Qatar"                : "QA",
    "Romania"              : "RO",
    "Russia"               : "RU",
    "Rwanda"               : "RW",
    "Saudi_Arabia"         : "SA",
    "Serbia"               : "RS",
    "Seychelles"           : "SC",
    "Singapore"            : "SG",
    "Slovakia"             : "SK",
    "Slovenia"             : "SI",
    "South_Africa"         : "ZA",
    "Spain"                : "ES",
    "Sri_Lanka"            : "LK",
    "Swaziland"            : "SZ",
    "Sweden"               : "SE",
    "Switzerland"          : "CH",
    "Tanzania"             : "TZ",
    "Thailand"             : "TH",
    "Trinidad_and_Tobago"  : "TT",
    "Tunisia"              : "TN",
    "Turkey"               : "TR",
    "Ukraine"              : "UA",
    "United_Arab_Emirates" : "AE",
    "United_Kingdom"       : "GB",
    "US"                   : "US",
    "Uruguay"              : "UY",
    "Venezuela"            : "VE",
    "Vietnam"              : "VN",
    "West_Bank_and_Gaza"   : "PS",
    "Zambia"               : "ZM",
    "Zimbabwe"             : "ZW",
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Load and reshape country-level data
# ════════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("STEP 1 — Load and reshape country-level data")
log("=" * 60)

df_ctry_wide = pd.read_csv(RAW_CTRY)
df_ctry_wide = df_ctry_wide.rename(columns={"Unnamed: 0": "year_date"})
log(f"Raw shape: {df_ctry_wide.shape[0]} rows x {df_ctry_wide.shape[1]} cols")
log(f"Year range: {df_ctry_wide['year_date'].iloc[0]} to {df_ctry_wide['year_date'].iloc[-1]}")

# Extract integer year from "YYYY-MM-DD" date string
df_ctry_wide["year"] = pd.to_numeric(
    df_ctry_wide["year_date"].str[:4], errors="coerce"
).astype("Int64")

# Reshape wide → long
country_cols = [c for c in df_ctry_wide.columns
                if c.endswith("_StockTraded_GDP")]
log(f"Country columns: {len(country_cols)}")

df_long = df_ctry_wide[["year"] + country_cols].melt(
    id_vars="year",
    var_name="country_col",
    value_name="stock_traded_gdp"
)

# Parse country name from column name (remove _StockTraded_GDP suffix)
df_long["country_name"] = df_long["country_col"].str.replace(
    "_StockTraded_GDP", "", regex=False
)

# Map to ISO2 code
df_long["country_code"] = df_long["country_name"].map(COUNTRY_MAP)

n_unmapped = df_long["country_code"].isna().sum()
if n_unmapped > 0:
    unmapped = df_long.loc[df_long["country_code"].isna(), "country_name"].unique()
    log(f"WARNING: {n_unmapped} rows with unmapped country names: {unmapped.tolist()}")

# Drop rows with no ISO2 mapping
df_long = df_long.dropna(subset=["country_code"])

# Keep only the merge key and value
df_ctry = df_long[["year", "country_code", "stock_traded_gdp"]].copy()
df_ctry["stock_traded_gdp"] = pd.to_numeric(df_ctry["stock_traded_gdp"], errors="coerce")

log(f"After reshape: {len(df_ctry):,} rows | "
    f"{df_ctry['country_code'].nunique()} unique country codes | "
    f"{df_ctry['year'].nunique()} years")

# Quick check: key countries present
for cc in ["CN", "GB", "US", "JP", "KR"]:
    n = (df_ctry["country_code"] == cc).sum()
    log(f"  {cc}: {n} year-rows")


# ════════════════════════════════════════════════════════════════════════════
# 2. Load deal_firm_full
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 2 — Load 04_deal_firm_full.csv")
log("=" * 60)

ORBIS_DTYPE = {c: str for c in [
    "tar_orbis_id_num", "acq_orbis_id_num", "ven_orbis_id_num",
    "tar_orbis_id_num_ovw", "acq_orbis_id_num_ovw",
    "tar_orbis_id_num_fin", "acq_orbis_id_num_fin",
    "tar_orbis_id_num_cf", "tar_orbis_id_num_leg", "acq_orbis_id_num_leg",
]}
df = pd.read_csv(os.path.join(MERGED, "04a_deal_firm_full.csv"),
                 encoding="utf-8-sig", low_memory=False, dtype=ORBIS_DTYPE)
BASE_ROWS = len(df)
log(f"Rows: {BASE_ROWS:,} | Columns: {df.shape[1]}")

# Prepare merge keys
# M1 FIXER R1: use announced_d_yr as fallback when completed_d_yr is missing
df["_merge_year"] = pd.to_numeric(df["completed_d_yr"], errors="coerce").astype("Int64")
df["_year_source"] = "completed_d_yr"  # track which column supplied the year
_n_missing_before = df["_merge_year"].isna().sum()
if "announced_d_yr" in df.columns:
    _fallback_mask = df["_merge_year"].isna() & df["announced_d_yr"].notna()
    df.loc[_fallback_mask, "_merge_year"] = pd.to_numeric(
        df.loc[_fallback_mask, "announced_d_yr"], errors="coerce"
    ).astype("Int64")
    df.loc[_fallback_mask, "_year_source"] = "announced_d_yr"
    _n_fallback = int(_fallback_mask.sum())
    log(f"M1 fix: used announced_d_yr as fallback year for {_n_fallback:,} deals "
        f"with missing completed_d_yr")
    log(f"  _merge_year missing before fallback: {_n_missing_before:,}")
    log(f"  _merge_year missing after fallback:  {df['_merge_year'].isna().sum():,}")
else:
    log(f"  M1: announced_d_yr column not found — no fallback applied")

# 目标国家的merge key
df["_merge_cc_tar"] = df["tar_country_code"].astype(str).str.strip()

# 收购方国家的merge key
df["_merge_cc_acq"] = df["acq_country_code"].astype(str).str.strip()

log(f"completed_d_yr non-missing: {df['_merge_year'].notna().sum():,} "
    f"({df['_merge_year'].notna().mean()*100:.1f}%)")

# m4 FIXER R1: diagnostics
_n_cc_notna    = df["tar_country_code"].notna().sum()
_n_cc_notempty = (df["tar_country_code"].fillna("").str.strip() != "").sum()
log(f"tar_country_code non-missing (NaN check): {_n_cc_notna:,}")
log(f"tar_country_code non-empty  (str check) : {_n_cc_notempty:,}")

_n_acq_notna    = df["acq_country_code"].notna().sum()
_n_acq_notempty = (df["acq_country_code"].fillna("").str.strip() != "").sum()
log(f"acq_country_code non-missing (NaN check): {_n_acq_notna:,}")
log(f"acq_country_code non-empty  (str check) : {_n_acq_notempty:,}")


# ════════════════════════════════════════════════════════════════════════════
# 3. Merge country data on (tar_country_code, completed_d_yr)
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 3 — Left merge on (country_code, year)")
log("=" * 60)


# ────────────────────────────────────────────────────────────────────────────
# 3.1 目标国家的 stock_traded_gdp
# ────────────────────────────────────────────────────────────────────────────
df_ctry_tar = df_ctry.rename(columns={
    "country_code": "_merge_cc_tar",
    "year": "_merge_year_tar",
    "stock_traded_gdp": "stock_traded_gdp_tar"
})
df_ctry_tar["_merge_year_tar"] = df_ctry_tar["_merge_year_tar"].astype("Int64")

df["_merge_year_tar"] = df["_merge_year"]

df = df.merge(
    df_ctry_tar[["_merge_cc_tar", "_merge_year_tar", "stock_traded_gdp_tar"]],
    on=["_merge_cc_tar", "_merge_year_tar"],
    how="left"
)

if len(df) != BASE_ROWS:
    raise AssertionError(
        f"Row count changed after tar merge: {BASE_ROWS:,} -> {len(df):,}"
    )

n_tar_matched = df["stock_traded_gdp_tar"].notna().sum()
log(f"stock_traded_gdp_tar non-missing: {n_tar_matched:,} / {BASE_ROWS:,} "
    f"({n_tar_matched/BASE_ROWS*100:.1f}%)")

# ────────────────────────────────────────────────────────────────────────────
# 3.2 收购方国家的 stock_traded_gdp
# ────────────────────────────────────────────────────────────────────────────
df_ctry_acq = df_ctry.rename(columns={
    "country_code": "_merge_cc_acq",
    "year": "_merge_year_acq",
    "stock_traded_gdp": "stock_traded_gdp_acq"
})
df_ctry_acq["_merge_year_acq"] = df_ctry_acq["_merge_year_acq"].astype("Int64")

df["_merge_year_acq"] = df["_merge_year"]

df = df.merge(
    df_ctry_acq[["_merge_cc_acq", "_merge_year_acq", "stock_traded_gdp_acq"]],
    on=["_merge_cc_acq", "_merge_year_acq"],
    how="left"
)

if len(df) != BASE_ROWS:
    raise AssertionError(
        f"Row count changed after acq merge: {BASE_ROWS:,} -> {len(df):,}"
    )

n_acq_matched = df["stock_traded_gdp_acq"].notna().sum()
log(f"stock_traded_gdp_acq non-missing: {n_acq_matched:,} / {BASE_ROWS:,} "
    f"({n_acq_matched/BASE_ROWS*100:.1f}%)")

# ────────────────────────────────────────────────────────────────────────────
# 3.3 清理临时列
# ────────────────────────────────────────────────────────────────────────────
df = df.drop(columns=[
    "_merge_year_tar", "_merge_cc_tar",
    "_merge_year_acq", "_merge_cc_acq",
    "_merge_year"
])

# 向后兼容：原有的 stock_traded_gdp 指向目标国家
#df["stock_traded_gdp"] = df["stock_traded_gdp_tar"]



# ════════════════════════════════════════════════════════════════════════════
# 4. Diagnostics
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 4. Diagnostics — 分别诊断tar和acq的匹配情况
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 4 — Diagnostics: tar vs acq match quality")
log("=" * 60)

# ────────────────────────────────────────────────────────────────────────────
# 4.1 目标国家的匹配诊断
# ────────────────────────────────────────────────────────────────────────────
df_tmp_tar = df.copy()
df_tmp_tar["_yr"] = pd.to_numeric(df_tmp_tar.get("completed_d_yr", pd.Series(dtype=float)),
                                   errors="coerce")
df_tmp_tar["_cc"] = df_tmp_tar.get("tar_country_code", pd.Series(dtype=str))
df_tmp_tar["_hit"] = df_tmp_tar["stock_traded_gdp_tar"].notna()

n_miss_yr_tar = df_tmp_tar["_yr"].isna().sum()
n_no_ctry_tar = df_tmp_tar["_cc"].isna().sum()

log(f"\n--- Target country stock_traded_gdp_tar ---")
log(f"  completed_d_yr missing: {n_miss_yr_tar:,}")
log(f"  tar_country_code missing: {n_no_ctry_tar:,}")

# 未映射国家
all_deal_ccs_tar = set(df_tmp_tar["_cc"].dropna().unique())
mapped_ccs = set(COUNTRY_MAP.values())
unmapped_tar = all_deal_ccs_tar - mapped_ccs - {"nan"}
if unmapped_tar:
    log(f"\n  Target countries NOT in country file: {sorted(unmapped_tar)}")
    for cc in sorted(unmapped_tar):
        n = (df_tmp_tar["_cc"] == cc).sum()
        log(f"    {cc}: {n:,} rows")

log(f"\n  Tar match rate by top 10 target countries:")
top_cc_tar = df_tmp_tar.groupby("_cc").size().nlargest(10).index
for cc in top_cc_tar:
    sub = df_tmp_tar[df_tmp_tar["_cc"] == cc]
    n_hit = sub["_hit"].sum()
    log(f"    {cc:<6s}: {n_hit:>5,} / {len(sub):>5,} ({n_hit/len(sub)*100:5.1f}%)")

# ────────────────────────────────────────────────────────────────────────────
# 4.2 收购方国家的匹配诊断
# ────────────────────────────────────────────────────────────────────────────
df_tmp_acq = df.copy()
df_tmp_acq["_yr"] = pd.to_numeric(df_tmp_acq.get("completed_d_yr", pd.Series(dtype=float)),
                                   errors="coerce")
df_tmp_acq["_cc"] = df_tmp_acq.get("acq_country_code", pd.Series(dtype=str))
df_tmp_acq["_hit"] = df_tmp_acq["stock_traded_gdp_acq"].notna()

n_miss_yr_acq = df_tmp_acq["_yr"].isna().sum()
n_no_ctry_acq = df_tmp_acq["_cc"].isna().sum()

log(f"\n--- Acquirer country stock_traded_gdp_acq ---")
log(f"  completed_d_yr missing: {n_miss_yr_acq:,}")
log(f"  acq_country_code missing: {n_no_ctry_acq:,}")

all_deal_ccs_acq = set(df_tmp_acq["_cc"].dropna().unique())
unmapped_acq = all_deal_ccs_acq - mapped_ccs - {"nan"}
if unmapped_acq:
    log(f"\n  Acquirer countries NOT in country file: {sorted(unmapped_acq)}")
    for cc in sorted(unmapped_acq):
        n = (df_tmp_acq["_cc"] == cc).sum()
        log(f"    {cc}: {n:,} rows")

log(f"\n  Acq match rate by top 10 acquirer countries:")
top_cc_acq = df_tmp_acq.groupby("_cc").size().nlargest(10).index
for cc in top_cc_acq:
    sub = df_tmp_acq[df_tmp_acq["_cc"] == cc]
    n_hit = sub["_hit"].sum()
    log(f"    {cc:<6s}: {n_hit:>5,} / {len(sub):>5,} ({n_hit/len(sub)*100:5.1f}%)")

# ────────────────────────────────────────────────────────────────────────────
# 4.3 汇总统计
# ────────────────────────────────────────────────────────────────────────────
log(f"\n--- Summary statistics ---")
s_tar = df["stock_traded_gdp_tar"].dropna()
s_acq = df["stock_traded_gdp_acq"].dropna()
log(f"  stock_traded_gdp_tar: n={len(s_tar):,} | mean={s_tar.mean():.2f} | median={s_tar.median():.2f}")
log(f"  stock_traded_gdp_acq: n={len(s_acq):,} | mean={s_acq.mean():.2f} | median={s_acq.median():.2f}")

# 中国覆盖警告
log(f"\n--- China coverage ---")
_cn_tar_total = (df["tar_country_code"] == "CN").sum()
_cn_tar_hit = ((df["tar_country_code"] == "CN") & df["stock_traded_gdp_tar"].notna()).sum()
_cn_acq_total = (df["acq_country_code"] == "CN").sum()
_cn_acq_hit = ((df["acq_country_code"] == "CN") & df["stock_traded_gdp_acq"].notna()).sum()

log(f"  CN as target: {_cn_tar_hit:,}/{_cn_tar_total:,} matched ({_cn_tar_hit/max(_cn_tar_total,1)*100:.1f}%)")
log(f"  CN as acquirer: {_cn_acq_hit:,}/{_cn_acq_total:,} matched ({_cn_acq_hit/max(_cn_acq_total,1)*100:.1f}%)")

# ────────────────────────────────────────────────────────────────────────────
# 4.4 校验：标的&收购方国家一致时，两国股市指标应相等
# ────────────────────────────────────────────────────────────────────────────
log(f"\n--- Consistency Check: tar_country_code == acq_country_code ---")
# 筛选买卖双方同国样本
same_cc_mask = (df["tar_country_code"].notna()) & (df["acq_country_code"].notna())
same_cc_mask = same_cc_mask & (df["tar_country_code"] == df["acq_country_code"])
df_same_cc = df[same_cc_mask].copy()
n_same_cc_total = len(df_same_cc)
log(f"买卖双方同国交易总样本数：{n_same_cc_total:,}")

# 双方指标均不为空的子样本（才可对比差值）
both_notna_mask = df_same_cc["stock_traded_gdp_tar"].notna() & df_same_cc["stock_traded_gdp_acq"].notna()
df_compare = df_same_cc[both_notna_mask].copy()
n_compare = len(df_compare)
log(f"同国且两国股市指标均非空样本：{n_compare:,}")

# 计算差值，判定不一致（浮点误差容忍0.0001）
df_compare["gdp_diff"] = abs(df_compare["stock_traded_gdp_tar"] - df_compare["stock_traded_gdp_acq"])
inconsistent_mask = df_compare["gdp_diff"] > 1e-4
df_inconsistent = df_compare[inconsistent_mask].copy()
n_inconsistent = len(df_inconsistent)
consistent_rate = ((n_compare - n_inconsistent) / n_compare * 100) if n_compare > 0 else 100

log(f"指标不一致样本数量：{n_inconsistent:,} | 一致率：{consistent_rate:.2f}%")

if n_inconsistent > 0:
    log(f"\n【警告】存在买卖双方同国但股市指标不匹配交易，前20条明细：")
    show_cols = ["_row_id", "tar_country_code", "acq_country_code",
                 "stock_traded_gdp_tar", "stock_traded_gdp_acq", "gdp_diff",
                 "completed_d_yr", "_year_source"]
    sample_show = df_inconsistent[show_cols].head(20)
    log(sample_show.to_string(index=False))

    # 统计不匹配国家分布
    cc_bad_cnt = df_inconsistent["tar_country_code"].value_counts()
    log(f"\n不匹配样本分国家统计：")
    for cc, cnt in cc_bad_cnt.items():
        log(f"    {cc}: {cnt:,} 条")
else:
    log(f"✅ 所有买卖双方同国交易，股市指标完全匹配，无逻辑冲突")

# 统计同国但至少一方指标缺失样本
missing_one = df_same_cc[~both_notna_mask]
n_missing_one = len(missing_one)
log(f"\n同国但标的/收购方股市指标至少一方缺失样本：{n_missing_one:,}")

# ════════════════════════════════════════════════════════════════════════════
# 5. Save outputs
# ════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5 — Save outputs")
log("=" * 60)

# M1 FIX R2: retain _year_source so Stata can identify fallback-year observations
# and run robustness checks excluding them. "completed_d_yr" = used completion year;
# "announced_d_yr" = fallback used for Completed Assumed deals.

# Add _row_id: sequential integer index, unique per row.
# Assigned here (Script 04b) so that 04b_deal_firm_country.csv carries a stable
# row identifier. Script 05 and Script 06 both read this file; having _row_id in
# the source file guarantees the two scripts reference the same row by the same ID,
# without depending on CSV read order being consistent.
df["_row_id"] = range(len(df))
assert df["_row_id"].nunique() == len(df), "_row_id not unique — check for duplicate rows"
log(f"_row_id assigned: 0 to {len(df)-1:,} (all unique)")

out_path = os.path.join(MERGED, "04b_deal_firm_country.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")
log(f"Saved -> {out_path}")
log(f"Size  : {os.path.getsize(out_path)/1024/1024:.1f} MB")
log(f"Rows  : {len(df):,} | Columns: {df.shape[1]}")

diag_path = os.path.join(MERGED, "04b_deal_firm_country_diagnostics.txt")
with open(diag_path, "w", encoding="utf-8") as f:
    f.write("\n".join(diag_lines))
log(f"Saved -> {diag_path}")

log("\nScript 04b complete.")
