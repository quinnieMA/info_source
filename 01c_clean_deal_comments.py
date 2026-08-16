"""
Extract M&A Deal Comment Text Features with Loughran-McDonald Financial Dictionary
Author: CC  Date: 2026-08-08
Input Source: raw/MA_deal/acquisition_comments.csv
Output Target: data/cleaned/01c_comments_features.csv
Diagnostic Log: data/merged/01c_comments_diagnostics.txt

Core Processing Logic:
This script extracts structured numerical and textual sentiment metrics from Zephyr's raw deal comment editorial narratives,
and eliminates bulky unstructured raw text columns before export to control file storage size.

1. Text Sentiment Pipeline
    Load standard Loughran-McDonald (2011) financial word dictionary; build word-bound regular expressions
    for seven textual semantic categories: Positive, Negative, Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining.
    Calculate raw word count and normalized word density (word count / total text tokens) for each category,
    plus net sentiment density (positive density minus negative density) as aggregate managerial tone proxy.

2. Deal Timeline & Negotiation Feature Extraction
    Use regular expressions to parse all date strings within comment text; derive earliest/latest event dates
    and total timeline span (days from first reported rumour to latest closing/update event).
    Binary indicator flags for key transaction milestones: rumour emergence, target board rejection,
    unconditional offer, deal completion, Go-shop clause existence, Phase 2 antitrust investigation,
    regulatory remedy/divestment requirements, debt financing arrangements.
    Count metrics: total currency price mentions, unique regulatory bodies, competitive rival bidders.

3. Transaction Complexity Composite Index
    Aggregate weighted score (deal_complexity_score) combining competitive bidding, regulatory hurdles,
    multi-round price revisions, debt financing and textual uncertainty density to measure overall M&A friction.

4. Data Clean & Storage Optimization
    Drop raw long-text fields (editorial, Deal rationale) before saving output CSV to avoid GB-level oversized files.
    Only retain derived numerical features for subsequent master dataset merge in Script 02_merge_deal_master.py.

Unit of Observation: Single editorial comment record (one row per Zephyr news entry per deal).
Merge Key for Downstream: Integer deal_num (consistent with all other clean module files).

Dictionary Reference:
Loughran, T., & McDonald, B. (2011). When are liability risk disclosures informative? Journal of Finance.
Regulatory Keyword Library: Predefined list of global antitrust, financial and industrial supervisory authorities.
"""
import re
import os

import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ====================== 路径配置 ======================
# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = r"D:\MA"
CLEANED = os.path.join(BASE, "data", "cleaned")
MERGED  = os.path.join(BASE, "data", "merged")
os.makedirs(MERGED, exist_ok=True)

comment_path = os.path.join(BASE, "raw", "MA_deal", "acquisition_comments.csv")
lm_dict_path = os.path.join(BASE, "raw", "Loughran-McDonald_MasterDictionary_1993-2021.csv")
#out_path = r"D:\FlashCenter\LQ\raw\01-deals\comments\comments_features.csv"
# 自动创建文件夹
os.makedirs(CLEANED, exist_ok=True)
os.makedirs(MERGED, exist_ok=True)

# 简易日志打印函数
def log(msg):
    print(msg)
# ====================== 1 加载Loughran-McDonald金融词典【完全重写修复】 ======================
lm_df = pd.read_csv(lm_dict_path)
lm_df["Word_lower"] = lm_df["Word"].str.lower().str.strip()

# 正确筛选：列值 !=0 代表该词属于对应分类（非0是首次收录年份）
pos_words = lm_df[lm_df["Positive"] != 0]["Word_lower"].dropna().tolist()
neg_words = lm_df[lm_df["Negative"] != 0]["Word_lower"].dropna().tolist()
uncert_words = lm_df[lm_df["Uncertainty"] != 0]["Word_lower"].dropna().tolist()
litigious_words = lm_df[lm_df["Litigious"] != 0]["Word_lower"].dropna().tolist()
strong_modal = lm_df[lm_df["Strong_Modal"] != 0]["Word_lower"].dropna().tolist()
weak_modal = lm_df[lm_df["Weak_Modal"] != 0]["Word_lower"].dropna().tolist()
constrain_words = lm_df[lm_df["Constraining"] != 0]["Word_lower"].dropna().tolist()

# 打印校验词表长度，确认不为空
# 打印校验词表长度，确认不为空
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
    # (?:...) 非捕获分组，避免findall重复计数bug
    pat_str = r"\b(?:{}) \b".format("|".join(word_esc))
    return re.compile(pat_str, re.IGNORECASE)

pos_pat = build_word_re(pos_words)
neg_pat = build_word_re(neg_words)
uncert_pat = build_word_re(uncert_words)
litigious_pat = build_word_re(litigious_words)
strong_modal_pat = build_word_re(strong_modal)
weak_modal_pat = build_word_re(weak_modal)
constrain_pat = build_word_re(constrain_words)

# ====================== 2 客观交易固定正则（保留不变） ======================
date_pat = re.compile(r"(\d{2}/\d{2}(?:/\d{2,4})?)")
reg_keywords = [
    "European Commission", "EU Commission", "CMA", "FTC", "DOJ",
    "CADE", "Fiscalia Nacional Economica", "SAMR", "Korea's FTC",
    "Federal Communications Commission", "Hart-Scott-Rodino", "Phase 1", "Phase 2"
]
reg_pat = re.compile("|".join([re.escape(k) for k in reg_keywords]), re.IGNORECASE)
price_pat = re.compile(r"(EUR|USD|GBP)\s*[\d,.]+(?:\s+billion|\s*bn|\s*m)", re.IGNORECASE)
premium_pat = re.compile(r"premium of\s*(\d+\.\d+)\s*per cent|premium.*?(\d+\.\d+)", re.IGNORECASE)
rival_pat = re.compile(r"rival|suitor|white knight|bid for|poised to bid", re.IGNORECASE)
rumour_pat = re.compile(r"reported that|rumour|speculation", re.IGNORECASE)
reject_pat = re.compile(r"rejected", re.IGNORECASE)
unconditional_pat = re.compile(r"unconditional", re.IGNORECASE)
complete_pat = re.compile(r"completed|closed the deal", re.IGNORECASE)
go_shop_pat = re.compile(r"go-shop", re.IGNORECASE)
debt_fin_pat = re.compile(r"loan|syndicated debt|assumption of debt", re.IGNORECASE)
remedy_pat = re.compile(r"divest|sell off asset|remedy|concessions", re.IGNORECASE)

# ====================== 3 通用工具函数 ======================
def parse_date(d_str: str):
    """
    适配欧洲DD/MM/YY / DD/MM/YYYY
    兼容三段：13/11/99、23/01/2004；两段：06/02（无年份直接丢弃）
    """
    try:
        parts = d_str.split("/")
        day = int(parts[0])
        month = int(parts[1])

        # 只处理三段带年份的日期，两段无年份直接返回None
        if len(parts) != 3:
            return None
        
        year_raw = int(parts[-1])
        if len(str(year_raw)) == 4:
            # 4位完整年份，直接使用
            year = year_raw
        else:
            # 2位年份判断
            if year_raw <= 30:
                year = 2000 + year_raw
            else:
                year = 1900 + year_raw

        # 严格日期合法性校验，过滤 31/04、30/02 这类无效日期
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        
        # 构建日期对象（自动校验当月最大天数，比如2月30会报错）
        return datetime(year, month, day)
    except Exception:
        # 数字越界、非法日期全部返回空
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

# ====================== 4 特征抽取函数【核心修复区】 ======================
def extract_features(row):
    raw_text = str(row["comments"]).strip()
    # ========== 修复点1：删除 text_lower = raw_text.lower() ==========
    # 正则自带 re.IGNORECASE，无需手动小写，避免破坏单词边界\b匹配
    if len(raw_text) < 10:
        return {}
    word_total = len(raw_text.split())

    dates = extract_all_dates(raw_text)
    dt_min = dates[0] if len(dates) > 0 else None
    dt_max = dates[-1] if len(dates) > 0 else None
    total_days = (dt_max - dt_min).days if (dt_min is not None and dt_max is not None) else None

    # 基础交易事实
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

    # ========== 修复点2：全部传入 raw_text，不再传入小写文本 ==========
    pos_cnt = count_lm_term(raw_text, pos_pat)
    neg_cnt = count_lm_term(raw_text, neg_pat)
    uncert_cnt = count_lm_term(raw_text, uncert_pat)
    lit_cnt = count_lm_term(raw_text, litigious_pat)
    strong_mod_cnt = count_lm_term(raw_text, strong_modal_pat)
    weak_mod_cnt = count_lm_term(raw_text, weak_modal_pat)
    constrain_cnt = count_lm_term(raw_text, constrain_pat)

    # 分词修复，解决密度>1异常
    all_tokens = [t for t in re.split(r"\s+", raw_text) if t.strip()]
    word_total = len(all_tokens)
    word_cnt = word_total

    # 密度指标
    pos_dens = pos_cnt / word_total if word_total > 0 else 0
    neg_dens = neg_cnt / word_total if word_total > 0 else 0
    uncert_dens = uncert_cnt / word_total if word_total > 0 else 0
    lit_dens = lit_cnt / word_total if word_total > 0 else 0
    strong_mod_dens = strong_mod_cnt / word_total if word_total > 0 else 0
    weak_mod_dens = weak_mod_cnt / word_total if word_total > 0 else 0
    constrain_dens = constrain_cnt / word_total if word_total > 0 else 0
    net_sentiment = pos_dens - neg_dens

    feat = {
        # 时序
        "dt_first": dt_min,
        "dt_last": dt_max,
        "total_timeline_days": total_days,
        "has_rumour": has_rumour,
        "has_target_reject": has_reject,
        "has_unconditional_offer": has_unconditional,
        "has_deal_complete": has_complete,
        # 溢价
        "price_event_num": price_count,
        "max_premium_pct": max_prem,
        "avg_premium_pct": prem_mean,
        # 竞标
        "rival_bidder_num": rival_count,
        "has_goshop": has_goshop,
        # 监管
        "reg_event_count": reg_count,
        "has_phase2_investigation": has_phase2,
        "has_reg_remedy": has_remedy,
        "reg_entity_list": reg_entities,
        "num_unique_reg": len(reg_entities),
        # 融资
        "has_debt_assumption": has_debt_fin,
        # 文本基础
        "comment_char_length": char_len,
        "comment_wordcount": word_cnt,
        "sentence_event_count": event_tokens,
        # LM计数
        "lm_pos_count": pos_cnt,
        "lm_neg_count": neg_cnt,
        "lm_uncertain_count": uncert_cnt,
        "lm_litigious_count": lit_cnt,
        "lm_strongmodal_count": strong_mod_cnt,
        "lm_weakmodal_count": weak_mod_cnt,
        "lm_constrain_count": constrain_cnt,
        # LM密度（主回归变量）
        "lm_pos_density": pos_dens,
        "lm_neg_density": neg_dens,
        "lm_uncertain_density": uncert_dens,
        "lm_litigious_density": lit_dens,
        "lm_strongmodal_density": strong_mod_dens,
        "lm_weakmodal_density": weak_mod_dens,
        "lm_constrain_density": constrain_dens,
        "lm_net_sentiment": net_sentiment
    }
    return feat

# 重写计数函数，用finditer精准计数，规避findall缺陷
def count_lm_term(text: str, pat: re.Pattern) -> int:
    cnt = 0
    for _ in pat.finditer(text):
        cnt +=1
    return cnt

# ====================== 5 交易复杂度函数 ======================
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
    # 处理NaN空值，缺失则按0计算
    unc_val = row["lm_uncertain_density"] if pd.notna(row["lm_uncertain_density"]) else 0
    score += round(unc_val * 10)
    return score
# ====================== 6 主执行流程 ======================
df = pd.read_csv(
    comment_path,
    encoding="utf-8-sig",
    usecols=["Deal Number","Date of editorial", "Deal comments" ],
    engine="python",
    on_bad_lines="skip",
    quoting=3
)

df.rename(columns={
    "Deal Number": "deal_num",
    "Date of editorial": "edit_date",
    "Deal comments": "comments"
}, inplace=True)

print("开始抽取Deal comments特征（LM金融词典版）...")
feat_result = df.apply(lambda row: extract_features(row), axis=1)
feat_df = pd.DataFrame(feat_result.tolist())
# 修复：必须加axis=1横向拼接原始表+特征表
df_full = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

# 计算综合复杂度
df_full["deal_complexity_score"] = df_full.apply(calc_complex, axis=1)

# ========== 新增：导出前删除原始超长文本字段 ==========
drop_raw_text = [
    "editorial",
    "Deal rationale"
]
drop_cols = [c for c in drop_raw_text if c in df_full.columns]
if len(drop_cols) > 0:
    df_full = df_full.drop(columns=drop_cols)
    log(f"Dropped raw heavy text columns before export: {drop_cols}")
    
# 输出文件标准化路径
out_comments = os.path.join(CLEANED, "01c_comments_features.csv")
df_full.to_csv(out_comments, index=False, encoding="utf-8-sig")
log(f"\nSaved → {out_comments}")
log(f"File size: {os.path.getsize(out_comments)/1024:.1f} MB")

# 简易诊断日志（可选，如需统计可扩充diag_lines）
diag_lines = [
    f"Total rows processed: {len(df_full)}",
    f"Unique deal_num: {df_full['deal_num'].nunique()}",
    f"Valid non-empty comments count: {(df_full['comment_wordcount']>10).sum()}",
    f"Mean deal_complexity: {df_full['deal_complexity_score'].mean():.2f}"
]
diag_path = os.path.join(MERGED, "01c_comments_diagnostics.txt")
with open(diag_path, "w", encoding="utf-8") as f:
    f.write("\n".join(diag_lines))
log(f"\nDiagnostics saved → {diag_path}")
log("\nScript untitled1 complete.")