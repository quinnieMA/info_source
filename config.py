# -*- coding: utf-8 -*-
"""config.py — 修复版（2026-09-05）
改动：
  1. 删除重复函数定义
  2. procedural_hardness / legal_hedge 废弃（依赖已删除的 lm_*_density，
     且 Litigious 词表与监管审查存在定义重叠 → 套娃）
  3. 监管摩擦改用结构化 reg_body_count（国家中性，覆盖率 12.8%）
  4. 数据源优先 .dta，回退 csv
"""
import os
import pandas as pd
import numpy as np
from global_config import MERGED, CLEANED, OUTPUT

DTA_PATH = os.path.join(MERGED, "08b_deal_firm_analysis.dta")
CSV_PATH = os.path.join(MERGED, "08b_temp_export.csv")
OUT_DIR  = OUTPUT
os.makedirs(OUT_DIR, exist_ok=True)


def load_df(path=None):
    if path is None:
        path = DTA_PATH if os.path.exists(DTA_PATH) else CSV_PATH
    if str(path).lower().endswith(".dta"):
        df = pd.read_stata(path, convert_categoricals=False)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[load] {os.path.basename(path)}: {len(df):,} rows × {len(df.columns)} cols")
    return df


def apply_sample(df, verbose=True):
    n0 = len(df)
    df = df[(df["tar_listed"] == 0) & (df["acq_listed"] == 1)].copy()
    if verbose:
        print(f"[sample] {n0:,} → {len(df):,} (tar_listed=0 & acq_listed=1)")
    return df


def build_derived(df):
    df = attach_stake(df)      # ← 加这一行
    # ── 信息暴露 ──
    df["info_exposure"] = pd.to_numeric(df.get("total_info_source"), errors="coerce")

    # ── 监管类型（01b 结构化，国家中性）──
    for c in ["reg_antitrust", "reg_securities", "reg_state_assets",
              "reg_foreign_invest", "reg_financial", "reg_defense_tech"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["hard_reg_exposure"] = (
        (df.get("reg_antitrust", 0) == 1) |
        (df.get("reg_state_assets", 0) == 1) |
        (df.get("reg_defense_tech", 0) == 1)
    ).astype(int)
    df["soft_reg_exposure"] = (
        (df.get("reg_securities", 0) == 1) & (df["hard_reg_exposure"] == 0)
    ).astype(int)

    # 2×2 监管单元（本次分析主口径）
    df["reg_cell"] = (df.get("reg_antitrust", 0).astype(str) + "_" +
                      df.get("reg_securities", 0).astype(str))

    # ── 监管摩擦：结构化，替代已废弃的文本 hardness ──
    df["reg_friction"] = pd.to_numeric(df.get("reg_body_count"), errors="coerce")

    # ── 文本变量（仅描述性，不作因果）──
    # 注：LM Litigious 为法律语域词表，含 antitrust/court/claim/acquirors 等，
    #     与监管审查定义重叠 → 仅用于构念效度检验
    if "has_lm_litigious" in df.columns:
        df["has_comment"] = df["has_lm_litigious"].notna().astype(int)
    df["log_comment_wordcount"] = pd.to_numeric(
        df.get("log_comment_wordcount"), errors="coerce")

    # ── 中国 / 跨境 ──
    df["tar_china"]    = (df["tar_country_code"] == "CN").astype(int)
    df["acq_china"]    = (df["acq_country_code"] == "CN").astype(int)
    df["cross_border"] = (df["tar_country_code"] != df["acq_country_code"]).astype(int)

    # ── 交易规模 ──
    dv = pd.to_numeric(df.get("deal_value"), errors="coerce")
    df["ln_deal_value"] = np.where(dv > 0, np.log(dv), np.nan)

    print(f"[derived] N={len(df):,} | hard_reg={df['hard_reg_exposure'].mean():.3f} "
          f"| soft_reg={df['soft_reg_exposure'].mean():.3f} "
          f"| reg_friction={df['reg_friction'].mean():.3f}")
    return df

STAKE_SRC = os.path.join(MERGED, "07_deal_firm_benchmark.csv")

def attach_stake(df):
    """07 有 stake_acq_pct / stake_final_pct，08b 导出时未包含 → 按行序或 _row_id 补回"""
    if "stake_acq_pct" in df.columns:
        return df
    src = pd.read_csv(STAKE_SRC, usecols=["stake_acq_pct", "stake_final_pct"],
                      low_memory=False)
    if "_row_id" in df.columns:
        src["_row_id"] = pd.read_csv(STAKE_SRC, usecols=["_row_id"],
                                     low_memory=False)["_row_id"]
        df = df.merge(src, on="_row_id", how="left")
        print("[stake] 按 _row_id 并入")
    elif len(src) == len(df):
        for c in ["stake_acq_pct", "stake_final_pct"]:
            df[c] = src[c].values
        print("[stake] 按行序并入")
    else:
        print("[stake] ⚠️ 无法并入")
    return df