# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 08:37:04 2026

@author: 13601
"""

# -*- coding: utf-8 -*-
"""
paylib.py — 支付方式分类的单一定义源 (2026-09-07)

用法:
    from paylib import ensure_pay_vars
    df = ensure_pay_vars(df)

背景:
    Zephyr 一条支付方式占一行。01a Module C 在 deal_num 去重【之前】
    把全部取值聚合成竖线分隔集合 deal_pay_method_all。
    实测：去重口径 Shares 13.4% → 聚合口径 18.6%
          （2,238 笔交易的换股信息被 keep-first 去重丢弃）。

设计（关键）:
    先按【类别】归并（Cash 与 Cash Reserves 同属现金类，不算混合），
    再判断纯/混。若直接按取值个数判断，3,244 笔 Cash|Cash Reserves
    会被误判为混合支付，严重低估纯现金组。
"""
import numpy as np
import pandas as pd

# 取值 → 类别（覆盖 01a 观察到的全部 15 个取值）
PAY_CLASS_MAP = {
    "Cash":               "cash",
    "Cash Reserves":      "cash",
    "Shares":             "shares",      # 触发证券审查的关键组
    "Third party shares": "shares",
    "Liabilities":        "debt",
    "Converted Debt":     "debt",
    "Bonds":              "debt",
    "Deferred payment":   "other",
    "Earn-out":           "other",
    "Business assets":    "other",
    "Dividend":           "other",
    "Services":           "other",
    "Other":              "other",
    "Cash assumed":       "other",       # 承担标的现金，语义模糊，保守归 other
}

SRC_PREF = ("deal_pay_method_all", "deal_pay_method")


def ensure_pay_vars(df, verbose=True):
    """为 df 补齐支付方式变量。缺源列直接报错（不静默降级）。"""
    src = next((c for c in SRC_PREF if c in df.columns), None)
    if src is None:
        raise ValueError(
            f"缺支付方式源列（{SRC_PREF}）。请重跑 01a（生成 deal_pay_method_all）"
            f"并确认该列已流入 08b 的 dta 导出白名单。")

    if verbose:
        print(f"[paylib] 源列 = {src}")

    unmapped = set()

    def _to_classes(s):
        if pd.isna(s) or not str(s).strip():
            return frozenset()
        out = set()
        for v in (x.strip() for x in str(s).split("|")):
            if not v:
                continue
            c = PAY_CLASS_MAP.get(v)
            if c is None:
                unmapped.add(v)
            else:
                out.add(c)
        return frozenset(out)

    ps = df[src].apply(_to_classes)

    if unmapped and verbose:
        print(f"[paylib] ⚠️ 未映射取值 {len(unmapped)} 个: {sorted(unmapped)}")
        print(f"         请补进 PAY_CLASS_MAP 后重跑，否则这些交易会落入错误类别")

    # 非互斥：是否含某类
    df["pay_has_cash"]   = ps.apply(lambda c: int("cash"   in c))
    df["pay_has_shares"] = ps.apply(lambda c: int("shares" in c))
    df["pay_has_debt"]   = ps.apply(lambda c: int("debt"   in c))
    df["pay_has_other"]  = ps.apply(lambda c: int("other"  in c))
    df["pay_n_class"]    = ps.apply(len)

    # 互斥全集（6 个哑变量之和恒为 1）
    df["pay_pure_cash"]   = ps.apply(lambda c: int(c == {"cash"}))
    df["pay_pure_shares"] = ps.apply(lambda c: int(c == {"shares"}))
    df["pay_pure_debt"]   = ps.apply(lambda c: int(c == {"debt"}))
    df["pay_pure_other"]  = ps.apply(lambda c: int(c == {"other"}))
    df["pay_mix"]         = ps.apply(lambda c: int(len(c) > 1))
    df["pay_unknown"]     = ps.apply(lambda c: int(len(c) == 0))

    # 混合支付细分
    df["pay_mix_cash_shares"] = ps.apply(lambda c: int(c == {"cash", "shares"}))
    df["pay_mix_with_shares"] = ps.apply(lambda c: int(len(c) > 1 and "shares" in c))
    df["pay_mix_with_cash"]   = ps.apply(lambda c: int(len(c) > 1 and "cash"   in c))
    df["pay_mix_with_debt"]   = ps.apply(lambda c: int(len(c) > 1 and "debt"   in c))

    # 互斥类别标签（回归用 C(pay_class)，基准组 cash）
    df["pay_class"] = np.select(
        [df["pay_unknown"] == 1,
         df["pay_pure_cash"] == 1,
         df["pay_pure_shares"] == 1,
         df["pay_pure_debt"] == 1,
         df["pay_pure_other"] == 1],
        ["unknown", "cash", "shares", "debt", "other"],
        default="mix")

    # 校验：必须构成全集且互斥
    chk = df[["pay_pure_cash", "pay_pure_shares", "pay_pure_debt",
              "pay_pure_other", "pay_mix", "pay_unknown"]].sum(axis=1)
    if not (chk == 1).all():
        raise ValueError(f"支付方式哑变量未构成全集（{int((chk != 1).sum())} 行异常）")

    if verbose:
        print(f"[paylib] 含换股 {df['pay_has_shares'].mean():.1%}"
              f" | 混合 {df['pay_mix'].mean():.1%}"
              f" | 无记录 {df['pay_unknown'].mean():.1%}")
    return df