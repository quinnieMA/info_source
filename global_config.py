# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:53:49 2026

@author: 13601
"""

# global_config.py
# ===============================
# MA pipeline global config
# Coauthor: change BASE only
# ===============================

import os
from pathlib import Path

# ── 1. ROOT PATH ─────────────────────────────────────────────
BASE = r"D:\MA"          # ← 唯一需要改的地方
BASE = Path(BASE)

# ── 2. Derived paths ─────────────────────────────────────────
RAW      = BASE / "raw"
CLEANED  = BASE / "data" / "cleaned"
MERGED   = BASE / "data" / "merged"
MODELS   = BASE / "data" / "models"
OUTPUT   = BASE / "output"

for p in (CLEANED, MERGED, MODELS, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)

# ── 3. Orbis ID dtype (所有脚本共用) ───────────────────────
ORBIS_DTYPE = {
    "tar_orbis_id_num": str,
    "acq_orbis_id_num": str,
    "ven_orbis_id_num": str,
    "tar_orbis_id_num_ovw": str,
    "acq_orbis_id_num_ovw": str,
}

# ── 4. Triple merge key (firm modules) ─────────────────────
MERGE_KEYS = [
    "deal_num",
    "tar_bvd_id_num", "tar_orbis_id_num",
    "acq_bvd_id_num", "acq_orbis_id_num"
]

# ── 5. pandas display niceties ─────────────────────────────
import pandas as pd
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 300)
pd.set_option("mode.chained_assignment", None)

# ── 6. Pipeline switches (可选) ────────────────────────────
DEBUG_N = None          # None = 全跑；1000 = 试跑
RUN_TEXT = True         # False = 跳过 06/07
RUN_08 = True           # False = 只到 benchmark

# ── 7. Text benchmark defaults ─────────────────────────────
TEXT_K = 10
TEXT_MIN_PEERS = 3
TEXT_SUFFIX = ""        # "_K5" 之类 robustness 用

# ── 8. 时间戳（diagnostic 用） ────────────────────────────
from datetime import datetime
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M")

# ── 9. 给脚本 import 用 ───────────────────────────────────
SCRIPTS_DIR = BASE / "scripts"