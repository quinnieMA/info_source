# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 11:55:17 2026

@author: 13601
"""

# run_all.py
"""
MA pipeline runner
Usage: python run_all.py
"""

import subprocess, sys, time, os

from global_config import SCRIPTS_DIR, RUN_TEXT, RUN_08
MA_ROOT = r"D:\MA"
os.chdir(MA_ROOT)
# 脚本顺序 = 论文数据生成顺序（别动）
STEPS = [
    "01_clean_deal_modules",
    "01b_clean_deal_overview",
    "01c_clean_deal_comments",
    "02_merge_deal_master",
    "03_clean_firm_modules",
    "04a_merge_firm_to_deal",
    "04b_merge_country",
    "05_sample_filter",
]

TEXT_STEPS = [
    "06_text_benchmark",
    "07_sic_benchmark",
]

FINAL_STEPS = [
    "08b_variable_construction",
]

print("\n" + "="*60)
print("MA PIPELINE RUNNER")
print("="*60)
print(f"Scripts dir: {SCRIPTS_DIR}\n")

def run_step(script_name):
    """每个脚本开独立子进程，跑完自动释放内存"""
    script_path = os.path.join(MA_ROOT, "data", f"{script_name}.py")
    print(f"\n▶ {script_name}")
    t0 = time.time()
    
    # 关键：subprocess 隔离，每个脚本独立进程
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
        cwd=MA_ROOT,
    )
    
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✅ done in {elapsed:.1f}s")
    else:
        print(f"  ❌ FAILED (exit code {result.returncode})")
        raise SystemExit(f"Pipeline stopped at {script_name}")
        
        
# ── 1. Core deal + firm pipeline ─────────────────────
for s in STEPS:
    run_step(s)

# ── 2. Text benchmark (可开关) ───────────────────────
if RUN_TEXT:
    for s in TEXT_STEPS:
        run_step(s)
else:
    print("\n⚠️ Text steps skipped (RUN_TEXT = False)")

# ── 3. Final variable construction ───────────────────
if RUN_08:
    run_step(FINAL_STEPS[0])
else:
    print("\n⚠️ 08b skipped (RUN_08 = False)")

print("\n" + "="*60)
print("✅ ALL DONE — outputs in data/merged/")
print("="*60)