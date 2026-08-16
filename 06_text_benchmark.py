# -*- coding: utf-8 -*-
"""
06_text_benchmark.py
Date: 2026-04-14 (PATCHED: 2026-08-03)
Purpose: Train Doc2Vec on full corpus and compute text-based valuation benchmarks.
         HP2025 (JF 2025) parameters. Rolling window peer selection (deal_year <= t)
         ensures no look-ahead bias. Uses trained document vectors (model.dv), not
         infer_vector(), for full reproducibility.

NEW (PATCH 2026-08-03):
    - Computes acq_tar_similarity: cosine similarity between acquirer and target overviews
    - This is a DIRECT similarity score for each transaction (not KNN-based)

Inputs:
    data/merged/05_deal_firm_filtered.csv   -- focal deals (2000-2024)
    data/merged/04b_deal_firm_country.csv   -- full corpus (all deals)

Outputs:
    data/merged/07_deal_firm_benchmark{suffix}.csv  -- focal deals + benchmark columns
    data/models/doc2vec_full_corpus.model           -- trained Doc2Vec model (reusable)
    data/merged/06_text_benchmark_log{suffix}.txt   -- diagnostics

New columns added to 07_deal_firm_benchmark{suffix}.csv:
    text_bench_rev, text_bench_ebitda, text_bench_ebit
    text_peer_ids        -- pipe-separated _row_ids of the K nearest neighbors
    text_peer_sim_mean   -- mean cosine similarity of the K peers
    n_valid_text_peers_rev, n_valid_text_peers_ebitda, n_valid_text_peers_ebit
    ind_text_diversity   -- within SIC-3 × year std of pairwise text similarities
    acq_tar_similarity   -- DIRECT similarity between acquirer and target (NEW)
"""

import os
import re
import sys
import argparse
import logging
import random
from bisect import bisect_right

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# ── CLI args (robustness variants) ──────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--K", type=int, default=10, help="number of nearest neighbors")
_parser.add_argument("--min-peers", type=int, default=3,
                      help="minimum peers with valid multiple for non-NaN benchmark")
_parser.add_argument("--suffix", type=str, default="",
                      help="suffix appended to output/log filenames, e.g. _K5")
_args = _parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = r"D:\MA"
MERGED = os.path.join(ROOT, "data", "merged")
MODELS = os.path.join(ROOT, "data", "models")
LIT_SW = os.path.join(ROOT, "raw", "stop_words.txt")
LOG_PATH = os.path.join(MERGED, f"06_text_benchmark_log{_args.suffix}.txt")

os.makedirs(MODELS, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("06_text_benchmark")
log.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(fh)
log.addHandler(ch)

log.info("=== Script 06: Text Benchmark (Doc2Vec) ===")
if _args.suffix:
    log.info(f"  Robustness variant run: K={_args.K}  min_peers={_args.min_peers}  suffix={_args.suffix}")

# ── Constants ─────────────────────────────────────────────────────────────────
K = _args.K
MIN_PEERS = _args.min_peers
MIN_TOKENS = 10

# ── Stopwords ─────────────────────────────────────────────────────────────────
log.info("Loading stopwords ...")
with open(LIT_SW, "r", encoding="utf-8") as f:
    hp2025_stop = {w.strip().lower() for w in f if w.strip()}

BUSINESS_BOILERPLATE = {
    "company", "companies", "corporation", "incorporated", "limited",
    "operations", "business", "businesses", "headquartered", "registered",
    "office", "offices", "subsidiary", "subsidiaries", "holding",
    "provides", "provide", "offering", "offers", "offer", "services",
    "service", "products", "product", "include", "includes", "including",
    "primarily", "engaged", "engage", "operates", "operate", "conduct",
    "conducts", "based", "located", "founded", "established", "known",
    "formerly", "also", "through", "well", "various", "range", "wide",
    "full", "number", "related", "activities", "activity",
    "head", "involved", "provision", "providing", "addition",
    "specialises", "specializes", "active", "leading", "principally",
    "serves", "strategically", "employs", "firm", "group",
    "keting", "kets", "management", "international", "corporate", "solutions",
}

STOP_WORDS = hp2025_stop | BUSINESS_BOILERPLATE
log.info(f"  HP2025 stop words: {len(hp2025_stop)}, boilerplate: {len(BUSINESS_BOILERPLATE)}, combined: {len(STOP_WORDS)}")


# ── Text Preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    text = re.sub(
        r"(?:(?:2(?:[0-4][0-9]|5[0-5])|[0-1]?[0-9]?[0-9]|(\*?))\.){3}"
        r"(?:(?:2([0-4][0-9]|5[0-5])|[0-1]?[0-9]?[0-9]|(\*))?)",
        " ", text,
    )
    text = re.sub(
        r"((\d\d?(st|nd|rd|th)?([\-\./])?)?\s?"
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)"
        r"(\s?([\-\./])?(,)?\s?(\d\d\d?\d?))?)"
        r"|(([\d]+)([\-])([\d]+)([\-])([\d]+))"
        r"|(([\d]+)([/])([\d]+)([/])([\d]+))"
        r"|(([\d]+)([.])([\d]+)([.])([\d]+))",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(<script(\s|\S)*?<\/script>)|(<style(\s|\S)*?<\/style>)"
        r"|(<!--(\s|\S)*?-->)|(<\/?(\s|\S)*?>)",
        " ", text,
    )
    text = re.sub(
        r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
        r"@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[eE]([+-]?\d+))?", " ", text)
    text = re.sub(r"[`~@$^&\\;\'<>.,!?()#\"|=\-:/_*\[\]+%]", " ", text)
    text = text.lower()
    return [w for w in text.split() if len(w) >= 4 and w not in STOP_WORDS]


# ── Load Data ─────────────────────────────────────────────────────────────────
log.info("Loading data ...")
df_filtered = pd.read_csv(os.path.join(MERGED, "05_deal_firm_filtered.csv"), low_memory=False)
df_full = pd.read_csv(os.path.join(MERGED, "04b_deal_firm_country.csv"), low_memory=False)
df_filtered["_row_id"] = df_filtered["_row_id"].astype(int)
df_full["_row_id"] = df_full["_row_id"].astype(int)
log.info(f"  05_deal_firm_filtered.csv:  {len(df_filtered):,} rows")
log.info(f"  04b_deal_firm_country.csv:  {len(df_full):,} rows")

# ── _row_id Consistency Verification ─────────────────────────────────────────
log.info("Verifying _row_id consistency ...")
assert "_row_id" in df_full.columns, "_row_id missing from full corpus"
assert "_row_id" in df_filtered.columns, "_row_id missing from filtered"
assert df_full["_row_id"].nunique() == len(df_full), "_row_id not unique in full corpus"

sample_ids = df_filtered["_row_id"].sample(5, random_state=SEED).tolist()
for rid in sample_ids:
    name_filt = df_filtered.loc[df_filtered["_row_id"] == rid, "tar_name"].values[0]
    name_full = df_full.loc[df_full["_row_id"] == rid, "tar_name"].values[0]
    assert name_filt == name_full, f"_row_id={rid}: tar_name mismatch!"
log.info("  _row_id consistency check: PASS")

# ── Derive deal_year for full corpus ──────────────────────────────────────────
log.info("Deriving deal_year for full corpus ...")
df_full["deal_year"] = df_full["completed_d_yr"].where(
    df_full["completed_d_yr"].notna(), df_full["announced_d_yr"]
)
log.info(f"  deal_year notna: {df_full['deal_year'].notna().sum():,}, null: {df_full['deal_year'].isna().sum():,}")

# ── Preprocess All Texts ──────────────────────────────────────────────────────
log.info("Preprocessing tar_overview texts for full corpus ...")
df_full["tokens"] = df_full["tar_overview"].apply(lambda x: preprocess_text(x) if pd.notna(x) else [])
df_full["token_count"] = df_full["tokens"].apply(len)

notna_mask = df_full["tar_overview"].notna()
log.info(f"  Rows with tar_overview notna: {notna_mask.sum():,}")
log.info(f"  Rows with token_count >= {MIN_TOKENS}: {(df_full['token_count'] >= MIN_TOKENS).sum():,}")

tc = df_full.loc[df_full["token_count"] > 0, "token_count"]
log.info(f"  Token count (non-zero): min={tc.min()}, p25={tc.quantile(0.25):.0f}, p50={tc.quantile(0.50):.0f}, p75={tc.quantile(0.75):.0f}, max={tc.max()}")

# ── Doc2Vec Training ──────────────────────────────────────────────────────────
model_path = os.path.join(MODELS, "doc2vec_full_corpus.model")

train_mask = df_full["tar_overview"].notna() & df_full["deal_year"].notna()
train_df = df_full[train_mask].copy()
log.info(f"Training corpus: {len(train_df):,} documents")

tagged_docs = [TaggedDocument(words=row["tokens"], tags=[int(row["_row_id"])]) for _, row in train_df.iterrows()]
log.info(f"  Tagged documents built: {len(tagged_docs):,}")

trained_rids = {int(doc.tags[0]) for doc in tagged_docs}

if os.path.exists(model_path):
    log.info(f"Loading existing model: {model_path}")
    model = Doc2Vec.load(model_path)
else:
    log.info("Training Doc2Vec (HP2025 parameters: PV-DBOW, 300d, w=15, epochs=40) ...")
    model = Doc2Vec(
        documents=tagged_docs,
        dm=0,
        vector_size=300,
        window=15,
        min_count=3,
        dbow_words=1,
        hs=1,
        negative=0,
        sample=1e-5,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=40,
        workers=1,
        seed=SEED,
    )
    model.save(model_path)
    log.info(f"  Model saved: {model_path}")

log.info(f"  Vocabulary size: {len(model.wv.key_to_index):,}")
log.info(f"  Trained document vectors: {len(trained_rids):,}")

# ── Sanity Check ──────────────────────────────────────────────────────────────
log.info("Doc2Vec sanity check (nearest neighbors for 3 focal deals):")
valid_tags = trained_rids
for _, row in df_filtered.sample(3, random_state=SEED).iterrows():
    rid = int(row["_row_id"])
    if rid not in valid_tags:
        log.info(f"  {row.get('tar_name','?')} (rid={rid}): no trained vector")
        continue
    try:
        similar = model.dv.most_similar(rid, topn=3)
        peer_strs = []
        for sim_rid, sim_score in similar:
            match = df_full.loc[df_full["_row_id"] == sim_rid, "tar_name"]
            name = match.values[0] if len(match) > 0 else "?"
            peer_strs.append(f"{name} ({sim_score:.3f})")
        log.info(f"  Focal: {row.get('tar_name','?')} (yr={row['deal_year']:.0f})")
        log.info(f"    Top-3 text peers: {' | '.join(peer_strs)}")
    except Exception as e:
        log.warning(f"  Sanity check error for rid={rid}: {e}")

# ============================================================================
# PATCH 1: Pre-process acquirer overviews for all focal deals
# ============================================================================
log.info("Preprocessing acquirer overviews for focal deals ...")
df_filtered["acq_tokens"] = df_filtered["acq_overview"].apply(
    lambda x: preprocess_text(x) if pd.notna(x) else []
)
df_filtered["acq_token_count"] = df_filtered["acq_tokens"].apply(len)
log.info(f"  Acquirer token count >= {MIN_TOKENS}: {(df_filtered['acq_token_count'] >= MIN_TOKENS).sum():,}")
# ============================================================================

# ── Build Peer Candidate Pool ─────────────────────────────────────────────────
log.info("Building peer candidate pool ...")
pool_mask = (
    df_full["_row_id"].isin(valid_tags)
    & (df_full["token_count"] >= MIN_TOKENS)
    & df_full["deal_year"].notna()
)
df_pool = df_full[pool_mask].copy().reset_index(drop=True)
log.info(f"  Peer pool: {len(df_pool):,} rows")
assert all(int(rid) in trained_rids for rid in df_pool["_row_id"].values), "FATAL: pool contains rows with untrained vectors!"
log.info(f"  Pool year range: {int(df_pool['deal_year'].min())}--{int(df_pool['deal_year'].max())}")

# ── Expanding-window winsorisation ───────────────────────────────────────────
MUL_COLS = ["pre_rev_mul_ly", "pre_ebitda_mul_ly", "pre_ebit_mul_ly"]
focal_years_sorted = sorted(df_filtered["deal_year"].dropna().unique())

year_bounds = {}
for t in focal_years_sorted:
    pool_up_to_t = df_pool[df_pool["deal_year"] <= t]
    bounds = {}
    for _col in MUL_COLS:
        _lo = pool_up_to_t[_col].quantile(0.01)
        _hi = pool_up_to_t[_col].quantile(0.99)
        bounds[_col] = (_lo, _hi)
    year_bounds[t] = bounds

first_fy = focal_years_sorted[0]
last_fy = focal_years_sorted[-1]
for _col in MUL_COLS:
    lo0, hi0 = year_bounds[first_fy][_col]
    lo1, hi1 = year_bounds[last_fy][_col]
    log.info(f"  Expanding-window winsorise {_col}: year={int(first_fy)} [{lo0:.3f}, {hi0:.3f}]  |  year={int(last_fy)} [{lo1:.3f}, {hi1:.3f}]")

sorted_focal_years_arr = np.array(focal_years_sorted)

def _get_bounds_for_year(deal_yr, col):
    candidates = sorted_focal_years_arr[sorted_focal_years_arr <= deal_yr]
    if len(candidates) == 0:
        t_use = focal_years_sorted[0]
    else:
        t_use = float(candidates[-1])
    return year_bounds[t_use][col]

for _col in MUL_COLS:
    raw_vals = df_pool[_col].values.astype(float)
    lo_vals = np.array([_get_bounds_for_year(dy, _col)[0] for dy in df_pool["deal_year"].values])
    hi_vals = np.array([_get_bounds_for_year(dy, _col)[1] for dy in df_pool["deal_year"].values])
    df_pool[_col] = np.where(np.isnan(raw_vals), raw_vals, np.clip(raw_vals, lo_vals, hi_vals))

log.info(f"  Expanding-window winsorisation applied to {len(MUL_COLS)} multiple columns across {len(focal_years_sorted)} focal years.")

# ── Sort pool by deal_year ──────────────────────────────────────────────────
sort_idx = df_pool["deal_year"].argsort(kind="stable").values
pool_row_ids_s = df_pool["_row_id"].values[sort_idx]
pool_years_s = df_pool["deal_year"].values[sort_idx]
pool_mul_rev_s = df_pool["pre_rev_mul_ly"].values[sort_idx]
pool_mul_ebitda_s = df_pool["pre_ebitda_mul_ly"].values[sort_idx]
pool_mul_ebit_s = df_pool["pre_ebit_mul_ly"].values[sort_idx]
pool_bvd_ids_s = df_pool["tar_bvd_id_num"].values[sort_idx]

# ── Build normalized vector matrix for pool ──────────────────────────────────
log.info("Retrieving and normalizing pool document vectors ...")
pool_vecs_raw = np.array([model.dv[int(rid)] for rid in pool_row_ids_s], dtype=np.float32)
norms = np.linalg.norm(pool_vecs_raw, axis=1, keepdims=True)
pre_norm_mean = float(np.mean(norms))
pre_norm_sd = float(np.std(norms))
norms[norms == 0] = 1.0
pool_vecs_s = pool_vecs_raw / norms

log.info(f"  Pool vector matrix shape: {pool_vecs_s.shape}")
log.info(f"  Pool vector norms (pre-norm): mean={pre_norm_mean:.4f}, sd={pre_norm_sd:.4f}")

pool_years_list = pool_years_s.tolist()

n_focal_no_vec = sum(1 for rid in df_filtered["_row_id"].values if rid not in valid_tags)
if n_focal_no_vec > 0:
    log.warning(f"  {n_focal_no_vec} focal deals have no trained vector -- NaN benchmarks")
else:
    log.info("  All focal deals have trained vectors: OK")

# ── Rolling Window KNN Computation ───────────────────────────────────────────
log.info(f"Computing text benchmarks (K={K}, min_peers={MIN_PEERS}, rolling window) ...")

results = {}
focal_years = sorted(df_filtered["deal_year"].dropna().unique())
log.info(f"  Focal year range: {int(min(focal_years))}--{int(max(focal_years))} ({len(focal_years)} unique years)")

n_processed = 0
for year_t in focal_years:
    pool_cutoff = bisect_right(pool_years_list, year_t)
    if pool_cutoff < K:
        continue

    active_row_ids = pool_row_ids_s[:pool_cutoff]
    active_vecs = pool_vecs_s[:pool_cutoff]
    active_mul_rev = pool_mul_rev_s[:pool_cutoff]
    active_mul_ebitda = pool_mul_ebitda_s[:pool_cutoff]
    active_mul_ebit = pool_mul_ebit_s[:pool_cutoff]

    focal_in_year = df_filtered[df_filtered["deal_year"] == year_t]
    focal_row_ids_year = focal_in_year["_row_id"].values.astype(int)
    focal_bvd_ids_year = focal_in_year["tar_bvd_id_num"].values
    n_focal_year = len(focal_row_ids_year)

    # ========================================================================
    # PATCH 2: Use ACQUIRER vectors (infer_vector) for similarity
    # ========================================================================
    focal_vecs_list = []
    focal_has_vec = []
    for _, row in focal_in_year.iterrows():
        acq_tokens = row["acq_tokens"]
        if len(acq_tokens) >= MIN_TOKENS:
            vec = model.infer_vector(acq_tokens)
            focal_vecs_list.append(vec)
            focal_has_vec.append(True)
        else:
            focal_vecs_list.append(np.zeros(300, dtype=np.float32))
            focal_has_vec.append(False)
    # ========================================================================

    focal_vecs_raw = np.array(focal_vecs_list, dtype=np.float32)
    focal_norms = np.linalg.norm(focal_vecs_raw, axis=1, keepdims=True)
    focal_norms[focal_norms == 0] = 1.0
    focal_vecs_norm = focal_vecs_raw / focal_norms

    # sim_matrix is now ACQUIRER × TARGET (not target × target)
    sim_matrix = focal_vecs_norm @ active_vecs.T

    for i, focal_rid in enumerate(focal_row_ids_year):
        if not focal_has_vec[i]:
            results[focal_rid] = {
                "text_bench_rev": np.nan, "text_bench_ebitda": np.nan,
                "text_bench_ebit": np.nan, "text_peer_ids": "", "text_peer_sim_mean": np.nan,
                "text_peer_sim_list": "",
                "n_valid_text_peers_rev": np.nan, "n_valid_text_peers_ebitda": np.nan,
                "n_valid_text_peers_ebit": np.nan,
                "n_text_peers_selected": 0,
            }
            continue

        sim_scores = sim_matrix[i].copy()

        self_mask = active_row_ids == focal_rid
        focal_bvd = focal_bvd_ids_year[i]
        if pd.notna(focal_bvd) and focal_bvd != "":
            bvd_mask = pool_bvd_ids_s[:pool_cutoff] == focal_bvd
            self_mask = self_mask | bvd_mask
        sim_scores[self_mask] = -2.0

        n_candidates = pool_cutoff - int(self_mask.sum())
        k_actual = min(K, n_candidates)
        if k_actual == 0:
            results[focal_rid] = {
                "text_bench_rev": np.nan, "text_bench_ebitda": np.nan,
                "text_bench_ebit": np.nan, "text_peer_ids": "", "text_peer_sim_mean": np.nan,
                "text_peer_sim_list": "",
                "n_valid_text_peers_rev": 0, "n_valid_text_peers_ebitda": 0,
                "n_valid_text_peers_ebit": 0,
                "n_text_peers_selected": 0,
            }
            continue

        top_k_raw = np.argpartition(sim_scores, -k_actual)[-k_actual:]
        top_k_idx = top_k_raw[np.argsort(sim_scores[top_k_raw])[::-1]]

        peer_row_ids = active_row_ids[top_k_idx]
        peer_sims = sim_scores[top_k_idx]

        bench = {}
        n_valid_peers = {}
        for m_name, peer_mul in [
            ("rev", active_mul_rev[top_k_idx]),
            ("ebitda", active_mul_ebitda[top_k_idx]),
            ("ebit", active_mul_ebit[top_k_idx]),
        ]:
            valid_mul = peer_mul[peer_mul > 0]
            n_valid_peers[f"n_valid_text_peers_{m_name}"] = int(len(valid_mul))
            bench[f"text_bench_{m_name}"] = float(np.median(valid_mul)) if len(valid_mul) >= MIN_PEERS else np.nan

        results[focal_rid] = {
            **bench,
            **n_valid_peers,
            "text_peer_ids": "|".join(str(rid) for rid in peer_row_ids),
            "text_peer_sim_mean": float(np.mean(peer_sims)),
            "text_peer_sim_list": "|".join(f"{s:.6f}" for s in peer_sims),
            "n_text_peers_selected": len(peer_row_ids),
        }
        n_processed += 1

    if int(year_t) % 5 == 0 or year_t == focal_years[-1]:
        log.info(f"  Year {int(year_t)}: pool={pool_cutoff:,}, focal_in_year={n_focal_year}, cumulative={n_processed:,}")

log.info(f"KNN complete: {n_processed:,} focal deals processed")

# ── Apply results to df_filtered ─────────────────────────────────────────────
for col, default in [
    ("text_bench_rev", np.nan), ("text_bench_ebitda", np.nan),
    ("text_bench_ebit", np.nan), ("text_peer_ids", ""),
    ("text_peer_sim_mean", np.nan), ("text_peer_sim_list", ""),
    ("n_valid_text_peers_rev", np.nan), ("n_valid_text_peers_ebitda", np.nan),
    ("n_valid_text_peers_ebit", np.nan), ("n_text_peers_selected", np.nan),
]:
    col_map = {rid: v[col] for rid, v in results.items()}
    df_filtered[col] = df_filtered["_row_id"].map(col_map)

df_filtered["text_peer_ids"] = df_filtered["text_peer_ids"].fillna("").astype(str)
df_filtered["text_peer_sim_list"] = df_filtered["text_peer_sim_list"].fillna("").astype(str)

# ── Coverage Statistics ───────────────────────────────────────────────────────
log.info("=== Text Benchmark Coverage ===")
n_total = len(df_filtered)
for m in ["rev", "ebitda", "ebit"]:
    n_valid = df_filtered[f"text_bench_{m}"].notna().sum()
    log.info(f"  text_bench_{m}: {n_valid:,}/{n_total:,} ({100.0 * n_valid / n_total:.1f}%)")

sim_col = df_filtered["text_peer_sim_mean"].dropna()
log.info(f"  text_peer_sim_mean: mean={sim_col.mean():.4f}, sd={sim_col.std():.4f}, p10={sim_col.quantile(0.10):.4f}, p50={sim_col.quantile(0.50):.4f}, p90={sim_col.quantile(0.90):.4f}")

# ── Look-ahead Bias Check ─────────────────────────────────────────────────────
log.info("Look-ahead bias check (sample of 200 focal deals) ...")
pool_year_lookup = dict(zip(df_pool["_row_id"].values, df_pool["deal_year"].values))

has_peers = df_filtered["text_peer_ids"].str.len() > 0
sample_check = df_filtered[has_peers].sample(min(200, has_peers.sum()), random_state=SEED)
n_violations = 0
for _, row in sample_check.iterrows():
    focal_yr = row["deal_year"]
    for pid_str in row["text_peer_ids"].split("|"):
        if pid_str:
            peer_yr = pool_year_lookup.get(int(pid_str), np.nan)
            if pd.notna(peer_yr) and peer_yr > focal_yr:
                n_violations += 1

if n_violations > 0:
    log.error(f"  LOOK-AHEAD BIAS: {n_violations} violations detected!")
else:
    log.info("  Look-ahead bias check: PASS (0 violations in 200-deal sample)")

# ── Spot-check ────────────────────────────────────────────────────────────────
log.info("=== Spot-check: 5 focal deals with text peers ===")
has_peers_mask = df_filtered["text_peer_ids"].str.len() > 0
spot = df_filtered[has_peers_mask].sample(5, random_state=SEED)
for _, row in spot.iterrows():
    peer_ids = [int(x) for x in row["text_peer_ids"].split("|") if x]
    peer_names = df_full.loc[df_full["_row_id"].isin(peer_ids), ["_row_id", "tar_name"]].set_index("_row_id")["tar_name"]
    top3 = [peer_names.get(pid, "?") for pid in peer_ids[:3]]
    log.info(f"  Focal: {row.get('tar_name','?')} | yr={row['deal_year']:.0f} | sic3={row.get('tar_sic3','?')}")
    log.info(f"    bench: rev={row['text_bench_rev']:.3f}, ebitda={row['text_bench_ebitda']:.3f}, ebit={row['text_bench_ebit']:.3f} | sim_mean={row['text_peer_sim_mean']:.3f}")
    log.info(f"    Top-3 text peers: {' | '.join(top3)}")

# ── Compute ind_text_diversity ──────────────────────────────────────────────
log.info("=== Computing ind_text_diversity ===")

pool_sic4_float = pd.to_numeric(df_pool["tar_primary_sic_code"], errors="coerce")
pool_valid_sic = pool_sic4_float.between(100, 9999, inclusive="both")
df_pool["_tar_sic3"] = np.where(pool_valid_sic, np.floor(pool_sic4_float / 10), np.nan)

pool_rid_to_vec = {int(pool_row_ids_s[i]): pool_vecs_s[i] for i in range(len(pool_row_ids_s))}

diversity_dict = {}
groups = df_pool.groupby(["_tar_sic3", "deal_year"])
n_cells_computed = 0
n_cells_skipped = 0
for (sic3, yr), grp in groups:
    if np.isnan(sic3) or np.isnan(yr):
        n_cells_skipped += 1
        continue
    rids = [int(r) for r in grp["_row_id"].values if int(r) in pool_rid_to_vec]
    if len(rids) < 2:
        diversity_dict[(sic3, yr)] = np.nan
        continue
    vecs = np.array([pool_rid_to_vec[r] for r in rids], dtype=np.float32)
    sim_mat = vecs @ vecs.T
    triu_i, triu_j = np.triu_indices(len(rids), k=1)
    pairwise = sim_mat[triu_i, triu_j]
    diversity_dict[(sic3, yr)] = float(np.std(pairwise))
    n_cells_computed += 1

log.info(f"  SIC-3 x year cells computed: {n_cells_computed:,}  skipped (NaN key): {n_cells_skipped:,}")

df_filtered["_tar_sic3_num"] = pd.to_numeric(df_filtered["tar_sic3"], errors="coerce")
df_filtered["ind_text_diversity"] = [
    diversity_dict.get((row["_tar_sic3_num"], row["deal_year"]), np.nan)
    for _, row in df_filtered.iterrows()
]
df_filtered.drop(columns=["_tar_sic3_num"], inplace=True)
df_pool.drop(columns=["_tar_sic3"], inplace=True)

n_itd = df_filtered["ind_text_diversity"].notna().sum()
log.info(f"  ind_text_diversity: {n_itd:,}/{len(df_filtered):,} notna ({100*n_itd/len(df_filtered):.1f}%)")
log.info(f"  ind_text_diversity: mean={df_filtered['ind_text_diversity'].mean():.4f}  sd={df_filtered['ind_text_diversity'].std():.4f}  p50={df_filtered['ind_text_diversity'].median():.4f}")

# ============================================================================
# PATCH 3: Compute Acquirer-Target Direct Similarity (acq_tar_similarity)
# ============================================================================
log.info("=" * 60)
log.info("Computing ACQUIRER-TARGET direct similarity (acq_tar_similarity) ...")

tar_vec_lookup = {}
for rid in trained_rids:
    tar_vec_lookup[rid] = model.dv[rid]

acq_tar_sim_results = []
n_acq_valid = 0
n_acq_no_text = 0
n_tar_no_vec = 0

for _, row in df_filtered.iterrows():
    rid = int(row["_row_id"])
    
    if rid not in tar_vec_lookup:
        acq_tar_sim_results.append({"target_row_id": rid, "acq_tar_similarity": np.nan})
        n_tar_no_vec += 1
        continue
    
    tar_vec = tar_vec_lookup[rid]
    tar_vec_norm = tar_vec / (np.linalg.norm(tar_vec) + 1e-8)
    
    acq_tokens = row["acq_tokens"]
    if len(acq_tokens) < MIN_TOKENS:
        acq_tar_sim_results.append({"target_row_id": rid, "acq_tar_similarity": np.nan})
        n_acq_no_text += 1
        continue
    
    acq_vec = model.infer_vector(acq_tokens)
    acq_vec_norm = acq_vec / (np.linalg.norm(acq_vec) + 1e-8)
    
    sim = float(np.dot(acq_vec_norm, tar_vec_norm))
    sim = max(-1.0, min(1.0, sim))
    
    acq_tar_sim_results.append({"target_row_id": rid, "acq_tar_similarity": sim})
    n_acq_valid += 1

acq_tar_sim_df = pd.DataFrame(acq_tar_sim_results)
df_filtered = df_filtered.merge(
    acq_tar_sim_df[["target_row_id", "acq_tar_similarity"]],
    left_on="_row_id",
    right_on="target_row_id",
    how="left"
)
df_filtered = df_filtered.drop(columns=["target_row_id"])

log.info(f"  acq_tar_similarity valid: {n_acq_valid:,} / {len(df_filtered):,} ({100*n_acq_valid/len(df_filtered):.1f}%)")
log.info(f"    - Acquirer no/insufficient text: {n_acq_no_text:,}")
log.info(f"    - Target no trained vector: {n_tar_no_vec:,}")

sim_vals = df_filtered["acq_tar_similarity"].dropna()
if len(sim_vals) > 0:
    log.info(f"  acq_tar_similarity distribution:")
    log.info(f"    mean={sim_vals.mean():.4f}, sd={sim_vals.std():.4f}")
    log.info(f"    p25={sim_vals.quantile(0.25):.4f}, p50={sim_vals.quantile(0.50):.4f}, p75={sim_vals.quantile(0.75):.4f}")
log.info("=" * 60)
# ============================================================================

# ── Save Output ───────────────────────────────────────────────────────────────
out_path = os.path.join(MERGED, f"07_deal_firm_benchmark{_args.suffix}.csv")
df_filtered.to_csv(out_path, index=False, encoding="utf-8-sig")
log.info(f"Saved: {out_path}  ({len(df_filtered):,} rows)")

# Row count assertion
n_filtered_orig = len(pd.read_csv(os.path.join(MERGED, "05_deal_firm_filtered.csv"), usecols=["_row_id"], low_memory=False))
assert len(df_filtered) == n_filtered_orig, f"Row count mismatch! benchmark={len(df_filtered)}, filtered={n_filtered_orig}"
log.info(f"Row count assertion: PASS ({len(df_filtered):,} rows unchanged)")

log.info("=== Script 06 complete ===")
logging.shutdown()