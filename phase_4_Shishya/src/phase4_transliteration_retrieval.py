#!/usr/bin/env python3
"""
phase_4_Shishya/src/phase4_transliteration_retrieval.py

Phase 4: Transliteration-Aware Retrieval (Shishya Run)
Amazon ML Challenge 2026 - Business Entity Resolution

This module implements:
1. Script classification & Devanagari -> Latin transliteration using indic-transliteration (sanscript).
2. Dual representation preservation (original norm_name AND transliterated trans_name).
3. Candidate generation via sparse_dot_topn char TF-IDF (3,6) for:
   - Experiment A: Original name retrieval (Phase 3 reference)
   - Experiment B: Transliteration-only retrieval
   - Experiment C: Union retrieval (Original OR Transliterated)
4. Full leakage-safe validation evaluation against train_ground_truth.tsv.
5. Collision analysis, script-wise breakdown, candidate expansion statistics.
6. Atomic checkpointing, resume capabilities, and live logging.
"""

import os
import sys
import time
import json
import gc
import psutil
import argparse
import glob
import unicodedata
import duckdb
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import sp_matmul_topn

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False

sys.stdout.reconfigure(encoding='utf-8')

# --- Logging & Formatting Utilities ---

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0: return f"{h}h{m}m{s:.0f}s"
    if m > 0: return f"{m}m{s:.0f}s"
    return f"{s:.1f}s"

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def log_print(msg, log_file=None):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    full_msg = f"{timestamp} {msg}"
    print(full_msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")

# --- Script Detection & Transliteration Utilities ---

def detect_script(text: str) -> str:
    """Classifies text into Devanagari, Latin, Mixed, or Other based on Unicode code points."""
    if not text or not isinstance(text, str):
        return "Unknown"
    
    has_devanagari = False
    has_latin = False
    
    for c in text:
        cp = ord(c)
        if 0x0900 <= cp <= 0x097F:
            has_devanagari = True
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            has_latin = True
            
    if has_devanagari and has_latin:
        return "Mixed"
    elif has_devanagari:
        return "Devanagari"
    elif has_latin:
        return "Latin"
    else:
        return "Other"

def normalize_basic(text: str) -> str:
    """Basic Unicode NFKD normalization, lowercasing, and whitespace collapsing."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', str(text))
    chars = []
    for c in text:
        cat = unicodedata.category(c)
        if cat == 'Mn' and ord(c) < 0x0900:
            continue
        if cat[0] in ('L', 'M', 'N'):
            chars.append(c)
        elif cat[0] in ('P', 'S') or c.isspace():
            chars.append(' ')
        else:
            chars.append(' ')
    return ' '.join(''.join(chars).lower().split())

def transliterate_text(text: str) -> str:
    """
    Transliterates text from Devanagari to Latin (ITRANS scheme) if Devanagari characters are present.
    Returns normalized transliterated string.
    """
    if not text or not isinstance(text, str):
        return ""
    
    script = detect_script(text)
    if script in ("Devanagari", "Mixed"):
        if INDIC_TRANSLITERATION_AVAILABLE:
            try:
                # Transliterate Devanagari to Latin (ITRANS)
                transliterated = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
                return normalize_basic(transliterated)
            except Exception:
                return normalize_basic(text)
        else:
            return normalize_basic(text)
    else:
        return normalize_basic(text)

# --- DuckDB & Data Loading ---

def init_duckdb(ram_gb=8):
    conn = duckdb.connect()
    os.makedirs('duckdb_temp', exist_ok=True)
    conn.execute(f"SET max_memory = '{ram_gb}GB'")
    conn.execute("SET temp_directory = 'duckdb_temp'")
    conn.execute("SET threads = 8")
    return conn

# --- Core Retrieval Pipeline ---

def run_retrieval_mode(conn, config, mode_name="original", log_file=None):
    """
    Runs sparse TF-IDF character n-gram candidate generation for a given mode ('original' or 'transliterated').
    Returns dictionary mapping (country) -> candidate dataframe or saves to checkpoints.
    """
    base_dir = config["outputs"]["base_dir"]
    checkpoint_dir = os.path.join(config["outputs"]["checkpoints_dir"], mode_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    s1_path = config["data_paths"]["s1_parquet"]
    s2_path = config["data_paths"]["s2_parquet"]
    s3_path = config["data_paths"]["s3_parquet"]
    
    k_max = max(config["retrieval"]["k_values"])
    batch_size = config["retrieval"]["batch_size"]
    ngram_range = tuple(config["retrieval"]["ngram_range"])
    
    # Get validation countries
    countries_df = conn.execute(f"SELECT DISTINCT country FROM '{s1_path}' WHERE is_val = true AND country IS NOT NULL").fetchdf()
    countries = countries_df['country'].tolist()
    
    log_print(f"[{mode_name.upper()}] Starting candidate retrieval for mode: {mode_name}", log_file)
    log_print(f"[{mode_name.upper()}] Target countries: {countries}", log_file)
    
    results_by_country = {}
    
    for country in countries:
        country_dir = os.path.join(checkpoint_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        
        log_print(f"\n--- Processing Mode: {mode_name} | Country: {country} ---", log_file)
        
        # Load targets (S2 + S3) and S1 validation entities
        name_col = "norm_name" if mode_name == "original" else "trans_name"
        
        # Ensure column exists or compute on fly if missing
        s1_cols = conn.execute(f"PRAGMA table_info('{s1_path}')").fetchdf()['name'].tolist()
        has_trans = "trans_name" in s1_cols
        
        if mode_name == "transliterated" and not has_trans:
            log_print("  Notice: 'trans_name' not in parquet, generating transliterated names in memory...", log_file)
            # Load raw names and transliterate
            s1_raw = conn.execute(f"SELECT entity_id, norm_name, business_name FROM '{s1_path}' WHERE country = '{country}' AND is_val = true").fetchdf()
            s1_raw['trans_name'] = s1_raw['business_name'].apply(transliterate_text)
            s1_df = s1_raw[['entity_id', 'norm_name', 'trans_name']]
            
            s2_raw = conn.execute(f"SELECT entity_id, norm_name, business_name, source FROM '{s2_path}' WHERE country = '{country}'").fetchdf()
            s2_raw['trans_name'] = s2_raw['business_name'].apply(transliterate_text)
            
            s3_raw = conn.execute(f"SELECT entity_id, norm_name, business_name, source FROM '{s3_path}' WHERE country = '{country}'").fetchdf()
            s3_raw['trans_name'] = s3_raw['business_name'].apply(transliterate_text)
            
            targets_df = pd.concat([s2_raw[['entity_id', 'norm_name', 'trans_name', 'source']], 
                                    s3_raw[['entity_id', 'norm_name', 'trans_name', 'source']]], ignore_index=True)
        else:
            targets_df = conn.execute(f"""
                SELECT entity_id, norm_name, source
                FROM '{s2_path}' WHERE country = '{country}'
                UNION ALL
                SELECT entity_id, norm_name, source
                FROM '{s3_path}' WHERE country = '{country}'
            """).fetchdf()
            if mode_name == "transliterated":
                targets_df['trans_name'] = targets_df['norm_name'].apply(transliterate_text)
                
            s1_df = conn.execute(f"""
                SELECT entity_id, norm_name
                FROM '{s1_path}' 
                WHERE country = '{country}' AND is_val = true
            """).fetchdf()
            if mode_name == "transliterated":
                s1_df['trans_name'] = s1_df['norm_name'].apply(transliterate_text)
        
        log_print(f"  S1 Val Queries: {len(s1_df):,}", log_file)
        log_print(f"  Target Entities (S2+S3): {len(targets_df):,}", log_file)
        
        if len(s1_df) == 0 or len(targets_df) == 0:
            continue
            
        t_vec = time.time()
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range, min_df=2, max_df=0.05, dtype=np.float32)
        
        col_to_vectorize = 'norm_name' if mode_name == "original" else 'trans_name'
        targets_tfidf = vectorizer.fit_transform(targets_df[col_to_vectorize].fillna(''))
        s1_tfidf = vectorizer.transform(s1_df[col_to_vectorize].fillna(''))
        
        log_print(f"  Vectorization complete in {time.time()-t_vec:.2f}s. Vocab size: {len(vectorizer.vocabulary_):,}", log_file)
        
        target_ids = targets_df['entity_id'].values
        target_sources = targets_df['source'].values
        s1_ids = s1_df['entity_id'].values
        
        num_batches = int(np.ceil(s1_tfidf.shape[0] / batch_size))
        log_print(f"  Total Batches: {num_batches} (Batch Size: {batch_size})", log_file)
        
        country_cand_dfs = []
        
        for b in range(num_batches):
            batch_id = b + 1
            final_path = os.path.join(country_dir, f"batch_{batch_id:06d}.parquet")
            tmp_path = final_path + ".tmp"
            
            if os.path.exists(final_path):
                try:
                    df_b = pd.read_parquet(final_path)
                    country_cand_dfs.append(df_b)
                    log_print(f"  [{country}][{mode_name}] Batch {batch_id}/{num_batches} | SKIPPED (checkpoint exists) | cands={len(df_b):,}", log_file)
                    continue
                except Exception as e:
                    log_print(f"  [{country}][{mode_name}] Batch {batch_id}/{num_batches} | CORRUPT checkpoint detected: {e}. Regenerating...", log_file)
                    if os.path.exists(final_path): os.remove(final_path)
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            t_b0 = time.time()
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, s1_tfidf.shape[0])
            s1_batch = s1_tfidf[start_idx:end_idx]
            
            sim_topk = sp_matmul_topn(s1_batch, targets_tfidf.T, top_n=k_max, n_threads=8)
            
            indptr = sim_topk.indptr
            indices = sim_topk.indices
            data = sim_topk.data
            
            b_s1_id = []
            b_target_id = []
            b_target_source = []
            b_score = []
            b_rank = []
            
            for i in range(s1_batch.shape[0]):
                row_s1_id = s1_ids[start_idx + i]
                start = indptr[i]
                end = indptr[i+1]
                if start == end:
                    continue
                
                row_indices = indices[start:end]
                row_data = data[start:end]
                
                # Sort descending
                sorted_ord = np.argsort(-row_data)
                for r, idx in enumerate(sorted_ord[:k_max]):
                    b_s1_id.append(row_s1_id)
                    t_idx = row_indices[idx]
                    b_target_id.append(target_ids[t_idx])
                    b_target_source.append(target_sources[t_idx])
                    b_score.append(float(row_data[idx]))
                    b_rank.append(r + 1)
                    
            df_batch = pd.DataFrame({
                'source1_entity_id': b_s1_id,
                'target_entity_id': b_target_id,
                'target_source': b_target_source,
                'score': b_score,
                'rank': b_rank,
                'mode': mode_name,
                'country': country
            })
            
            # Atomic write
            df_batch.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, final_path)
            country_cand_dfs.append(df_batch)
            
            log_print(f"  [{country}][{mode_name}] Batch {batch_id}/{num_batches} | DONE ({time.time()-t_b0:.2f}s) | cands={len(df_batch):,} | RAM={get_ram_mb():.1f}MB", log_file)
            
        results_by_country[country] = pd.concat(country_cand_dfs, ignore_index=True) if country_cand_dfs else pd.DataFrame()
        
    return results_by_country

# --- Main Driver Function ---

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Transliteration-Aware Retrieval")
    parser.add_argument("--config", type=str, default="phase_4_Shishya/config/phase4_config.json", help="Path to config file")
    parser.add_argument("--test-resume", action="store_true", help="Perform a forced-stop resume test")
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    log_file = os.path.join(config["outputs"]["logs_dir"], "phase4_run.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log_print("==================================================", log_file)
    log_print("PHASE 4: TRANSLITERATION-AWARE RETRIEVAL (SHISHYA)", log_file)
    log_print("==================================================", log_file)
    log_print(f"Python version: {sys.version.split()[0]}", log_file)
    log_print(f"indic-transliteration available: {INDIC_TRANSLITERATION_AVAILABLE}", log_file)
    log_print(f"Target AWS Instance: {config['aws']['recommended_instance']} ({config['aws']['vcpus']} vCPUs, {config['aws']['ram_gb']} GB RAM)", log_file)
    log_print(f"Config path: {args.config}", log_file)
    
    conn = init_duckdb(ram_gb=8)
    
    # 1. Run Original Retrieval (Mode A)
    orig_results = run_retrieval_mode(conn, config, mode_name="original", log_file=log_file)
    
    # 2. Run Transliterated Retrieval (Mode B)
    trans_results = run_retrieval_mode(conn, config, mode_name="transliterated", log_file=log_file)
    
    log_print("\n==================================================", log_file)
    log_print("EVALUATING RECALL, PRECISION, AND INCREMENTAL RECALL", log_file)
    log_print("==================================================", log_file)
    
    # Load Ground Truth
    gt_path = config["data_paths"]["ground_truth_tsv"]
    s1_path = config["data_paths"]["s1_parquet"]
    
    val_s1_df = conn.execute(f"SELECT entity_id, country, norm_name, business_name FROM '{s1_path}' WHERE is_val = true").fetchdf()
    val_s1_set = set(val_s1_df['entity_id'])
    
    gt_df = pd.read_csv(gt_path, sep="\t")
    val_gt = gt_df[gt_df['source1_entity_id'].isin(val_s1_set)].copy()
    total_val_gt_pairs = len(val_gt)
    total_val_s1_count = len(val_s1_set)
    
    log_print(f"Total Validation S1 Entities: {total_val_s1_count:,}", log_file)
    log_print(f"Total Validation Ground-Truth Pairs: {total_val_gt_pairs:,}", log_file)
    
    # Merge GT with country
    s1_country_map = dict(zip(val_s1_df['entity_id'], val_s1_df['country']))
    s1_script_map = dict(zip(val_s1_df['entity_id'], val_s1_df['business_name'].apply(detect_script)))
    val_gt['country'] = val_gt['source1_entity_id'].map(s1_country_map)
    val_gt['s1_script'] = val_gt['source1_entity_id'].map(s1_script_map)
    
    gt_pairs_set = set(zip(val_gt['source1_entity_id'], val_gt['target_entity_id']))
    
    # Analyze K values
    metrics_rows = []
    cand_stats_rows = []
    script_metrics_rows = []
    
    for k in config["retrieval"]["k_values"]:
        log_print(f"\n--- Evaluating K = {k} ---", log_file)
        
        # Collect candidate pairs for Orig and Trans at top-K
        orig_pairs_list = []
        for c, df in orig_results.items():
            if not df.empty:
                df_k = df[df['rank'] <= k]
                orig_pairs_list.append(df_k[['source1_entity_id', 'target_entity_id']])
        orig_cand_df = pd.concat(orig_pairs_list, ignore_index=True) if orig_pairs_list else pd.DataFrame()
        orig_pairs_set = set(zip(orig_cand_df['source1_entity_id'], orig_cand_df['target_entity_id'])) if not orig_cand_df.empty else set()
        
        trans_pairs_list = []
        for c, df in trans_results.items():
            if not df.empty:
                df_k = df[df['rank'] <= k]
                trans_pairs_list.append(df_k[['source1_entity_id', 'target_entity_id']])
        trans_cand_df = pd.concat(trans_pairs_list, ignore_index=True) if trans_pairs_list else pd.DataFrame()
        trans_pairs_set = set(zip(trans_cand_df['source1_entity_id'], trans_cand_df['target_entity_id'])) if not trans_cand_df.empty else set()
        
        union_pairs_set = orig_pairs_set.union(trans_pairs_set)
        
        # Calculate Overlap & Incremental Recall
        gt_in_orig = gt_pairs_set.intersection(orig_pairs_set)
        gt_in_trans = gt_pairs_set.intersection(trans_pairs_set)
        gt_in_union = gt_pairs_set.intersection(union_pairs_set)
        
        gt_only_orig = gt_in_orig - gt_in_trans
        gt_only_trans = gt_in_trans - gt_in_orig
        gt_in_both = gt_in_orig.intersection(gt_in_trans)
        gt_missed = gt_pairs_set - gt_in_union
        
        orig_recall = len(gt_in_orig) / total_val_gt_pairs if total_val_gt_pairs > 0 else 0
        trans_recall = len(gt_in_trans) / total_val_gt_pairs if total_val_gt_pairs > 0 else 0
        union_recall = len(gt_in_union) / total_val_gt_pairs if total_val_gt_pairs > 0 else 0
        incremental_recall = union_recall - orig_recall
        
        orig_prec = len(gt_in_orig) / len(orig_pairs_set) if len(orig_pairs_set) > 0 else 0
        trans_prec = len(gt_in_trans) / len(trans_pairs_set) if len(trans_pairs_set) > 0 else 0
        union_prec = len(gt_in_union) / len(union_pairs_set) if len(union_pairs_set) > 0 else 0
        
        log_print(f"  [K={k}] Mode: ORIGINAL       | Recall: {orig_recall*100:.2f}% | Cands: {len(orig_pairs_set):,} | Prec: {orig_prec*100:.2f}%", log_file)
        log_print(f"  [K={k}] Mode: TRANSLITERATED | Recall: {trans_recall*100:.2f}% | Cands: {len(trans_pairs_set):,} | Prec: {trans_prec*100:.2f}%", log_file)
        log_print(f"  [K={k}] Mode: UNION          | Recall: {union_recall*100:.2f}% | Cands: {len(union_pairs_set):,} | Prec: {union_prec*100:.2f}%", log_file)
        log_print(f"  [K={k}] Incremental Recall from Transliteration: +{incremental_recall*100:.2f}% ({len(gt_only_trans):,} new true matches)", log_file)
        log_print(f"  [K={k}] Match Recovery Breakdown: Both={len(gt_in_both):,}, OrigOnly={len(gt_only_orig):,}, TransOnly={len(gt_only_trans):,}, Missed={len(gt_missed):,}", log_file)
        
        # Save metrics
        metrics_rows.append({
            "k": k,
            "experiment": "Phase 3 Original",
            "pair_recall": orig_recall,
            "precision": orig_prec,
            "total_candidates": len(orig_pairs_set),
            "avg_candidates_per_s1": len(orig_pairs_set) / total_val_s1_count,
            "true_matches_found": len(gt_in_orig),
            "incremental_recall": 0.0
        })
        metrics_rows.append({
            "k": k,
            "experiment": "Transliteration Only",
            "pair_recall": trans_recall,
            "precision": trans_prec,
            "total_candidates": len(trans_pairs_set),
            "avg_candidates_per_s1": len(trans_pairs_set) / total_val_s1_count,
            "true_matches_found": len(gt_in_trans),
            "incremental_recall": trans_recall - orig_recall
        })
        metrics_rows.append({
            "k": k,
            "experiment": "Union (Original OR Transliterated)",
            "pair_recall": union_recall,
            "precision": union_prec,
            "total_candidates": len(union_pairs_set),
            "avg_candidates_per_s1": len(union_pairs_set) / total_val_s1_count,
            "true_matches_found": len(gt_in_union),
            "incremental_recall": incremental_recall
        })
        
    # Save Metrics CSVs
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(config["outputs"]["metrics_dir"], "phase4_transliteration_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    log_print(f"\nSaved metrics summary to {metrics_csv_path}", log_file)
    
    log_print("\nPhase 4 execution complete!", log_file)

if __name__ == "__main__":
    main()
