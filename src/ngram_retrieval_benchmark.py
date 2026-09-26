#!/usr/bin/env python3
"""
src/ngram_retrieval_benchmark.py
Phase 3 Redesign Benchmark: Compares old vs. new ngram retrieval.
"""
import os
import sys
import time
import gc
import duckdb
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:
    pass

def init_duckdb():
    conn = duckdb.connect()
    os.makedirs('duckdb_temp', exist_ok=True)
    conn.execute("SET max_memory = '4GB'")
    conn.execute("SET temp_directory = 'duckdb_temp'")
    conn.execute("SET threads = 8")
    return conn

# Old implementation bottleneck snippet
def get_top_k_old(s1_batch_tfidf: csr_matrix, targets_tfidf: csr_matrix, k=50):
    sim = s1_batch_tfidf.dot(targets_tfidf.T)
    top_k_indices = []
    top_k_scores = []
    for i in range(sim.shape[0]):
        row = sim.getrow(i)
        data = row.data
        indices = row.indices
        if len(data) == 0:
            top_k_indices.append(np.array([], dtype=np.int32))
            top_k_scores.append(np.array([], dtype=np.float32))
            continue
        if len(data) <= k:
            sort_idx = np.argsort(-data)
            top_k_indices.append(indices[sort_idx])
            top_k_scores.append(data[sort_idx])
        else:
            part_idx = np.argpartition(data, -k)[-k:]
            sort_idx = np.argsort(-data[part_idx])
            top_k_indices.append(indices[part_idx][sort_idx])
            top_k_scores.append(data[part_idx][sort_idx])
    return top_k_indices, top_k_scores

def run_benchmark():
    conn = init_duckdb()
    
    # 1. Fetch small subset
    country = 'US'
    target_limit = 500000
    s1_limit = 2000
    
    print(f"Loading benchmark data for {country} (Targets: {target_limit}, S1: {s1_limit})...")
    targets_df = conn.execute(f"""
        (SELECT entity_id, norm_name, source
        FROM 'data/train_s2.parquet' WHERE country = '{country}' LIMIT {target_limit // 2})
        UNION ALL
        (SELECT entity_id, norm_name, source
        FROM 'data/train_s3.parquet' WHERE country = '{country}' LIMIT {target_limit // 2})
    """).fetchdf()
    
    s1_df = conn.execute(f"""
        SELECT entity_id, norm_name
        FROM 'data/train_s1.parquet' 
        WHERE country = '{country}' AND is_val = true LIMIT {s1_limit}
    """).fetchdf()
    
    print("Vectorizing...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2, max_df=0.05, dtype=np.float32)
    targets_names = targets_df['norm_name'].fillna('')
    targets_tfidf = vectorizer.fit_transform(targets_names)
    s1_names = s1_df['norm_name'].fillna('')
    s1_tfidf = vectorizer.transform(s1_names)
    print(f"Vectorization done in {time.time()-t0:.2f}s. Vocab size: {len(vectorizer.vocabulary_)}")
    
    target_ids = targets_df['entity_id'].values
    target_sources = targets_df['source'].values
    s1_ids = s1_df['entity_id'].values
    
    process = psutil.Process(os.getpid())
    
    print("\n--- Running OLD algorithm ---")
    mem_before_old = process.memory_info().rss
    t0_old = time.time()
    batch_size = 1000
    num_batches = int(np.ceil(s1_tfidf.shape[0] / batch_size))
    
    old_results = []
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, s1_tfidf.shape[0])
        s1_batch = s1_tfidf[start_idx:end_idx]
        
        # Old
        idxs, scores = get_top_k_old(s1_batch, targets_tfidf, k=20)
        
        for i in range(len(idxs)):
            sid = s1_ids[start_idx + i]
            for rank, (idx, score) in enumerate(zip(idxs[i], scores[i]), 1):
                old_results.append((sid, target_ids[idx], rank, float(score)))
                
    old_time = time.time() - t0_old
    mem_after_old = process.memory_info().rss
    print(f"Old approach took {old_time:.2f}s. Generated {len(old_results)} candidates.")
    
    print("\n--- Running NEW algorithm ---")
    mem_before_new = process.memory_info().rss
    t0_new = time.time()
    
    new_results = []
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, s1_tfidf.shape[0])
        s1_batch = s1_tfidf[start_idx:end_idx]
        
        # New
        sim_topk = sp_matmul_topn(s1_batch, targets_tfidf.T, top_n=20, n_threads=8)
        
        # Extract
        indptr = sim_topk.indptr
        indices = sim_topk.indices
        data = sim_topk.data
        
        for i in range(s1_batch.shape[0]):
            start = indptr[i]
            end = indptr[i+1]
            if start == end: continue
            
            row_data = data[start:end]
            row_indices = indices[start:end]
            
            sid = s1_ids[start_idx + i]
            for rank, (idx, score) in enumerate(zip(row_indices, row_data), 1):
                new_results.append((sid, target_ids[idx], rank, float(score)))

    new_time = time.time() - t0_new
    mem_after_new = process.memory_info().rss
    print(f"New approach took {new_time:.2f}s. Generated {len(new_results)} candidates.")
    
    # Validation
    df_old = pd.DataFrame(old_results, columns=['s1_id', 'target_id', 'rank', 'score'])
    df_new = pd.DataFrame(new_results, columns=['s1_id', 'target_id', 'rank', 'score'])
    
    # Note: rank might differ if scores are exactly identical, so sort by s1, target
    df_old_sorted = df_old.sort_values(['s1_id', 'target_id']).reset_index(drop=True)
    df_new_sorted = df_new.sort_values(['s1_id', 'target_id']).reset_index(drop=True)
    
    overlap = len(pd.merge(df_old[['s1_id', 'target_id']], df_new[['s1_id', 'target_id']], on=['s1_id', 'target_id']))
    print(f"\nCandidate Overlap: {overlap} / {len(df_old)}")
    
    print("\n--- Memory (rough estimates) ---")
    print(f"Before Old: {mem_before_old / 1024**2:.1f} MB, After Old: {mem_after_old / 1024**2:.1f} MB")
    print(f"Before New: {mem_before_new / 1024**2:.1f} MB, After New: {mem_after_new / 1024**2:.1f} MB")
    
if __name__ == '__main__':
    run_benchmark()
