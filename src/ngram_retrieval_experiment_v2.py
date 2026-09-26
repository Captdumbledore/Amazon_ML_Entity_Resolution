#!/usr/bin/env python3
"""
src/ngram_retrieval_experiment_v2.py
Redesigned Phase 3 script with sparse_dot_topn and chunk-level checkpointing.
"""

import os
import sys
import time
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import pandas as pd
import numpy as np
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import sp_matmul_topn

CHECKPOINT_DIR = "data/phase3_checkpoints"

def init_duckdb():
    conn = duckdb.connect()
    os.makedirs('duckdb_temp', exist_ok=True)
    conn.execute("SET max_memory = '4GB'")
    conn.execute("SET temp_directory = 'duckdb_temp'")
    conn.execute("SET threads = 8")
    return conn

def process_country_config(conn, country, ngr_range, k=50, batch_size=2000, test_crash_at_batch=None, test_mode=False):
    ngr_str = f"{ngr_range[0]}_{ngr_range[1]}"
    country_dir = os.path.join(CHECKPOINT_DIR, ngr_str, country)
    os.makedirs(country_dir, exist_ok=True)
    
    print(f"\n--- {country} | Config: {ngr_range} ---", flush=True)
    
    # Load targets
    targets_df = conn.execute(f"""
        SELECT entity_id, norm_name, source
        FROM 'data/train_s2.parquet' WHERE country = '{country}'
        UNION ALL
        SELECT entity_id, norm_name, source
        FROM 'data/train_s3.parquet' WHERE country = '{country}'
    """).fetchdf()
    
    if len(targets_df) == 0:
        print("No targets found.")
        return
        
    s1_df = conn.execute(f"""
        SELECT entity_id, norm_name
        FROM 'data/train_s1.parquet' 
        WHERE country = '{country}' AND is_val = true
    """).fetchdf()
    
    if len(s1_df) == 0:
        print("No S1 found.", flush=True)
        return
        
    if test_mode:
        # In test mode, shrink the data to make vectorization fast
        targets_df = targets_df.head(20000)
        s1_df = s1_df.head(10000)
        batch_size = 2000
        
    print(f"Targets: {len(targets_df):,}, S1: {len(s1_df):,}", flush=True)
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngr_range, min_df=2, max_df=0.05, dtype=np.float32)
    t0 = time.time()
    targets_tfidf = vectorizer.fit_transform(targets_df['norm_name'].fillna(''))
    s1_tfidf = vectorizer.transform(s1_df['norm_name'].fillna(''))
    print(f"Vectorized in {time.time()-t0:.1f}s. Vocab: {len(vectorizer.vocabulary_):,}")
    
    target_ids = targets_df['entity_id'].values
    target_sources = targets_df['source'].values
    s1_ids = s1_df['entity_id'].values
    
    num_batches = int(np.ceil(s1_tfidf.shape[0] / batch_size))
    
    schema = pa.schema([
        ('source1_entity_id', pa.string()),
        ('target_entity_id', pa.string()),
        ('target_source', pa.string()),
        ('country', pa.string()),
        ('ngram_config', pa.string()),
        ('score', pa.float32()),
        ('rank', pa.int32())
    ])
    
    for b in range(num_batches):
        batch_id = b + 1
        final_path = os.path.join(country_dir, f"batch_{batch_id:06d}.parquet")
        tmp_path = final_path + ".tmp"
        
        if os.path.exists(final_path):
            print(f"Batch {batch_id}/{num_batches} already completed. Skipping.")
            continue
            
        print(f"Processing batch {batch_id}/{num_batches}...")
        
        if test_crash_at_batch == batch_id:
            print("TEST MODE: Simulating a crash mid-batch!")
            sys.exit(99)
            
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, s1_tfidf.shape[0])
        s1_batch = s1_tfidf[start_idx:end_idx]
        
        sim_topk = sp_matmul_topn(s1_batch, targets_tfidf.T, top_n=k, n_threads=8)
        
        indptr = sim_topk.indptr
        indices = sim_topk.indices
        data = sim_topk.data
        
        b_s1_id = []
        b_target_id = []
        b_target_source = []
        b_score = []
        b_rank = []
        
        for i in range(s1_batch.shape[0]):
            start = indptr[i]
            end = indptr[i+1]
            if start == end: continue
            
            row_data = data[start:end]
            row_indices = indices[start:end]
            sid = s1_ids[start_idx + i]
            
            sort_order = np.argsort(-row_data)
            row_data = row_data[sort_order]
            row_indices = row_indices[sort_order]
            
            for rank, (idx, score) in enumerate(zip(row_indices, row_data), 1):
                b_s1_id.append(sid)
                b_target_id.append(target_ids[idx])
                b_target_source.append(target_sources[idx])
                b_score.append(float(score))
                b_rank.append(rank)
                
        if len(b_s1_id) > 0:
            df_out = pd.DataFrame({
                'source1_entity_id': b_s1_id,
                'target_entity_id': b_target_id,
                'target_source': b_target_source,
                'country': country,
                'ngram_config': ngr_str,
                'score': np.array(b_score, dtype=np.float32),
                'rank': np.array(b_rank, dtype=np.int32)
            })
            table = pa.Table.from_pandas(df_out, schema=schema)
            pq.write_table(table, tmp_path, compression='snappy')
        else:
            pq.write_table(schema.empty_table(), tmp_path, compression='snappy')
            
        os.replace(tmp_path, final_path)
        
if __name__ == '__main__':
    print("This is the V2 implementation. Run from main orchestrator.")
