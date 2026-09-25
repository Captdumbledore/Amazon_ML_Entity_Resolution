#!/usr/bin/env python3
"""
src/ngram_retrieval_experiment.py

Phase 3: Character N-Gram Name Retrieval Experiment
Tests whether character-level n-gram retrieval on business names recovers S1->S2/S3 matches 
missed by exact string matching.

Uses DuckDB for data loading/evaluation and Scikit-learn sparse TF-IDF for retrieval.
Evaluates at K=5, 10, 20, 50.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def init_duckdb():
    conn = duckdb.connect()
    os.makedirs('duckdb_temp', exist_ok=True)
    conn.execute("SET max_memory = '4GB'")
    conn.execute("SET temp_directory = 'duckdb_temp'")
    conn.execute("SET threads = 8")
    return conn

def get_top_k(s1_batch_tfidf: csr_matrix, targets_tfidf: csr_matrix, k=50):
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

def generate_candidates(conn, ngram_range):
    ngram_str = f"{ngram_range[0]}_{ngram_range[1]}"
    out_parquet = os.path.join(DATA_DIR, f"phase3_candidates_{ngram_str}.parquet")
    
    print(f"\n==================================================")
    print(f"GENERATING CANDIDATES: char n-grams {ngram_range}")
    print(f"==================================================")
    t0 = time.time()
    
    countries = conn.execute("SELECT DISTINCT country FROM 'data/train_s1.parquet' WHERE is_val = true;").fetchall()
    countries = [c[0] for c in countries if c[0]]
    
    schema = pa.schema([
        ('s1_id', pa.string()),
        ('target_id', pa.string()),
        ('target_source', pa.string()),
        ('rank', pa.int32()),
        ('score', pa.float32())
    ])
    writer = pq.ParquetWriter(out_parquet, schema, compression='snappy')
    
    total_candidates = 0
    total_s1 = 0
    
    for country in countries:
        print(f"\n--- Processing Country: {country} ---")
        t_country = time.time()
        
        # Load targets
        targets_df = conn.execute(f"""
            SELECT entity_id, norm_name, source
            FROM 'data/train_s2.parquet' WHERE country = '{country}'
            UNION ALL
            SELECT entity_id, norm_name, source
            FROM 'data/train_s3.parquet' WHERE country = '{country}'
        """).fetchdf()
        
        if len(targets_df) == 0:
            print("  No targets found. Skipping.")
            continue
            
        print(f"  Loaded {len(targets_df):,} targets.")
        
        # Load S1 validation
        s1_df = conn.execute(f"""
            SELECT entity_id, norm_name
            FROM 'data/train_s1.parquet' 
            WHERE country = '{country}' AND is_val = true
        """).fetchdf()
        
        if len(s1_df) == 0:
            print("  No S1 validation found. Skipping.")
            continue
            
        print(f"  Loaded {len(s1_df):,} S1 validation entities.")
        
        print("  Vectorizing targets...", flush=True)
        t_vec = time.time()
        vectorizer = TfidfVectorizer(
            analyzer='char', 
            ngram_range=ngram_range, 
            min_df=2, 
            max_df=0.05,
            dtype=np.float32
        )
        # Fill NA just in case
        targets_names = targets_df['norm_name'].fillna('')
        targets_tfidf = vectorizer.fit_transform(targets_names)
        print(f"  Vectorized targets in {time.time()-t_vec:.1f}s. Vocab size: {len(vectorizer.vocabulary_):,}")
        
        s1_names = s1_df['norm_name'].fillna('')
        target_ids = targets_df['entity_id'].values
        target_sources = targets_df['source'].values
        s1_ids = s1_df['entity_id'].values
        
        del targets_names, targets_df
        gc.collect()
        
        batch_size = 1000
        num_batches = int(np.ceil(len(s1_names) / batch_size))
        
        country_candidates = 0
        t_search = time.time()
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(s1_names))
            s1_batch_names = s1_names[start_idx:end_idx]
            s1_batch_ids = s1_ids[start_idx:end_idx]
            
            s1_batch_tfidf = vectorizer.transform(s1_batch_names)
            top_k_indices, top_k_scores = get_top_k(s1_batch_tfidf, targets_tfidf, k=50)
            
            b_s1_id = []
            b_target_id = []
            b_target_source = []
            b_rank = []
            b_score = []
            
            for i in range(len(s1_batch_ids)):
                idxs = top_k_indices[i]
                scores = top_k_scores[i]
                sid = s1_batch_ids[i]
                for rank, (idx, score) in enumerate(zip(idxs, scores), 1):
                    b_s1_id.append(sid)
                    b_target_id.append(target_ids[idx])
                    b_target_source.append(target_sources[idx])
                    b_rank.append(rank)
                    b_score.append(score)
            
            if b_s1_id:
                batch_df = pd.DataFrame({
                    's1_id': b_s1_id,
                    'target_id': b_target_id,
                    'target_source': b_target_source,
                    'rank': np.array(b_rank, dtype=np.int32),
                    'score': np.array(b_score, dtype=np.float32)
                })
                table = pa.Table.from_pandas(batch_df, schema=schema)
                writer.write_table(table)
                country_candidates += len(b_s1_id)
                
            if (b + 1) % 10 == 0 or (b + 1) == num_batches:
                print(f"    Batch {b+1}/{num_batches} | Candidates: {country_candidates:,} | {time.time()-t_search:.1f}s", flush=True)
                
        total_candidates += country_candidates
        total_s1 += len(s1_df)
        print(f"  Finished {country} in {time.time()-t_country:.1f}s. Generated {country_candidates:,} candidates.")
        
        del targets_tfidf, vectorizer, s1_names
        gc.collect()
        
    writer.close()
    print(f"Finished generating {total_candidates:,} candidates for S1={total_s1:,} in {time.time()-t0:.1f}s.")
    return out_parquet

def evaluate_candidates(conn, parquet_path, ngram_range):
    ngram_str = f"{ngram_range[0]}_{ngram_range[1]}"
    print(f"\n==================================================")
    print(f"EVALUATING CANDIDATES: char n-grams {ngram_range}")
    print(f"==================================================")
    
    conn.execute(f"CREATE OR REPLACE VIEW cands AS SELECT * FROM '{parquet_path}';")
    
    # GT pair count for validation split S1 entities (only those with > 0 targets)
    conn.execute("""
        CREATE OR REPLACE VIEW gt_val AS 
        SELECT source1_entity_id, target_entity_id, target_source 
        FROM 'data/train_gt_pairs.parquet'
        WHERE source1_entity_id IN (SELECT entity_id FROM 'data/train_s1.parquet' WHERE is_val = true);
    """)
    
    conn.execute("""
        CREATE OR REPLACE VIEW gt_val_counts AS
        SELECT source1_entity_id as s1_id, count(*) as total_gt
        FROM gt_val
        GROUP BY source1_entity_id;
    """)
    
    total_gt = conn.execute("SELECT sum(total_gt) FROM gt_val_counts;").fetchone()[0] or 0
    total_s1_with_gt = conn.execute("SELECT count(*) FROM gt_val_counts;").fetchone()[0] or 0
    total_s1_val = conn.execute("SELECT count(*) FROM 'data/train_s1.parquet' WHERE is_val = true;").fetchone()[0] or 0
    
    print(f"Total Val S1 (with >=1 target): {total_s1_with_gt:,} / {total_s1_val:,}")
    print(f"Total Val GT pairs: {total_gt:,}")
    
    results = []
    
    for k in [5, 10, 20, 50]:
        t0 = time.time()
        
        # 1. Pair Recall at K
        tp_query = f"""
            SELECT count(*) 
            FROM cands c 
            JOIN gt_val g 
              ON c.s1_id = g.source1_entity_id 
             AND c.target_id = g.target_entity_id
            WHERE c.rank <= {k}
        """
        tp_k = conn.execute(tp_query).fetchone()[0] or 0
        pair_recall = tp_k / total_gt if total_gt > 0 else 0
        
        # 2. Complete Recovery Rate at K
        # Fraction of S1 entities where ALL ground truth targets are retrieved within rank K
        recovery_query = f"""
            WITH retrieved_counts AS (
                SELECT c.s1_id, count(*) as tp_count
                FROM cands c
                JOIN gt_val g 
                  ON c.s1_id = g.source1_entity_id 
                 AND c.target_id = g.target_entity_id
                WHERE c.rank <= {k}
                GROUP BY c.s1_id
            )
            SELECT sum(case when r.tp_count = g.total_gt then 1 else 0 end)
            FROM gt_val_counts g
            LEFT JOIN retrieved_counts r ON g.s1_id = r.s1_id
        """
        complete_k = conn.execute(recovery_query).fetchone()[0] or 0
        recovery_rate = complete_k / total_s1_with_gt if total_s1_with_gt > 0 else 0
        
        # 3. Precision at K (Pair level)
        total_pred_k = conn.execute(f"SELECT count(*) FROM cands WHERE rank <= {k}").fetchone()[0] or 0
        precision = tp_k / total_pred_k if total_pred_k > 0 else 0
        
        elapsed = time.time() - t0
        print(f"  K={k:<2} | Pair Recall: {pair_recall*100:5.2f}% | S1 Complete Recovery: {recovery_rate*100:5.2f}% | Pair Precision: {precision*100:5.2f}% | TP: {tp_k:,} | Candidates: {total_pred_k:,} ({elapsed:.1f}s)")
        
        results.append({
            "ngram_range": ngram_str,
            "k": k,
            "pair_recall": pair_recall,
            "s1_complete_recovery": recovery_rate,
            "pair_precision": precision,
            "total_tp": tp_k,
            "total_candidates": total_pred_k
        })
        
    return results

def main():
    conn = init_duckdb()
    
    all_results = []
    
    ngram_ranges = [(3, 5), (3, 6)]
    for ngr in ngram_ranges:
        parquet_path = generate_candidates(conn, ngr)
        metrics = evaluate_candidates(conn, parquet_path, ngr)
        all_results.extend(metrics)
        
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(REPORTS_DIR, "phase3_ngram_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")
    
    # Simple markdown report
    md_path = os.path.join(REPORTS_DIR, "phase3_ngram_retrieval.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3: Character N-Gram Name Retrieval Experiment\n\n")
        f.write("## Overview\n")
        f.write("Evaluates character n-gram TF-IDF retrieval (cosine similarity) on normalized business names to recover S1->S2/S3 matches missed by exact string matching.\n\n")
        f.write("## Results\n\n")
        f.write("| N-Gram | K | Pair Recall (%) | S1 Complete Recovery (%) | Pair Precision (%) | Total Candidates |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in all_results:
            f.write(f"| {r['ngram_range']} | {r['k']} | {r['pair_recall']*100:.2f} | {r['s1_complete_recovery']*100:.2f} | {r['pair_precision']*100:.2f} | {r['total_candidates']:,} |\n")
    print(f"Saved report to {md_path}")
    
    conn.close()

if __name__ == '__main__':
    main()
