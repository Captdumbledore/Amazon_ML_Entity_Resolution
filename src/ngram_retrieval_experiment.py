#!/usr/bin/env python3
import os
import sys
import time
import gc
import psutil
import glob
import duckdb
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import sp_matmul_topn

CHECKPOINT_DIR = "data/phase3_checkpoints"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def init_duckdb():
    conn = duckdb.connect()
    os.makedirs('duckdb_temp', exist_ok=True)
    conn.execute("SET max_memory = '8GB'")
    conn.execute("SET temp_directory = 'duckdb_temp'")
    conn.execute("SET threads = 8")
    return conn

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0: return f"{h}h{m}m{s:.0f}s"
    if m > 0: return f"{m}m{s:.0f}s"
    return f"{s:.1f}s"

def generate_candidates(conn, ngr_range, k=50, batch_size=2000):
    t_start = time.time()
    ngr_str = f"{ngr_range[0]}_{ngr_range[1]}"
    
    countries_df = conn.execute("SELECT DISTINCT country FROM 'data/train_s1.parquet' WHERE is_val = true AND country IS NOT NULL").fetchdf()
    countries = countries_df['country'].tolist()
    
    total_candidates_all = 0
    total_s1_all = 0
    
    print(f"\n==================================================", flush=True)
    print(f"STARTING CONFIGURATION: {ngr_range} | K={k}", flush=True)
    print(f"Countries to process: {countries}", flush=True)
    print(f"==================================================", flush=True)
    
    schema = pa.schema([
        ('source1_entity_id', pa.string()),
        ('target_entity_id', pa.string()),
        ('target_source', pa.string()),
        ('country', pa.string()),
        ('ngram_config', pa.string()),
        ('score', pa.float32()),
        ('rank', pa.int32())
    ])
    
    for country in countries:
        country_dir = os.path.join(CHECKPOINT_DIR, ngr_str, country)
        os.makedirs(country_dir, exist_ok=True)
        
        print(f"\n--- Country: {country} ---", flush=True)
        
        targets_df = conn.execute(f"""
            SELECT entity_id, norm_name, source
            FROM 'data/train_s2.parquet' WHERE country = '{country}'
            UNION ALL
            SELECT entity_id, norm_name, source
            FROM 'data/train_s3.parquet' WHERE country = '{country}'
        """).fetchdf()
        
        s1_df = conn.execute(f"""
            SELECT entity_id, norm_name
            FROM 'data/train_s1.parquet' 
            WHERE country = '{country}' AND is_val = true
        """).fetchdf()
        
        print(f"S1 Validation Queries: {len(s1_df):,}", flush=True)
        print(f"Target Entities (S2+S3): {len(targets_df):,}", flush=True)
        
        if len(s1_df) == 0 or len(targets_df) == 0:
            print("Skipping (no data).", flush=True)
            continue
            
        print("Vectorizing...", flush=True)
        t_vec = time.time()
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngr_range, min_df=2, max_df=0.05, dtype=np.float32)
        targets_tfidf = vectorizer.fit_transform(targets_df['norm_name'].fillna(''))
        s1_tfidf = vectorizer.transform(s1_df['norm_name'].fillna(''))
        print(f"Vectorization complete in {time.time()-t_vec:.1f}s. Vocabulary size: {len(vectorizer.vocabulary_):,}", flush=True)
        
        target_ids = targets_df['entity_id'].values
        target_sources = targets_df['source'].values
        s1_ids = s1_df['entity_id'].values
        
        num_batches = int(np.ceil(s1_tfidf.shape[0] / batch_size))
        print(f"Total Batches: {num_batches} (Batch Size: {batch_size})", flush=True)
        
        country_candidates = 0
        t_proc_start = time.time()
        batches_processed = 0
        
        for b in range(num_batches):
            batch_id = b + 1
            final_path = os.path.join(country_dir, f"batch_{batch_id:06d}.parquet")
            tmp_path = final_path + ".tmp"
            
            if os.path.exists(final_path):
                try:
                    cands = pq.read_metadata(final_path).num_rows
                    country_candidates += cands
                    print(f"[{country}][{ngr_str}][K={k}] Batch {batch_id}/{num_batches} | SKIPPED (already complete) | candidates={cands:,}", flush=True)
                except Exception as e:
                    print(f"[{country}][{ngr_str}][K={k}] Batch {batch_id}/{num_batches} | CORRUPTED checkpoint detected, regenerating... Error: {e}", flush=True)
                    os.remove(final_path)
                else:
                    continue
            
            if os.path.exists(tmp_path):
                print(f"[{country}][{ngr_str}][K={k}] Batch {batch_id}/{num_batches} | INCOMPLETE checkpoint detected, regenerating...", flush=True)
                os.remove(tmp_path)
                
            t_batch_start = time.time()
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
            
            num_cands = len(b_s1_id)
            country_candidates += num_cands
            
            if num_cands > 0:
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
            
            t_batch_end = time.time()
            elapsed_total = t_batch_end - t_proc_start
            batches_processed += 1
            avg_batch_time = elapsed_total / batches_processed
            eta = avg_batch_time * (num_batches - batch_id)
            ram_mb = get_ram_mb()
            
            print(f"[{country}][{ngr_str}][K={k}] Batch {batch_id}/{num_batches} | candidates={num_cands:,} | elapsed={format_time(t_batch_end - t_batch_start)} | ETA={format_time(eta)} | RAM={ram_mb:.0f}MB", flush=True)
            
        total_candidates_all += country_candidates
        total_s1_all += len(s1_df)
        print(f"Finished {country} | Candidates: {country_candidates:,}", flush=True)
        
        del targets_tfidf, s1_tfidf, vectorizer
        gc.collect()
        
    total_runtime = time.time() - t_start
    print(f"Configuration {ngr_range} Completed! Total S1: {total_s1_all:,} | Total Candidates: {total_candidates_all:,} | Runtime: {format_time(total_runtime)}\n", flush=True)
    return os.path.join(CHECKPOINT_DIR, ngr_str, "*", "*.parquet"), total_runtime, get_ram_mb()

def evaluate_config(conn, glob_path, ngr_range, config_runtime, config_ram, k_values=[5,10,20,50]):
    ngr_str = f"{ngr_range[0]}_{ngr_range[1]}"
    print(f"\n==================================================", flush=True)
    print(f"EVALUATING CONFIGURATION: {ngr_range}", flush=True)
    print(f"==================================================", flush=True)
    
    conn.execute(f"CREATE OR REPLACE VIEW cands AS SELECT * FROM read_parquet('{glob_path}');")
    
    conn.execute("""
        CREATE OR REPLACE VIEW gt_val AS 
        SELECT g.source1_entity_id, g.target_entity_id, g.target_source, coalesce(s1.country, 'Unknown') as country 
        FROM 'data/train_gt_pairs.parquet' g
        JOIN 'data/train_s1.parquet' s1 ON g.source1_entity_id = s1.entity_id
        WHERE s1.is_val = true;
    """)
    conn.execute("""
        CREATE OR REPLACE VIEW gt_val_counts AS
        SELECT source1_entity_id as s1_id, country, count(*) as total_gt
        FROM gt_val
        GROUP BY source1_entity_id, country;
    """)
    
    metrics = []
    
    for k in k_values:
        print(f"Evaluating K={k}...", flush=True)
        cands_k_view = f"cands_k_{k}"
        conn.execute(f"CREATE OR REPLACE VIEW {cands_k_view} AS SELECT * FROM cands WHERE rank <= {k};")
        
        # We need metrics OVERALL and per COUNTRY
        
        queries = [
            ("OVERALL", f"""
                SELECT 
                    'OVERALL' as grouping,
                    (SELECT count(*) FROM gt_val) as total_gt,
                    (SELECT count(*) FROM gt_val_counts) as total_s1_with_gt,
                    (SELECT count(*) FROM {cands_k_view} c JOIN gt_val g ON c.source1_entity_id = g.source1_entity_id AND c.target_entity_id = g.target_entity_id) as tp_k,
                    (
                        WITH retrieved_counts AS (
                            SELECT c.source1_entity_id as s1_id, count(*) as tp_count
                            FROM {cands_k_view} c JOIN gt_val g ON c.source1_entity_id = g.source1_entity_id AND c.target_entity_id = g.target_entity_id
                            GROUP BY c.source1_entity_id
                        )
                        SELECT sum(case when coalesce(r.tp_count,0) >= g.total_gt then 1 else 0 end)
                        FROM gt_val_counts g LEFT JOIN retrieved_counts r ON g.s1_id = r.s1_id
                    ) as complete_k,
                    (SELECT sum(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true GROUP BY s.entity_id)) as total_cands,
                    (SELECT avg(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true GROUP BY s.entity_id)) as avg_cands,
                    (SELECT median(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true GROUP BY s.entity_id)) as med_cands,
                    (SELECT quantile_cont(num_cands, 0.95) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true GROUP BY s.entity_id)) as p95,
                    (SELECT quantile_cont(num_cands, 0.99) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true GROUP BY s.entity_id)) as p99,
                    (SELECT max(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true GROUP BY s.entity_id)) as max_cands,
                    (SELECT sum(case when num_cands = 0 then 1 else 0 end) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true GROUP BY s.entity_id)) as zero_cands
            """),
        ]
        
        countries = conn.execute("SELECT DISTINCT country FROM 'data/train_s1.parquet' WHERE is_val = true AND country IS NOT NULL").df()['country'].tolist()
        for c in countries:
            queries.append((c, f"""
                SELECT 
                    '{c}' as grouping,
                    (SELECT count(*) FROM gt_val WHERE country = '{c}') as total_gt,
                    (SELECT count(*) FROM gt_val_counts WHERE country = '{c}') as total_s1_with_gt,
                    (SELECT count(*) FROM {cands_k_view} c JOIN gt_val g ON c.source1_entity_id = g.source1_entity_id AND c.target_entity_id = g.target_entity_id WHERE c.country = '{c}') as tp_k,
                    (
                        WITH retrieved_counts AS (
                            SELECT c.source1_entity_id as s1_id, count(*) as tp_count
                            FROM {cands_k_view} c JOIN gt_val g ON c.source1_entity_id = g.source1_entity_id AND c.target_entity_id = g.target_entity_id
                            WHERE c.country = '{c}'
                            GROUP BY c.source1_entity_id
                        )
                        SELECT sum(case when coalesce(r.tp_count,0) >= g.total_gt then 1 else 0 end)
                        FROM gt_val_counts g LEFT JOIN retrieved_counts r ON g.s1_id = r.s1_id
                        WHERE g.country = '{c}'
                    ) as complete_k,
                    (SELECT sum(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true AND s.country = '{c}' GROUP BY s.entity_id)) as total_cands,
                    (SELECT avg(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true AND s.country = '{c}' GROUP BY s.entity_id)) as avg_cands,
                    (SELECT median(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true AND s.country = '{c}' GROUP BY s.entity_id)) as med_cands,
                    (SELECT quantile_cont(num_cands, 0.95) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true AND s.country = '{c}' GROUP BY s.entity_id)) as p95,
                    (SELECT quantile_cont(num_cands, 0.99) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true AND s.country = '{c}' GROUP BY s.entity_id)) as p99,
                    (SELECT max(num_cands) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true AND s.country = '{c}' GROUP BY s.entity_id)) as max_cands,
                    (SELECT sum(case when num_cands = 0 then 1 else 0 end) FROM (SELECT s.entity_id as s1_id, coalesce(count(c.target_entity_id), 0) as num_cands FROM 'data/train_s1.parquet' s LEFT JOIN {cands_k_view} c ON s.entity_id = c.source1_entity_id WHERE s.is_val = true AND s.country = '{c}' GROUP BY s.entity_id)) as zero_cands
            """))
            
        for grp, query in queries:
            row = conn.execute(query).fetchone()
            if row is None or row[1] is None:
                continue
                
            total_gt_grp = row[1]
            total_s1_with_gt_grp = row[2]
            tp_k_grp = row[3]
            complete_k_grp = row[4]
            total_cands_grp = row[5] or 0
            avg_cands_grp = row[6] or 0
            med_cands_grp = row[7] or 0
            p95_grp = row[8] or 0
            p99_grp = row[9] or 0
            max_cands_grp = row[10] or 0
            zero_cands_grp = row[11] or 0
            
            pair_recall = tp_k_grp / total_gt_grp if total_gt_grp > 0 else 0
            recovery_rate = complete_k_grp / total_s1_with_gt_grp if total_s1_with_gt_grp > 0 else 0
            precision = tp_k_grp / total_cands_grp if total_cands_grp > 0 else 0
            
            print(f"  [{grp}][K={k}] Recall: {pair_recall*100:.2f}% | S1 Rec: {recovery_rate*100:.2f}% | Cands: {total_cands_grp:,}", flush=True)
            
            metrics.append({
                'ngram_config': ngr_str,
                'grouping': grp,
                'k': k,
                'pair_recall': pair_recall,
                's1_complete_recovery': recovery_rate,
                'precision': precision,
                'total_candidates': total_cands_grp,
                'avg_candidates_per_s1': avg_cands_grp,
                'median_candidates': med_cands_grp,
                'p95_candidates': p95_grp,
                'p99_candidates': p99_grp,
                'max_candidates': max_cands_grp,
                'zero_candidate_s1s': zero_cands_grp,
                'config_runtime_s': config_runtime if grp == 'OVERALL' else 0,
                'config_peak_ram_mb': config_ram if grp == 'OVERALL' else 0
            })
            
    return metrics

def main():
    conn = init_duckdb()
    configs = [(3,5), (3,6)]
    
    print("\n==================================================", flush=True)
    print("PHASE 3 EXPERIMENT - FULL VALIDATION RUN", flush=True)
    print("==================================================", flush=True)
    
    all_metrics = []
    
    for cfg in configs:
        glob_path, runtime, ram = generate_candidates(conn, cfg, k=50, batch_size=2000)
        metrics = evaluate_config(conn, glob_path, cfg, runtime, ram)
        all_metrics.extend(metrics)
        
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(REPORTS_DIR, 'phase3_ngram_metrics.csv'), index=False)
    
    # Save a markdown report
    with open(os.path.join(REPORTS_DIR, 'phase3_ngram_retrieval.md'), 'w') as f:
        f.write("# Phase 3: Character N-Gram Retrieval Final Metrics\n\n")
        f.write("## Overview\nRedesigned highly optimized sparse_dot_topn retrieval.\n\n")
        f.write("## Results (OVERALL)\n\n")
        f.write("| Config | K | Recall | S1 Recovery | Precision | Total Cands | Avg Cands | Zero S1 | Runtime |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        
        overall = df_metrics[df_metrics['grouping'] == 'OVERALL']
        for _, r in overall.iterrows():
            f.write(f"| {r['ngram_config']} | {r['k']} | {r['pair_recall']*100:.2f}% | {r['s1_complete_recovery']*100:.2f}% | {r['precision']*100:.2f}% | {r['total_candidates']:,} | {r['avg_candidates_per_s1']:.1f} | {r['zero_candidate_s1s']:,} | {format_time(r['config_runtime_s'])} |\n")
            
        f.write("\n## Comparison to Phase 2 Baselines\n")
        try:
            baseline_df = pd.read_csv(os.path.join(REPORTS_DIR, 'baseline_metrics.csv'))
            f.write(baseline_df.to_markdown(index=False))
        except:
            f.write("*Baseline metrics file not found to display inline.*")
            
    # Also save detailed candidate statistics as requested
    df_metrics.to_csv(os.path.join(REPORTS_DIR, 'phase3_candidate_statistics.csv'), index=False)
            
    print("\nPhase 3 processing fully complete. Results saved in reports/.", flush=True)

if __name__ == '__main__':
    main()
