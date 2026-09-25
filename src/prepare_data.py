#!/usr/bin/env python3
"""
src/prepare_data.py

Preprocesses training data for the Amazon ML Challenge 2026 Business Entity Resolution project.
Performs:
1. Leakage-safe, reproducible 80/20 train/validation split at the S1 entity level
   stratified by country and match count category.
2. Standardized normalization:
   - norm_basic: Unicode NFKD, Latin diacritic stripping (preserving Indic marks),
     punctuation/symbol replacement, lowercasing, whitespace collapse.
   - norm_name_legal: norm_basic + stripping of common legal entity suffixes (US/India).
   - norm_addr: norm_basic on addresses (empty addresses preserved as "").
3. High-performance, memory-safe Parquet export using PyArrow and chunked multiprocessing.
"""

import os
import sys
import gc
import time
import re
import unicodedata
import multiprocessing as mp
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = r"Dataset/student_resource/dataset/train"
OUTPUT_DIR = r"data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGAL_SUFFIX_RE = re.compile(
    r'\b(pvt\s+ltd|private\s+limited|limited\s+liability\s+partnership|'
    r'ltd|limited|llc|inc|corp|corporation|co|company|pllc|llp|pc|gmbh|sa|srl|lp|'
    r'प्राइवेट\s+लिमिटेड|लिमिटेड|एलएलपी)\b$',
    re.IGNORECASE
)

def normalize_text(text: str) -> str:
    """Normalizes text by decomposing Unicode, stripping Latin diacritics, 
    preserving Indic characters/matras, replacing punctuation with space, 
    lowercasing, and collapsing whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', str(text))
    chars = []
    for c in text:
        cat = unicodedata.category(c)
        # Strip Latin combining marks (< 0x0900), preserve Indic marks (>= 0x0900)
        if cat == 'Mn' and ord(c) < 0x0900:
            continue
        if cat[0] in ('L', 'M', 'N'):
            chars.append(c)
        elif cat[0] in ('P', 'S') or c.isspace():
            chars.append(' ')
        else:
            chars.append(' ')
    return ' '.join(''.join(chars).lower().split())

def normalize_legal(norm_name: str) -> str:
    """Strips terminal legal entity suffixes from a pre-normalized business name."""
    if not norm_name:
        return ""
    stripped = LEGAL_SUFFIX_RE.sub('', norm_name).strip()
    return stripped if stripped else norm_name

def process_chunk(chunk_tuples):
    """Worker function for multiprocessing."""
    # chunk_tuples: list of (entity_id, business_name, business_address, country)
    out = []
    for eid, bname, baddr, country in chunk_tuples:
        nb = normalize_text(bname)
        nl = normalize_legal(nb)
        na = normalize_text(baddr)
        out.append((eid, bname, baddr, country, nb, nl, na))
    return out

def convert_tsv_to_parquet(tsv_path, parquet_path, source_tag=None, val_ids=None, chunk_size=150000, num_workers=6):
    """Reads a large TSV in chunks, normalizes fields in parallel, and writes to Parquet."""
    print(f"\n--- Processing {os.path.basename(tsv_path)} -> {os.path.basename(parquet_path)} ---")
    t0 = time.time()
    
    schema_fields = [
        ('entity_id', pa.string()),
        ('business_name', pa.string()),
        ('business_address', pa.string()),
        ('country', pa.string()),
        ('norm_name', pa.string()),
        ('norm_name_legal', pa.string()),
        ('norm_addr', pa.string())
    ]
    if source_tag is not None:
        schema_fields.append(('source', pa.string()))
    if val_ids is not None:
        schema_fields.append(('is_val', pa.bool_()))
        
    schema = pa.schema(schema_fields)
    writer = pq.ParquetWriter(parquet_path, schema, compression='snappy')
    
    total_rows = 0
    pool = mp.Pool(processes=num_workers)
    
    try:
        # Read in chunks
        chunk_iter = pd.read_csv(tsv_path, sep='\t', dtype=str, keep_default_na=False, chunksize=chunk_size)
        sub_chunk_size = 25000
        
        for chunk_idx, chunk in enumerate(chunk_iter):
            t_chunk_start = time.time()
            rows = list(zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']))
            
            # Split rows into sub-chunks for worker processes
            sub_chunks = [rows[i:i + sub_chunk_size] for i in range(0, len(rows), sub_chunk_size)]
            results = pool.map(process_chunk, sub_chunks)
            
            # Flatten results
            flat = [item for sublist in results for item in sublist]
            
            # Build PyArrow RecordBatch
            eids = [r[0] for r in flat]
            bnames = [r[1] for r in flat]
            baddrs = [r[2] for r in flat]
            countries = [r[3] for r in flat]
            norm_names = [r[4] for r in flat]
            norm_legals = [r[5] for r in flat]
            norm_addrs = [r[6] for r in flat]
            
            arrays = [
                pa.array(eids, type=pa.string()),
                pa.array(bnames, type=pa.string()),
                pa.array(baddrs, type=pa.string()),
                pa.array(countries, type=pa.string()),
                pa.array(norm_names, type=pa.string()),
                pa.array(norm_legals, type=pa.string()),
                pa.array(norm_addrs, type=pa.string())
            ]
            
            if source_tag is not None:
                arrays.append(pa.array([source_tag] * len(flat), type=pa.string()))
            if val_ids is not None:
                is_val_arr = [eid in val_ids for eid in eids]
                arrays.append(pa.array(is_val_arr, type=pa.bool_()))
                
            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            writer.write_batch(batch)
            
            total_rows += len(flat)
            print(f"  Chunk {chunk_idx+1}: {len(flat):,} rows (Total: {total_rows:,}) in {time.time()-t_chunk_start:.1f}s")
            del rows, sub_chunks, results, flat, arrays, batch
            gc.collect()
            
    finally:
        pool.close()
        pool.join()
        writer.close()
        
    print(f"Finished {total_rows:,} rows in {time.time()-t0:.1f}s. Saved to {parquet_path}")

def process_ground_truth(gt_path, s1_country_map, val_ids):
    """Processes train_ground_truth.tsv into:
    1. data/train_gt_pairs.parquet (all pairs: s1_id, target_id, target_source)
    2. data/train_gt_s1.parquet (s1_id, country, match_count, is_singleton, is_val)
    """
    print(f"\n--- Processing Ground Truth: {gt_path} ---")
    t0 = time.time()
    
    gt_df = pd.read_csv(gt_path, sep='\t', dtype=str, keep_default_na=False)
    print(f"Loaded {len(gt_df):,} ground truth rows in {time.time()-t0:.1f}s", flush=True)
    
    # Process S1 summary
    s1_ids = gt_df['source1_entity_id'].tolist()
    matched_lists = [
        [m.strip() for m in x.split(',') if m.strip()] if x.strip() else []
        for x in gt_df['matched_entity_ids']
    ]
    counts = np.array([len(m) for m in matched_lists], dtype=np.int32)
    singletons = (counts == 0)
    
    countries = [s1_country_map.get(sid, 'UNKNOWN') for sid in s1_ids]
    is_val = np.array([sid in val_ids for sid in s1_ids], dtype=bool)
    
    # Write S1 GT table
    gt_s1_df = pd.DataFrame({
        'source1_entity_id': s1_ids,
        'country': countries,
        'match_count': counts,
        'is_singleton': singletons,
        'is_val': is_val
    })
    gt_s1_path = os.path.join(OUTPUT_DIR, 'train_gt_s1.parquet')
    gt_s1_df.to_parquet(gt_s1_path, index=False, compression='snappy')
    print(f"Saved {len(gt_s1_df):,} S1 GT rows to {gt_s1_path}", flush=True)
    
    # Explode for pairs table
    t1 = time.time()
    pair_s1 = []
    pair_target = []
    pair_source = []
    
    for sid, mlist in zip(s1_ids, matched_lists):
        for mid in mlist:
            pair_s1.append(sid)
            pair_target.append(mid)
            pair_source.append('S2' if mid.startswith('S2-') else 'S3')
            
    pairs_df = pd.DataFrame({
        'source1_entity_id': pair_s1,
        'target_entity_id': pair_target,
        'target_source': pair_source
    })
    gt_pairs_path = os.path.join(OUTPUT_DIR, 'train_gt_pairs.parquet')
    pairs_df.to_parquet(gt_pairs_path, index=False, compression='snappy')
    print(f"Saved {len(pairs_df):,} GT pairs to {gt_pairs_path} in {time.time()-t1:.1f}s")

def main():
    print("==================================================")
    print("AMAZON ML CHALLENGE 2026 - DATA PREPARATION")
    print("==================================================")
    total_start = time.time()
    
    # 1. Read S1 IDs and Country to create stratified validation split
    print("\n[Step 1] Creating reproducible, leakage-safe validation split (80/20)...")
    s1_meta = pd.read_csv(
        os.path.join(DATASET_DIR, 'train_source1.tsv'),
        sep='\t', usecols=['entity_id', 'country'], dtype=str
    )
    gt_meta = pd.read_csv(
        os.path.join(DATASET_DIR, 'train_ground_truth.tsv'),
        sep='\t', dtype=str, keep_default_na=False
    )
    
    gt_meta['match_count'] = gt_meta['matched_entity_ids'].apply(
        lambda x: len([i for i in x.split(',') if i.strip()]) if x.strip() else 0
    )
    s1_merged = s1_meta.merge(gt_meta[['source1_entity_id', 'match_count']], left_on='entity_id', right_on='source1_entity_id')
    
    def get_stratum(row):
        cnt = row['match_count']
        c_cat = '0' if cnt == 0 else ('1' if cnt == 1 else 'multi')
        return f"{row['country']}_{c_cat}"
        
    s1_merged['stratum'] = s1_merged.apply(get_stratum, axis=1)
    
    entity_ids = np.array(s1_merged['entity_id'].tolist(), dtype=object)
    strata = np.array(s1_merged['stratum'].tolist(), dtype=object)
    
    train_ids, val_ids_list = train_test_split(
        entity_ids,
        test_size=0.20,
        random_state=42,
        stratify=strata
    )
    val_ids = set(val_ids_list)
    print(f"Total S1: {len(s1_merged):,}", flush=True)
    print(f"Train S1: {len(train_ids):,} (80.0%)", flush=True)
    print(f"Val S1  : {len(val_ids):,} (20.0%)", flush=True)
    
    s1_country_map = dict(zip(s1_merged['entity_id'].tolist(), s1_merged['country'].tolist()))
    
    # 2. Process Ground Truth
    print("\n[Step 2] Processing Ground Truth into Parquet...", flush=True)
    process_ground_truth(
        os.path.join(DATASET_DIR, 'train_ground_truth.tsv'),
        s1_country_map,
        val_ids
    )
    
    # 3. Process Source 1
    print("\n[Step 3] Processing Source 1 into Parquet...")
    convert_tsv_to_parquet(
        os.path.join(DATASET_DIR, 'train_source1.tsv'),
        os.path.join(OUTPUT_DIR, 'train_s1.parquet'),
        source_tag='S1',
        val_ids=val_ids,
        chunk_size=200000,
        num_workers=6
    )
    
    # 4. Process Source 2
    print("\n[Step 4] Processing Source 2 into Parquet...")
    convert_tsv_to_parquet(
        os.path.join(DATASET_DIR, 'train_source2.tsv'),
        os.path.join(OUTPUT_DIR, 'train_s2.parquet'),
        source_tag='S2',
        val_ids=None,
        chunk_size=200000,
        num_workers=6
    )
    
    # 5. Process Source 3
    print("\n[Step 5] Processing Source 3 into Parquet...")
    convert_tsv_to_parquet(
        os.path.join(DATASET_DIR, 'train_source3.tsv'),
        os.path.join(OUTPUT_DIR, 'train_s3.parquet'),
        source_tag='S3',
        val_ids=None,
        chunk_size=200000,
        num_workers=6
    )
    
    print(f"\nData preparation completed in {time.time()-total_start:.1f}s.")

if __name__ == '__main__':
    main()
