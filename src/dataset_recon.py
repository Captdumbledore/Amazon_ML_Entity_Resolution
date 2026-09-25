#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Dataset Reconnaissance
Performs deep analysis and generates comprehensive markdown and JSON reports.
"""
import pandas as pd
import numpy as np
import os, sys, json, unicodedata, re, time
from collections import Counter
from itertools import chain

sys.stdout.reconfigure(encoding='utf-8')

BASE = r'c:\Users\jisto\Documents\Amazon_ML\Dataset\student_resource\dataset'
REPORTS_DIR = r'c:\Users\jisto\Documents\Amazon_ML\reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# Helper for basic normalization
def normalize_text(series):
    s = series.fillna('').str.strip().str.lower()
    s = s.apply(lambda x: unicodedata.normalize('NFKD', x) if x else '')
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def calc_f05(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

print("Starting deep dataset reconnaissance...")

# 1. DISCOVER ACTUAL DATASET
print("\n[1] Discovering dataset files...")
discovered_files = []
for root, dirs, files in os.walk(BASE):
    for f in files:
        if f.endswith('.tsv') or f.endswith('.py') or f.endswith('.md'):
            discovered_files.append(os.path.join(root, f))
print(f"Found {len(discovered_files)} files.")

# Load Data
print("\nLoading datasets into memory...")
dtypes = {'entity_id': str, 'business_name': str, 'business_address': str, 'country': str}
t1 = time.time()
train_s1 = pd.read_csv(f'{BASE}/train/train_source1.tsv', sep='\t', dtype=dtypes, keep_default_na=False)
train_s2 = pd.read_csv(f'{BASE}/train/train_source2.tsv', sep='\t', dtype=dtypes, keep_default_na=False)
train_s3 = pd.read_csv(f'{BASE}/train/train_source3.tsv', sep='\t', dtype=dtypes, keep_default_na=False)
test_s1 = pd.read_csv(f'{BASE}/test/test_source1.tsv', sep='\t', dtype=dtypes, keep_default_na=False)
test_s2 = pd.read_csv(f'{BASE}/test/test_source2.tsv', sep='\t', dtype=dtypes, keep_default_na=False)
test_s3 = pd.read_csv(f'{BASE}/test/test_source3.tsv', sep='\t', dtype=dtypes, keep_default_na=False)
gt = pd.read_csv(f'{BASE}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
print(f"Loaded in {time.time()-t1:.1f}s")

# 2. DATASET SIZE & 3. DATA QUALITY & 4. COUNTRY DISTRIBUTION
print("\n[2-4] Analyzing size, quality, and distributions...")
datasets = {
    'train_s1': train_s1, 'train_s2': train_s2, 'train_s3': train_s3,
    'test_s1': test_s1, 'test_s2': test_s2, 'test_s3': test_s3
}

stats = {}
for name, df in datasets.items():
    mem_usage = df.memory_usage(deep=True).sum() / (1024*1024)
    name_len = df['business_name'].str.len()
    addr_len = df['business_address'].str.len()
    stats[name] = {
        'rows': len(df),
        'columns': list(df.columns),
        'unique_ids': df['entity_id'].nunique(),
        'duplicate_ids': len(df) - df['entity_id'].nunique(),
        'memory_mb': round(mem_usage, 2),
        'missing_names': int((df['business_name'] == '').sum()),
        'missing_addrs': int((df['business_address'] == '').sum()),
        'unique_names': df['business_name'].nunique(),
        'unique_addrs': df['business_address'].nunique(),
        'name_len_min': int(name_len.min()) if len(name_len)>0 else 0,
        'name_len_max': int(name_len.max()) if len(name_len)>0 else 0,
        'name_len_mean': round(float(name_len.mean()), 2) if len(name_len)>0 else 0,
        'addr_len_min': int(addr_len.min()) if len(addr_len)>0 else 0,
        'addr_len_max': int(addr_len.max()) if len(addr_len)>0 else 0,
        'addr_len_mean': round(float(addr_len.mean()), 2) if len(addr_len)>0 else 0,
        'country_counts': df['country'].value_counts().to_dict()
    }

brute_force = {
    'S1xS2': stats['train_s1']['rows'] * stats['train_s2']['rows'],
    'S1xS3': stats['train_s1']['rows'] * stats['train_s3']['rows'],
    'S1x(S2+S3)': stats['train_s1']['rows'] * (stats['train_s2']['rows'] + stats['train_s3']['rows'])
}

# 5. GROUND TRUTH ANALYSIS
print("\n[5] Ground Truth Analysis...")
gt['match_list'] = gt['matched_entity_ids'].apply(lambda x: [i.strip() for i in x.split(',')] if x.strip() else [])
gt['match_count'] = gt['match_list'].apply(len)

gt_stats = {
    'total_s1': len(gt),
    '0_matches': int((gt['match_count'] == 0).sum()),
    '1_match': int((gt['match_count'] == 1).sum()),
    '2_matches': int((gt['match_count'] == 2).sum()),
    '3_matches': int((gt['match_count'] == 3).sum()),
    '4plus_matches': int((gt['match_count'] >= 4).sum()),
    'max_matches': int(gt['match_count'].max()),
}

# Explode GT for S2/S3 analysis
exploded = gt[['source1_entity_id', 'match_list']].explode('match_list')
exploded = exploded[exploded['match_list'] != '']
exploded['is_s2'] = exploded['match_list'].str.startswith('S2-')
exploded['is_s3'] = exploded['match_list'].str.startswith('S3-')

per_s1 = exploded.groupby('source1_entity_id').agg(s2_c=('is_s2', 'sum'), s3_c=('is_s3', 'sum'))
gt_stats['s2_only'] = int(((per_s1['s2_c'] > 0) & (per_s1['s3_c'] == 0)).sum())
gt_stats['s3_only'] = int(((per_s1['s2_c'] == 0) & (per_s1['s3_c'] > 0)).sum())
gt_stats['both_s2_s3'] = int(((per_s1['s2_c'] > 0) & (per_s1['s3_c'] > 0)).sum())

gt_pairs_set = set(zip(exploded['source1_entity_id'], exploded['match_list']))

# 6. EXACT MATCH BASELINES
print("\n[6] Exact Match Baselines (Train Data Only - 10% Sample)...")
train_s1['norm_name'] = normalize_text(train_s1['business_name'])
train_s1['norm_addr'] = normalize_text(train_s1['business_address'])
train_s2['norm_name'] = normalize_text(train_s2['business_name'])
train_s2['norm_addr'] = normalize_text(train_s2['business_address'])
train_s3['norm_name'] = normalize_text(train_s3['business_name'])
train_s3['norm_addr'] = normalize_text(train_s3['business_address'])

train_cand = pd.concat([train_s2, train_s3], ignore_index=True)
train_cand = train_cand.rename(columns={'entity_id': 'match_id'})

# Sample S1 to avoid memory explosion
s1_sample = train_s1.sample(frac=0.1, random_state=42)
sampled_s1_ids = set(s1_sample['entity_id'])
gt_pairs_set_sampled = {p for p in gt_pairs_set if p[0] in sampled_s1_ids}
total_true_pairs_sampled = len(gt_pairs_set_sampled)

def evaluate_merge(s1_df, cand_df, merge_keys, name):
    print(f"  Evaluating {name}...")
    # Drop empty keys
    s1_valid = s1_df[s1_df[merge_keys[0]] != '']
    cand_valid = cand_df[cand_df[merge_keys[0]] != '']
    
    # Filter highly frequent keys to avoid explosion (keys appearing > 100,000 times in candidates)
    key_counts = cand_valid[merge_keys[0]].value_counts()
    frequent_keys = set(key_counts[key_counts > 100000].index)
    if frequent_keys:
        print(f"    Filtering {len(frequent_keys)} overly frequent keys from {merge_keys[0]}")
        s1_valid = s1_valid[~s1_valid[merge_keys[0]].isin(frequent_keys)]
        cand_valid = cand_valid[~cand_valid[merge_keys[0]].isin(frequent_keys)]
    
    merged = pd.merge(s1_valid[['entity_id', 'country'] + merge_keys], 
                      cand_valid[['match_id', 'country'] + merge_keys], 
                      on=['country'] + merge_keys, how='inner')
    
    cands_generated = len(merged)
         
    generated_set = set(zip(merged['entity_id'], merged['match_id']))
    tp = len(generated_set.intersection(gt_pairs_set_sampled))
    fp = len(generated_set) - tp
    fn = total_true_pairs_sampled - tp
    
    return {
        'candidate_count': cands_generated * 10, # Extrapolate to 100%
        'tp': tp * 10,
        'fp': fp * 10,
        'fn': fn * 10,
        'precision': tp/(tp+fp) if (tp+fp)>0 else 0,
        'recall': tp/(tp+fn) if (tp+fn)>0 else 0,
        'f05': calc_f05(tp, fp, fn)
    }

baselines = {}
baselines['raw_name'] = evaluate_merge(s1_sample, train_cand, ['business_name'], "Raw Name")
baselines['norm_name'] = evaluate_merge(s1_sample, train_cand, ['norm_name'], "Norm Name")
baselines['raw_addr'] = evaluate_merge(s1_sample, train_cand, ['business_address'], "Raw Addr")
baselines['norm_addr'] = evaluate_merge(s1_sample, train_cand, ['norm_addr'], "Norm Addr")
baselines['norm_name_addr'] = evaluate_merge(s1_sample, train_cand, ['norm_name', 'norm_addr'], "Norm Name + Addr")


# 9. HARD NEGATIVES
print("\n[9] Hard Negatives...")
# Find pairs that share exact norm name but are NOT in ground truth
s1_samp = train_s1.sample(200000, random_state=42) # Sample to avoid huge memory
hn_merged = pd.merge(s1_samp[['entity_id', 'country', 'norm_name', 'business_name', 'business_address']], 
                     train_cand[['match_id', 'country', 'norm_name', 'business_name', 'business_address']], 
                     on=['country', 'norm_name'], how='inner')
hn_merged['is_gt'] = hn_merged.apply(lambda x: (x['entity_id'], x['match_id']) in gt_pairs_set, axis=1)
hard_negatives = hn_merged[~hn_merged['is_gt']]

# Write Reports
print("\n[14] Writing Reports...")

summary = {
    'dataset_scale': {
        'S1_rows': stats['train_s1']['rows'],
        'S2_rows': stats['train_s2']['rows'],
        'S3_rows': stats['train_s3']['rows']
    },
    'ground_truth': gt_stats,
    'data_quality': {
        'train_s1': stats['train_s1'],
        'train_s2': stats['train_s2'],
        'train_s3': stats['train_s3']
    },
    'baselines': baselines
}

with open(f"{REPORTS_DIR}/dataset_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

with open(f"{REPORTS_DIR}/hard_negative_analysis.md", 'w', encoding='utf-8') as f:
    f.write("# Hard Negatives Analysis\n\n")
    f.write("Pairs with identical normalized names but NOT in ground truth (False Positives for Name Blocking):\n\n")
    for _, row in hard_negatives.head(50).iterrows():
        f.write(f"**S1**: {row['business_name']} | {row['business_address']}\n")
        f.write(f"**S2/3**: {row['business_name_y']} | {row['business_address_y']}\n")
        f.write("---\n")

print("Reconnaissance script complete. Data saved to reports/ directory.")
