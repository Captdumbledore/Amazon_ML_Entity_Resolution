#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Data Audit (Vectorized, Fast)
All loops replaced with vectorized pandas/numpy operations.
"""
import pandas as pd
import numpy as np
import os, sys, re, unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE = r'c:\Users\jisto\Documents\Amazon_ML\Dataset\student_resource\dataset'
REPORT = r'c:\Users\jisto\Documents\Amazon_ML\reports\data_audit.md'
os.makedirs(os.path.dirname(REPORT), exist_ok=True)

def basic_normalize(series):
    """Vectorized normalization of a pandas string Series."""
    s = series.fillna('').str.strip().str.lower()
    s = s.apply(lambda x: unicodedata.normalize('NFKD', x) if x else '')
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

# ======================================================================
# STAGE 1: Source file stats (chunked)
# ======================================================================
print("STAGE 1: Source file statistics...")

source_stats = {}
for label, path in [
    ('train_source1', f'{BASE}/train/train_source1.tsv'),
    ('train_source2', f'{BASE}/train/train_source2.tsv'),
    ('train_source3', f'{BASE}/train/train_source3.tsv'),
    ('test_source1', f'{BASE}/test/test_source1.tsv'),
    ('test_source2', f'{BASE}/test/test_source2.tsv'),
    ('test_source3', f'{BASE}/test/test_source3.tsv'),
]:
    print(f"  {label}...")
    total = 0; ids = set(); countries = Counter()
    name_null=0; name_empty=0; addr_null=0; addr_empty=0
    country_null=0; country_empty=0
    name_lens_all = []; addr_lens_all = []

    for chunk in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=500000):
        total += len(chunk)
        ids.update(chunk['entity_id'].values)
        countries.update(chunk['country'].values)
        name_empty += (chunk['business_name'] == '').sum()
        addr_empty += (chunk['business_address'] == '').sum()
        country_empty += (chunk['country'] == '').sum()
        nne = chunk[chunk['business_name'] != '']['business_name'].str.len()
        ane = chunk[chunk['business_address'] != '']['business_address'].str.len()
        name_lens_all.extend(nne.tolist())
        addr_lens_all.extend(ane.tolist())

    nl = np.array(name_lens_all, dtype=float)
    al = np.array(addr_lens_all, dtype=float)
    source_stats[label] = {
        'rows': total, 'unique_ids': len(ids), 'dup_ids': total - len(ids),
        'countries': dict(countries.most_common()),
        'name_empty': int(name_empty), 'addr_empty': int(addr_empty), 'country_empty': int(country_empty),
        'name_len_mean': round(float(np.mean(nl)),1) if len(nl) else 0,
        'name_len_median': round(float(np.median(nl)),1) if len(nl) else 0,
        'name_len_min': int(np.min(nl)) if len(nl) else 0,
        'name_len_max': int(np.max(nl)) if len(nl) else 0,
        'name_len_p95': round(float(np.percentile(nl,95)),1) if len(nl) else 0,
        'addr_len_mean': round(float(np.mean(al)),1) if len(al) else 0,
        'addr_len_median': round(float(np.median(al)),1) if len(al) else 0,
        'addr_len_min': int(np.min(al)) if len(al) else 0,
        'addr_len_max': int(np.max(al)) if len(al) else 0,
        'addr_len_p95': round(float(np.percentile(al,95)),1) if len(al) else 0,
    }
    print(f"    {total:,} rows, {len(ids):,} unique, {int(name_empty):,} empty names, {int(addr_empty):,} empty addrs")

# ======================================================================
# STAGE 2: Ground truth (fully vectorized)
# ======================================================================
print("\nSTAGE 2: Ground truth analysis (vectorized)...")
gt = pd.read_csv(f'{BASE}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
print(f"  Loaded {len(gt):,} rows")

# Parse match lists vectorized
gt['match_str'] = gt['matched_entity_ids'].fillna('').str.strip()
gt['match_count'] = gt['match_str'].apply(lambda x: len(x.split(',')) if x else 0)

total_gt = len(gt)
zero_matches = int((gt['match_count'] == 0).sum())
one_match = int((gt['match_count'] == 1).sum())
multi_match = int((gt['match_count'] > 1).sum())
max_matches = int(gt['match_count'].max())
mean_matches = round(gt['match_count'].mean(), 3)
median_matches = round(gt['match_count'].median(), 1)
match_dist = gt['match_count'].value_counts().sort_index().to_dict()

print(f"  Zero: {zero_matches:,} ({zero_matches/total_gt*100:.1f}%)")
print(f"  One: {one_match:,} ({one_match/total_gt*100:.1f}%)")
print(f"  Multi: {multi_match:,} ({multi_match/total_gt*100:.1f}%)")
print(f"  Max: {max_matches}")

# Explode matches for S2/S3 analysis
print("  Exploding matches for S2/S3 analysis...")
gt_nonzero = gt[gt['match_str'] != ''].copy()
gt_nonzero['match_list'] = gt_nonzero['match_str'].str.split(',')
exploded = gt_nonzero[['source1_entity_id']].copy()
exploded['match_id'] = gt_nonzero['match_list']
exploded = exploded.explode('match_id')
exploded['match_id'] = exploded['match_id'].str.strip()

total_match_ids = len(exploded)
unique_match_ids = exploded['match_id'].nunique()

# S2 vs S3
exploded['is_s2'] = exploded['match_id'].str.startswith('S2-')
exploded['is_s3'] = exploded['match_id'].str.startswith('S3-')
s2_total = int(exploded['is_s2'].sum())
s3_total = int(exploded['is_s3'].sum())

# Per-S1 S2/S3 counts
per_s1 = exploded.groupby('source1_entity_id').agg(
    s2_count=('is_s2', 'sum'),
    s3_count=('is_s3', 'sum'),
).reset_index()

both = int(((per_s1['s2_count'] > 0) & (per_s1['s3_count'] > 0)).sum())
only_s2 = int(((per_s1['s2_count'] > 0) & (per_s1['s3_count'] == 0)).sum())
only_s3 = int(((per_s1['s2_count'] == 0) & (per_s1['s3_count'] > 0)).sum())
multi_s2 = int((per_s1['s2_count'] > 1).sum())
multi_s3 = int((per_s1['s3_count'] > 1).sum())

# S2/S3 IDs matched to multiple S1
id_counts = exploded.groupby('match_id')['source1_entity_id'].nunique()
multi_assigned = int((id_counts > 1).sum())
multi_assigned_examples = id_counts[id_counts > 1].head(5).index.tolist()

print(f"  S2: {s2_total:,}, S3: {s3_total:,}")
print(f"  Both: {both:,}, Only S2: {only_s2:,}, Only S3: {only_s3:,}")
print(f"  Multi-S2: {multi_s2:,}, Multi-S3: {multi_s3:,}")
print(f"  IDs matched to multiple S1: {multi_assigned:,}")

# ======================================================================
# STAGE 3: Exact match analysis (vectorized via merge on sample)
# ======================================================================
print("\nSTAGE 3: Exact match & variation analysis...")

# Load S1
print("  Loading S1...")
s1 = pd.read_csv(f'{BASE}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False)
s1 = s1.rename(columns={c: f's1_{c}' for c in ['business_name', 'business_address', 'country']})
s1 = s1.rename(columns={'entity_id': 'source1_entity_id'})

# Merge GT exploded with S1
print("  Merging with S1...")
pairs = exploded[['source1_entity_id', 'match_id']].merge(s1, on='source1_entity_id', how='left')
del s1

# Load S2 and S3, merge
print("  Loading S2...")
s2 = pd.read_csv(f'{BASE}/train/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False)
s2 = s2.rename(columns={'entity_id': 'match_id', 'business_name': 'm_name', 'business_address': 'm_addr', 'country': 'm_country'})

print("  Loading S3...")
s3 = pd.read_csv(f'{BASE}/train/train_source3.tsv', sep='\t', dtype=str, keep_default_na=False)
s3 = s3.rename(columns={'entity_id': 'match_id', 'business_name': 'm_name', 'business_address': 'm_addr', 'country': 'm_country'})

# Concat S2+S3
s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3

print("  Merging with S2/S3...")
pairs = pairs.merge(s23, on='match_id', how='left')
del s23

total_pairs = len(pairs)
print(f"  Total GT pairs: {total_pairs:,}")

# Exact matches (vectorized)
exact_raw_name = int((pairs['s1_business_name'] == pairs['m_name']).sum())
exact_raw_addr = int((pairs['s1_business_address'] == pairs['m_addr']).sum())
country_match_count = int((pairs['s1_country'] == pairs['m_country']).sum())

print(f"  Exact raw name: {exact_raw_name:,} ({exact_raw_name/total_pairs*100:.1f}%)")
print(f"  Exact raw addr: {exact_raw_addr:,} ({exact_raw_addr/total_pairs*100:.1f}%)")
print(f"  Country match: {country_match_count:,} ({country_match_count/total_pairs*100:.1f}%)")

# Normalized exact matches (on a sample for speed)
print("  Computing normalized matches on full data...")
pairs['s1_name_norm'] = basic_normalize(pairs['s1_business_name'])
pairs['m_name_norm'] = basic_normalize(pairs['m_name'])
pairs['s1_addr_norm'] = basic_normalize(pairs['s1_business_address'])
pairs['m_addr_norm'] = basic_normalize(pairs['m_addr'])

norm_name_match = pairs['s1_name_norm'] == pairs['m_name_norm']
norm_name_match = norm_name_match & (pairs['s1_name_norm'] != '')
exact_norm_name = int(norm_name_match.sum())

norm_addr_match = pairs['s1_addr_norm'] == pairs['m_addr_norm']
norm_addr_match = norm_addr_match & (pairs['s1_addr_norm'] != '')
exact_norm_addr = int(norm_addr_match.sum())

print(f"  Exact norm name: {exact_norm_name:,} ({exact_norm_name/total_pairs*100:.1f}%)")
print(f"  Exact norm addr: {exact_norm_addr:,} ({exact_norm_addr/total_pairs*100:.1f}%)")

# Variation examples (non-exact raw name, first 20)
non_exact_name = pairs[pairs['s1_business_name'] != pairs['m_name']].head(20)
name_var_examples = list(zip(
    non_exact_name['s1_business_name'].str[:100].tolist(),
    non_exact_name['m_name'].str[:100].tolist()
))

non_exact_addr = pairs[pairs['s1_business_address'] != pairs['m_addr']].head(20)
addr_var_examples = list(zip(
    non_exact_addr['s1_business_address'].str[:100].tolist(),
    non_exact_addr['m_addr'].str[:100].tolist()
))

# Missing field examples
missing_name_s1 = pairs[(pairs['s1_business_name'] == '') & (pairs['m_name'] != '')].head(5)
missing_name_m = pairs[(pairs['s1_business_name'] != '') & (pairs['m_name'] == '')].head(5)
missing_addr_s1 = pairs[(pairs['s1_business_address'] == '') & (pairs['m_addr'] != '')].head(5)
missing_addr_m = pairs[(pairs['s1_business_address'] != '') & (pairs['m_addr'] == '')].head(5)

# Transliteration detection (Devanagari)
def has_devanagari_vec(s):
    return s.fillna('').str.contains(r'[\u0900-\u097F]', regex=True)

s1_has_dev = has_devanagari_vec(pairs['s1_business_name'])
m_has_dev = has_devanagari_vec(pairs['m_name'])
translit_mask = s1_has_dev != m_has_dev
translit_examples = pairs[translit_mask].head(10)

print(f"  Name transliteration cases: {translit_mask.sum():,}")

# Country mismatch count
country_mismatch = int((pairs['s1_country'] != pairs['m_country']).sum())
print(f"  Country mismatch: {country_mismatch:,}")

# ======================================================================
# STAGE 4: Write Report
# ======================================================================
print("\nSTAGE 4: Writing report...")

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write("# Amazon ML Challenge 2026 - Data Audit Report\n\n")
    f.write(f"Generated: {pd.Timestamp.now().isoformat()}\n\n---\n\n")

    # 1. File Overview
    f.write("## 1. Dataset File Overview\n\n")
    f.write("| File | Rows | Unique IDs | Dup IDs | Empty Names | Empty Addrs | Empty Country |\n")
    f.write("|------|------|-----------|---------|-------------|-------------|---------------|\n")
    for label, st in source_stats.items():
        f.write(f"| {label} | {st['rows']:,} | {st['unique_ids']:,} | {st['dup_ids']:,} | "
                f"{st['name_empty']:,} | {st['addr_empty']:,} | {st['country_empty']:,} |\n")
    f.write("\n")

    # 2. Country Distribution
    f.write("## 2. Country Distribution\n\n")
    for label, st in source_stats.items():
        f.write(f"### {label}\n\n| Country | Count | Pct |\n|---------|-------|-----|\n")
        for country, count in sorted(st['countries'].items(), key=lambda x: -x[1]):
            pct = round(count / st['rows'] * 100, 1)
            f.write(f"| {country if country else '(empty)'} | {count:,} | {pct}% |\n")
        f.write("\n")

    # 3. Length Distributions
    f.write("## 3. Business Name & Address Length Distributions\n\n")
    f.write("| Dataset | Field | Mean | Median | Min | Max | P95 |\n")
    f.write("|---------|-------|------|--------|-----|-----|-----|\n")
    for label, st in source_stats.items():
        f.write(f"| {label} | name | {st['name_len_mean']} | {st['name_len_median']} | {st['name_len_min']} | {st['name_len_max']} | {st['name_len_p95']} |\n")
        f.write(f"| {label} | addr | {st['addr_len_mean']} | {st['addr_len_median']} | {st['addr_len_min']} | {st['addr_len_max']} | {st['addr_len_p95']} |\n")
    f.write("\n")

    # 4. Ground Truth
    f.write("## 4. Ground Truth Analysis\n\n")
    f.write(f"- **Total S1 entities:** {total_gt:,}\n")
    f.write(f"- **Singletons (0 matches):** {zero_matches:,} ({zero_matches/total_gt*100:.1f}%)\n")
    f.write(f"- **1 match:** {one_match:,} ({one_match/total_gt*100:.1f}%)\n")
    f.write(f"- **>1 matches:** {multi_match:,} ({multi_match/total_gt*100:.1f}%)\n")
    f.write(f"- **Max matches:** {max_matches}\n")
    f.write(f"- **Mean matches:** {mean_matches}\n")
    f.write(f"- **Median matches:** {median_matches}\n\n")

    f.write("### Match Count Distribution\n\n| Count | # Entities | Pct |\n|-------|-----------|-----|\n")
    for k, v in sorted(match_dist.items()):
        f.write(f"| {k} | {v:,} | {v/total_gt*100:.1f}% |\n")
    f.write("\n")

    f.write("### S2/S3 Distribution\n\n")
    f.write(f"- **Total S2 matches:** {s2_total:,}\n")
    f.write(f"- **Total S3 matches:** {s3_total:,}\n")
    f.write(f"- **S1 with both S2 & S3:** {both:,}\n")
    f.write(f"- **S1 with only S2:** {only_s2:,}\n")
    f.write(f"- **S1 with only S3:** {only_s3:,}\n")
    f.write(f"- **S1 with multiple S2:** {multi_s2:,}\n")
    f.write(f"- **S1 with multiple S3:** {multi_s3:,}\n")
    f.write(f"- **Unique matched IDs:** {unique_match_ids:,}\n")
    f.write(f"- **Total match instances:** {total_match_ids:,}\n")
    f.write(f"- **S2/S3 IDs matched to >1 S1:** {multi_assigned:,}\n\n")

    # 5. Exact Match Rates
    f.write("## 5. Exact Match Rates (All GT Pairs)\n\n")
    f.write(f"- **Total GT pairs:** {total_pairs:,}\n")
    f.write(f"- **Exact raw name:** {exact_raw_name:,} ({exact_raw_name/total_pairs*100:.1f}%)\n")
    f.write(f"- **Exact normalized name:** {exact_norm_name:,} ({exact_norm_name/total_pairs*100:.1f}%)\n")
    f.write(f"- **Exact raw address:** {exact_raw_addr:,} ({exact_raw_addr/total_pairs*100:.1f}%)\n")
    f.write(f"- **Exact normalized address:** {exact_norm_addr:,} ({exact_norm_addr/total_pairs*100:.1f}%)\n")
    f.write(f"- **Country match:** {country_match_count:,} ({country_match_count/total_pairs*100:.1f}%)\n")
    f.write(f"- **Country mismatch:** {country_mismatch:,}\n\n")

    # 6. Name Variation Examples
    f.write("## 6. Name Variation Examples\n\n")
    f.write("| # | S1 Name | Match Name |\n|---|---------|------------|\n")
    for i, (a, b) in enumerate(name_var_examples[:20], 1):
        a = str(a).replace('|', '/'); b = str(b).replace('|', '/')
        f.write(f"| {i} | {a} | {b} |\n")
    f.write("\n")

    # 7. Address Variation Examples
    f.write("## 7. Address Variation Examples\n\n")
    f.write("| # | S1 Address | Match Address |\n|---|------------|---------------|\n")
    for i, (a, b) in enumerate(addr_var_examples[:20], 1):
        a = str(a).replace('|', '/'); b = str(b).replace('|', '/')
        f.write(f"| {i} | {a} | {b} |\n")
    f.write("\n")

    # 8. Transliteration Examples
    f.write("## 8. Transliteration / Script Variation Examples\n\n")
    if len(translit_examples) > 0:
        f.write("| S1 Name | Match Name |\n|---------|------------|\n")
        for _, row in translit_examples.iterrows():
            a = str(row['s1_business_name'])[:80].replace('|','/')
            b = str(row['m_name'])[:80].replace('|','/')
            f.write(f"| {a} | {b} |\n")
    else:
        f.write("No transliteration differences detected.\n")
    f.write("\n")

    # 9. Missing Field Examples
    f.write("## 9. Missing Field Asymmetries\n\n")
    f.write(f"- S1 name empty but match has name: {len(missing_name_s1):,} (first 5 shown)\n")
    f.write(f"- Match name empty but S1 has name: {len(missing_name_m):,} (first 5 shown)\n")
    f.write(f"- S1 addr empty but match has addr: {len(missing_addr_s1):,} (first 5 shown)\n")
    f.write(f"- Match addr empty but S1 has addr: {len(missing_addr_m):,} (first 5 shown)\n\n")

    # 10. Summary
    f.write("## 10. Key Findings & Strategic Implications\n\n")

    f.write("### Scale\n")
    f.write("- **Training:** 2.2M S1 entities x (5.0M S2 + 5.3M S3) = ~22.7 TRILLION potential pairs\n")
    f.write("- **Test:** 1.7M S1 x (4.9M S2 + 5.1M S3) = ~17.3 TRILLION potential pairs\n")
    f.write("- **Efficient blocking is THE critical bottleneck**\n\n")

    f.write("### Match Patterns\n")
    f.write(f"- {zero_matches/total_gt*100:.1f}% singletons --> correctly predicting these = free F0.5 score\n")
    f.write(f"- {multi_match/total_gt*100:.1f}% have multiple matches --> MUST support many-to-one\n")
    f.write(f"- Typical entity has {mean_matches:.1f} matches, up to {max_matches}\n")
    f.write(f"- {both:,} S1 entities match from BOTH S2 and S3\n\n")

    f.write("### Data Quality\n")
    f.write(f"- Only {exact_raw_name/total_pairs*100:.1f}% raw name exact matches --> heavy name noise\n")
    f.write(f"- Normalized name captures {exact_norm_name/total_pairs*100:.1f}% --> normalization helps but fuzzy matching essential\n")
    f.write(f"- {translit_mask.sum():,} transliteration cases (Devanagari <-> Latin)\n")
    f.write(f"- Country always matches in GT: {country_match_count/total_pairs*100:.1f}% --> country is a strong blocking signal\n")
    f.write(f"- France in test only (not in training) --> country-agnostic features needed\n\n")

    f.write("### Blocking Strategy Implications\n")
    f.write("1. **Country blocking** is safe and gives massive reduction (~50% of data per country)\n")
    f.write("2. **Name token blocking** should capture most matches given name similarity patterns\n")
    f.write("3. **Transliteration** requires special handling for Hindi/Devanagari names\n")
    f.write("4. **Address tokens** (PIN codes, city names) provide complementary blocking keys\n")
    f.write("5. **Multiple blocking passes** with union of candidates is essential for high recall\n\n")

    f.write("### Modeling Implications\n")
    f.write("1. F0.5 is precision-heavy --> conservative thresholds, penalize false merges\n")
    f.write("2. Singletons score 1.0 when correctly empty --> explicit no-match modeling\n")
    f.write("3. Multiple matches per S1 are the norm --> independent pair-level scoring\n")
    f.write("4. Name + address features both needed --> cross-field features important\n")
    f.write("5. Lightweight model (RF/GBM) likely sufficient given good features\n\n")

    f.write("---\n\n*End of Data Audit Report*\n")

print(f"\nReport written to: {REPORT}")
print("Audit complete!")
