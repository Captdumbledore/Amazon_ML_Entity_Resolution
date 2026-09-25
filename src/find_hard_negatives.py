#!/usr/bin/env python3
import pandas as pd
import json

BASE = r'c:\Users\jisto\Documents\Amazon_ML\Dataset\student_resource\dataset'

# Load GT
gt = pd.read_csv(f'{BASE}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
gt_pairs = set()
for _, row in gt.iterrows():
    if row['matched_entity_ids']:
        for m in row['matched_entity_ids'].split(','):
            gt_pairs.add((row['source1_entity_id'], m.strip()))

# Load small sample of S1 and S2
s1 = pd.read_csv(f'{BASE}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False).sample(50000, random_state=42)
s2 = pd.read_csv(f'{BASE}/train/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False)

s1['norm'] = s1['business_name'].str.lower().str.strip()
s2['norm'] = s2['business_name'].str.lower().str.strip()

merged = pd.merge(s1, s2, on=['norm', 'country'], suffixes=('_s1', '_s2'))

hard_negatives = []
for _, row in merged.iterrows():
    if (row['entity_id_s1'], row['entity_id_s2']) not in gt_pairs:
        if len(row['norm']) > 8: # Avoid generic short names
            hard_negatives.append({
                's1_name': row['business_name_s1'],
                's2_name': row['business_name_s2'],
                's1_addr': row['business_address_s1'],
                's2_addr': row['business_address_s2']
            })
    if len(hard_negatives) >= 20:
        break

with open('reports/hard_negatives.json', 'w', encoding='utf-8') as f:
    json.dump(hard_negatives, f, indent=2)

print("Hard negatives found.")
