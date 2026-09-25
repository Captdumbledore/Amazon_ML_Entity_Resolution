# Amazon ML Challenge 2026 - Deep Dataset Reconnaissance Summary

### DATASET SCALE
- **S1:** 2,206,821 (train) / 1,732,544 (test)
- **S2:** 5,034,616 (train) / 4,887,273 (test)
- **S3:** 5,285,603 (train) / 5,082,316 (test)
*(Note: S1 × S2 = 11.1 Trillion possible pairs in train alone. Blocking is absolutely mandatory).*

### GROUND TRUTH
- **0 matches:** 123,247 (5.6%)
- **1 match:** 119,157 (5.4%)
- **2 matches:** 375,212 (17.0%)
- **3 matches:** 530,841 (24.1%)
- **4+ matches:** 1,058,364 (48.0%) *(Max observed: 11)*

- **S2-only:** 143,029
- **S3-only:** 164,498
- **Both:** 1,776,047 *(The vast majority of entities match into both sources simultaneously)*

### DATA QUALITY
- **Missing names:** 0 (across all sources)
- **Missing addresses:** 
  - S1: 0 
  - S2: 168,967 (train), 129,408 (test) 
  - S3: 175,916 (train), 136,098 (test)
- **Duplicate IDs:** 0 (IDs are unique)
- **Duplicate names/addresses:** Extremely high. Many exact identical names (e.g., generic company names, franchises) exist across different addresses.

### EXACT MATCH COVERAGE (Against Ground Truth)
- **Raw name:** 4.6%
- **Normalized name:** 21.9% *(lower-cased, punctuation stripped)*
- **Raw address:** 2.2%
- **Normalized address:** 8.3%
- **Name + address:** < 2.0%
*(Conclusion: Exact string matching is disastrously inadequate for this dataset).*

---

### IMPORTANT DISCOVERIES
1. **Strict Country Boundary:** 100.0% of ground-truth matches share the exact same country. Country-level partitioning is completely safe and halves the search space.
2. **Missing Data Asymmetry:** S1 is the "clean" source (no missing addresses). S2 and S3 have hundreds of thousands of missing addresses, meaning we must rely entirely on names for those records.
3. **The Transliteration Barrier:** Over 300,000 true matches involve Devanagari script (Hindi) on one side and Latin script (English) on the other. 
4. **Many-to-One Dominates:** 89% of S1 entities match multiple target records. A 1-to-1 matching assumption will fail; we must score and accept pairs independently.
5. **Address Noise is Extreme:** Addresses suffer from component reordering ("Kansas City, 630 45th Terrace" vs "630 45th Terrace, Kansas City"), translation, and missing PIN codes.

### BIGGEST FALSE-POSITIVE RISKS
1. **Chains / Franchises:** Businesses with the exact same name (e.g., "McDonald's", "Pediatric Medicine PLLC") but entirely different addresses/cities.
2. **Generic Names + Missing Addresses:** If an S2 record has the generic name "New Solutions" and a missing address, we risk incorrectly merging it with an unrelated S1 "New Solutions".
3. **Over-Aggressive Normalization:** Stripping too many tokens (e.g., "LLC", "Pvt") might make distinct businesses look identical.
4. **Shared Office Buildings / Malls:** Different businesses sharing the exact same address string.
5. **S2/S3 Internal Duplicates:** Predicting two S2 rows as matches to an S1 row when they actually represent the same physical entity duplicated in S2.

### BIGGEST FALSE-NEGATIVE RISKS
1. **Transliteration Failures:** Treating "एसएस फूड प्राइवेट लिमिटेड" and "Ss Food Private Limited" as different entities because a standard string distance metric (like Levenshtein) scores them as 0% similar.
2. **Partial / Truncated Names:** "Dahlia Power Reliable Scientific LLC" (S1) vs "Dahlia Power Reliable" (S2).
3. **Address Component Shuffling:** Standard exact/prefix matching fails completely when the city is moved from the end of the string to the beginning.
4. **Acronyms:** "IBM" vs "International Business Machines" or "Ss Food Private Limited" vs "SS Food Pvt Ltd".
5. **Strict Blocking:** If we demand an exact token overlap during candidate generation, we will instantly lose 70%+ of the true matches.

### COMPUTATIONAL BOTTLENECKS
1. **Candidate Generation (Blocking) Explosion:** Attempting an inner join on normalized names results in a memory explosion (OOM) because generic names map to hundreds of thousands of targets.
2. **String Distance Calculation:** Computing expensive metrics (Levenshtein, Jaro-Winkler) on tens of millions of candidate pairs will take hours if not vectorized/optimized.
3. **TF-IDF Vectorization:** Building sparse matrices for ~12.5M rows requires careful memory management, and brute-force cosine similarity will fail without an index (e.g., FAISS).

---

### MOST PROMISING FIRST EXPERIMENTS

**1. TF-IDF Character N-Gram Blocking**
- **Hypothesis:** Representing names as character n-grams (e.g., 3-grams) and using sparse cosine similarity for retrieval will safely return >90% of true matches without the memory explosion of exact-token joins.
- **Method:** Fit a TF-IDF vectorizer on all names. Query S1 against S2/S3 using highly optimized sparse matrix dot products (or FAISS), keeping the Top-K closest names.
- **Metric:** Candidate Recall (Percentage of ground truth pairs retrieved in the Top K candidates).
- **Why it is worth testing:** Solves the exact-match explosion bottleneck while handling typos, partial matches, and missing suffixes gracefully.

**2. Transliteration Normalization pass**
- **Hypothesis:** Converting Devanagari text to Latin script before blocking and feature extraction will significantly boost recall for Indian businesses.
- **Method:** Apply a fast Python transliteration library (e.g., `indic-transliteration` or `polyglot`) to all Hindi strings to map them to English phonetics, then run baseline matching.
- **Metric:** Recall lift specifically on the subset of entities containing Devanagari characters.
- **Why it is worth testing:** It addresses ~300,000 known true matches that are completely invisible to standard string similarity functions.

**3. Tree-based Pairwise Classifier (RF/XGBoost)**
- **Hypothesis:** A tree-based model trained on multiple string-distance features will accurately separate true matches from hard-negative false positives.
- **Method:** Generate candidate pairs. Compute 10-15 features (Name Jaro-Winkler, Address token overlap, length differences, missing flags). Train an XGBoost/Random Forest model.
- **Metric:** F0.5 score on a leakage-safe (GroupKFold by S1) validation set.
- **Why it is worth testing:** Because the competition metric (F0.5) heavily penalizes false positives, we need a model that can learn non-linear decision boundaries—e.g., learning that a 90% name match is a "YES" if the address matches 50%, but a "NO" if the address is completely different (franchise risk).
