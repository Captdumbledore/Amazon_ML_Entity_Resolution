# Amazon ML Challenge 2026 - Data Audit Report

Generated: 2026-09-25T00:42:42.123503

---

## 1. Dataset File Overview

| File | Rows | Unique IDs | Dup IDs | Empty Names | Empty Addrs | Empty Country |
|------|------|-----------|---------|-------------|-------------|---------------|
| train_source1 | 2,206,821 | 2,206,821 | 0 | 0 | 0 | 0 |
| train_source2 | 5,034,616 | 5,034,616 | 0 | 0 | 168,967 | 0 |
| train_source3 | 5,285,603 | 5,285,603 | 0 | 0 | 175,916 | 0 |
| test_source1 | 1,732,544 | 1,732,544 | 0 | 0 | 0 | 0 |
| test_source2 | 4,887,273 | 4,887,273 | 0 | 0 | 129,408 | 0 |
| test_source3 | 5,082,316 | 5,082,316 | 0 | 0 | 136,098 | 0 |

## 2. Country Distribution

### train_source1

| Country | Count | Pct |
|---------|-------|-----|
| US | 1,323,633 | 60.0% |
| India | 883,188 | 40.0% |

### train_source2

| Country | Count | Pct |
|---------|-------|-----|
| US | 3,016,817 | 59.9% |
| India | 2,017,799 | 40.1% |

### train_source3

| Country | Count | Pct |
|---------|-------|-----|
| US | 3,170,056 | 60.0% |
| India | 2,115,547 | 40.0% |

### test_source1

| Country | Count | Pct |
|---------|-------|-----|
| India | 809,986 | 46.8% |
| US | 663,106 | 38.3% |
| France | 259,452 | 15.0% |

### test_source2

| Country | Count | Pct |
|---------|-------|-----|
| India | 2,312,565 | 47.3% |
| US | 1,871,330 | 38.3% |
| France | 703,378 | 14.4% |

### test_source3

| Country | Count | Pct |
|---------|-------|-----|
| India | 2,405,000 | 47.3% |
| US | 1,945,701 | 38.3% |
| France | 731,615 | 14.4% |

## 3. Business Name & Address Length Distributions

| Dataset | Field | Mean | Median | Min | Max | P95 |
|---------|-------|------|--------|-----|-----|-----|
| train_source1 | name | 24.0 | 24.0 | 3 | 105 | 37.0 |
| train_source1 | addr | 52.1 | 41.0 | 11 | 256 | 103.0 |
| train_source2 | name | 25.1 | 25.0 | 2 | 104 | 40.0 |
| train_source2 | addr | 47.8 | 37.0 | 8 | 249 | 97.0 |
| train_source3 | name | 25.2 | 25.0 | 2 | 123 | 42.0 |
| train_source3 | addr | 48.3 | 42.0 | 2 | 240 | 92.0 |
| test_source1 | name | 23.8 | 24.0 | 3 | 92 | 36.0 |
| test_source1 | addr | 57.2 | 50.0 | 11 | 268 | 105.0 |
| test_source2 | name | 25.7 | 25.0 | 2 | 102 | 42.0 |
| test_source2 | addr | 51.8 | 43.0 | 5 | 269 | 99.0 |
| test_source3 | name | 25.7 | 25.0 | 2 | 103 | 42.0 |
| test_source3 | addr | 50.1 | 44.0 | 5 | 267 | 95.0 |

## 4. Ground Truth Analysis

- **Total S1 entities:** 2,206,821
- **Singletons (0 matches):** 123,247 (5.6%)
- **1 match:** 119,157 (5.4%)
- **>1 matches:** 1,964,417 (89.0%)
- **Max matches:** 11
- **Mean matches:** 3.461
- **Median matches:** 3.0

### Match Count Distribution

| Count | # Entities | Pct |
|-------|-----------|-----|
| 0 | 123,247 | 5.6% |
| 1 | 119,157 | 5.4% |
| 2 | 375,212 | 17.0% |
| 3 | 530,841 | 24.1% |
| 4 | 484,115 | 21.9% |
| 5 | 321,957 | 14.6% |
| 6 | 164,868 | 7.5% |
| 7 | 63,968 | 2.9% |
| 8 | 18,680 | 0.8% |
| 9 | 4,205 | 0.2% |
| 10 | 534 | 0.0% |
| 11 | 37 | 0.0% |

### S2/S3 Distribution

- **Total S2 matches:** 3,693,619
- **Total S3 matches:** 3,944,746
- **S1 with both S2 & S3:** 1,776,047
- **S1 with only S2:** 143,029
- **S1 with only S3:** 164,498
- **S1 with multiple S2:** 1,129,968
- **S1 with multiple S3:** 1,224,128
- **Unique matched IDs:** 7,638,365
- **Total match instances:** 7,638,365
- **S2/S3 IDs matched to >1 S1:** 0

## 5. Exact Match Rates (All GT Pairs)

- **Total GT pairs:** 7,638,365
- **Exact raw name:** 354,118 (4.6%)
- **Exact normalized name:** 1,674,157 (21.9%)
- **Exact raw address:** 170,074 (2.2%)
- **Exact normalized address:** 631,369 (8.3%)
- **Country match:** 7,638,365 (100.0%)
- **Country mismatch:** 0

## 6. Name Variation Examples

| # | S1 Name | Match Name |
|---|---------|------------|
| 1 | Maure Williams Colombier Inc | Maure Wilblims Colombier Inc |
| 2 | Maure Williams Colombier Inc | Maure Williams Colombier |
| 3 | Maure Williams Colombier Inc | Dréxkor |
| 4 | Maure Williams Colombier Inc | maurewilliamscolombier.com |
| 5 | Maure Williams Colombier Inc | Maure Williams Inc Center |
| 6 | Raj Investments LLP | ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி |
| 7 | Raj Investments LLP | Raj Investments எல்எல்பி |
| 8 | Raj Investments LLP | ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி |
| 9 | Dahlia Power Reliable Scientific LLC | Dahlia Power Reliable |
| 10 | Dahlia Power Reliable Scientific LLC | Dahlia Power Reliable Scientific |
| 11 | Dahlia Power Reliable Scientific LLC | Dahlia Ponr Reliable Scientific LLC |
| 12 | Ss Food Private Limited | एसएस फूड प्राइवेट लिमिटेड |
| 13 | Ss Food Private Limited | एसएस फूड प्राइवेट लिमिटेड |
| 14 | Ss Food Private Limited | एसएस फूड प्राइवेट लिमिटेड |
| 15 | Payne Enterprises | Payne Énterprises |
| 16 | Payne Enterprises | Payne Enterpires |
| 17 | Payne Enterprises | PAYNE-ENRTPRMISES |
| 18 | Payne Enterprises | Payne Etrepndiels |
| 19 | Payne Enterprises | Payne Énterprises |
| 20 | Payne Enterprises | Payne Enterprises  LLC |

## 7. Address Variation Examples

| # | S1 Address | Match Address |
|---|------------|---------------|
| 1 | 85 Wayne Avenue, Ticonderoga, NY |  |
| 2 | 85 Wayne Avenue, Ticonderoga, NY |  |
| 3 | 85 Wayne Avenue, Ticonderoga, NY | 85 Wanye Avenue, Ticonderoga Townshiip, New York |
| 4 | 85 Wayne Avenue, Ticonderoga, NY | Wayne Ave, Ticonderoga Townshiip, New York |
| 5 | 85 Wayne Avenue, Ticonderoga, NY |  |
| 6 | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu |
| 7 | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu |
| 8 | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu | 6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN |
| 9 | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu | 6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு |
| 10 | 630 45th Terrace, Kansas City, MO | KANSAS CITY, MO, 630 45ND TERRACE, null |
| 11 | 630 45th Terrace, Kansas City, MO | 45ND TERRACE, null, KANSAS CITY, MO |
| 12 | 630 45th Terrace, Kansas City, MO | Missouri, 630 45th Terrace, Kansas City |
| 13 | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh | AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश |
| 14 | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh | AF-0684, Uttar Pradesh, GHAZIABAD, 9487203 |
| 15 | Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh | Af-684, Ghaziabad, UP |
| 16 | 3315 Fremont Street, Peoria, IL | 3315 FREMONT ST, PEORIA, IL |
| 17 | 3315 Fremont Street, Peoria, IL | 3315 FREMONT ST, PEORIA, IL |
| 18 | 3315 Fremont Street, Peoria, IL | 3315 FREMONT SAINT, PEORIA, IL |
| 19 | 3315 Fremont Street, Peoria, IL | 3315 Fremont St, Peoria, Illinois |
| 20 | 3315 Fremont Street, Peoria, IL | 3315 Fremont Street, Peoria, Illinois |

## 8. Transliteration / Script Variation Examples

| S1 Name | Match Name |
|---------|------------|
| Ss Food Private Limited | एसएस फूड प्राइवेट लिमिटेड |
| Ss Food Private Limited | एसएस फूड प्राइवेट लिमिटेड |
| Ss Food Private Limited | एसएस फूड प्राइवेट लिमिटेड |
| Red Ventures Private Limited | रेड वेंचर्स प्राइवेट लिमिटेड |
| Hotel Enterprises Limited | होटल एंटरप्राइजेज लिमिटेड |
| Swastik Om Solutions LLP | स्वस्तिक ॐ सॉल्यूशंस एलएलपी |
| Swastik Om Solutions LLP | स्वस्तिक ॐ सॉल्यूशंस एलएलपी |
| New Solutions | न्यू सॉल्यूशंस |
| New Solutions | न्यू सॉल्यूशंस |
| New Solutions | न्यू सॉल्यूशंस |

## 9. Missing Field Asymmetries

- S1 name empty but match has name: 0 (first 5 shown)
- Match name empty but S1 has name: 0 (first 5 shown)
- S1 addr empty but match has addr: 0 (first 5 shown)
- Match addr empty but S1 has addr: 5 (first 5 shown)

## 10. Key Findings & Strategic Implications

### Scale
- **Training:** 2.2M S1 entities x (5.0M S2 + 5.3M S3) = ~22.7 TRILLION potential pairs
- **Test:** 1.7M S1 x (4.9M S2 + 5.1M S3) = ~17.3 TRILLION potential pairs
- **Efficient blocking is THE critical bottleneck**

### Match Patterns
- 5.6% singletons --> correctly predicting these = free F0.5 score
- 89.0% have multiple matches --> MUST support many-to-one
- Typical entity has 3.5 matches, up to 11
- 1,776,047 S1 entities match from BOTH S2 and S3

### Data Quality
- Only 4.6% raw name exact matches --> heavy name noise
- Normalized name captures 21.9% --> normalization helps but fuzzy matching essential
- 312,725 transliteration cases (Devanagari <-> Latin)
- Country always matches in GT: 100.0% --> country is a strong blocking signal
- France in test only (not in training) --> country-agnostic features needed

### Blocking Strategy Implications
1. **Country blocking** is safe and gives massive reduction (~50% of data per country)
2. **Name token blocking** should capture most matches given name similarity patterns
3. **Transliteration** requires special handling for Hindi/Devanagari names
4. **Address tokens** (PIN codes, city names) provide complementary blocking keys
5. **Multiple blocking passes** with union of candidates is essential for high recall

### Modeling Implications
1. F0.5 is precision-heavy --> conservative thresholds, penalize false merges
2. Singletons score 1.0 when correctly empty --> explicit no-match modeling
3. Multiple matches per S1 are the norm --> independent pair-level scoring
4. Name + address features both needed --> cross-field features important
5. Lightweight model (RF/GBM) likely sufficient given good features

---

*End of Data Audit Report*
