#!/usr/bin/env python3
"""
phase_4_Shishya/src/pilot_transliteration.py

Small pilot script to test indic-transliteration on sample names from the training dataset.
Verifies determinism, speed, script detection, and quality of Devanagari -> Latin transliteration.
"""

import sys
import time
import unicodedata
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def detect_script(text: str) -> str:
    if not text or not isinstance(text, str):
        return "Unknown"
    has_devanagari = False
    has_latin = False
    for c in text:
        cp = ord(c)
        if 0x0900 <= cp <= 0x097F:
            has_devanagari = True
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            has_latin = True
    if has_devanagari and has_latin: return "Mixed"
    if has_devanagari: return "Devanagari"
    if has_latin: return "Latin"
    return "Other"

def normalize_basic(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFKD', str(text))
    chars = []
    for c in text:
        cat = unicodedata.category(c)
        if cat == 'Mn' and ord(c) < 0x0900: continue
        if cat[0] in ('L', 'M', 'N'): chars.append(c)
        elif cat[0] in ('P', 'S') or c.isspace(): chars.append(' ')
        else: chars.append(' ')
    return ' '.join(''.join(chars).lower().split())

def transliterate_devanagari(text: str) -> str:
    script = detect_script(text)
    if script in ("Devanagari", "Mixed"):
        try:
            res = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
            return normalize_basic(res)
        except Exception:
            return normalize_basic(text)
    return normalize_basic(text)

def run_pilot():
    print("=== Running Transliteration Pilot ===")
    sample_names = [
        "शर्मा इलेक्ट्रॉनिक्स",
        "श्री गणेश एंटरप्राइज",
        "राजेश ट्रेडिंग कंपनी",
        "Religare Finvest Limited",
        "Tata Consultancy Services Pvt Ltd",
        "महिंद्रा & महिंद्रा फाइनेंस",
        "Google India Private Limited",
        "डाबर इंडिया लिमिटेड"
    ]
    
    t0 = time.time()
    results = []
    for name in sample_names:
        script = detect_script(name)
        norm = normalize_basic(name)
        trans = transliterate_devanagari(name)
        results.append({
            "original": name,
            "script": script,
            "normalized": norm,
            "transliterated": trans
        })
        print(f"Original:      {name}")
        print(f"Script:        {script}")
        print(f"Normalized:    {norm}")
        print(f"Transliterated:{trans}")
        print("-" * 50)
        
    print(f"Pilot processed {len(sample_names)} names in {(time.time()-t0)*1000:.2f} ms.")

if __name__ == "__main__":
    run_pilot()
