# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:13:06 2025

@author: Bayra
"""

# export.py
import pandas as pd

def export_to_csv(df, filename):
    df.to_csv(filename, index=False, encoding="utf-8")

def export_to_json(df, filename):
    df.to_json(filename, orient='records', indent=2, force_ascii=False)
