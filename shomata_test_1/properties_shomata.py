# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:05:48 2025

@author: Bayra
"""

import pandas as pd
import numpy as np


def shomate_properties(df: pd.DataFrame, T: float) -> dict:
    """Calculate Cp, H, and S at a given temperature T (in K)."""
    match = df[(df['T_min'] <= T) & (df['T_max'] >= T)]
    if match.empty:
        raise ValueError(f"No coefficients found for T = {T} K.")

    row = match.iloc[0]
    T_scaled = T / 1000

    A, B, C, D, E, F, G, H0 = row[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']]

    Cp = A + B*T_scaled + C*T_scaled**2 + D*T_scaled**3 + E / T_scaled**2
    H = A*T_scaled + B*T_scaled**2/2 + C*T_scaled**3/3 + D*T_scaled**4/4 - E/T_scaled + F - H0
    S = A*np.log(T_scaled) + B*T_scaled + C*T_scaled**2/2 + D*T_scaled**3/3 - E/(2*T_scaled**2) + G

    return {
        'Cp (J/mol·K)': Cp,
        'H - H° (kJ/mol)': H,
        'S (J/mol·K)': S,
        'T_used': T,
        'T_range': (row['T_min'], row['T_max'])
    }

def shomate_properties_vectorized(df, T_array):
    return [shomate_properties(df, T) for T in T_array]
