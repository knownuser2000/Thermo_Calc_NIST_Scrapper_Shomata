# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:16:26 2025

@author: Bayra
"""

# main.py

#import numpy as np
from fetch_shoma_coeff_2 import fetch_multiple_substances_parallel
#from fetch_shoma_coeff import fetch_multiple_substances_parallel

from properties_shomata import shomate_properties, shomate_properties_vectorized
# main.py
from fetch_shoma_coeff_2 import fetch_multiple_standard_entropies

import pandas as pd


def main(substances, temperature_input=None):
    """Fetch and calculate Shomate data for one or more substances over a temp range or single temp."""

    # Normalize substances input to a list
    substances = [substances] if isinstance(substances, str) else substances

    # Fetch Shomate coefficients for each substance
    substance_data = fetch_multiple_substances_parallel(substances)

    # Fetch standard entropy values
    standard_entropies = fetch_multiple_standard_entropies(substances)

    results = {}

    # Handle temperature input: range or single value
    for substance, df in substance_data.items():
        if isinstance(df, str):  # Error fetching Shomate data
            results[substance] = df
            continue

        try:
            # Determine temperature(s)
            if temperature_input is None:
                T = float(input(f"Enter temperature in K for {substance}: ").strip())
                props = shomate_properties(df, T)
            elif isinstance(temperature_input, (int, float)):
                props = shomate_properties(df, temperature_input)
            else:  # Assume it's an array-like range
                props = shomate_properties_vectorized(df, temperature_input)

            # Subtract standard entropy if available and props is dict (not vectorized)
            S_std = standard_entropies.get(substance)

        # Add S° and S - S° to results
            if isinstance(props, dict):  # scalar case
                props["S° (J/mol·K)"] = S_std
                if isinstance(S_std, (int, float)) and "S (J/mol·K)" in props:
                    props["S - S° (J/mol·K)"] = props["S (J/mol·K)"] - S_std
                else:
                    props["S - S° (J/mol·K)"] = "Unavailable"
            
            elif isinstance(props, pd.DataFrame):  # vectorized case
                props["S° (J/mol·K)"] = S_std
                if isinstance(S_std, (int, float)) and "S (J/mol·K)" in props.columns:
                    props["S - S° (J/mol·K)"] = props["S (J/mol·K)"] - S_std
                else:
                    props["S - S° (J/mol·K)"] = "Unavailable"

            
            results[substance] = props

        except Exception as e:
            results[substance] = f"Error during property calculation: {e}"

    return results, substance_data
