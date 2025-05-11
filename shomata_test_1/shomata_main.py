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

def main(substances, temperature_input=None):
    """Fetch and calculate Shomate data for one or more substances over a temp range or single temp."""

    # Normalize substances input to a list
    substances = [substances] if isinstance(substances, str) else substances

    # Fetch data for each substance
    substance_data = fetch_multiple_substances_parallel(substances)
    results = {}

    # Handle temperature input: range or single value
    for substance, df in substance_data.items():
        if isinstance(df, str):  # Error case
            results[substance] = df
            continue

        try:
            if temperature_input is None:
                T = float(input(f"Enter temperature in K for {substance}: ").strip())
                results[substance] = shomate_properties(df, T)
            elif isinstance(temperature_input, (int, float)):
                results[substance] = shomate_properties(df, temperature_input)
            else:  # Assume it's an array-like range
                results[substance] = shomate_properties_vectorized(df, temperature_input)
        except Exception as e:
            results[substance] = f"Error during property calculation: {e}"

    return results, substance_data

