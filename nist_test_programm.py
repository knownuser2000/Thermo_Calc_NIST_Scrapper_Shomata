# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:20:36 2025

@author: Bayra
"""
import requests
import concurrent.futures
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import string
import argparse


def fetch_shomate_html_async(substance: str):
    """Async fetch function for Shomate data."""
    base_url = "https://webbook.nist.gov"
    search_url = f"{base_url}/cgi/cbook.cgi?Name={substance.replace(' ', '+')}&Units=SI"
    soup = BeautifulSoup(requests.get(search_url).text, 'lxml')

    gas_link = soup.find('a', string=re.compile("Gas phase thermochemistry data"))
    if not gas_link:
        raise ValueError(f"No gas phase thermochemistry data found for '{substance}'.")

    gas_soup = BeautifulSoup(requests.get(base_url + gas_link['href']).text, 'lxml')
    for table in gas_soup.find_all('table', class_='data'):
        caption = table.find_previous_sibling('h3')
        if caption and 'Shomate Equation' in caption.text:
            return table
    raise ValueError(f"No Shomate equation table found for '{substance}'.")

def fetch_multiple_substances_parallel(substance_list):
    result = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_shomate_html_async, substance): substance for substance in substance_list}
        for future in concurrent.futures.as_completed(futures):
            substance = futures[future]
            try:
                table = future.result()
                raw_df = parse_shomate_table(table)
                clean_df = clean_shomate_dataframe(raw_df)
                result[substance] = clean_df
            except Exception as e:
                result[substance] = f"Error: {e}"
    return result

def parse_shomate_table(table):
    """Parse Shomate HTML table into a raw DataFrame (prior to cleaning)."""
    rows = table.find_all('tr')
    headers = [cell.get_text(strip=True) for cell in rows[0].find_all(['th', 'td'])][1:]

    data = {}
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        values = [cell.get_text(strip=True) or None for cell in cells[1:]]
        values += [None] * (len(headers) - len(values))  # pad if needed
        data[label] = values[:len(headers)]

    return pd.DataFrame.from_dict(data, orient='index', columns=headers)

def clean_shomate_dataframe(df):
    """Clean and transform the raw Shomate DataFrame with minimal logic change."""

    # Step 1: Reset index and shift all columns one to the right
    df_reset = df.reset_index()
    df_shifted = df_reset.shift(axis=1)

    # Step 2: Set row labels A, B, C, ...
    df_shifted.index = list(string.ascii_uppercase[:len(df_shifted)])

    # Step 3: Drop old 'index' column if present
    if 'index' in df_shifted.columns:
        df_shifted.drop(columns='index', inplace=True)

    # Step 4: Transpose to make column ranges into rows
    df_transposed = df_shifted.T

    # Step 5: Extract temperature ranges from the original index labels
    tmin_tmax = df_transposed.index.to_series().str.extract(r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)').astype(float)
    tmin_tmax.columns = ['T_min', 'T_max']

    # Step 6: Reset index to match dimensions for merge
    df_transposed = df_transposed.reset_index(drop=True)
    tmin_tmax = tmin_tmax.reset_index(drop=True)

    # Step 7: Combine temperature ranges with data
    df_transposed[['T_min', 'T_max']] = tmin_tmax

    # Step 8: Reorder columns to have T_min, T_max first
    front_cols = ['T_min', 'T_max']
    all_cols = front_cols + [col for col in df_transposed.columns if col not in front_cols]
    df_transposed = df_transposed[all_cols]

    # Step 9: Convert coefficient columns to float if they exist
    coeff_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for col in coeff_cols:
        if col in df_transposed.columns:
            df_transposed[col] = pd.to_numeric(df_transposed[col], errors='coerce')

    return df_transposed

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
        'H (kJ/mol)': H,
        'S (J/mol·K)': S,
        'T_used': T,
        'T_range': (row['T_min'], row['T_max'])
    }

def shomate_properties_vectorized(df, T_array):
    return [shomate_properties(df, T) for T in T_array]

def export_to_csv(df, filename):
    df.to_csv(filename, index=False)

def export_to_json(df, filename):
    df.to_json(filename, orient='records', indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and calculate Shomate thermodynamic properties.")
    parser.add_argument('--substances', nargs='+', required=True, help='Substance names or CAS numbers')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--temperature', type=float, help='Single temperature (K)')
    group.add_argument('--tmin', type=float, help='Minimum temperature (K)')
    
    # Only relevant if using a range
    parser.add_argument('--tmax', type=float, help='Maximum temperature (K)')
    parser.add_argument('--npoints', type=int, default=50, help='Number of temperature points')
    
    parser.add_argument('--output', type=str, help='Optional output file (JSON)')

    return parser.parse_args()


def main(substances, temperature_range=None):
    """Optimized function to fetch and process Shomate data."""
    
    # Ensure substances is a list, even if a single string is provided
    substances = [substances] if isinstance(substances, str) else substances
    
    # Fetch data for each substance in parallel
    substance_data = fetch_multiple_substances_parallel(substances)
    
    results = {}
    
    # Determine if we're calculating for a temperature range or a single temperature
    if temperature_range is not None:
        # Vectorized properties calculation for each substance over the temperature range
        for substance, df in substance_data.items():
            try:
                results[substance] = shomate_properties_vectorized(df, temperature_range)
            except Exception as e:
                print(f"Error calculating properties for {substance} over the temperature range: {e}")
    else:
        # Single temperature handling
        for substance, df in substance_data.items():
            T = temperature_range if isinstance(temperature_range, (int, float)) else float(input(f"Enter temperature for {substance} (in K): ").strip())
            try:
                results[substance] = shomate_properties(df, T)
            except Exception as e:
                print(f"Error calculating properties for {substance} at {T} K: {e}")

    return results

if __name__ == "__main__":
    substances = ["Cl2", "H2O", "7647-01-0", "O2"]
    temperature_range = np.linspace(300, 1000, 50)  # From 300K to 1000K, 50 points
    results = main(substances, temperature_range)

    # Print results (optional)
    for substance, properties in results.items():
        print(f"\nProperties for {substance}:")
        if isinstance(properties, list):  # For vectorized properties
            for prop in properties:
                print(prop)
        else:
            print(properties)
