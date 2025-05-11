# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:03:25 2025

@author: Bayra
"""

import requests
import concurrent.futures
from bs4 import BeautifulSoup
import pandas as pd
import re
import string

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
