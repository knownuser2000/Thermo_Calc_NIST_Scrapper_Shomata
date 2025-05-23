# -*- coding: utf-8 -*-
"""
Created on Sun May 18 12:44:24 2025

@author: Bayra
"""
import os
import pandas as pd
from fetch_shoma_coeff_2 import fetch_multiple_substances_parallel, fetch_multiple_standard_entropies
from properties_shomata import shomate_properties, shomate_properties_vectorized


# ==== Data I/O (CSV version) ====

def save_raw_data_to_file(name, coeffs, S_std, directory="nist_saved_data"):
    os.makedirs(directory, exist_ok=True)
    df = pd.DataFrame(coeffs)
    df["S_std"] = S_std  # Add S° to every row
    path = os.path.join(directory, f"{name}.csv")
    df.to_csv(path, index=False)

def load_local_data(name, directory="nist_saved_data"):
    path = os.path.join(directory, f"{name}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        S_std = df["S_std"].iloc[0]
        coeffs = df.drop(columns=["S_std"])
        return {"coeffs": coeffs, "S_std": S_std}
    return None


# ==== Property Calculation ====

def calculate_properties(substance, coeffs, S_std, T):
    if isinstance(T, (int, float)):
        props = shomate_properties(coeffs, T)
        if isinstance(props, dict):
            props = pd.DataFrame([props])
    else:
        props_list = shomate_properties_vectorized(coeffs, T)
        if isinstance(props_list, list) and all(isinstance(p, dict) for p in props_list):
            props = pd.DataFrame(props_list)
        else:
            return f"Invalid result format from vectorized function for {substance}"

    props["S° (J/mol·K)"] = S_std
    if "S (J/mol·K)" in props.columns:
        props["S - S° (J/mol·K)"] = props["S (J/mol·K)"] - S_std
    props = props.drop(columns="T_range", errors="ignore")

    return props


# ==== Species Class ====

class Species:
    def __init__(self, name, coeff_df, interp_df):
        self.name = name
        self.coeff_df = coeff_df
        self.interp_df = interp_df

    def shomate_properties(self, T):
        return shomate_properties(self.coeff_df, T)

    def interpolation_properties(self, T):
        row = self.interp_df[self.interp_df["T_used"] == T]
        if row.empty:
            raise ValueError(f"T={T} not found in interpolation table.")
        return row.iloc[0].to_dict()

    def get_Cp(self, T):
        return self.shomate_properties(T)["Cp (J/mol·K)"]

    def delta_H_from_interp(self, T):
        row = self.interpolation_properties(T)
        return row["H - H° (kJ/mol)"]

    def delta_S_from_interp(self, T):
        row = self.interpolation_properties(T)
        return row["S - S° (J/mol·K)"]

    def delta_G_from_interp(self, T):
        dH = self.delta_H_from_interp(T) * 1000  # J/mol
        dS = self.delta_S_from_interp(T)         # J/mol·K
        dG = dH - T * dS
        return dG / 1000  # kJ/mol


# ==== Main Program Entry Point ====

def main(substances, temperature_input=None):
    substances = [substances] if isinstance(substances, str) else substances

    local_data = {}
    missing_substances = []

    for s in substances:
        data = load_local_data(s)
        if data:
            local_data[s] = data
        else:
            missing_substances.append(s)

    fetched_data = fetch_multiple_substances_parallel(missing_substances)
    fetched_entropies = fetch_multiple_standard_entropies(missing_substances)

    substance_data = {
        **{k: v["coeffs"] for k, v in local_data.items()},
        **fetched_data
    }

    standard_entropies = {
        **{k: v["S_std"] for k, v in local_data.items()},
        **fetched_entropies
    }

    results = {}
    dataframes = {}
    species_objects = {}

    for substance, coeffs in substance_data.items():
        if isinstance(coeffs, str):
            results[substance] = coeffs
            continue

        try:
            if temperature_input is None:
                T = float(input(f"Enter temperature in K for {substance}: ").strip())
            elif isinstance(temperature_input, (list, range, tuple)):
                T = temperature_input
            else:
                T = [temperature_input]

            S_std = standard_entropies.get(substance)
            save_raw_data_to_file(substance, coeffs, S_std)

            props = calculate_properties(substance, coeffs, S_std, T)
            results[substance] = props
            dataframes[substance] = props

            interp_df = props.copy()
            species = Species(name=substance, coeff_df=coeffs, interp_df=interp_df)
            species_objects[substance] = species

        except Exception as e:
            results[substance] = f"Error during property calculation: {e}"

    return results, dataframes, species_objects

# =============================================================================
# 
# # ==== Example Usage ====
# 
# if __name__ == "__main__":
#     substances = ["Cl2", "H2O", "HCl", "O2"]
#     T = range(500, 1000, 1)
# 
#     results, dfs, species_objs = main(substances, T)
# 
#     for name, species in species_objs.items():
#         print(f"\n--- {name} at T=700 K ---")
#         print(species.get_Cp(700.15))
#         print(species.shomate_properties(800))
# 
# =============================================================================
