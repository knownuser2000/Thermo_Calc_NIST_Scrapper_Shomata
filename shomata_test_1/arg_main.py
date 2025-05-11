# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:15:27 2025

@author: Bayra
"""

# arg_main.py

import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
from shomata_main import main
from export_shomata import export_to_csv, export_to_json

def parse_args():
    parser = argparse.ArgumentParser(description="Shomate property calculator")
    parser.add_argument("-s", "--substances", nargs="+", required=True, help="List of substances or CAS numbers")
    parser.add_argument("-t", "--temperature", type=float, help="Single temperature in Kelvin")
    parser.add_argument("-tr", "--temperature_range", nargs=3, type=float,
                        help="Temperature range: start end steps (e.g., 300 1000 20)")
    parser.add_argument("--save", choices=["csv", "json", "both"], help="Save results in CSV, JSON, or both formats")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.temperature_range:
        temp_input = np.linspace(*args.temperature_range)
    elif args.temperature:
        temp_input = args.temperature
    else:
        temp_input = None

    results, substance_data = main(args.substances, temp_input)

    output_dir = "shomate_outputs"
    if args.save:
        os.makedirs(output_dir, exist_ok=True)

    for substance, props in results.items():
        print(f"\nProperties for {substance}:")

        if isinstance(props, (list, np.ndarray)):
            for entry in props:
                print(entry)
            df = pd.DataFrame(props)
        elif isinstance(props, dict):
            print(props)
            df = pd.DataFrame([props])
        else:
            print(f"{substance}: {props}")  # Likely an error message
            continue

        if args.save:
            safe_name = Path(substance.replace(" ", "_").replace("/", "-"))
            if args.save in ("csv", "both"):
                export_to_csv(df, f"{output_dir}/{safe_name}_shomate_properties.csv")
            if args.save in ("json", "both"):
                export_to_json(df, f"{output_dir}/{safe_name}_shomate_properties.json")


