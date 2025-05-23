# -*- coding: utf-8 -*-
"""
Created on Sun May 11 10:18:53 2025

@author: Bayra
"""

# main.py

from shomata_main import main
import numpy as np
from fetch_shoma_coeff_2 import get_standard_entropy
import pandas as pd

if __name__ == "__main__":
    #substances = ["Cl2", "H2O", "7647-01-0", "O2"]        # Can be single substance: "O2"
    #print(fetch_multiple_substances_parallel(["HCl"]))

    substances = ["Cl2", "H2O", "HCl", "O2"]        # Can be single substance: "O2"

    temperature_range = np.linspace(500, 1000, 50)        # Can be single value: 298.15
    #temperature_range = 298
    
    results,substance_data = main(substances, temperature_range)
    
    #entropy = get_standard_entropy("7647-01-0")
    #print(f"\Entropy for {entropy}:")



    for substance, props in results.items():
        print(f"\nProperties for {substance}:")
        if isinstance(props, (list, np.ndarray)):
            for entry in props:
                print(entry)
        else:
            print(props)
            
    df_test = pd.DataFrame(results['Cl2'])  # Assuming 'key1' contains the list of dictionaries

