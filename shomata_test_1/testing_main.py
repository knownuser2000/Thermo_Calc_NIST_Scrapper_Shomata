# -*- coding: utf-8 -*-
"""
Created on Sun May 11 10:18:53 2025

@author: Bayra
"""

# main.py

from shomata_main import main
import numpy as np
#from fetch_shoma_coeff import fetch_multiple_substances_parallel


if __name__ == "__main__":
    #substances = ["Cl2", "H2O", "7647-01-0", "O2"]        # Can be single substance: "O2"
    #print(fetch_multiple_substances_parallel(["HCl"]))

    substances = ["Cl2", "H2O", "HCl", "O2"]        # Can be single substance: "O2"

    temperature_range = np.linspace(500, 1000, 50)        # Can be single value: 298.15

    results,substance_data = main(substances, temperature_range)

    for substance, props in results.items():
        print(f"\nProperties for {substance}:")
        if isinstance(props, (list, np.ndarray)):
            for entry in props:
                print(entry)
        else:
            print(props)
