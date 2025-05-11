# -*- coding: utf-8 -*-
"""
Created on Sun May 11 12:25:44 2025

@author: Bayra
"""

# cas_lookup.py

import requests
import re

def get_cas_number(substance_name):
    """Get CAS number from a substance name using PubChem PUG-REST."""
    try:
        # Step 1: Get CID from substance name
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{substance_name}/cids/JSON"
        cid_response = requests.get(cid_url)
        cid_response.raise_for_status()
        cids = cid_response.json().get("IdentifierList", {}).get("CID", [])

        if not cids:
            return f"Error: No CID found for '{substance_name}'."

        cid = cids[0]

        # Step 2: Get synonyms from CID
        synonyms_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        synonyms_response = requests.get(synonyms_url)
        synonyms_response.raise_for_status()
        synonyms = synonyms_response.json().get("InformationList", {}).get("Information", [])[0].get("Synonym", [])

        # Step 3: Find CAS number via regex (format: NNNNN-NN-N)
        for synonym in synonyms:
            if re.match(r"^\d{2,7}-\d{2}-\d$", synonym):
                return synonym

        return f"Error: No CAS number found for '{substance_name}'."

    except requests.RequestException as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
# =============================================================================
# 
# # Optional: run directly
# if __name__ == "__main__":
#     substances = ["Cl2", "H2O", "HCl", "O2", "chlorine", "hydrochloric acid"]
#     for name in substances:
#         cas = get_cas_number(name)
#         print(f"{name} → {cas}")
# 
# =============================================================================
