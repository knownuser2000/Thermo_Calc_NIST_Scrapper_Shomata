# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:42:30 2025

@author: Bayra
"""

import numpy as np

def normalize(frac):
    frac = np.asarray(frac, float)
    s = frac.sum()
    if s == 0: raise ValueError("Fractions sum to zero.")
    return frac / s

def to_mole_fractions(y=None, w=None, p_i=None, P=None, M_i=None):
    if y is not None:
        return normalize(y)
    if w is not None:
        if M_i is None: raise ValueError("M_i required to convert w to y.")
        w = normalize(w)
        return normalize(w / np.asarray(M_i, float))
    if p_i is not None:
        if P is None: raise ValueError("P required with p_i.")
        p_i = np.asarray(p_i, float)
        if np.any(p_i > P): raise ValueError("Partial pressures cannot exceed total pressure.")
        return normalize(p_i / P)
    raise ValueError("Must provide y, w, or p_i.")

def mole_to_mass_fractions(y, M_i):
    y, M_i = np.asarray(y, float), np.asarray(M_i, float)
    w = y * M_i
    return w / w.sum()

def avg_molar_mass(y, M_i):
    return np.dot(np.asarray(y, float), np.asarray(M_i, float))

def weighted_property(prop_i, y=None, w=None, p_i=None, P=None, M_i=None,
                      weight_by='mole', prop_basis='mole', output_basis=None):
    y = to_mole_fractions(y, w, p_i, P, M_i)
    prop_i = np.asarray(prop_i, float)
    if prop_basis not in ('mole', 'mass') or weight_by not in ('mole', 'mass'):
        raise ValueError("Invalid prop_basis or weight_by.")

    M_i = None if M_i is None else np.asarray(M_i, float)

    # Convert all properties internally to mole basis for weighting
    if prop_basis == 'mass':
        if M_i is None: raise ValueError("M_i required for mass->mole conversion.")
        prop_mol = prop_i * M_i
    else:
        prop_mol = prop_i

    # Select weights
    if weight_by == 'mole':
        weighted = np.dot(y, prop_mol)
    else:
        if M_i is None: raise ValueError("M_i required for mass weighting.")
        w_frac = mole_to_mass_fractions(y, M_i)
        weighted = np.dot(w_frac, prop_mol)

    # Convert output as requested
    if output_basis is None or output_basis == prop_basis:
        if prop_basis == 'mass':
            return weighted / avg_molar_mass(y, M_i)
        return weighted

    if output_basis == 'mole':
        return weighted
    if output_basis == 'mass':
        if prop_basis == 'mass':
            return weighted
        return weighted / avg_molar_mass(y, M_i)

    raise ValueError("output_basis must be None, 'mole', or 'mass'.")

def mixture_properties(y=None, w=None, p_i=None, P=None, M_i=None, properties=None,
                       weight_by='mole', prop_basis_dict=None, output_basis_dict=None):
    y = to_mole_fractions(y, w, p_i, P, M_i)
    M_i = None if M_i is None else np.asarray(M_i, float)
    w = mole_to_mass_fractions(y, M_i) if M_i is not None else None
    M_mix = avg_molar_mass(y, M_i) if M_i is not None else None

    result = {'y': y}
    if w is not None: result.update({'w': w, 'M_mix': M_mix})

    prop_basis_dict = prop_basis_dict or {}
    output_basis_dict = output_basis_dict or {}

    if properties:
        for k, prop in properties.items():
            result[f"{k}_mix"] = weighted_property(
                prop, y=y, w=w, M_i=M_i, weight_by=weight_by,
                prop_basis=prop_basis_dict.get(k, 'mole'),
                output_basis=output_basis_dict.get(k)
            )
    return result

def normalize_fraction(frac):
    """Normalize fractions to sum to 1, handle None."""
    frac = np.asarray(frac, dtype=float)
    s = np.sum(frac)
    if s == 0:
        raise ValueError("Fractions sum to zero.")
    return frac / s

def convert_to_mole_fractions(y=None, w=None, p_i=None, P=None, M_i=None):
    """
    Convert given composition to mole fractions (y).
    Provide exactly one of y, w, p_i.
    
    Parameters:
        y : array-like or None - mole fractions (should sum to 1)
        w : array-like or None - mass fractions (should sum to 1)
        p_i : array-like or None - partial pressures [Pa]
        P : float or None - total pressure [Pa], required if p_i given
        M_i : array-like or None - molar masses [kg/mol], required if w given
    
    Returns:
        y : numpy array of mole fractions (sum to 1)
    """
    if y is not None:
        y = normalize_fraction(y)
        return y
    elif w is not None:
        if M_i is None:
            raise ValueError("Molar masses M_i required to convert mass fractions w to mole fractions y.")
        w = normalize_fraction(w)
        M_i = np.asarray(M_i, dtype=float)
        y_unnormalized = w / M_i
        y = normalize_fraction(y_unnormalized)
        return y
    elif p_i is not None:
        if P is None:
            raise ValueError("Total pressure P required with partial pressures p_i.")
        p_i = np.asarray(p_i, dtype=float)
        if np.any(p_i > P):
            raise ValueError("Partial pressures cannot exceed total pressure P.")
        y = p_i / P
        return normalize_fraction(y)
    else:
        raise ValueError("Must provide one of mole fractions (y), mass fractions (w), or partial pressures (p_i).")

def mole_fractions_to_mass_fractions(y, M_i):
    """
    Convert mole fractions to mass fractions.
    
    Parameters:
        y : array-like - mole fractions
        M_i : array-like - molar masses [kg/mol]
    
    Returns:
        w : numpy array of mass fractions
    """
    y = np.asarray(y, dtype=float)
    M_i = np.asarray(M_i, dtype=float)
    w_unnormalized = y * M_i
    w = w_unnormalized / np.sum(w_unnormalized)
    return w

def average_molar_mass(y, M_i):
    """
    Calculate average molar mass of mixture.
    
    Parameters:
        y : mole fractions
        M_i : molar masses [kg/mol]
    
    Returns:
        M_mix : float average molar mass [kg/mol]
    """
    y = np.asarray(y, dtype=float)
    M_i = np.asarray(M_i, dtype=float)
    return np.sum(y * M_i)

def average_Cp(y, Cp_i):
    """
    Calculate average molar heat capacity Cp of mixture.
    
    Parameters:
        y : mole fractions
        Cp_i : array-like of Cp values [J/(mol·K)]
    
    Returns:
        Cp_mix : float average Cp [J/(mol·K)]
    """
    y = np.asarray(y, dtype=float)
    Cp_i = np.asarray(Cp_i, dtype=float)
    return np.sum(y * Cp_i)

# Example usage:
props = {
    'H': [10000, 20000, 30000],   # J/mol
    'Cp': [450, 500, 550],        # J/(kg·K)
}
prop_basis = {'H': 'mole', 'Cp': 'mass'}
output_basis = {'H': 'mole', 'Cp': 'mass'}

res = mixture_properties(
    w=[0.2, 0.3, 0.5],
    M_i=[0.02, 0.03, 0.04],
    properties=props,
    weight_by='mole',
    prop_basis_dict=prop_basis,
    output_basis_dict=output_basis
)
print(res)
