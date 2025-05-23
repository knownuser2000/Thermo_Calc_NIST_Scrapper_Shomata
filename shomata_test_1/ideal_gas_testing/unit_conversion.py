# -*- coding: utf-8 -*-
"""
Created on Wed May 21 13:51:04 2025

@author: Bayra
"""


def to_SI(values, unit):
    """
    Converts a given scalar or array of values from supported units to SI base units.
    Supports vectorized inputs (numpy arrays or lists) and scalars.
    """
    conversions = {
        # Pressure
        "Pa": 1,
        "kPa": 1e3,
        "MPa": 1e6,
        "GPa": 1e9,
        "bar": 1e5,
        "atm": 101325,
        "psi": 6894.76,
        "Torr": 133.322,
        "mmHg": 133.322,  # equivalent to Torr
        
        # Volume
        "m3": 1,
        "L": 1e-3,
        "mL": 1e-6,
        "cm3": 1e-6,  # cm³ to m³
        "cm³": 1e-6,
        "dm3": 1e-3,
        "dm³": 1e-3,
        
        # Mass
        "kg": 1,
        "g": 1e-3,
        "mg": 1e-6,
        
        # Temperature
        "K": ("K", lambda x: x),
        "C": ("K", lambda x: x + 273.15),
        "°C": ("K", lambda x: x + 273.15),
        "F": ("K", lambda x: (x - 32) * 5/9 + 273.15),  # Fahrenheit to Kelvin
        
        # Molar mass
        "kg/mol": 1,
        "g/mol": 1e-3,
        
        # Amount of substance
        "mol": 1,
        "kmol": 1e3,
        
        # Density
        "kg/m3": 1,
        "g/L": 1,
        "g/mL": 1000,
        
        # Concentration (molarity)
        "mol/m3": 1,
        "mol/L": 1e3,
        "mmol/L": 1,     # 1 mmol/L = 1 mol/m3
        "mM": 1,         # shorthand for mmol/L
        "M": 1e3         # Molar (mol/L)
    }
    
    if unit not in conversions:
        raise ValueError(f"Unsupported unit: {unit}")
    
    conversion = conversions[unit]
    values = np.asarray(values)
    
    if isinstance(conversion, tuple):  # temperature conversions
        return conversion[1](values)
    
    return values * conversion