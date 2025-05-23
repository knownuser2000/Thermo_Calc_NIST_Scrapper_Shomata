# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:30:26 2025

@author: Bayra
"""

# mixture.py

import numpy as np

class Mixture:
    """
    Represents an ideal gas mixture of chemical species with given mole fractions.
    Can calculate mixture Cp, H, S, and U as mole-fraction-weighted averages.
    """

    def __init__(self, species: list, mole_fractions: list[float]):
        """
        Initialize with a list of Species objects and their corresponding mole fractions.
        """
        if len(species) != len(mole_fractions):
            raise ValueError("Species and mole fraction lists must be the same length.")
        if not np.isclose(sum(mole_fractions), 1.0):
            raise ValueError("Mole fractions must sum to 1.")

        self.species = species
        self.x = mole_fractions

    def Cp(self, T: float) -> float:
        """
        Returns the mixture heat capacity at constant pressure (J/mol·K).
        """
        return sum(xi * sp.Cp(T) for sp, xi in zip(self.species, self.x))

    def H(self, T: float) -> float:
        """
        Returns the mixture enthalpy (J/mol) at temperature T.
        """
        return sum(xi * sp.H(T) for sp, xi in zip(self.species, self.x))

    def S(self, T: float, p: float = None, p0: float = 1e5) -> float:
        """
        Returns the total entropy (J/mol·K) of the mixture at T.
        Includes mixing entropy and optional pressure correction.
        """
        base = sum(xi * sp.S(T) for sp, xi in zip(self.species, self.x))  # pure-species entropy
        mix = -8.3145 * sum(xi * np.log(xi) for xi in self.x if xi > 0)    # ideal mixing entropy
        pressure_term = 0 if p is None else -8.3145 * np.log(p / p0)      # pressure correction
        return base + mix + pressure_term

    def U(self, T: float) -> float:
        """
        Returns internal energy (J/mol) as U = H - R·T for ideal gases.
        """
        return self.H(T) - 8.3145 * T
