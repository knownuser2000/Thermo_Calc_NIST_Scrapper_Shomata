# -*- coding: utf-8 -*-
"""
Created on Mon May 12 08:29:13 2025

@author: Bayra
"""

# species.py

from typing import Union
import numpy as np
import pandas as pd
#from properties_shomate import shomate_properties


class Species:
    """
    Represents a chemical species with precomputed thermodynamic properties
    (ΔH, ΔS) as a function of temperature.
    """

    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.df = df.sort_values("T_used").reset_index(drop=True)
        if "T_used" not in df.columns:
            raise ValueError("DataFrame must contain 'T_used' column.")

    def __repr__(self):
        Tmin, Tmax = self.df["T_used"].min(), self.df["T_used"].max()
        return f"<Species: {self.name}, T range: {Tmin}-{Tmax} K>"

    def _interpolate(self, T: float) -> dict:
        """
        Linearly interpolate ΔH and ΔS for a given temperature T.
        """
        if not (self.df["T_used"].min() <= T <= self.df["T_used"].max()):
            raise ValueError(f"T = {T} K is outside the data range for {self.name}.")

        result = {}
        for col in ["H - H° (kJ/mol)", "S - S° (J/mol·K)"]:
            result[col] = np.interp(T, self.df["T_used"], self.df[col])
        result["T_used"] = T
        return result

    def delta_H(self, T: float) -> float:
        return self._interpolate(T)["H - H° (kJ/mol)"]

    def delta_S(self, T: float) -> float:
        return self._interpolate(T)["S - S° (J/mol·K)"]

    def properties(self, T: float) -> dict:
        """
        Return both delta H and S at temperature T.
        """
        return self._interpolate(T)

    def properties_vectorized(self, T_array: Union[list, np.ndarray]) -> pd.DataFrame:
        """
        Return a DataFrame of delta H and S across a temperature array.
        """
        return pd.DataFrame([self._interpolate(T) for T in T_array])

    def plot_property(self, prop: str, T_range: Union[list, np.ndarray]):
        """
        Plot a thermodynamic property (ΔH or ΔS) across a temperature range.
        """
        import matplotlib.pyplot as plt

        values = [self._interpolate(T)[prop] for T in T_range]
        plt.plot(T_range, values)
        plt.title(f"{prop} vs. Temperature for {self.name}")
        plt.xlabel("Temperature (K)")
        plt.ylabel(prop)
        plt.grid(True)
        plt.show()
