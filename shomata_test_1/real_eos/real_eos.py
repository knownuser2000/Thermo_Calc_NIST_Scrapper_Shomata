# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:01:25 2025

@author: Bayra
"""

from abc import ABC, abstractmethod
import numpy as np
import sympy as sp
import pandas as pd

class InputParser:
    """
    Utility class to parse various input types into a consistent format.
    Supports scalar values, numpy arrays, pandas Series/DataFrames, and dictionaries.
    """
    @staticmethod
    def parse(value):
        if isinstance(value, (float, int, sp.Basic)):
            return value
        elif isinstance(value, dict):
            return value.get('value', None)
        elif isinstance(value, pd.Series):
            return value.values
        elif isinstance(value, pd.DataFrame):
            return value.to_numpy()
        elif isinstance(value, (np.ndarray, list, tuple)):
            return np.array(value)
        else:
            raise TypeError(f"Unsupported input type: {type(value)}")

class BaseEOS(ABC):
    """
    Abstract base class for cubic equations of state.
    Provides interface and common methods to compute compressibility factor (Z),
    and calculate pressure.
    """
    def __init__(self, a, b, R=8.314):
        self.a = a
        self.b = b
        self.R = R

    @abstractmethod
    def alpha(self, T):
        """Temperature-dependent alpha function for attractive parameter a."""
        pass

    @abstractmethod
    def A(self, T, P):
        """Reduced attraction parameter A = a*alpha*P / (R^2 * T^2)."""
        pass

    @abstractmethod
    def B(self, T, P):
        """Reduced repulsion parameter B = b*P / (R*T)."""
        pass

    @abstractmethod
    def pressure(self, T, V):
        """Compute pressure using the EOS equation."""
        pass

    def cubic_eq_coeffs(self, T, P):
        A = self.A(T, P)
        B = self.B(T, P)
        return [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]

    def solve_Z(self, T, P, phase='both'):
        T = InputParser.parse(T)
        P = InputParser.parse(P)
        T_arr = np.atleast_1d(T)
        P_arr = np.atleast_1d(P)
        Z_solutions = []
        for t, p in zip(np.broadcast_to(T_arr, T_arr.shape), np.broadcast_to(P_arr, T_arr.shape)):
            coeffs = self.cubic_eq_coeffs(t, p)
            roots = np.roots(coeffs)
            real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-8])
            if not real_roots:
                Z_solutions.append(None)
            elif phase == 'liquid':
                Z_solutions.append(real_roots[0])
            elif phase == 'vapor':
                Z_solutions.append(real_roots[-1])
            else:
                Z_solutions.append(real_roots)
        return Z_solutions if len(Z_solutions) > 1 else Z_solutions[0]

    def calculate_properties(self, T, P, V=None):
        T = InputParser.parse(T)
        P = InputParser.parse(P)
        results = {}
        Z = self.solve_Z(T, P)
        results['Z'] = Z
        if V is not None:
            results['P_calc'] = self.pressure(T, V)
        return results

    def residual_enthalpy(self, T, P, Z, dimensionless=False):
        """
        Residual enthalpy H^R.
        Parameters:
            T: Temperatur (K)
            P: Druck (Pa)
            Z: Kompressibilitätsfaktor
            dimensionless: bool, ob Wert dimensionslos (True) oder in J/mol (False) zurückgegeben wird.
        Returns:
            Residualenthalpie (J/mol) oder dimensionslose Größe.
        """
        A = self.A(T, P)
        B = self.B(T, P)
        R = self.R
        val = Z - 1 - (A / B) * np.log(1 + B / Z)
        if dimensionless:
            return val
        else:
            return R * T * val

    def residual_entropy(self, T, P, Z, dimensionless=False):
        """
        Residual entropy S^R.
        Parameters:
            T: Temperatur (K)
            P: Druck (Pa)
            Z: Kompressibilitätsfaktor
            dimensionless: bool, ob Wert dimensionslos (True) oder in J/(mol·K) (False) zurückgegeben wird.
        Returns:
            Residualentropie (J/(mol·K)) oder dimensionslose Größe.
        """
        A = self.A(T, P)
        B = self.B(T, P)
        R = self.R

        # TODO: Einheitlichkeit prüfen
        # np.sqrt(T) kann dimensional problematisch sein, evtl. reduzierte Temperatur (T/Tc) verwenden.

        val = np.log(Z - B) - (A / (B * np.sqrt(T))) * np.log(1 + B / Z)
        if dimensionless:
            return val
        else:
            return R * val

    def fugacity_coefficient(self, T, P, Z):
        """
        Berechnet den Fugazitätskoeffizienten.
        Achtung: Numerische Robustheit prüfen - log(Z - B) kann bei kleinen Werten problematisch sein.
        """
        A = self.A(T, P)
        B = self.B(T, P)

        # TODO: Prüfe ob Z > B, um log() Fehler zu vermeiden
        if np.any(np.array(Z) <= B):
            raise ValueError("Z must be greater than B to avoid log of non-positive number.")

        R = self.R
        return np.exp(Z - 1 - np.log(Z - B) - A / (B * np.sqrt(T)) * np.log(1 + B / Z))

class VanDerWaalsEOS(BaseEOS):
    def alpha(self, T):
        return 1

    def A(self, T, P):
        a = InputParser.parse(self.a)
        R = self.R
        return a * P / (R**2 * T**2)

    def B(self, T, P):
        b = InputParser.parse(self.b)
        R = self.R
        return b * P / (R * T)

    def pressure(self, T, V):
        a = InputParser.parse(self.a)
        b = InputParser.parse(self.b)
        R = self.R
        T = InputParser.parse(T)
        V = InputParser.parse(V)
        return R * T / (V - b) - a / V**2

class RedlichKwongEOS(BaseEOS):
    def alpha(self, T):
        return 1 / np.sqrt(T)

    def A(self, T, P):
        a = InputParser.parse(self.a)
        alpha = self.alpha(T)
        R = self.R
        return a * alpha * P / (R**2 * T**2.5)

    def B(self, T, P):
        b = InputParser.parse(self.b)
        R = self.R
        return b * P / (R * T)

    def pressure(self, T, V):
        a = InputParser.parse(self.a)
        b = InputParser.parse(self.b)
        R = self.R
        alpha = self.alpha(T)
        return R * T / (V - b) - a * alpha / (V * (V + b))

class SoaveRedlichKwongEOS(BaseEOS):
    def __init__(self, a, b, kappa, Tc, R=8.314):
        super().__init__(a, b, R)
        self.kappa = kappa
        self.Tc = Tc

    def alpha(self, T):
        Tr = T / self.Tc
        return (1 + self.kappa * (1 - np.sqrt(Tr)))**2

    def A(self, T, P):
        a = InputParser.parse(self.a)
        alpha = self.alpha(T)
        R = self.R
        return a * alpha * P / (R**2 * T**2)

    def B(self, T, P):
        b = InputParser.parse(self.b)
        R = self.R
        return b * P / (R * T)

    def pressure(self, T, V):
        a = InputParser.parse(self.a)
        b = InputParser.parse(self.b)
        R = self.R
        alpha = self.alpha(T)
        return R * T / (V - b) - a * alpha / (V * (V + b))

    def cubic_eq_coeffs(self, T, P):
        A = self.A(T, P)
        B = self.B(T, P)
        return [1, -(1 + B), A, -A * B]

class PengRobinsonEOS(BaseEOS):
    def __init__(self, a, b, kappa, Tc, R=8.314):
        super().__init__(a, b, R)
        self.kappa = kappa
        self.Tc = Tc

    def alpha(self, T):
        Tr = T / self.Tc
        return (1 + self.kappa * (1 - np.sqrt(Tr)))**2

    def A(self, T, P):
        a = InputParser.parse(self.a)
        alpha = self.alpha(T)
        R = self.R
        return a * alpha * P / (R**2 * T**2)

    def B(self, T, P):
        b = InputParser.parse(self.b)
        R = self.R
        return b * P / (R * T)

    def pressure(self, T, V):
        a = InputParser.parse(self.a)
        b = InputParser.parse(self.b)
        R = self.R
        alpha = self.alpha(T)
        return R * T / (V - b) - a * alpha / (V**2 + 2*b*V - b**2)

    def cubic_eq_coeffs(self, T, P):
        A = self.A(T, P)
        B = self.B(T, P)
        return [1, -(1 + B - B**2), A + 3*B**2 - 2*B, -(A*B + B**2 + B**3)]

if __name__ == "__main__":
    eos = VanDerWaalsEOS(a=0.421875, b=0.125)
    T = np.array([300, 350])
    P = np.array([1e5, 1.5e5])
    Z_roots = eos.solve_Z(T, P, phase='both')
    print("Z roots (vdW):", Z_roots)

    V = 0.003
    P_calc = eos.pressure(300, V)
    print("Calculated pressure (vdW):", P_calc)

    eos_pr = PengRobinsonEOS(a=1.457, b=0.0778, kappa=0.37464, Tc=304.2)
    results = eos_pr.calculate_properties(300, 1e5, V=0.003)
    print("Peng-Robinson EOS results:", results)
    print("Residual Enthalpy (PR):", eos_pr.residual_enthalpy(300, 1e5, results['Z']))
    print("Residual Entropy (PR):", eos_pr.residual_entropy(300, 1e5, results['Z']))
    print("Fugacity Coefficient (PR):", eos_pr.fugacity_coefficient(300, 1e5, results['Z']))
