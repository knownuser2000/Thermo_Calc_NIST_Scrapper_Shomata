# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:51:44 2025

@author: Bayra
"""

from .base import BaseEOS, InputParser
import numpy as np

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
