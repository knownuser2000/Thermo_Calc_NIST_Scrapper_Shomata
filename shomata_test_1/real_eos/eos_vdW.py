# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:48:52 2025

@author: Bayra
"""

from .base import BaseEOS, InputParser

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
