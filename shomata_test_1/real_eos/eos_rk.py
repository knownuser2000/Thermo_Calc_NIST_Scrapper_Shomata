# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:49:48 2025

@author: Bayra
"""

from .base import BaseEOS, InputParser
import numpy as np

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
