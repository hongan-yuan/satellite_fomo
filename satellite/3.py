# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import numpy as np

k = 1.38e-23  # Boltzmann constant in J/K
T_b = 6000  # Solar brightness temperature in K
T_0 = 1000  # Typical system noise temperature in K (assumed value, can be changed)
T_CMB = 2.725  # Cosmic Microwave Background temperature in K
Bandwidth = 193e12 * 0.02
lamda = 1550e-9  # wavelength
# Total noise temperature
T_total = T_b + T_0 + T_CMB
P_N = k * T_total * Bandwidth
p_len = 100000
FSPL = ((4 * np.pi * p_len) / lamda) ** 2
receiver_power = 0.1 * 0.8 * 0.1 / 0.1 * FSPL
DataRate = Bandwidth * np.log2(1 + (receiver_power / P_N))
