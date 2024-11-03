# -*- coding: utf-8 -*-
"""
@Time: 6/11/2024 11:27 AM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import re
from typing import List

import networkx as nx
from networkx import Graph
import numpy as np

np.random.seed = 54


def update_satellite_position(initial_r, initial_theta, initial_phi, omega, time_interval_s):
    # phi_new = initial_phi + np.rad2deg(omega * time_interval_s)
    phi_new = initial_phi + omega * time_interval_s
    return initial_r, initial_theta, phi_new


def extract_satellite(text, start_char="=", end_char="S"):
    # 使用正则表达式进行匹配
    pattern = re.escape(start_char) + '(.*?)' + re.escape(end_char)
    matches = re.findall(pattern, text)
    satellite_id = int(matches[0])
    return satellite_id


def generate_microservice_deployment_matrix(rows, cols):
    col_id_list = [i for i in range(cols)]
    # 初始化全为 0 的矩阵
    matrix = np.zeros((rows, cols), dtype=int)  # 300 * 400
    random_position_list = np.random.choice(col_id_list, rows, replace=True)
    for i, rpl in enumerate(random_position_list):
        matrix[i, rpl] = 1
    return matrix
