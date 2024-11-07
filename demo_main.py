# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import numpy as np

np.seed = 42

mean = 5
std_dev = 1
total_iter = 100
sate_num = [4, 5, 6]
total_batch_size = 1024
mini_batch_size = [256, 128, 64, 32, 16]

# mid_data_vol = mini_batch_size * 8 * 202 * 276
# data_rate1 = np.random.normal(mean, std_dev, 4) * 10e9
# data_rate2 = np.random.normal(mean, std_dev, 5) * 10e9
# data_rate3 = np.random.normal(mean, std_dev, 8) * 10e9

# lat_1 = mid_data_vol / data_rate1
# lat_2 = mid_data_vol / data_rate2
# lat_3 = mid_data_vol / data_rate3
T10 = [11400, 4980, 2170, 1110, 564]
# T25 = 30800
# T20 = 26200
# T16 = 1330
# T12 = 781
for i in range(len(T10)):
    mid_data_vol = mini_batch_size[i] * 8 * 202 * 276
    data_rate1 = np.random.normal(mean, std_dev, 10) * 10e9
    lat_1 = mid_data_vol / data_rate1
    total_lat1 = (T10[i] + (max(lat_1) * 1000)) * (10 + total_batch_size / mini_batch_size[i] - 1)
    print(total_lat1)
# total_lat1 = (T25 + (max(lat_1) * 1000)) * (4 + total_batch_size / mini_batch_size - 1)
# total_lat2 = (T20 + (max(lat_2) * 1000)) * (5 + total_batch_size / mini_batch_size - 1)
# total_lat3 = (T16 + (max(lat_3) * 1000)) * 5 + (T20 + (max(lat_3) * 1000)) * (total_batch_size / mini_batch_size)
# total_lat3 = (T12 + (max(lat_3) * 1000)) * 7 + (T16 + (max(lat_3) * 1000)) * (total_batch_size / mini_batch_size)

# print(sum(lat_1) * 1000)
# print(sum(lat_2) * 1000)
# print(sum(lat_3) * 1000)
# print("************************")
# print(max(lat_1) * 1000)
# print(max(lat_2) * 1000)
# print(max(lat_3) * 1000)
print("************************")
# print(total_lat1)
# print(total_lat2)
# print(total_lat3)
