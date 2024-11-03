# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""

import scipy.io
import numpy as np

# 读取.mat文件
mat_file_path = 'polar-Telesat-position.mat'  # 替换为你的.mat文件路径
mat_contents = scipy.io.loadmat(mat_file_path)

# 显示.mat文件中的所有变量
print("MAT文件中的变量:")
for var in mat_contents:
    print(var)

# 将.mat文件中的数据存储到列表中
data_list = []
for var in mat_contents:
    # 排除.mat文件中的默认元数据变量
    # if '_' not in var:
    if not var.startswith('_'):
        data_list.append(mat_contents[var])

print("\n存储在列表中的数据:")
spherical_position_data = np.array(data_list[0])
cartesian_position_data = np.array(data_list[1])

SATELLITE_TOTAL_NUM = 72
TIME_STEP = 6299
SATELLITE_POSITION_DATA = []

for satellite_id in range(SATELLITE_TOTAL_NUM):
    SATELLITE_POSITION_DATA.append([])
    satellite_position_data = spherical_position_data[satellite_id][0]  # 3*6299
    for time_step in range(TIME_STEP):
        SATELLITE_POSITION_DATA[satellite_id].append(
            {
                time_step:
                    [satellite_position_data[0][time_step],
                     satellite_position_data[1][time_step],
                     satellite_position_data[2][time_step]]
            }
        )
print(f"SATELLITE_POSITION_DATA ... len is ... {len(SATELLITE_POSITION_DATA)}")
print(f"SATELLITE_POSITION_DATA[0] ... len is ... {len(SATELLITE_POSITION_DATA[0])}")
