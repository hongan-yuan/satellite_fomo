# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import torch

y_pred_list = []
y_pred_r = torch.randn(32, 101)
y_pred = y_pred_r[:, [-1]]
for i in range(200):
    y_pred_list.append(torch.randn(32, 101))

last_y_pred_list = [y_pred[:, [-1]] for y_pred in y_pred_list]   # list of [B, 1], length T,
last_y_pred_array = torch.cat(last_y_pred_list, dim=1)  # [B, T]

print(f"y_pred_list[0] ..is.. {y_pred_list[0]}")
print(f"last_y_pred_list[0] ..is.. {last_y_pred_list[0]}")
print(f"last_y_pred_list[1] ..is.. {last_y_pred_list[1]}")
print(f"last_y_pred_array.shape ..is.. {last_y_pred_array.shape}")
print(f"last_y_pred_array[0] ..is.. {last_y_pred_array[0]}")
print(f"last_y_pred_list[0].shape ..is.. {last_y_pred_list[0].shape}")
# print(f"y_pred_r.shape is .. {y_pred_r.shape}")
# print(f"y_pred.shape is .. {y_pred.shape}")
