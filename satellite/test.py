# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import torch

device = torch.device('cuda:0')
batch_size = 1280

dt_tensor = torch.randint(
    low=0,
    high=20,
    size=(1280, 2 ** (4 + 1) - 1),  # size=(1280, 31)
    device=torch.device('cuda:0')
)
target_tensor = torch.randn(dt_tensor.shape, device=torch.device('cuda:0'))

xs_b = torch.randn(1280, 101, 20).to(device)
ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)

for i in range(xs_b.shape[0]):
    xs_bool = xs_b[i] > 0
    if batch_size == 1:
        dt = dt_tensor[0]
        target = target_tensor[0]
        print(f"... batch_size ... {batch_size}")
    else:
        dt = dt_tensor[i]
        target = target_tensor[i]
    cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()

    for j in range(4):
        cur_coords = dt[cur_nodes]
        cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
        cur_nodes = 2 * cur_nodes + 1 + cur_decisions

    ys_b[i] = target[cur_nodes]


print(xs_b.shape)
print(ys_b.shape)
print(xs_b[0].shape)
print(ys_b[0].shape)
print(xs_b[0][0])
print(ys_b[0][0])
print("###########################################################################")
print(xs_b[0])
print(ys_b[0])
#
# print(xs_b.shape[0])
# print(xs_b.shape[1])

# print(dt_tensor)
