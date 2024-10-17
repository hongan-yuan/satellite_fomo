# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import os
import sys
import json
import numpy as np
from quinine import QuinineArgumentParser
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append('../scripts')
from models import TransformerModelLooped
from nano_gpt import GPT2Model, GPT2Config
from jupyter_notebooks.utils import aggregate_metrics, get_model, eval_unlooped_model, eval_looped_model

torch.manual_seed(42)
device = torch.device('cuda:0')
fig_hparam = {
    'figsize': (8, 5),
    'labelsize': 28,
    'ticksize': 20,
    'linewidth': 5,
    'fontsize': 15,
    'titlesize': 20,
    'markersize': 15
}

# font specification
fontdict = {
    'family': 'serif',
    'size': fig_hparam['fontsize'],
}


class DecisionTree:
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, depth=4):
        """
        batch_size: 1280
        n_points: 101
        n_dims: 20
        n_dims_truncated: 20
        device: torch.device('cuda:0')
        depth: 4
        """
        self.batch_size = batch_size
        self.n_points = n_points
        self.n_dims = n_dims
        self.n_dims_truncated = n_dims_truncated
        self.depth = depth

        # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
        # dt_tensor stores the coordinate used at each node of the decision tree.
        # Only indices corresponding to non-leaf nodes are relevant
        self.decisionTree_tensor = torch.randint(
            low=0,
            high=n_dims,
            size=(batch_size, 2 ** (depth + 1) - 1)  # size=(1280, 31)
        )

        # Target value at the leaf nodes.
        # Only indices corresponding to leaf nodes are relevant.
        self.target_tensor = torch.randn(self.decisionTree_tensor.shape)

        self.xs = torch.randn(batch_size, n_points, n_dims).to(device)  # [B, n, d]
        self.ys = self.evaluate(self.xs)

    def evaluate(self, xs_b):
        dt_tensor = self.decisionTree_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)

        ys_b = torch.zeros(
            xs_b.shape[0],  # 1280
            xs_b.shape[1],  # 101
            device=xs_b.device
        )

        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.batch_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]
            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b


if __name__ == '__main__':
    sample_size = 1280
    batch_size = 32
    n_points = 101
    n_dims_truncated = 20
    n_dims = 20
    result_dir = '../results2/decision_tree_loop'
    run_id = '0926061635-DT_loop_L1_endsb70_T15-0602'

    real_task = DecisionTree(
        batch_size=sample_size,  # 1280
        n_points=n_points,  # 101
        n_dims=n_dims,  # 20
        n_dims_truncated=n_dims_truncated,  # 20
        device=torch.device('cuda:0')
    )

    xs, ys = real_task.xs, real_task.ys

    model = TransformerModelLooped(
        n_dims=20,
        n_positions=101,
        n_embd=256,
        n_layer=12,
        n_head=8
    )
    step = -1
    model = get_model(model, result_dir, run_id, step)
    model = model.to(device)
    T = 200
    with torch.no_grad():
        y_pred_total = torch.zeros(1280, 101)  # [N, n]
        y_pred_last = torch.zeros(1280, T)  # [N, T]
        for batch_idx in range(sample_size // batch_size):
            xs_train = xs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            ys_train = ys[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y_pred_list = model(xs_train, ys_train, 0, T)  # list of [B, n], length T
            y_pred_total[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred_list[-1].detach()
            tmp_list = [y_pred[:, [-1]] for y_pred in y_pred_list]  # list of [B, 1], length T
            tmp_array = torch.cat(tmp_list, dim=1)  # [B, T]
            y_pred_last[batch_idx * batch_size: (batch_idx + 1) * batch_size] = tmp_array
        err = (y_pred_total - ys.cpu()).square()  # [n,]
        loop_err = (y_pred_last - ys.cpu()[:, [-1]]).square()  # [N, T] - [N, 1]
    print(err)
