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
from jupyter_notebooks.utils import aggregate_metrics, get_model, eval_unlooped_model, eval_looped_model

sys.path.append('../scripts')
from nano_gpt import GPT2Model, GPT2Config
from models import TransformerModelLooped

torch.manual_seed(42)
device = torch.device('cuda:0')
# device = 'cpu'

"""
"""
SAMPLE_SIZE = 1280
BATCH_SIZE = 32
CONTEXT_SIZE = 101
INPUT_DIMS = 20
#  -----------------------------------------------------------------------------
N_DIMS_TRUNCATED = 20
#  -----------------------------------------------------------------------------
LOOP_ITER_NUM = 200
EMBEDDING_DIM = 256
N_HEAD = 8
N_LAYER = 12
N_LAYER_LOOP = 1


class DecisionTree:
    def __init__(self,
                 batch_size, n_points, n_dims, n_dims_truncated, device, depth=4):
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
    result_dir = '../results2/decision_tree_loop'
    run_id = '0926061635-DT_loop_L1_endsb70_T15-0602'

    decision_tree_task = DecisionTree(
        batch_size=SAMPLE_SIZE,  # 1280
        n_points=CONTEXT_SIZE,  # 101
        n_dims=INPUT_DIMS,  # 20
        n_dims_truncated=N_DIMS_TRUNCATED,  # 20
        device=torch.device('cuda:0')
    )

    loop_model = TransformerModelLooped(
        n_dims=INPUT_DIMS,
        n_positions=CONTEXT_SIZE,
        n_embd=EMBEDDING_DIM,
        n_layer=N_LAYER_LOOP,
        n_head=N_HEAD
    )
    step = -1

    loop_model = get_model(loop_model, result_dir, run_id, step)
    loop_model = loop_model.to(device)
    xs, ys = decision_tree_task.xs, decision_tree_task.ys

    xs_train = xs[0: 4]
    ys_train = ys[0: 4]

    result_y = loop_model(xs_train, ys_train, 0, LOOP_ITER_NUM)  # [1280, 101]
    print(result_y.shape)
    """
    with torch.no_grad():
        y_pred_total = torch.zeros(1280, 101)  # [N, n]
        y_pred_last = torch.zeros(1280, 200)  # [N, T]  T refers to the number of loops.
        for batch_idx in range(1280 // 32):
            xs_train = xs[batch_idx * 32: (batch_idx + 1) * 32]
            ys_train = ys[batch_idx * 32: (batch_idx + 1) * 32]
            # Record the results of each loop iteration.
            y_pred_list = loop_model(xs_train, ys_train, 0, 200)  # list of [B, n], length T

            #  get the last y_value from the list whose length equals to 101
            last_y_pred_list = [y_pred[:, [-1]] for y_pred in y_pred_list]  # list of [B, 1], length T,
            # Record the last y_value for each looped iteration can contact
            last_y_pred_array = torch.cat(last_y_pred_list, dim=1)  # [B, T]
            print(f">>>>>> ... last_y_pred_array.shape ..is.. {last_y_pred_array.shape}")
            y_pred_last[batch_idx * 32: (batch_idx + 1) * 32] = last_y_pred_array

            # The computation result of last iteration for (xs & ys) ==> the predicted ys' ==>  y_pred_list[-1].detach()
            # y_pred_list[-1].shape ... is torch.Size([32, 101])
            y_pred_total[batch_idx * 32: (batch_idx + 1) * 32] = y_pred_list[-1].detach()
            print(f"y_pred_total.length ... is {len(y_pred_total)}")

        total_err = (y_pred_total - ys.cpu()).square()  # [n,]
        loop_iter_err = (y_pred_last - ys.cpu()[:, [-1]]).square()  # [N, T] - [N, 1]
        """
