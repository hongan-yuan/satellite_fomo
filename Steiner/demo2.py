# -*- coding: utf-8 -*-
"""
@Time: 6/13/2024 3:50 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import networkx as nx
from scipy.sparse.csgraph import dijkstra
import numpy as np


def directed_steiner_tree(graph, source, terminals):
    """
    使用贪心算法近似求解有向斯坦纳树问题
    :param graph: 有向图（networkx.DiGraph）
    :param source: 源节点
    :param terminals: 终端节点列表
    :return: 包含源节点和所有终端节点的近似斯坦纳树（networkx.DiGraph）
    """
    # 初始化斯坦纳树
    steiner_tree = nx.DiGraph()

    # 初始化未覆盖的终端节点集合
    uncovered_terminals = set(terminals)

    # 使用Dijkstra算法计算从源节点到所有节点的最短路径
    lengths, paths = nx.single_source_dijkstra(graph, source)

    # 贪心选择覆盖未覆盖的终端节点
    while uncovered_terminals:
        # 找到最近的未覆盖的终端节点
        closest_terminal = min(uncovered_terminals, key=lambda t: lengths[t])
        # 获取从源节点到最近终端节点的最短路径
        path = paths[closest_terminal]
        # 将路径上的所有节点和边添加到斯坦纳树中
        for i in range(len(path) - 1):
            steiner_tree.add_edge(path[i], path[i + 1], weight=graph[path[i]][path[i + 1]]['weight'])
        # 从未覆盖的终端节点集合中移除该终端节点
        uncovered_terminals.remove(closest_terminal)

        # 更新未覆盖的终端节点的最短路径
        new_lengths, new_paths = nx.single_source_dijkstra(graph, closest_terminal)
        for t in uncovered_terminals:
            if lengths[t] > new_lengths[t]:
                lengths[t] = new_lengths[t]
                paths[t] = new_paths[t]

    return steiner_tree


# 示例：创建有向图并求解有向斯坦纳树
G = nx.DiGraph()

# 添加有向边及其权重
edges = [
    ('A', 'B', 1), ('A', 'C', 2), ('B', 'C', 1),
    ('B', 'D', 2), ('C', 'D', 1), ('C', 'E', 3),
    ('D', 'E', 1), ('D', 'G', 2), ('E', 'F', 2)
]
G.add_weighted_edges_from(edges)

source = 'A'
terminals = ['F', 'G']

steiner_tree = directed_steiner_tree(G, source, terminals)

# 打印斯坦纳树的节点和边
print("斯坦纳树的节点:")
print(steiner_tree.nodes())
print("斯坦纳树的边:")
print(steiner_tree.edges(data=True))
