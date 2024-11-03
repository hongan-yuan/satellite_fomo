# -*- coding: utf-8 -*-
"""
@Time: 5/7/2024 4:31 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""

import networkx as nx


def steiner_tree(graph, terminals):
    steiner_tree = nx.DiGraph()
    for terminal in terminals:
        # 求解每个终端到其他终端的最短路径
        shortest_paths = nx.single_source_shortest_path_length(graph, terminal)
        for node, dist in shortest_paths.items():
            if node in terminals and node != terminal:
                steiner_tree.add_edge(terminal, node, weight=dist)
    # 使用最小生成树算法找到最小的有向斯坦纳树
    mst = nx.minimum_spanning_tree(steiner_tree.to_undirected())
    return mst.to_directed()


# 创建一个有向图
graph = nx.Graph()
graph.add_weighted_edges_from([(1, 2, 1), (1, 3, 2), (2, 3, 1), (2, 4, 3), (3, 4, 1)])
print(graph.edges)

# 指定终端节点
terminals = [1, 2, 3]

# 求解有向斯坦纳树
steiner_tree = steiner_tree(graph, terminals)
print("Edges in Steiner Tree:", steiner_tree.edges(data=True))
