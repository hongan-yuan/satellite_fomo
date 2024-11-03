# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
from typing import List

import networkx as nx
from networkx import DiGraph


def get_shortest_path_subgraph(G, source_node, target_node):
    try:
        shortest_path = nx.shortest_path(G, source=source_node, target=target_node, weight='weight')
    except nx.NetworkXNoPath:
        return None
    sub_graph = nx.DiGraph()
    for i in range(len(shortest_path) - 1):
        u = shortest_path[i]
        v = shortest_path[i + 1]
        weight = G[u][v]['weight']
        sub_graph.add_edge(u, v, weight=weight)
    return sub_graph


def merge_graphs(graphs):
    merged_graph = nx.DiGraph()
    for u, v, data in graphs[0].edges(data=True):
        merged_graph.add_edge(u, v, weight=data['weight'])
    for g in graphs[1:]:
        for u, v, data in g.edges(data=True):
            if merged_graph.has_edge(u, v):
                merged_graph[u][v]['weight'] = min(merged_graph[u][v]['weight'], data['weight'])
            else:
                merged_graph.add_edge(u, v, weight=data['weight'])
    return merged_graph


def show_shortest_paths(graph: DiGraph, source_node: int, target_node: List):
    shortest_path = []
    for i, target in enumerate(target_node):
        # sp_len = nx.shortest_path_length(graph, source=source_node, target=target, weight="weight")
        sp = nx.shortest_path(graph, source=source_node, target=target, weight="weight")
        shortest_path.append(sp)
        # print(f"From source {source_node} --> target {target} shortest path is: {sp}")
        # print(f"From source {source_node} --> target {target} the shortest path length is: {sp_len}")


def shortest_path_merge_graph(graph: DiGraph, source_node: int, target_node: List):
    shortest_path_sub_graphs = []
    for target in target_node:
        shortest_path_sub_graph = get_shortest_path_subgraph(graph, source_node, target)
        shortest_path_sub_graphs.append(shortest_path_sub_graph)
    merged_shortest_path_graph = merge_graphs(shortest_path_sub_graphs)
    return merged_shortest_path_graph
