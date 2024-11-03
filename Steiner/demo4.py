# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加带权重的边
edges = [
    ('A', 'B', 1),
    ('A', 'C', 2),
    ('B', 'C', 1),
    ('B', 'D', 2),
    ('C', 'D', 1),
    ('C', 'E', 2),
    ('D', 'E', 1)
]
G.add_weighted_edges_from(edges)

# 指定源节点和目标节点
source = 'A'
target = 'E'

# 找到从源节点到目标节点的最短路径
shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')

# 打印最短路径
print(f"从节点 {source} 到节点 {target} 的最短路径为: {shortest_path}")
