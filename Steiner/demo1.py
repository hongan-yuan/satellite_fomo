# -*- coding: utf-8 -*-
"""
@Time: 5/9/2024 4:56 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import matplotlib.pyplot as plt
import networkx as nx

# 生成具有10个节点，36条边的无向图
G = nx.gnm_random_graph(10, 36, directed=True)

# 绘制图形
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=12, font_weight='bold')
plt.title("Random Undirected Graph")
plt.show()

# 使用 Prim 算法计算最小生成树
prim_mst = nx.minimum_spanning_tree(G, algorithm='prim')

# 使用 Kruskal 算法计算最小生成树
kruskal_mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

# 绘制 Prim 算法计算得到的最小生成树
plt.figure(figsize=(8, 6))
nx.draw(prim_mst, with_labels=True, node_size=500, node_color='lightgreen', font_size=12, font_weight='bold')
plt.title("Minimum Spanning Tree (Prim's Algorithm)")
plt.show()

# 绘制 Kruskal 算法计算得到的最小生成树
plt.figure(figsize=(8, 6))
nx.draw(kruskal_mst, with_labels=True, node_size=500, node_color='lightcoral', font_size=12, font_weight='bold')
plt.title("Minimum Spanning Tree (Kruskal's Algorithm)")
plt.show()
