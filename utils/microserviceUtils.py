# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import random
from typing import List

import networkx as nx
import numpy as np

from BaseObject.constellation import Constellation
from Steiner.shortestPathMerge import merge_graphs
from utils.constellationUtils import *

np.random.seed = 54


def generate_user_request_chain(microservice_total_num, chain_len):
    """
    :param microservice_total_num: The total number of Microservices
    :param chain_len: The length of Chain of User requested
    :return: Chain-of-Service: List
    """
    ret_chain = np.random.choice(microservice_total_num, chain_len, replace=False)  # np.array([3, 4, 5, 6])
    ret_chain.sort()
    ret_chain = ret_chain.tolist()
    return ret_chain


def generate_user_request_tree(microservice_total_num, tree_nodes):
    """
    :param microservice_total_num: The total number of Microservices
    :param tree_nodes: The number of nodes of Tree of User requested
    :return: Tree-of-Service: List[List]
    """
    if tree_nodes < 10:
        return None
    ret_tree = []
    tree_nodes_list = np.random.choice(microservice_total_num, tree_nodes, replace=False)
    tree_nodes_list.sort()
    tree_nodes_list = tree_nodes_list.tolist()
    # print(f">>>>>>>>> ... tree_nodes_list ... {tree_nodes_list}")
    idx_1 = random.randint(int(tree_nodes / 3), int(tree_nodes / 3) + 4)
    idx_2 = random.randint(int(2 * tree_nodes / 3), int(2 * tree_nodes / 3) + 4)
    idx_3 = random.randint(int(2 * tree_nodes / 3) + 4, tree_nodes)
    # print(f">>>>>>>>> ... idx_1 ... {idx_1}")
    # print(f">>>>>>>>> ... idx_2 ... {idx_2}")

    ret_tree.append(tree_nodes_list[0:idx_1])
    ret_tree.append(tree_nodes_list[idx_1:idx_2])
    ret_tree.append(tree_nodes_list[idx_2:idx_3])
    ret_tree.append(tree_nodes_list[idx_3:])
    filtered_tree = [sublist for sublist in ret_tree if sublist]
    return filtered_tree


def generate_fixed_topo_user_request_tree(microservice_total_num, total_node_num):
    """
    fixed topology
    :param microservice_total_num: The total number of Microservices
    :param total_node_num: The number of nodes of Tree of User requested
    :return: nx.DiGraph
    """
    tree_nodes = 25
    nodes_list = np.random.choice(microservice_total_num, total_node_num, replace=False)
    nodes_list = nodes_list[:25]
    nodes_list.sort()
    service_nodes_list = nodes_list.tolist()
    print(f">>>> service_nodes_list is: {service_nodes_list}")
    nodes = [i for i in range(tree_nodes)]
    edges = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (4, 7), (5, 8), (5, 9), (6, 10), (7, 11), (8, 12), (9, 13),
        (10, 14), (10, 15), (11, 16), (13, 17), (14, 18), (15, 19), (17, 20), (18, 21), (18, 22), (19, 23), (20, 24),
    ]
    print(f">>>>>>>>>>>>>>>>>>>...edges len is: {len(edges)}")

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    service_idx_map = {i: service_nodes_list[i] for i in nodes}
    ret_tree = nx.relabel_nodes(G, service_idx_map, copy=True)
    return ret_tree


def construct_satellite_microservice_chain_colocation_di_graph(
        constellation: Constellation, microservice_cluster: List, user_request_service_chain, constellation_topo_graph):
    service_satellite_pair = {}
    for service in user_request_service_chain:
        service_satellite_pair[service] = []
        for satellite in constellation.satellites_group:
            if service in satellite.microservice_set:
                service_satellite_pair[service].append(satellite.id)
    # print(f"Microservice-Satellite colocation: {service_satellite_pair}")

    # construct the satellite-microservice colocation graph
    service_satellite_edges = []
    service_satellite_edge_weights = []
    # contacting the started satellite with the first service in chain
    satellite_sp = constellation.satellites_group[0]  # satellite start-point no.1

    for i, sa in enumerate(service_satellite_pair[user_request_service_chain[0]]):
        communication_cost = get_satellite_pair_communication_cost(
            constellation=constellation, constellation_topo_graph=constellation_topo_graph,
            source_satellite=satellite_sp.id, target_satellite=sa,
            microservice_cluster=microservice_cluster, microservice_id=user_request_service_chain[0]
        )
        # >>> ... Calculating the communication cost between the first satellite and the first requested microservice
        service_satellite_edges.append((satellite_sp.id, f"{user_request_service_chain[0]}-{2 * i + 1}"))
        service_satellite_edge_weights.append(communication_cost)

    for pre_service in user_request_service_chain[:-1]:  # service_id  0 -> len -2
        for i, pre_sa in enumerate(service_satellite_pair[pre_service]):  # satellite_id where service deployed

            # post_service = user_request_service_chain[i + 1]  # service_id
            post_service = user_request_service_chain[user_request_service_chain.index(pre_service) + 1]  # service_id
            # pre_service_comp_cost = (microservice_cluster[pre_service].dataVol / constellation.satellites_group[pre_sa].computing_cab) / 2
            pre_service_comp_cost = get_microservice_computing_cost(
                constellation=constellation,
                satellite=pre_sa,
                microservice_cluster=microservice_cluster,
                microservice_id=pre_service
            )
            # 1^1 -> 3 -> 1^2
            service_satellite_edges.append((f"{pre_service}-{2 * i + 1}", f"{pre_service}M={pre_sa}S"))
            service_satellite_edge_weights.append(pre_service_comp_cost)

            service_satellite_edges.append((f"{pre_service}M={pre_sa}S", f"{pre_service}-{2 * i + 2}"))
            service_satellite_edge_weights.append(pre_service_comp_cost)

            for j, post_sa in enumerate(service_satellite_pair[post_service]):  # satellite_id
                # post_service_comp_cost = (microservice_cluster[post_service].dataVol / constellation.satellites_group[post_sa].computing_cab) / 2
                # shortest_path_length = nx.shortest_path_length(constellation_topo_graph, source=pre_sa, target=post_sa)
                # communication_cost = (constellation.get_satellite_transmitter_power_by_id(pre_sa, shortest_path_length) *
                #                       (constellation.link_data_rate / microservice_cluster[pre_service].dataVol))
                communication_cost = get_satellite_pair_communication_cost(
                    constellation=constellation, constellation_topo_graph=constellation_topo_graph,
                    source_satellite=pre_sa, target_satellite=post_sa,
                    microservice_cluster=microservice_cluster, microservice_id=pre_service
                )
                service_satellite_edges.append((f"{pre_service}-{2 * i + 2}", f"{post_service}-{2 * j + 1}"))
                service_satellite_edge_weights.append(communication_cost)

    last_service = user_request_service_chain[-1]
    for i, satellite in enumerate(service_satellite_pair[last_service]):
        # service_comp_cost = (microservice_cluster[last_service].dataVol / constellation.satellites_group[satellite].computing_cab) / 2
        service_comp_cost = get_microservice_computing_cost(
            constellation=constellation,
            satellite=satellite,
            microservice_cluster=microservice_cluster,
            microservice_id=last_service
        )
        service_satellite_edges.append((f"{last_service}-{2 * i + 1}", f"{last_service}M={satellite}S"))
        service_satellite_edge_weights.append(service_comp_cost)
        service_satellite_edges.append((f"{last_service}M={satellite}S", f"{last_service}-{2 * i + 2}"))
        service_satellite_edge_weights.append(service_comp_cost)

    satellite_ep = constellation.satellites_group[-1]  # satellite end-point no.-1
    for i, sa in enumerate(service_satellite_pair[user_request_service_chain[-1]]):
        # service_comp_cost = (microservice_cluster[user_request_service_chain[-1]].dataVol /
        #                      constellation.satellites_group[user_request_service_chain[-1]].computing_cab) / 2
        # shortest_path_length = nx.shortest_path_length(constellation_topo_graph, source=sa, target=satellite_ep.id)
        # communication_cost = (constellation.get_satellite_transmitter_power_by_id(sa, shortest_path_length) *
        #                       (constellation.link_data_rate / microservice_cluster[user_request_service_chain[-1]].dataVol))
        communication_cost = get_satellite_pair_communication_cost(
            constellation=constellation, constellation_topo_graph=constellation_topo_graph,
            source_satellite=sa, target_satellite=satellite_ep.id,
            microservice_cluster=microservice_cluster, microservice_id=user_request_service_chain[-1]
        )
        service_satellite_edges.append((f"{user_request_service_chain[-1]}-{2 * i + 2}", satellite_ep.id))
        service_satellite_edge_weights.append(communication_cost)

    # print(f"service_satellite_edges:\n"
    #       f"{service_satellite_edges}")
    # print(f"service_satellite_edge_weights:\n"
    #       f"{service_satellite_edge_weights}")
    # print(len(service_satellite_edges) == len(service_satellite_edge_weights))

    if len(service_satellite_edges) == len(service_satellite_edge_weights):
        colocationDiG = construct_satellite_microservice_colocation_di_graph(service_satellite_edges, service_satellite_edge_weights)
        return colocationDiG
    else:
        print(f"len(service_satellite_edges) != len(service_satellite_edge_weights)")
        return None


def construct_satellite_microservice_tree_colocation_di_graph(
        constellation: Constellation, microservice_cluster: List, user_request_service_tree, constellation_topo_graph):
    microservice_satellite_pairs = {}
    for child_tree in user_request_service_tree:
        for service in child_tree:
            microservice_satellite_pairs[service] = []
            for satellite in constellation.satellites_group:
                if service in satellite.microservice_set:
                    microservice_satellite_pairs[service].append(satellite.id)
    child_tree_graphs = []

    for i, child_tree in enumerate(user_request_service_tree):
        end_satellite_id = constellation.satellites_total_sum - 1
        if i == 0:
            CTG = construct_satellite_microservice_chain_colocation_di_graph(constellation, microservice_cluster, child_tree, constellation_topo_graph)
            CTG.remove_node(end_satellite_id)
        else:
            CTG = construct_satellite_microservice_chain_colocation_di_graph(constellation, microservice_cluster, child_tree, constellation_topo_graph)
            CTG.remove_node(0)
            new_end_satellite_id = f"{end_satellite_id}-{i}"
            mapping = {end_satellite_id: new_end_satellite_id}
            CTG = nx.relabel_nodes(CTG, mapping, copy=True)
        child_tree_graphs.append(CTG)

    # re-construct endpoint satellite for each branch of tree, replace the last satellite number with "number-idx"
    # satellite_ep = constellation.satellites_group[-1]  # satellite end-point no.-1
    # for i, child_tree_graph in enumerate(child_tree_graphs[1:]):
    #     pre_service = user_request_service_tree[i + 1][-1]
    #     add_ep_service_satellite_edges = []
    #     add_ep_service_satellite_edge_weights = []
    #
    #     for satellite in microservice_satellite_pairs[pre_service]:
    #         service_comp_cost = (microservice_cluster[pre_service].dataVol /
    #                              constellation.satellites_group[pre_service].computing_cab) / 2
    #         shortest_path_length = nx.shortest_path_length(constellation_topo_graph, source=satellite, target=satellite_ep.id)
    #         communication_cost = (constellation.get_satellite_transmitter_power_by_id(satellite, shortest_path_length) *
    #                               (constellation.link_data_rate / microservice_cluster[pre_service].dataVol))
    #         add_ep_service_satellite_edges.append((f"{pre_service}-{2 * i + 2}", f"{satellite_ep.id}-{i + 1}"))
    #         add_ep_service_satellite_edge_weights.append(service_comp_cost + communication_cost)
    #     child_tree_graph.add_edges_from(add_ep_service_satellite_edges)
    #     for edge, weight in zip(add_ep_service_satellite_edges, add_ep_service_satellite_edge_weights):
    #         child_tree_graph.edges[edge]['weight'] = weight

    cross_service = user_request_service_tree[0][-1]
    service_needs_to_reconnect = [user_request_service_tree[i][0] for i in range(1, len(user_request_service_tree))]

    add_service_satellite_edges = []
    add_service_satellite_edge_weights = []
    for post_service in service_needs_to_reconnect:
        for i, pre_sa in enumerate(microservice_satellite_pairs[cross_service]):
            for j, post_sa in enumerate(microservice_satellite_pairs[post_service]):  # satellite_id
                # shortest_path_length = nx.shortest_path_length(constellation_topo_graph, source=pre_sa, target=post_sa)
                # communication_cost = (constellation.get_satellite_transmitter_power_by_id(pre_sa, shortest_path_length) *
                #                       (constellation.link_data_rate / microservice_cluster[cross_service].dataVol))
                communication_cost = get_satellite_pair_communication_cost(
                    constellation=constellation, constellation_topo_graph=constellation_topo_graph,
                    source_satellite=pre_sa, target_satellite=post_sa,
                    microservice_cluster=microservice_cluster, microservice_id=cross_service
                )
                add_service_satellite_edges.append((f"{cross_service}-{2 * i + 2}", f"{post_service}-{2 * j + 1}"))
                add_service_satellite_edge_weights.append(communication_cost)
    # ret_tree = nx.compose_all(child_tree_graphs)
    ret_tree = merge_graphs(child_tree_graphs)
    ret_tree.add_edges_from(add_service_satellite_edges)
    for edge, weight in zip(add_service_satellite_edges, add_service_satellite_edge_weights):
        ret_tree.edges[edge]['weight'] = weight
    return ret_tree


def construct_satellite_microservice_chain_colocation_di_graph2(
        constellation: Constellation, microservice_cluster: List, user_request_service_chain, constellation_topo_graph):
    service_satellite_pair = {}
    for service in user_request_service_chain:
        service_satellite_pair[service] = []
        for satellite in constellation.satellites_group:
            if service in satellite.microservice_set:
                service_satellite_pair[service].append(satellite.id)
    # print(f"Microservice-Satellite colocation: {service_satellite_pair}")

    # construct the satellite-microservice colocation graph
    service_satellite_edges = []
    service_satellite_edge_weights = []
    pre_service_comp_cost, post_service_comp_cost = [], []

    satellite_sp = constellation.satellites_group[0]  # satellite start-point no.1
    for i, sa in enumerate(service_satellite_pair[user_request_service_chain[0]]):
        # >>> ... Calculating the communication cost between the first satellite and the first requested microservice
        communication_cost = get_satellite_pair_communication_cost(
            constellation=constellation, constellation_topo_graph=constellation_topo_graph,
            source_satellite=satellite_sp.id, target_satellite=sa,
            microservice_cluster=microservice_cluster, microservice_id=user_request_service_chain[0]
        )
        service_comp_cost = get_microservice_computing_cost(
            constellation=constellation, satellite=sa,
            microservice_cluster=microservice_cluster, microservice_id=user_request_service_chain[0])
        service_satellite_edges.append((satellite_sp.id, f"{user_request_service_chain[0]}M={sa}S"))
        service_satellite_edge_weights.append(communication_cost + service_comp_cost)

        pre_service_comp_cost.append(service_comp_cost)

    for pre_service in user_request_service_chain[:-1]:  # service_id
        for i, pre_sa in enumerate(service_satellite_pair[pre_service]):  # satellite_id where service deployed
            post_service = user_request_service_chain[user_request_service_chain.index(pre_service) + 1]  # service_id
            for j, post_sa in enumerate(service_satellite_pair[post_service]):  # satellite_id
                service_comp_cost = get_microservice_computing_cost(
                    constellation=constellation, satellite=post_sa,
                    microservice_cluster=microservice_cluster, microservice_id=post_service)
                communication_cost = get_satellite_pair_communication_cost(
                    constellation=constellation, constellation_topo_graph=constellation_topo_graph,
                    source_satellite=pre_sa, target_satellite=post_sa,
                    microservice_cluster=microservice_cluster, microservice_id=pre_service
                )
                service_satellite_edges.append((f"{pre_service}M={pre_sa}S", f"{post_service}M={post_sa}S"))
                service_satellite_edge_weights.append(pre_service_comp_cost[i] + communication_cost + service_comp_cost)
                post_service_comp_cost.append(service_comp_cost)
        pre_service_comp_cost = post_service_comp_cost[
                                0: len(service_satellite_pair[user_request_service_chain[user_request_service_chain.index(pre_service) + 1]])]

    satellite_ep = constellation.satellites_group[-1]  # satellite end-point no.-1
    for i, sa in enumerate(service_satellite_pair[user_request_service_chain[-1]]):
        communication_cost = get_satellite_pair_communication_cost(
            constellation=constellation, constellation_topo_graph=constellation_topo_graph,
            source_satellite=sa, target_satellite=satellite_ep.id,
            microservice_cluster=microservice_cluster, microservice_id=user_request_service_chain[-1]
        )
        service_satellite_edges.append((f"{user_request_service_chain[-1]}M={sa}S", satellite_ep.id))
        service_satellite_edge_weights.append(pre_service_comp_cost[i] + communication_cost)

    if len(service_satellite_edges) == len(service_satellite_edge_weights):
        colocationDiG = construct_satellite_microservice_colocation_di_graph(service_satellite_edges, service_satellite_edge_weights)
        return colocationDiG
    else:
        print(f"len(service_satellite_edges) != len(service_satellite_edge_weights)")
        return None


def construct_satellite_microservice_tree_colocation_di_graph2(
        constellation: Constellation, microservice_cluster: List, user_request_service_tree, constellation_topo_graph):
    microservice_satellite_pairs = {}
    for child_tree in user_request_service_tree:
        for service in child_tree:
            microservice_satellite_pairs[service] = []
            for satellite in constellation.satellites_group:
                if service in satellite.microservice_set:
                    microservice_satellite_pairs[service].append(satellite.id)
    child_tree_graphs = []

    first_child_tree = user_request_service_tree[0]
    for i, child_tree in enumerate(user_request_service_tree[1:]):
        CTG_Chain = first_child_tree + child_tree
        CTG = construct_satellite_microservice_chain_colocation_di_graph2(constellation, microservice_cluster, CTG_Chain, constellation_topo_graph)
        new_end_satellite_id = f"{constellation.satellites_group[-1].id}-{i + 1}"
        mapping = {constellation.satellites_group[-1].id: new_end_satellite_id}
        CTG = nx.relabel_nodes(CTG, mapping, copy=True)
        child_tree_graphs.append(CTG)

    # for ctg in child_tree_graphs:
        # print(f">>>>>>>>>>>>>>>>>>>>>>... child_tree_graph(communication greedy) is: {ctg} ")
        # print(ctg.nodes)
        # print(ctg.edges)
    ret_tree = merge_graphs(child_tree_graphs)
    # print(f">>>>>>>>>>>>>>>>>>>>>>... ret_tree(communication greedy) is: {ret_tree} ")
    # print(ret_tree.nodes)
    # print(ret_tree.edges)
    return ret_tree


def construct_satellite_microservice_fix_topo_tree_colocation_di_graph(
        constellation: Constellation, microservice_cluster: List, user_request_service_tree, constellation_topo_graph):
    leaf_nodes = [node for node in user_request_service_tree.nodes() if user_request_service_tree.out_degree(node) == 0]
    head_nodes = [node for node in user_request_service_tree.nodes() if user_request_service_tree.in_degree(node) == 0]
    head_leaf_node_pairs = [[head_nodes[0], leaf_node] for leaf_node in leaf_nodes]
    child_tree_chains = [
        nx.shortest_path(user_request_service_tree, source=head_tail_pair[0], target=head_tail_pair[1]) for head_tail_pair in head_leaf_node_pairs]

    microservice_satellite_pairs = {}
    for child_chain in child_tree_chains:
        for service in child_chain:
            microservice_satellite_pairs[service] = []
            for satellite in constellation.satellites_group:
                if service in satellite.microservice_set:
                    microservice_satellite_pairs[service].append(satellite.id)

    child_tree_graphs = []
    for i, child_tree in enumerate(child_tree_chains):
        CTG = construct_satellite_microservice_chain_colocation_di_graph2(constellation, microservice_cluster, child_tree, constellation_topo_graph)
        mapping = {constellation.satellites_group[-1].id: f"{constellation.satellites_group[-1].id}-{i + 1}"}
        CTG = nx.relabel_nodes(CTG, mapping, copy=True)
        child_tree_graphs.append(CTG)

    # for ctg in child_tree_graphs:
    #     print(f">>>>>>>>>>>>>>>>>>>>>>... child_tree_graph(communication greedy) is: {ctg} ")
    #     print(ctg.nodes)
    #     print(ctg.edges)
    augmented_graph = merge_graphs(child_tree_graphs)
    # print(f">>>>>>>>>>>>>>>>>>>>>>... ret_tree(communication greedy) is: {ret_tree} ")
    # print(ret_tree.nodes)
    # print(ret_tree.edges)
    return augmented_graph
