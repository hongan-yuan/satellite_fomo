# -*- coding: utf-8 -*-
"""
@Time: 6/12/2024 3:28 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import random
from typing import List

import networkx as nx
import numpy as np
from networkx import Graph

from BaseObject.constellation import Constellation


def construct_constellation_graph(constellation: Constellation):
    graph = nx.Graph()
    # add satellite node (id)
    for satellite in constellation.satellites_group:
        graph.add_node(satellite.id)
    # add edge with neighbor for each satellite
    for satellite in constellation.satellites_group:
        for neighbor_id in satellite.neighbor:
            graph.add_edge(satellite.id, neighbor_id, weight=satellite.neighbor[neighbor_id])
    return graph


def construct_satellite_microservice_colocation_di_graph(edges, weights):
    diGraph = nx.DiGraph()
    # add satellite node
    diGraph.add_edges_from(edges)
    for edge, weight in zip(edges, weights):
        diGraph.edges[edge]['weight'] = weight
    return diGraph


def get_satellite_pair_communication_cost(
        constellation: Constellation,
        constellation_topo_graph: Graph,
        source_satellite: int,
        target_satellite: int,
        microservice_cluster: List,
        microservice_id: int,
):
    k = 1.38e-23  # Boltzmann constant in J/K
    T_b = 6000  # Solar brightness temperature in K
    T_0 = 1000  # Typical system noise temperature in K (assumed value, can be changed)
    T_CMB = 2.725  # Cosmic Microwave Background temperature in K
    Bandwidth = 193e12 * 0.02
    lamada = 1550e-9  # wavelength
    # Total noise temperature
    T_total = T_b + T_0 + T_CMB
    P_N = k * T_total * Bandwidth
    communication_cost = 0
    if source_satellite == target_satellite:
        return 0
    else:
        # source_satellite_obj = constellation.satellites_group[source_satellite]
        # target_satellite_obj = constellation.satellites_group[target_satellite]
        try:
            satellite_path = nx.shortest_path(G=constellation_topo_graph, source=source_satellite, target=target_satellite, weight="weight")
        except Exception as exp:
            print(f"find the shortest path error: \n{exp}")
            print(constellation_topo_graph.nodes)
        if len(satellite_path) == 2:
            p_len = constellation.satellites_group[source_satellite].neighbor[target_satellite]
            FSPL = ((4 * np.pi * p_len) / lamada) ** 2
            receiver_power = constellation.satellites_group[source_satellite].transmitter_power * 0.8 * constellation.satellites_group[
                source_satellite].transmit_gain * \
                             constellation.satellites_group[target_satellite].receiver_gain * FSPL
            DataRate = Bandwidth * np.log2(1 + (receiver_power / P_N))
            # print(f"DataRate is ... {DataRate}")
            communication_cost = constellation.satellites_group[source_satellite].transmitter_power * (
                    microservice_cluster[microservice_id].dataVol / DataRate)
            return communication_cost
        else:
            for s_id, satellite_node in enumerate(satellite_path[:-1]):
                p_len = constellation.satellites_group[satellite_node].neighbor[satellite_path[s_id + 1]]
                FSPL = ((4 * np.pi * p_len) / lamada) ** 2
                receiver_power = constellation.satellites_group[satellite_node].transmitter_power * 0.8 * \
                                 constellation.satellites_group[satellite_node].transmit_gain * \
                                 constellation.satellites_group[satellite_path[s_id + 1]].receiver_gain * FSPL
                DataRate = Bandwidth * np.log2(1 + (receiver_power / P_N))
                communication_cost += (constellation.satellites_group[satellite_node].transmitter_power *
                                       (microservice_cluster[microservice_id].dataVol / DataRate))
            return communication_cost


def get_microservice_computing_cost(
        constellation: Constellation,
        satellite: int,
        microservice_cluster: List,
        microservice_id: int,
):
    computing_cost = 0
    if microservice_id in constellation.satellites_group[satellite].microservice_set:
        computing_cost = (constellation.satellites_group[satellite].power *
                          (microservice_cluster[microservice_id].dataVol /
                           constellation.satellites_group[satellite].computing_cab))
    else:
        print(f"Microservice ... {microservice_id} NOT IN Satellite ... {satellite}. Please double check !!!")
        raise Exception
    computing_cost = computing_cost / 2
    return computing_cost
