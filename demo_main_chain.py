# -*- coding: utf-8 -*-
"""
@Time: 6/12/2024 10:15 AM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import csv

import numpy as np

from BaseObject.earthInfo import EarthInfo
from BaseObject.constellation import ConstellationConf
from BaseObject.microservice import Microservice
from utils.microserviceUtils import *

np.random.seed = 30

microservice_cnt = 72
dataVol_mean = 0
dataVol_variance = 0
std_dev = np.sqrt(dataVol_variance)
micro_dataVols = np.round(np.random.normal(loc=dataVol_mean, scale=std_dev, size=microservice_cnt), 2)
MicroserviceCluster = []
earthInfo = EarthInfo()
constellationConf = ConstellationConf(
    orbit_num=6,
    orbit_idx=[0, 1, 2, 3, 4, 5],
    orbit_altitude=[1000e3, 1000e3, 1000e3, 1000e3, 1000e3, 1000e3],
    orbit_inclinations=[99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
    satellites_num_distribution=[12, 12, 12, 12, 12, 12],
    satellites_per_orbit=12,
    link_data_rate=10
)  # init a Constellation


def chain_of_service():
    # Generating a chain of service randomly
    user_request_service_chain_len = 10
    user_request_service_chain = generate_user_request_chain(microservice_cnt, user_request_service_chain_len)
    # user_request_service_chain = [10, 21, 24]
    print(f"user_request_service_chain ... {user_request_service_chain}")

    chain_graph = construct_satellite_microservice_chain_colocation_di_graph(constellation, MicroserviceCluster, user_request_service_chain, constellation_g)
    print(f"chain_graph ... {chain_graph}")
    print(f"chain_graph edges ... {chain_graph.edges}")
    print(f"chain_graph nodes ... {chain_graph.nodes}")
    chain_shortest_path_cost = nx.shortest_path_length(G=chain_graph,
                                                       source=constellation.satellites_group[0].id,
                                                       target=constellation.satellites_group[-1].id,
                                                       weight="weight")
    chain_shortest_path = nx.shortest_path(G=chain_graph,
                                           source=constellation.satellites_group[0].id,
                                           target=constellation.satellites_group[-1].id,
                                           weight="weight")
    print(f"chain_graph ... {chain_graph}")
    print(f"chain_shortest_path {chain_shortest_path}")
    print('The total chain_shortest_path_cost is {:.5e} J'.format(chain_shortest_path_cost))
    print("----------------------------------------------------------------")
    all_cost = [chain_shortest_path_cost]
    return user_request_service_chain, all_cost


if __name__ == '__main__':
    constellation = Constellation(constellation_conf=constellationConf, earth=earthInfo)
    for i in range(microservice_cnt):
        microservice = Microservice(id=i,
                                    dataVol=micro_dataVols[i],
                                    replica_num=np.random.randint(2, 6),
                                    isStartPoint=False,
                                    isEndPoint=False,
                                    successor={0: 0},
                                    predecessor={0: 0})
        MicroserviceCluster.append(microservice)
    # deploy microservice on satellites
    for service in MicroserviceCluster:
        constellation.satellites_group[service.id].microservice_set.append(service.id)
        service.deployment.append(service.id)
    # constellation.update_satellite_pos(random.randint(100, 1000))
    constellation_g = construct_constellation_graph(constellation)


    cnt = 0
    err_list = []
    cost_list = []
    chain_list = []
    for i in range(10):
        service_tree, total_costs = chain_of_service()
        chain_list.append(service_tree)
        cost_list.append(cost for cost in total_costs)
        min_cost = min(total_costs)
        if total_costs.index(min_cost) != 3:
            cnt += 1
            err_list.append(total_costs)
    print(f"$$$$$$$$$$$$$$$$$$$$$$>>> .. cnt is:{cnt}")
    for el in err_list:
        print(el)
