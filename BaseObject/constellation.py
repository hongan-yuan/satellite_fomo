# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import random
import threading
from typing import List, Optional

import numpy as np
import scipy.io
from pydantic import BaseModel

from BaseObject.earthInfo import EarthInfo
from BaseObject.orbit import Orbit
from BaseObject.satellite import Satellite, SatelliteRunData
from utils.coordinationUtils import cartesian_to_spherical, spherical_to_cartesian, get_spherical_dist
from utils.satelliteUtils import update_satellite_position

seed = 42
random.seed(seed)


class ConstellationConf(BaseModel):
    """
    Define a Constellation.
    """
    name: Optional[str] = "Telesat"
    orbit_num: int = 4
    orbit_idx: List = [0, 1, 2, 3]
    orbit_altitude: List = [600e3, 620e3, 600e3, 620e3]  # the altitude of satellites (km)
    orbit_inclinations: List = [30, 35, 40, 45]
    satellites_num_distribution: List = [20, 20, 20, 20]
    satellites_per_orbit: int = 20
    satellites_computing_cap: List = [100, 160, 220, 280, 340, 400]
    update_interval: int = 30
    link_data_rate: int = 5  # Gbps
    other_communication_loss: int = 10


class Constellation:
    def __init__(self, constellation_conf: ConstellationConf, earth: EarthInfo):
        self.constellation_conf = constellation_conf
        self.name = constellation_conf.name
        self.earth = earth
        self.orbit_num = constellation_conf.orbit_num
        self.orbit_idx = constellation_conf.orbit_idx
        self.orbit_altitude = constellation_conf.orbit_altitude
        self.orbit_inclinations = constellation_conf.orbit_inclinations
        self.satellites_per_orbit = constellation_conf.satellites_per_orbit
        self.satellites_computing_cap = constellation_conf.satellites_computing_cap
        self.satellites_distribution = constellation_conf.satellites_num_distribution
        self.other_communication_loss = constellation_conf.other_communication_loss
        self.link_data_rate = constellation_conf.link_data_rate
        self.satellites_total_sum = sum(self.satellites_distribution)
        self.init_constellation()
        self.update_timer = threading.Timer(constellation_conf.update_interval, self.update_satellite_pos,
                                            args=[constellation_conf.update_interval])

    def init_constellation(self):
        self._init_satellite_neighbor()
        # print("--- 1 ... Satellite neighbors initialization done.")
        self._init_orbits()
        # print("--- 2 ... Orbits initialization done.")
        self._init_satellites()
        self._colocate_satellite_orbit()
        # print("--- 3 ... Satellite-Orbit colocation initialization done.")
        self._init_satellite_conf()
        # print("--- 4 ... Satellite Info initialization done.")

    def _init_satellite_neighbor(self):
        self.satellite_neighbor = {}
        for i in range(self.satellites_total_sum):
            if i < self.satellites_per_orbit:
                if i == 0:
                    self.satellite_neighbor[i] = [1,
                                                  self.satellites_per_orbit - 1,
                                                  self.satellites_per_orbit]
                elif i == (self.satellites_per_orbit - 1):
                    self.satellite_neighbor[i] = [0,
                                                  self.satellites_per_orbit - 2,
                                                  2 * self.satellites_per_orbit - 1]
                else:
                    self.satellite_neighbor[i] = [i - 1,
                                                  i + 1,
                                                  i + self.satellites_per_orbit]
            elif self.satellites_per_orbit <= i < ((len(self.satellites_distribution) - 1) * self.satellites_per_orbit):
                if i % self.satellites_per_orbit == 0:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i + 1,
                                                  i + self.satellites_per_orbit - 1,
                                                  i + self.satellites_per_orbit]
                elif i % self.satellites_per_orbit == (self.satellites_per_orbit - 1):
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - self.satellites_per_orbit + 1,
                                                  i - 1,
                                                  i + self.satellites_per_orbit]
                else:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - 1,
                                                  i + 1,
                                                  i + self.satellites_per_orbit]
            else:
                if i % self.satellites_per_orbit == 0:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i + 1,
                                                  i + self.satellites_per_orbit - 1]

                elif i % self.satellites_per_orbit == (self.satellites_per_orbit - 1):
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - self.satellites_per_orbit + 1,
                                                  i - 1]
                else:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - 1,
                                                  i + 1]

    def _init_orbits(self):
        self.orbit_group = []
        for orbit_idx in range(self.orbit_num):
            orbit_R = self.earth.radius + self.orbit_altitude[orbit_idx]
            orbit_T = 2 * np.pi * np.sqrt((orbit_R ** 3) / (self.earth.mass * self.earth.gravitation_const))
            self.orbit_group.append(
                Orbit(id=orbit_idx,
                      altitude=orbit_R,
                      inclination=self.orbit_inclinations[orbit_idx],
                      period=orbit_T))

    def _init_satellites(self):
        # init the computing capability of satellites
        min_gain = 10 ** (81.67 / 10)
        max_gain = 10 ** (81.67 / 10)
        transmit_gains = [random.uniform(min_gain, max_gain) for _ in range(self.satellites_per_orbit * self.orbit_num)]
        receiver_gains = [random.uniform(min_gain, max_gain) for _ in range(self.satellites_per_orbit * self.orbit_num)]
        self.satellites_group = []
        for satellite_idx in range(self.satellites_total_sum):
            s_comp_cap = self.satellites_computing_cap[random.randint(0, len(self.satellites_computing_cap) - 1)]
            s_power = s_comp_cap / 100  # rated computing power
            self.satellites_group.append(
                # Satellite(id=satellite_idx, computing_cab=s_comp_cap, power=s_power, transmit_gain=10000, receiver_gain=10000))
                Satellite(id=satellite_idx, computing_cab=s_comp_cap, power=s_power, transmit_gain=transmit_gains, receiver_gain=receiver_gains))
        self.satellites_group[0].isStartPoint = True
        self.satellites_group[-1].isEndPoint = True

    def _colocate_satellite_orbit(self):
        # bind satellite and orbit
        satellite_start = 0
        for i, num_satellites in enumerate(self.satellites_distribution):
            satellite_end = satellite_start + num_satellites
            for satellite in self.satellites_group[satellite_start:satellite_end]:
                satellite.orbit = self.orbit_group[i]
            satellite_start = satellite_end
        # bind orbit and satellite
        satellite_start = 0
        for i, num_satellites in enumerate(self.satellites_distribution):
            satellite_end = satellite_start + num_satellites
            self.orbit_group[i].satelliteSet = self.satellites_group[satellite_start:satellite_end]
            satellite_start = satellite_end

    def _init_satellite_conf(self):
        """
        cartesian coordination (x, y, z)
        orbit_period
        velocity
        angular_velocity
        inclination
        """
        for o_i, orbit in enumerate(self.orbit_group):
            inclination_rad = np.radians(orbit.inclination)  # transfer degree to radian
            # satellite_angles_start_point = random.random()
            satellite_angles_start_point = o_i / 10
            satellite_angles = np.linspace(satellite_angles_start_point * np.pi,
                                           2 * np.pi + satellite_angles_start_point * np.pi,
                                           self.satellites_per_orbit,
                                           endpoint=False)
            satellite_angles %= 2 * np.pi
            for s_i, satellite in enumerate(orbit.satelliteSet):
                satellite.inclination = orbit.inclination

                satellite.orbit_altitude = orbit.altitude
                satellite.orbit_period = orbit.period

                satellite.velocity = np.sqrt(self.earth.gravitation_const * self.earth.mass / orbit.altitude)
                satellite.cartesian_x = orbit.altitude * np.cos(satellite_angles[s_i])
                satellite.cartesian_y = orbit.altitude * np.sin(satellite_angles[s_i]) * np.cos(inclination_rad)
                satellite.cartesian_z = orbit.altitude * np.sin(satellite_angles[s_i]) * np.sin(inclination_rad)

                satellite.angular_velocity = satellite.velocity / satellite.orbit_altitude
                satellite.spherical_r, satellite.spherical_theta, satellite.spherical_phi = cartesian_to_spherical(
                    satellite.cartesian_x,
                    satellite.cartesian_y,
                    satellite.cartesian_z)
        for idx, satellite_item in enumerate(self.satellites_group):
            for neighbor_id in self.satellite_neighbor[idx]:
                los_dist = (np.sqrt((satellite_item.orbit_altitude - self.earth.radius) * (
                        satellite_item.orbit_altitude + self.earth.radius)) +
                            np.sqrt(
                                (self.satellites_group[neighbor_id].orbit_altitude - self.earth.radius) * (
                                        self.satellites_group[neighbor_id].orbit_altitude + self.earth.radius)))
                dist = get_spherical_dist(satellite_item, self.satellites_group[neighbor_id])
                if dist < los_dist:
                    satellite_item.neighbor[neighbor_id] = get_spherical_dist(satellite_item,
                                                                              self.satellites_group[neighbor_id])

    def update_satellite_neighbor(self):
        for idx, satellite_item in enumerate(self.satellites_group):
            for neighbor_id in self.satellite_neighbor[idx]:
                los_dist = np.sqrt((satellite_item.orbit_altitude - self.earth.radius) * (
                        satellite_item.orbit_altitude + self.earth.radius)) + np.sqrt(
                    (self.satellites_group[neighbor_id].orbit_altitude - self.earth.radius) * (
                            self.satellites_group[neighbor_id].orbit_altitude + self.earth.radius))
                dist = get_spherical_dist(satellite_item, self.satellites_group[neighbor_id])
                if dist < los_dist:
                    satellite_item.neighbor[neighbor_id] = get_spherical_dist(satellite_item,
                                                                              self.satellites_group[neighbor_id])

    def update_satellite_pos(self, interval):
        for o_i, orbit in enumerate(self.orbit_group):
            for s_i, satellite in enumerate(orbit.satelliteSet):
                satellite.spherical_r, satellite.spherical_theta, satellite.spherical_phi = update_satellite_position(
                    initial_r=satellite.spherical_r,
                    initial_theta=satellite.spherical_theta,
                    initial_phi=satellite.spherical_phi,
                    omega=satellite.angular_velocity,
                    time_interval_s=interval
                )
                satellite.cartesian_x, satellite.cartesian_y, satellite.cartesian_z = spherical_to_cartesian(
                    r=satellite.spherical_r,
                    theta=satellite.spherical_theta,
                    phi=satellite.spherical_phi
                )
        self.update_satellite_neighbor()

    def update_satellite_pos_by_matlab(self, spherical_satellite_data, cartesian_satellite_data, interval):
        for s_i, satellite in enumerate(self.satellites_group):
            satellite.spherical_r, satellite.spherical_theta, satellite.spherical_phi = (
                spherical_satellite_data[s_i][interval][0], spherical_satellite_data[s_i][interval][1], spherical_satellite_data[s_i][interval][2])
            satellite.cartesian_x, satellite.cartesian_y, satellite.cartesian_z = (
                cartesian_satellite_data[s_i][interval][0], cartesian_satellite_data[s_i][interval][1], cartesian_satellite_data[s_i][interval][2])
        self.update_satellite_neighbor()

    def get_satellite_transmitter_power(self, source_satellite: Satellite, distance):
        t_power = source_satellite.transmitter_power * distance * self.other_communication_loss
        return t_power

    def get_satellite_transmitter_power_by_id(self, source_satellite: int, distance):
        path_loss = ((4 * np.pi * distance) / 1.5e-6) ** 2
        t_power = self.satellites_group[source_satellite].transmitter_power * path_loss * self.other_communication_loss
        return t_power


class TelesatConstellation(Constellation):
    def __init__(self, constellation_conf: ConstellationConf, earth: EarthInfo, total_time_step, running_data_path):
        # super().__init__(constellation_conf, earth)
        self.constellation_conf = constellation_conf
        self.name = constellation_conf.name
        self.earth = earth
        self.total_time_step = total_time_step
        self.orbit_num = constellation_conf.orbit_num
        self.orbit_idx = constellation_conf.orbit_idx
        self.orbit_altitude = constellation_conf.orbit_altitude
        self.orbit_inclinations = constellation_conf.orbit_inclinations
        self.satellites_per_orbit = constellation_conf.satellites_per_orbit
        self.satellites_computing_cap = constellation_conf.satellites_computing_cap
        self.satellites_distribution = constellation_conf.satellites_num_distribution
        self.other_communication_loss = constellation_conf.other_communication_loss
        self.link_data_rate = constellation_conf.link_data_rate
        self.satellites_total_sum = sum(self.satellites_distribution)
        self.SPHERICAL_SATELLITE_DATA = []
        self.CARTESIAN_SPHERICAL_SATELLITE_DATA = []
        self.init_satellite_position_data(running_data_path)
        self.init_constellation()

    def init_satellite_position_data(self, data_path):
        matlab_contents = scipy.io.loadmat(data_path)
        data_list = []
        for var in matlab_contents:
            if not var.startswith('_'):
                data_list.append(matlab_contents[var])
        spherical_position_data = np.array(data_list[0])
        cartesian_position_data = np.array(data_list[1])
        for satellite_id in range(self.satellites_total_sum):
            self.SPHERICAL_SATELLITE_DATA.append([])
            self.CARTESIAN_SPHERICAL_SATELLITE_DATA.append([])
            spherical_satellite_position_data = spherical_position_data[satellite_id][0]  # 3*6299
            cartesian_satellite_position_data = cartesian_position_data[satellite_id][0]  # 3*6299
            for time_step in range(self.total_time_step):
                self.SPHERICAL_SATELLITE_DATA[satellite_id].append(
                    [spherical_satellite_position_data[0][time_step],
                     spherical_satellite_position_data[1][time_step],
                     spherical_satellite_position_data[2][time_step]]
                )
                self.CARTESIAN_SPHERICAL_SATELLITE_DATA[satellite_id].append(
                    [cartesian_satellite_position_data[0][time_step],
                     cartesian_satellite_position_data[1][time_step],
                     cartesian_satellite_position_data[2][time_step]]
                )

    def init_constellation(self):
        self._init_satellite_neighbor()
        # print("--- 1 ... Satellite neighbors initialization done.")
        self._init_orbits()
        # print("--- 2 ... Orbits initialization done.")
        self._init_satellites_computation_capability()
        self._colocate_satellite_orbit()
        # print("--- 3 ... Satellite-Orbit colocation initialization done.")
        self._init_satellite_position()
        # print("--- 4 ... Satellite Info initialization done.")

    def _init_satellite_neighbor(self):
        self.satellite_neighbor = {}
        for i in range(self.satellites_total_sum):
            if i < self.satellites_per_orbit:
                if i == 0:
                    self.satellite_neighbor[i] = [1,
                                                  self.satellites_per_orbit - 1,
                                                  self.satellites_per_orbit]
                elif i == (self.satellites_per_orbit - 1):
                    self.satellite_neighbor[i] = [0,
                                                  self.satellites_per_orbit - 2,
                                                  2 * self.satellites_per_orbit - 1]
                else:
                    self.satellite_neighbor[i] = [i - 1,
                                                  i + 1,
                                                  i + self.satellites_per_orbit]
            elif self.satellites_per_orbit <= i < ((len(self.satellites_distribution) - 1) * self.satellites_per_orbit):
                if i % self.satellites_per_orbit == 0:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i + 1,
                                                  i + self.satellites_per_orbit - 1,
                                                  i + self.satellites_per_orbit]
                elif i % self.satellites_per_orbit == (self.satellites_per_orbit - 1):
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - self.satellites_per_orbit + 1,
                                                  i - 1,
                                                  i + self.satellites_per_orbit]
                else:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - 1,
                                                  i + 1,
                                                  i + self.satellites_per_orbit]
            else:
                if i % self.satellites_per_orbit == 0:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i + 1,
                                                  i + self.satellites_per_orbit - 1]

                elif i % self.satellites_per_orbit == (self.satellites_per_orbit - 1):
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - self.satellites_per_orbit + 1,
                                                  i - 1]
                else:
                    self.satellite_neighbor[i] = [i - self.satellites_per_orbit,
                                                  i - 1,
                                                  i + 1]

    def _init_orbits(self):
        self.orbit_group = []
        for orbit_idx in range(self.orbit_num):
            orbit_R = self.earth.radius + self.orbit_altitude[orbit_idx]
            orbit_T = -1
            self.orbit_group.append(
                Orbit(id=orbit_idx,
                      altitude=orbit_R,
                      inclination=self.orbit_inclinations[orbit_idx],
                      period=orbit_T))

    def _init_satellites_computation_capability(self):
        # init the computing capability of satellites
        self.satellites_group = []
        for satellite_idx in range(self.satellites_total_sum):
            s_comp_cap = self.satellites_computing_cap[random.randint(0, len(self.satellites_computing_cap) - 1)]
            s_power = s_comp_cap * 0.1
            self.satellites_group.append(
                Satellite(id=satellite_idx, computing_cab=s_comp_cap, power=s_power))
        self.satellites_group[0].isStartPoint = True
        self.satellites_group[-1].isEndPoint = True

    def _colocate_satellite_orbit(self):
        # bind satellite and orbit
        satellite_start = 0
        for i, num_satellites in enumerate(self.satellites_distribution):
            satellite_end = satellite_start + num_satellites
            for satellite in self.satellites_group[satellite_start:satellite_end]:
                satellite.orbit = self.orbit_group[i]
            satellite_start = satellite_end
        # bind orbit and satellite
        satellite_start = 0
        for i, num_satellites in enumerate(self.satellites_distribution):
            satellite_end = satellite_start + num_satellites
            self.orbit_group[i].satelliteSet = self.satellites_group[satellite_start:satellite_end]
            satellite_start = satellite_end

    def _init_satellite_position(self):
        """
        cartesian coordination (x, y, z)
        orbit_period
        velocity
        angular_velocity
        inclination
        """
        for i, satellite in enumerate(self.satellites_group):
            satellite.orbit_altitude = self.orbit_altitude[0]
            # satellite.inclination = self.orbit_inclinations
            satellite.spherical_r = self.SPHERICAL_SATELLITE_DATA[i][0][0]
            satellite.spherical_phi = self.SPHERICAL_SATELLITE_DATA[i][0][1]
            satellite.spherical_theta = self.SPHERICAL_SATELLITE_DATA[i][0][2]
            satellite.cartesian_x = self.CARTESIAN_SPHERICAL_SATELLITE_DATA[i][0][0]
            satellite.cartesian_y = self.CARTESIAN_SPHERICAL_SATELLITE_DATA[i][0][1]
            satellite.cartesian_z = self.CARTESIAN_SPHERICAL_SATELLITE_DATA[i][0][2]

        for idx, satellite_item in enumerate(self.satellites_group):
            for neighbor_id in self.satellite_neighbor[idx]:
                los_dist = (np.sqrt((satellite_item.orbit_altitude - self.earth.radius) * (
                        satellite_item.orbit_altitude + self.earth.radius)) +
                            np.sqrt(
                                (self.satellites_group[neighbor_id].orbit_altitude - self.earth.radius) * (
                                        self.satellites_group[neighbor_id].orbit_altitude + self.earth.radius)))
                dist = get_spherical_dist(satellite_item, self.satellites_group[neighbor_id])
                if dist < los_dist:
                    satellite_item.neighbor[neighbor_id] = get_spherical_dist(satellite_item,
                                                                              self.satellites_group[neighbor_id])

    def update_satellite_neighbor(self):
        for idx, satellite_item in enumerate(self.satellites_group):
            for neighbor_id in self.satellite_neighbor[idx]:
                los_dist = np.sqrt((satellite_item.orbit_altitude - self.earth.radius) * (
                        satellite_item.orbit_altitude + self.earth.radius)) + np.sqrt(
                    (self.satellites_group[neighbor_id].orbit_altitude - self.earth.radius) * (
                            self.satellites_group[neighbor_id].orbit_altitude + self.earth.radius))
                dist = get_spherical_dist(satellite_item, self.satellites_group[neighbor_id])
                if dist < los_dist:
                    satellite_item.neighbor[neighbor_id] = get_spherical_dist(satellite_item,
                                                                              self.satellites_group[neighbor_id])

    def update_satellite_pos(self, cur_step, interval):
        for i, satellite in enumerate(self.satellites_group):
            satellite.spherical_r = self.SPHERICAL_SATELLITE_DATA[i][cur_step + interval][0]
            satellite.spherical_phi = self.SPHERICAL_SATELLITE_DATA[i][cur_step + interval][1]
            satellite.spherical_theta = self.SPHERICAL_SATELLITE_DATA[i][cur_step + interval][2]
            satellite.cartesian_x = self.CARTESIAN_SPHERICAL_SATELLITE_DATA[i][cur_step + interval][0]
            satellite.cartesian_y = self.CARTESIAN_SPHERICAL_SATELLITE_DATA[i][cur_step + interval][1]
            satellite.cartesian_z = self.CARTESIAN_SPHERICAL_SATELLITE_DATA[i][cur_step + interval][2]
        self.update_satellite_neighbor()

    def get_satellite_transmitter_power(self, source_satellite: Satellite, distance):
        t_power = source_satellite.transmitter_power * distance * self.other_communication_loss
        return t_power

    def get_satellite_transmitter_power_by_id(self, source_satellite: int, distance):
        path_loss = ((4 * np.pi * distance) / 1.5e-6) ** 2
        t_power = self.satellites_group[source_satellite].transmitter_power * path_loss * self.other_communication_loss
        return t_power


if __name__ == '__main__':
    constellationConf = ConstellationConf(
        orbit_num=4,
        orbit_idx=[0, 1, 2, 3],
        orbit_altitude=[600e3, 620e3, 600e3, 620e3],
        orbit_inclinations=[30, 40, 50, 60],
        satellites_num_distribution=[10, 10, 10, 10],
        satellites_per_orbit=10
    )  # init a Constellation
    earthInfo = EarthInfo()  # init an Earth
    # Constructing a Constellation
    constellation = Constellation(constellation_conf=constellationConf, earth=earthInfo)
    # constellation.initConstellation()
    # constellation.update_timer.start()

    # print(constellation.orbit_group[0].satelliteSet)
    # for sate in constellation.orbit_group[3].satelliteSet:
    #     print(sate.velocity)

    SatelliteRunningData = []
    constellation.update_satellite_pos(5)
    constellation.update_satellite_neighbor()
    for satellite_child in constellation.satellites_group:
        satellite_run_data = SatelliteRunData(
            id=satellite_child.id,
            isStartPoint=satellite_child.isStartPoint,
            isEndPoint=satellite_child.isEndPoint,
            orbit=satellite_child.orbit.id,
            orbit_period=satellite_child.orbit_period,
            orbit_altitude=satellite_child.orbit_altitude,
            inclination=satellite_child.inclination,
            velocity=satellite_child.velocity,
            cartesian_x=satellite_child.cartesian_x,
            cartesian_y=satellite_child.cartesian_y,
            cartesian_z=satellite_child.cartesian_z,
            angular_velocity=satellite_child.angular_velocity,
            spherical_r=satellite_child.spherical_r,
            spherical_theta=satellite_child.spherical_theta,
            spherical_phi=satellite_child.spherical_phi,
            neighbor=satellite_child.neighbor
        )
        SatelliteRunningData.append(satellite_run_data.model_dump_json())
    print("***************************************************************")
    print(SatelliteRunningData[0])
    print(SatelliteRunningData[1])
    print(SatelliteRunningData[-1])
    # while True:
    #
    #     time.sleep(5)
    # print(constellation.satellites_group[1].orbit)
    # print(constellation.satellites_group[2].orbit)
    # print(constellation.satellites_group[10].orbit)
    # print(constellation.satellites_group[22].orbit)
    # print(constellation.satellites_group[32].orbit)
