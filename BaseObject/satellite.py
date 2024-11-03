# -*- coding: utf-8 -*-
"""
@Time : 5/7/2024 4:22 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description:
    Satellites
"""
import random
from typing import Dict, Optional, List, Literal

from pydantic import BaseModel

from BaseObject.microservice import Microservice
from BaseObject.orbit import Orbit


class Satellite(BaseModel):
    """
    Satellite
    """
    id: int = -1  # number of satellite
    computing_cab: Optional[int] = -1  # GFLOPS, Literal[100, 160, 220, 280, 340]  # GFLOPS
    power: Optional[float] = 0.1  # Power
    transmitter_power: float = 0.1  # battery ... unit W
    transmit_gain: Optional[float] = 0.1
    receiver_gain: Optional[float] = 0.1

    orbit: Optional[Orbit] = None  # number of orbit
    orbit_period: float = 0  # s
    orbit_altitude: float = 0  # m
    inclination: float = 0  # deg

    velocity: float = 0  # m/s
    cartesian_x: float = 0
    cartesian_y: float = 0
    cartesian_z: float = 0

    angular_velocity: float = 0
    spherical_r: float = 0
    spherical_theta: float = 0
    spherical_phi: float = 0

    neighbor: Optional[Dict] = {}  # reachable satellites, format: no.of satellite: distance

    microservice_set: Optional[List] = []  # set of deployed microservice

    isStartPoint: bool = False
    isEndPoint: bool = False


class SatelliteRunData(BaseModel):
    id: int = -1  # number of satellite
    computing_cab: Optional[int] = -1  # GFLOPS, Literal[100, 160, 220, 280, 340]  # GFLOPS
    isStartPoint: bool = False
    isEndPoint: bool = False
    orbit: int = 0  # number of orbit
    orbit_period: float = 0  # s
    orbit_altitude: float = 0  # m
    inclination: float = 0  # deg
    velocity: float = 0  # m/s
    cartesian_x: float = 0
    cartesian_y: float = 0
    cartesian_z: float = 0

    angular_velocity: float = 0
    spherical_r: float = 0
    spherical_theta: float = 0
    spherical_phi: float = 0
    neighbor: Optional[Dict] = {}  # reachable satellites, format: #.satellite: distance


if __name__ == '__main__':
    BO = BaseModel
    satellite = Satellite(id=1, power=1000, )
    print(satellite.model_fields)
