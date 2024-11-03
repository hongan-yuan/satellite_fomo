# -*- coding: utf-8 -*-
"""
@Time : 
@Author: Honggang Yuan
@Email: hn_yuanhg@163.com
Description:
    
"""
import time

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union


class EarthInfo(BaseModel):
    radius: float = 6371e3  # the Radius of Earth
    mass: float = 5964e21
    gravitation_const: float = 667428e-16
    kepler_const: float = 3986e11
    earth_rotation_angular_velocity: float = 729211510e-12
