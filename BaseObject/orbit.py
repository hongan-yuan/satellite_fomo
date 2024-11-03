# -*- coding: utf-8 -*-
"""
@Time : 5/7/2024 4:22 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description:
    Orbit
"""
from typing import List, Optional

from pydantic import BaseModel


class Orbit(BaseModel):
    """
    Orbit
    """

    id: int = -1  # number of orbit
    altitude: float = 0  # the height of Orbit ... unit: km
    period: float = 0
    inclination: float = 0
    satelliteSet: Optional[List] = []  # the satellite set belongs to the orbit ... [satellite-1, satellite-2, ...]
