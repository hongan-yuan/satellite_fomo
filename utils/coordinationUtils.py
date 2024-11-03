# -*- coding: utf-8 -*-
"""
@Time: 6/11/2024 11:05 AM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import numpy as np

from BaseObject.satellite import Satellite


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def get_cartesian_dist(satellite1: Satellite, satellite2: Satellite) -> float:
    cartesian_dist = np.sqrt(
        (satellite2.cartesian_x - satellite1.cartesian_x) ** 2 +
        (satellite2.cartesian_y - satellite1.cartesian_y) ** 2 +
        (satellite2.cartesian_z - satellite1.cartesian_z) ** 2
    )
    return cartesian_dist


def get_spherical_dist(satellite1: Satellite, satellite2: Satellite) -> float:
    spherical_dist = np.sqrt(
        satellite1.spherical_r ** 2 + satellite2.spherical_r ** 2 -
        2 * satellite1.spherical_r * satellite2.spherical_r *
        (np.sin(satellite1.spherical_theta) * np.sin(satellite2.spherical_theta) * np.cos(satellite1.spherical_phi - satellite2.spherical_phi) +
         np.cos(satellite1.spherical_theta) * np.cos(satellite2.spherical_theta))
    )
    return spherical_dist
