"""Planet model for trajectory simulation.

A PlanetModel bundles the physical parameters of a celestial body — radius,
gravitational parameter, rotation rate, and atmosphere — that affect vehicle
trajectories during atmospheric flight.

The module defines standard Earth models (rotating and non-rotating variants).
Additional planets (Mars, Venus, etc.) can be added as new PlanetModel instances.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

from reentrykit.atmosphere import (
    MAX_EXTENDED_ALTITUDE as _EARTH_MAX_ATMOSPHERE,
    AtmosphereState,
    us1976,
)


class PlanetModel(NamedTuple):
    """Physical properties of a planetary body for trajectory simulation.

    Attributes
    ----------
    name : str
        Human-readable identifier for the planet.
    radius : float
        Equatorial radius [m]. Used in altitude-to-radius conversion and
        centripetal acceleration calculations.
    gravitational_parameter : float
        Standard gravitational parameter G*M [m^3/s^2]. Gravity at radius r
        is computed as g(r) = gravitational_parameter / r^2.
    rotation_rate : float
        Angular velocity about the polar axis [rad/s]. Set to 0.0 for
        non-rotating-planet simulations (test/regression use).
    atmosphere : callable
        Function mapping altitude [m] to AtmosphereState. Provides density,
        pressure, temperature, and speed of sound at each altitude.
    max_atmosphere_altitude : float
        Highest altitude [m] at which the atmosphere function is defined.
        Above this altitude the vehicle is treated as being in vacuum.
    """

    name: str
    radius: float                                     # [m]
    gravitational_parameter: float                    # [m^3/s^2]
    rotation_rate: float                              # [rad/s]
    atmosphere: Callable[[float], AtmosphereState]
    max_atmosphere_altitude: float                    # [m]

    def gravity(self, altitude: float) -> float:
        """Gravitational acceleration [m/s^2] at the given altitude."""
        r = self.radius + altitude
        return self.gravitational_parameter / (r * r)


# --- Standard planets -------------------------------------------------------

EARTH = PlanetModel(
    name="Earth",
    radius=6378137.0,                                 # WGS-84 equatorial radius
    gravitational_parameter=3.986004418e14,          # GM_E, CODATA 2018
    rotation_rate=7.2921159e-5,                      # sidereal day
    atmosphere=us1976,
    max_atmosphere_altitude=_EARTH_MAX_ATMOSPHERE,
)

EARTH_NON_ROTATING = EARTH._replace(rotation_rate=0.0)
"""Non-rotating Earth, useful for test regression and reproducing older results."""