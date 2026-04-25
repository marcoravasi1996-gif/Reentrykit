"""Tests for the planet module."""
from __future__ import annotations

import numpy as np
import pytest

from reentrykit.atmosphere import AtmosphereState
from reentrykit.planet import EARTH, EARTH_NON_ROTATING, PlanetModel


# ---------------------------------------------------------------------------
# Earth model sanity checks
# ---------------------------------------------------------------------------


def test_earth_radius_matches_wgs84():
    """Earth equatorial radius matches WGS-84."""
    assert EARTH.radius == pytest.approx(6378137.0, rel=1e-12)


def test_earth_gravitational_parameter_codata_2018():
    """GM_E matches CODATA 2018."""
    assert EARTH.gravitational_parameter == pytest.approx(3.986004418e14, rel=1e-12)


def test_earth_rotation_rate_sidereal():
    """Earth rotation rate matches sidereal day (one rev / 86164 s)."""
    expected = 7.2921159e-5
    assert EARTH.rotation_rate == pytest.approx(expected, rel=1e-9)


def test_earth_gravity_at_sea_level():
    """g at sea level approximately 9.81 m/s^2."""
    g = EARTH.gravity(altitude=0.0)
    assert 9.79 < g < 9.83


def test_earth_gravity_decreases_with_altitude():
    """Gravity decreases monotonically with altitude (inverse-square)."""
    g_0 = EARTH.gravity(0.0)
    g_100 = EARTH.gravity(100_000.0)
    g_1000 = EARTH.gravity(1_000_000.0)
    assert g_100 < g_0
    assert g_1000 < g_100


def test_earth_gravity_inverse_square_form():
    """Doubling distance from center of Earth quarters gravity."""
    r0 = EARTH.radius
    g_at_surface = EARTH.gravity(0.0)
    g_at_double_radius = EARTH.gravity(r0)   # altitude = R, so r_total = 2R
    assert g_at_double_radius / g_at_surface == pytest.approx(0.25, rel=1e-12)


# ---------------------------------------------------------------------------
# Non-rotating Earth
# ---------------------------------------------------------------------------


def test_non_rotating_earth_has_zero_rotation():
    """Non-rotating Earth variant has rotation_rate = 0."""
    assert EARTH_NON_ROTATING.rotation_rate == 0.0


def test_non_rotating_earth_keeps_other_properties():
    """Only rotation_rate is changed; everything else matches EARTH."""
    assert EARTH_NON_ROTATING.radius == EARTH.radius
    assert EARTH_NON_ROTATING.gravitational_parameter == EARTH.gravitational_parameter
    assert EARTH_NON_ROTATING.atmosphere is EARTH.atmosphere
    assert EARTH_NON_ROTATING.max_atmosphere_altitude == EARTH.max_atmosphere_altitude
    assert EARTH_NON_ROTATING.name == "Earth"


# ---------------------------------------------------------------------------
# Atmosphere callable plumbing
# ---------------------------------------------------------------------------


def test_atmosphere_callable_returns_state():
    """The atmosphere attribute returns an AtmosphereState."""
    state = EARTH.atmosphere(0.0)
    assert isinstance(state, AtmosphereState)
    assert state.temperature > 0
    assert state.density > 0
    assert state.pressure > 0
    assert state.speed_of_sound > 0


def test_atmosphere_extends_to_max_altitude():
    """The atmosphere function works up to max_atmosphere_altitude."""
    state = EARTH.atmosphere(EARTH.max_atmosphere_altitude)
    assert state.density > 0


# ---------------------------------------------------------------------------
# PlanetModel as a generic abstraction
# ---------------------------------------------------------------------------


def test_planet_model_is_namedtuple_immutable():
    """PlanetModel cannot be mutated after creation (NamedTuple property)."""
    with pytest.raises(AttributeError):
        EARTH.radius = 0.0   # type: ignore


def test_custom_planet_model():
    """User can define a custom planet (e.g., for Mars-like body)."""
    def vacuum_atmosphere(altitude: float) -> AtmosphereState:
        return AtmosphereState(temperature=0.0, pressure=0.0, density=0.0,
                               speed_of_sound=0.0)

    test_planet = PlanetModel(
        name="TestBody",
        radius=3_000_000.0,
        gravitational_parameter=1e13,
        rotation_rate=0.0,
        atmosphere=vacuum_atmosphere,
        max_atmosphere_altitude=10_000.0,
    )
    assert test_planet.name == "TestBody"
    assert test_planet.gravity(0.0) > 0
