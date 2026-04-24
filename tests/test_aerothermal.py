"""Unit tests for the aerothermal heating module."""

from __future__ import annotations

import numpy as np
import pytest

from reentrykit.aerothermal import (
    SUTTON_GRAVES_K_EARTH,
    HeatingResult,
    heating_history,
    sutton_graves_heat_flux,
)
from reentrykit.trajectory import InitialState, Vehicle, simulate
from reentrykit.planet import EARTH_NON_ROTATING


# ---------------------------------------------------------------------------
# Sutton-Graves point function: known-value and invariant tests
# ---------------------------------------------------------------------------


def test_sutton_graves_known_value():
    """Hand-computed Sutton-Graves value at a representative operating point."""
    rho = 3.1e-4
    V = 7000.0
    R_N = 0.5

    q = sutton_graves_heat_flux(rho, V, R_N)

    expected = 1.7415e-4 * np.sqrt(rho / R_N) * V**3
    assert q == pytest.approx(expected, rel=1e-10)
    assert q > 1.0e6
    assert q < 2.0e6


def test_sutton_graves_zero_velocity_gives_zero_flux():
    q = sutton_graves_heat_flux(density=1.0, velocity=0.0, nose_radius=1.0)
    assert q == 0.0


def test_sutton_graves_zero_density_gives_zero_flux():
    q = sutton_graves_heat_flux(density=0.0, velocity=10_000.0, nose_radius=1.0)
    assert q == 0.0


def test_sutton_graves_scales_as_v_cubed():
    rho = 1e-4
    R_N = 0.3
    q1 = sutton_graves_heat_flux(rho, 5000.0, R_N)
    q2 = sutton_graves_heat_flux(rho, 10_000.0, R_N)
    assert q2 / q1 == pytest.approx(8.0, rel=1e-10)


def test_sutton_graves_scales_as_sqrt_rho():
    V = 8000.0
    R_N = 0.3
    q1 = sutton_graves_heat_flux(1e-4, V, R_N)
    q2 = sutton_graves_heat_flux(4e-4, V, R_N)
    assert q2 / q1 == pytest.approx(2.0, rel=1e-10)


def test_sutton_graves_scales_as_inv_sqrt_nose_radius():
    rho = 1e-4
    V = 8000.0
    q1 = sutton_graves_heat_flux(rho, V, 0.25)
    q2 = sutton_graves_heat_flux(rho, V, 1.00)
    assert q2 / q1 == pytest.approx(0.5, rel=1e-10)


def test_sutton_graves_rejects_zero_nose_radius():
    with pytest.raises(ValueError, match="nose_radius"):
        sutton_graves_heat_flux(density=1e-4, velocity=7000.0, nose_radius=0.0)
    with pytest.raises(ValueError, match="nose_radius"):
        sutton_graves_heat_flux(density=1e-4, velocity=7000.0, nose_radius=-0.5)


def test_sutton_graves_rejects_negative_density():
    with pytest.raises(ValueError, match="density"):
        sutton_graves_heat_flux(density=-1e-4, velocity=7000.0, nose_radius=0.5)


def test_sutton_graves_rejects_negative_velocity():
    with pytest.raises(ValueError, match="velocity"):
        sutton_graves_heat_flux(density=1e-4, velocity=-7000.0, nose_radius=0.5)


# ---------------------------------------------------------------------------
# Trajectory-based heating history
# ---------------------------------------------------------------------------


@pytest.fixture
def stardust_trajectory():
    """Run a Stardust-class trajectory for heating tests."""
    vehicle = Vehicle.from_mass_area_cd(
        mass=45.8,
        reference_area=np.pi * (0.811 / 2) ** 2,
        drag_coefficient=1.0,
        lift_to_drag_ratio=0.0,
        nose_radius=0.2202,
    )
    state = InitialState(
        altitude=125_000.0,
        velocity=12_300.0,
        flight_path_angle=np.deg2rad(-8.2),
        heading=np.deg2rad(15.0),
        latitude=np.deg2rad(41.0),
        longitude=np.deg2rad(-128.0),
    )
    return simulate(
        vehicle, state,
        planet=EARTH_NON_ROTATING,
        max_time=500.0, dt_output=0.1,
    )


def test_heating_history_returns_correct_shape(stardust_trajectory):
    result = heating_history(stardust_trajectory, nose_radius=0.2202)

    assert isinstance(result, HeatingResult)
    assert len(result.heat_flux) == len(stardust_trajectory.time)
    assert len(result.heat_load) == len(stardust_trajectory.time)
    assert result.time is stardust_trajectory.time


def test_heating_history_peak_is_positive(stardust_trajectory):
    result = heating_history(stardust_trajectory, nose_radius=0.2202)
    assert result.peak_heat_flux > 0.0
    assert result.total_heat_load > 0.0


def test_heating_history_heat_load_is_monotonic(stardust_trajectory):
    result = heating_history(stardust_trajectory, nose_radius=0.2202)
    diffs = np.diff(result.heat_load)
    assert (diffs >= 0.0).all()


def test_heating_history_initial_heat_load_is_zero(stardust_trajectory):
    result = heating_history(stardust_trajectory, nose_radius=0.2202)
    assert result.heat_load[0] == 0.0


def test_heating_history_peak_time_occurs_within_trajectory(stardust_trajectory):
    result = heating_history(stardust_trajectory, nose_radius=0.2202)
    assert stardust_trajectory.time[0] <= result.peak_heat_flux_time
    assert result.peak_heat_flux_time <= stardust_trajectory.time[-1]


def test_heating_history_rejects_zero_nose_radius(stardust_trajectory):
    with pytest.raises(ValueError, match="nose_radius"):
        heating_history(stardust_trajectory, nose_radius=0.0)


# ---------------------------------------------------------------------------
# Published-value validation tests
# ---------------------------------------------------------------------------


def test_stardust_peak_heat_flux_order_of_magnitude(stardust_trajectory):
    """Stardust peak convective heat flux should be order 5-15 MW/m².

    Sutton-Graves gives convective heating only. Stardust's published total
    peak heat flux (~1200 W/cm² = 12 MW/m²) includes substantial radiative
    heating at V = 12.3 km/s. Our convective-only result should be lower
    but within the correct order of magnitude.
    """
    result = heating_history(stardust_trajectory, nose_radius=0.2202)

    assert 5e6 < result.peak_heat_flux < 1.5e7, (
        f"Stardust peak convective heat flux {result.peak_heat_flux:.2e} W/m² "
        f"outside expected order-of-magnitude range (5e6 to 1.5e7 W/m²)"
    )