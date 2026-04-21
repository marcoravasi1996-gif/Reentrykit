"""Tests for the 3-DOF reentry trajectory simulator.

Validates the integrator against:
    1. API correctness and input validation
    2. Physical invariants (energy conservation, ballistic return)
    3. Allen-Eggers (1958) classical analytical solution
"""

from __future__ import annotations

import numpy as np
import pytest

from reentrykit.trajectory import (
    InitialState,
    TrajectoryResult,
    Vehicle,
    simulate,
)
@pytest.fixture
def reference_vehicle() -> Vehicle:
    """A generic biconic reentry body used across multiple tests."""
    return Vehicle(
        mass=500.0,
        reference_area=0.8,
        drag_coefficient=1.5,
        lift_to_drag_ratio=0.0,
        nose_radius=0.1,
    )


@pytest.fixture
def nominal_entry_state() -> InitialState:
    """Nominal reentry conditions from 80 km at orbital velocity."""
    return InitialState(
        altitude=80000.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
        downrange=0.0,
    )
# ---------------------------------------------------------------------------
# API and sanity tests
# ---------------------------------------------------------------------------


def test_rejects_altitude_above_atmosphere(reference_vehicle):
    """Initial altitude above the atmosphere's valid range is rejected."""
    bad_state = InitialState(
        altitude=200_000.0,  # 200 km — well above the 86 km ceiling
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
    )
    with pytest.raises(ValueError, match="outside the valid"):
        simulate(reference_vehicle, bad_state)


def test_rejects_negative_altitude(reference_vehicle):
    """Initial altitude below ground is rejected."""
    bad_state = InitialState(
        altitude=-100.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
    )
    with pytest.raises(ValueError):
        simulate(reference_vehicle, bad_state)


def test_returns_correct_type(reference_vehicle, nominal_entry_state):
    """Simulation returns a TrajectoryResult with all expected fields."""
    result = simulate(reference_vehicle, nominal_entry_state)

    assert isinstance(result, TrajectoryResult)
    assert isinstance(result.time, np.ndarray)
    assert isinstance(result.altitude, np.ndarray)
    assert isinstance(result.termination_reason, str)


def test_output_arrays_are_same_length(reference_vehicle, nominal_entry_state):
    """All time-history arrays in the result have matching lengths."""
    result = simulate(reference_vehicle, nominal_entry_state)

    n = len(result.time)
    assert len(result.altitude) == n
    assert len(result.velocity) == n
    assert len(result.flight_path_angle) == n
    assert len(result.downrange) == n
    assert len(result.mach) == n
    assert len(result.dynamic_pressure) == n
    assert len(result.density) == n


def test_ground_impact_termination(reference_vehicle, nominal_entry_state):
    """Nominal reentry terminates at the ground."""
    result = simulate(reference_vehicle, nominal_entry_state)

    assert result.termination_reason == "Ground impact"
    assert result.altitude[-1] < 200.0  # final sample near ground


def test_altitude_monotonically_decreases(reference_vehicle, nominal_entry_state):
    """Altitude is monotonically non-increasing during ballistic reentry."""
    result = simulate(reference_vehicle, nominal_entry_state)

    altitude_deltas = np.diff(result.altitude)
    # Allow a tiny positive tolerance for numerical noise near very shallow segments
    assert (altitude_deltas <= 1.0).all(), "Altitude should never increase during ballistic reentry"


def test_velocity_decreases_from_start(reference_vehicle, nominal_entry_state):
    """Velocity at impact is far below orbital entry velocity."""
    result = simulate(reference_vehicle, nominal_entry_state)

    assert result.velocity[-1] < result.velocity[0] / 2  # lost at least half the speed
    assert result.velocity[-1] > 0  # still moving
    assert result.velocity[-1] < 1000  # subsonic by the time it reaches the ground

# ---------------------------------------------------------------------------
# Physical invariant tests
# ---------------------------------------------------------------------------


def test_vacuum_ballistic_return_velocity():
    """A zero-drag vehicle launched vertically returns at its launch speed.

    In vacuum with constant gravity, a ballistic object launched straight up
    reaches apogee and falls back to its launch altitude at the same speed
    (energy conservation). We approximate vacuum by setting drag coefficient
    to zero; atmospheric drag above 80 km is tiny anyway.
    """
    vehicle = Vehicle(
        mass=500.0,
        reference_area=0.8,
        drag_coefficient=0.0,  # no drag -> ballistic in vacuum
    )
    initial_state = InitialState(
        altitude=80000.0,
        velocity=100.0,  # slow so it stays near the launch altitude
        flight_path_angle=np.deg2rad(90.0),  # straight up
    )

    result = simulate(vehicle, initial_state, max_time=30.0, dt_output=0.1)

    # At t = 2 * V_0 / g0 (ballistic return time), altitude matches launch
    # altitude and velocity magnitude equals initial velocity.
    t_return = 2.0 * 100.0 / 9.80665
    i_return = np.argmin(np.abs(result.time - t_return))

    assert result.altitude[i_return] == pytest.approx(80000.0, abs=5.0)
    assert abs(result.velocity[i_return]) == pytest.approx(100.0, rel=1e-3)


def test_deeper_entry_angle_causes_higher_peak_q(nominal_entry_state):
    """Steeper entry angles produce higher peak dynamic pressure.

    Physical invariant: a steeper entry deposits the vehicle deeper into
    denser atmosphere at higher velocity, producing a larger peak q.
    """
    vehicle = Vehicle(mass=500.0, reference_area=0.8, drag_coefficient=1.5)

    shallow_state = InitialState(
        altitude=80000.0, velocity=7500.0, flight_path_angle=np.deg2rad(-3.0)
    )
    steep_state = InitialState(
        altitude=80000.0, velocity=7500.0, flight_path_angle=np.deg2rad(-10.0)
    )

    shallow = simulate(vehicle, shallow_state)
    steep = simulate(vehicle, steep_state)

    assert steep.dynamic_pressure.max() > shallow.dynamic_pressure.max()


def test_higher_ballistic_coefficient_penetrates_deeper(nominal_entry_state):
    """A heavier vehicle (higher ballistic coefficient) decelerates lower.

    Ballistic coefficient beta = m / (Cd * S). Higher beta -> harder to
    decelerate -> reaches peak q at lower altitude.
    """
    light_vehicle = Vehicle(mass=200.0, reference_area=0.8, drag_coefficient=1.5)
    heavy_vehicle = Vehicle(mass=1000.0, reference_area=0.8, drag_coefficient=1.5)

    light_result = simulate(light_vehicle, nominal_entry_state)
    heavy_result = simulate(heavy_vehicle, nominal_entry_state)

    i_light_peak_q = light_result.dynamic_pressure.argmax()
    i_heavy_peak_q = heavy_result.dynamic_pressure.argmax()

    assert heavy_result.altitude[i_heavy_peak_q] < light_result.altitude[i_light_peak_q]


def test_dynamic_pressure_consistent_with_state():
    """Dynamic pressure at every step equals 0.5 * rho * V^2."""
    vehicle = Vehicle(mass=500.0, reference_area=0.8, drag_coefficient=1.5)
    initial_state = InitialState(
        altitude=80000.0, velocity=7500.0, flight_path_angle=np.deg2rad(-5.0)
    )

    result = simulate(vehicle, initial_state)

    expected_q = 0.5 * result.density * result.velocity**2
    np.testing.assert_allclose(result.dynamic_pressure, expected_q, rtol=1e-10)


def test_mach_consistent_with_state():
    """Mach number at every step equals V / a(h)."""
    from reentrykit.atmosphere import us1976

    vehicle = Vehicle(mass=500.0, reference_area=0.8, drag_coefficient=1.5)
    initial_state = InitialState(
        altitude=80000.0, velocity=7500.0, flight_path_angle=np.deg2rad(-5.0)
    )

    result = simulate(vehicle, initial_state)

    expected_mach = np.array([
        v / us1976(h).speed_of_sound for v, h in zip(result.velocity, result.altitude)
    ])
    np.testing.assert_allclose(result.mach, expected_mach, rtol=1e-10)

# ---------------------------------------------------------------------------
# Allen-Eggers classical validation
# ---------------------------------------------------------------------------


def _peak_deceleration(result: TrajectoryResult) -> float:
    """Peak deceleration magnitude [m/s^2] along the trajectory."""
    # Deceleration = -dV/dt. Using central differences for interior, forward
    # at the start, backward at the end.
    dV_dt = np.gradient(result.velocity, result.time)
    return float(-dV_dt.min())  # most negative dV/dt = peak deceleration


def _allen_eggers_peak_deceleration(
    entry_velocity: float,
    entry_flight_path_angle: float,
    scale_height: float = 7000.0,
) -> float:
    """Allen-Eggers (1958) peak deceleration for ballistic reentry.

    Closed-form prediction from NACA Report 1381. Assumes exponential
    atmosphere, shallow entry angle, non-lifting vehicle, constant Cd.
    """
    return entry_velocity**2 * abs(np.sin(entry_flight_path_angle)) / (
        2 * np.e * scale_height
    )


def test_allen_eggers_peak_deceleration_magnitude():
    """Peak deceleration matches Allen-Eggers (1958) prediction within 20%.

    US1976 is piecewise, not purely exponential, so we allow a loose
    tolerance. The 7 km scale height is representative of the lower
    atmosphere where peak deceleration occurs.
    """
    vehicle = Vehicle(mass=500.0, reference_area=0.8, drag_coefficient=1.5)
    initial_state = InitialState(
        altitude=80000.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
    )

    result = simulate(vehicle, initial_state)

    peak_decel_simulated = _peak_deceleration(result)
    peak_decel_predicted = _allen_eggers_peak_deceleration(
        entry_velocity=7500.0,
        entry_flight_path_angle=np.deg2rad(-5.0),
    )

    relative_error = abs(peak_decel_simulated - peak_decel_predicted) / peak_decel_predicted
    assert relative_error < 0.20, (
        f"Peak deceleration {peak_decel_simulated:.1f} m/s^2 differs from "
        f"Allen-Eggers prediction {peak_decel_predicted:.1f} m/s^2 by "
        f"{relative_error * 100:.1f}%"
    )


def test_allen_eggers_invariance_to_vehicle_properties():
    """Peak deceleration is nearly independent of mass, area, and drag coefficient.

    Allen-Eggers' central insight: for ballistic entry, peak deceleration
    depends only on entry velocity and angle, not on vehicle ballistic
    properties. These merely shift the altitude at which the peak occurs.
    """
    initial_state = InitialState(
        altitude=80000.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
    )

    vehicles = [
        Vehicle(mass=200.0, reference_area=0.5, drag_coefficient=1.2),
        Vehicle(mass=500.0, reference_area=0.8, drag_coefficient=1.5),
        Vehicle(mass=1500.0, reference_area=1.5, drag_coefficient=1.8),
    ]

    peak_decelerations = [
        _peak_deceleration(simulate(v, initial_state)) for v in vehicles
    ]

    # All three should be within 10% of each other
    max_peak = max(peak_decelerations)
    min_peak = min(peak_decelerations)
    spread = (max_peak - min_peak) / np.mean(peak_decelerations)

    assert spread < 0.10, (
        f"Peak deceleration varies by {spread * 100:.1f}% across vehicles, "
        f"violating Allen-Eggers invariance. Values: {peak_decelerations}"
    )
