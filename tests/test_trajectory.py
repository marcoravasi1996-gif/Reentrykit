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
    return Vehicle.from_mass_area_cd(
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
    """Initial altitude above the extended atmosphere range is rejected."""
    bad_state = InitialState(
        altitude=300_000.0,  # above the 200 km extended ceiling
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
    assert result.altitude[-1] < 200.0


def test_altitude_monotonically_decreases(reference_vehicle, nominal_entry_state):
    """Altitude is monotonically non-increasing during ballistic reentry."""
    result = simulate(reference_vehicle, nominal_entry_state)

    altitude_deltas = np.diff(result.altitude)
    assert (altitude_deltas <= 1.0).all(), "Altitude should never increase during ballistic reentry"


def test_velocity_decreases_from_start(reference_vehicle, nominal_entry_state):
    """Velocity at impact is far below orbital entry velocity."""
    result = simulate(reference_vehicle, nominal_entry_state)

    assert result.velocity[-1] < result.velocity[0] / 2
    assert result.velocity[-1] > 0
    assert result.velocity[-1] < 1000


# ---------------------------------------------------------------------------
# Physical invariant tests
# ---------------------------------------------------------------------------


def test_vacuum_ballistic_return_velocity():
    """A drag-free vehicle launched vertically returns at its launch speed.

    In vacuum with constant gravity, a ballistic object launched straight up
    reaches apogee and falls back to its launch altitude at the same speed
    (energy conservation). We approximate vacuum by setting the ballistic
    coefficient to an enormous value; atmospheric drag above 80 km is tiny
    anyway.
    """
    vehicle = Vehicle(
        reference_area=0.8,
        mass=500.0,
        drag_coefficient=0.0,  # drag-free for vacuum test
        lift_to_drag_ratio=0.0,
        nose_radius=0.1,
    )
    initial_state = InitialState(
        altitude=80000.0,
        velocity=100.0,
        flight_path_angle=np.deg2rad(90.0),
    )

    result = simulate(vehicle, initial_state, max_time=30.0, dt_output=0.1)

    t_return = 2.0 * 100.0 / 9.80665
    i_return = np.argmin(np.abs(result.time - t_return))

    assert result.altitude[i_return] == pytest.approx(80000.0, abs=5.0)
    assert abs(result.velocity[i_return]) == pytest.approx(100.0, rel=1e-3)


def test_deeper_entry_angle_causes_higher_peak_q(nominal_entry_state):
    """Steeper entry angles produce higher peak dynamic pressure."""
    vehicle = Vehicle.from_mass_area_cd(mass=500.0, reference_area=0.8, drag_coefficient=1.5)

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
    """A higher ballistic coefficient decelerates lower in the atmosphere."""
    light_vehicle = Vehicle.from_mass_area_cd(mass=200.0, reference_area=0.8, drag_coefficient=1.5)
    heavy_vehicle = Vehicle.from_mass_area_cd(mass=1000.0, reference_area=0.8, drag_coefficient=1.5)

    light_result = simulate(light_vehicle, nominal_entry_state)
    heavy_result = simulate(heavy_vehicle, nominal_entry_state)

    i_light_peak_q = light_result.dynamic_pressure.argmax()
    i_heavy_peak_q = heavy_result.dynamic_pressure.argmax()

    assert heavy_result.altitude[i_heavy_peak_q] < light_result.altitude[i_light_peak_q]


def test_dynamic_pressure_consistent_with_state():
    """Dynamic pressure at every step equals 0.5 * rho * V^2."""
    vehicle = Vehicle.from_mass_area_cd(mass=500.0, reference_area=0.8, drag_coefficient=1.5)
    initial_state = InitialState(
        altitude=80000.0, velocity=7500.0, flight_path_angle=np.deg2rad(-5.0)
    )

    result = simulate(vehicle, initial_state)

    expected_q = 0.5 * result.density * result.velocity**2
    np.testing.assert_allclose(result.dynamic_pressure, expected_q, rtol=1e-10)


def test_mach_consistent_with_state():
    """Mach number at every step equals V / a(h)."""
    from reentrykit.atmosphere import us1976

    vehicle = Vehicle.from_mass_area_cd(mass=500.0, reference_area=0.8, drag_coefficient=1.5)
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
    dV_dt = np.gradient(result.velocity, result.time)
    return float(-dV_dt.min())


def _allen_eggers_peak_deceleration(
    entry_velocity: float,
    entry_flight_path_angle: float,
    scale_height: float = 7000.0,
) -> float:
    """Allen-Eggers (1958) peak deceleration for ballistic reentry."""
    return entry_velocity**2 * abs(np.sin(entry_flight_path_angle)) / (
        2 * np.e * scale_height
    )


def test_allen_eggers_peak_deceleration_magnitude():
    """Peak deceleration matches Allen-Eggers (1958) prediction within 20%."""
    vehicle = Vehicle.from_mass_area_cd(mass=500.0, reference_area=0.8, drag_coefficient=1.5)
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
    """Peak deceleration is nearly independent of mass, area, and drag coefficient."""
    initial_state = InitialState(
        altitude=80000.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
    )

    vehicles = [
        Vehicle.from_mass_area_cd(mass=200.0, reference_area=0.5, drag_coefficient=1.2),
        Vehicle.from_mass_area_cd(mass=500.0, reference_area=0.8, drag_coefficient=1.5),
        Vehicle.from_mass_area_cd(mass=1500.0, reference_area=1.5, drag_coefficient=1.8),
    ]

    peak_decelerations = [
        _peak_deceleration(simulate(v, initial_state)) for v in vehicles
    ]

    max_peak = max(peak_decelerations)
    min_peak = min(peak_decelerations)
    spread = (max_peak - min_peak) / np.mean(peak_decelerations)

    assert spread < 0.10, (
        f"Peak deceleration varies by {spread * 100:.1f}% across vehicles, "
        f"violating Allen-Eggers invariance. Values: {peak_decelerations}"
    )
# ---------------------------------------------------------------------------
# Time-varying lift-to-drag ratio
# ---------------------------------------------------------------------------


def test_constant_lift_schedule_matches_float_ratio(nominal_entry_state):
    """Passing lift_to_drag_ratio=0.3 as a constant and as `lambda t: 0.3`
    produces identical trajectories."""
    vehicle_float = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5, lift_to_drag_ratio=0.3
    )
    vehicle_callable = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5, lift_to_drag_ratio=lambda t: 0.3
    )

    result_float = simulate(vehicle_float, nominal_entry_state)
    result_callable = simulate(vehicle_callable, nominal_entry_state)

    np.testing.assert_allclose(result_float.velocity, result_callable.velocity, rtol=1e-10)
    np.testing.assert_allclose(result_float.altitude, result_callable.altitude, rtol=1e-10)


def test_time_varying_lift_differs_from_constant(nominal_entry_state):
    """A time-varying L/D schedule produces a different trajectory than a
    constant L/D at the mean value."""
    mean_ld = 0.3

    vehicle_constant = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=mean_ld,
    )
    # Step schedule: higher L/D first, then lower
    def step_schedule(t: float) -> float:
        return 0.5 if t < 100.0 else 0.1

    vehicle_varying = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=step_schedule,
    )

    result_constant = simulate(vehicle_constant, nominal_entry_state)
    result_varying = simulate(vehicle_varying, nominal_entry_state)

    # The trajectories should differ meaningfully in range
    range_constant = result_constant.downrange[-1]
    range_varying = result_varying.downrange[-1]
    relative_diff = abs(range_varying - range_constant) / range_constant

    assert relative_diff > 0.01, (
        f"Time-varying L/D gave range {range_varying/1000:.1f} km vs. "
        f"constant L/D range {range_constant/1000:.1f} km "
        f"({relative_diff*100:.2f}% difference). Expected >1%."
    )


def test_negative_lift_modulation_prevents_skipout():
    """A brief negative L/D pulse early in the trajectory prevents skip-out
    that would otherwise occur with constant positive L/D (Apollo-style
    guidance)."""
    # Apollo-like entry conditions, but starting below the ceiling so that
    # the constant +0.5 L/D case doesn't immediately skip to 200 km.
    entry_state = InitialState(
        altitude=85_000.0,
        velocity=11_137.0,
        flight_path_angle=np.deg2rad(-6.93),
    )

    # Constant high L/D: causes skip
    vehicle_skip = Vehicle.from_mass_area_cd(
        mass=5357.0, reference_area=12.0, drag_coefficient=1.2,
        lift_to_drag_ratio=0.5,
    )

    # Time-varying: brief negative pulse at start prevents skip
    def apollo_like_schedule(t: float) -> float:
        if t < 22.0:
            return -0.5  # down-control to prevent skip
        return 0.3  # gentle positive afterwards

    vehicle_guided = Vehicle.from_mass_area_cd(
        mass=5357.0, reference_area=12.0, drag_coefficient=1.2,
        lift_to_drag_ratio=apollo_like_schedule,
    )

    result_skip = simulate(vehicle_skip, entry_state, max_time=3000.0)
    result_guided = simulate(vehicle_guided, entry_state, max_time=3000.0)

    # The skip trajectory should either terminate by skipping out or reach
    # a much higher max altitude than the guided one.
    max_alt_skip = result_skip.altitude.max()
    max_alt_guided = result_guided.altitude.max()

    assert max_alt_skip > max_alt_guided + 10_000.0, (
        f"Constant +0.5 L/D reached {max_alt_skip/1000:.1f} km, "
        f"guided reached {max_alt_guided/1000:.1f} km. "
        f"Expected >10 km higher apogee without negative-lift modulation."
    )

    # Time-varying: brief negative pulse at start prevents skip
    def apollo_like_schedule(t: float) -> float:
        if t < 22.0:
            return -0.5  # down-control to prevent skip
        return 0.3  # gentle positive afterwards

    vehicle_guided = Vehicle.from_mass_area_cd(
        mass=5357.0, reference_area=12.0, drag_coefficient=1.2,
        lift_to_drag_ratio=apollo_like_schedule,
    )

    result_skip = simulate(vehicle_skip, entry_state, max_time=3000.0)
    result_guided = simulate(vehicle_guided, entry_state, max_time=3000.0)

    max_alt_skip = result_skip.altitude.max()
    max_alt_guided = result_guided.altitude.max()

    # Constant positive L/D causes skip above entry altitude
    assert max_alt_skip > 85_001.0, "Constant +0.5 L/D should cause skip-out"

    # Guided trajectory should stay at or near entry altitude
    assert max_alt_guided < max_alt_skip, (
        f"Guided trajectory reached {max_alt_guided/1000:.1f} km, "
        f"constant L/D reached {max_alt_skip/1000:.1f} km. "
        f"Negative lift pulse should suppress skip-out."
    )
# ---------------------------------------------------------------------------
# Mach-dependent drag coefficient
# ---------------------------------------------------------------------------


def test_constant_cd_matches_callable_at_same_value(nominal_entry_state):
    """Cd=1.5 as a constant and `lambda m: 1.5` as a callable produce
    identical trajectories — no hidden side effects from the callable path."""
    vehicle_const = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
    )
    vehicle_callable = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=lambda m: 1.5,
    )

    result_const = simulate(vehicle_const, nominal_entry_state)
    result_callable = simulate(vehicle_callable, nominal_entry_state)

    np.testing.assert_allclose(result_const.velocity, result_callable.velocity, rtol=1e-10)
    np.testing.assert_allclose(result_const.altitude, result_callable.altitude, rtol=1e-10)
    np.testing.assert_allclose(result_const.downrange, result_callable.downrange, rtol=1e-10)


def test_mach_dependent_cd_differs_from_constant(nominal_entry_state):
    """A Mach-varying Cd produces different peak g than a constant Cd
    at one of the two endpoint values."""
    def stepped_cd(mach: float) -> float:
        """Sphere-cone-like profile: higher Cd at low Mach, lower at hypersonic."""
        if mach > 10.0:
            return 1.0
        elif mach > 1.5:
            return 1.3
        else:
            return 0.8

    vehicle_variable = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=stepped_cd,
    )
    vehicle_hypersonic = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.0,
    )
    vehicle_transonic = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.3,
    )

    result_variable = simulate(vehicle_variable, nominal_entry_state)
    result_hypersonic = simulate(vehicle_hypersonic, nominal_entry_state)
    result_transonic = simulate(vehicle_transonic, nominal_entry_state)

    # Peak decelerations
    peak_variable = -np.gradient(result_variable.velocity, result_variable.time).min()
    peak_hypersonic = -np.gradient(result_hypersonic.velocity, result_hypersonic.time).min()
    peak_transonic = -np.gradient(result_transonic.velocity, result_transonic.time).min()

    # Variable-Cd peak should differ from both endpoint-constant cases
    assert abs(peak_variable - peak_hypersonic) > 0.01 * peak_hypersonic
    assert abs(peak_variable - peak_transonic) > 0.01 * peak_transonic


def test_cd_callable_invoked_each_step():
    """Verify the Cd callable is actually called during integration, not just once."""
    call_count = {"count": 0, "mach_values": []}

    def counting_cd(mach: float) -> float:
        call_count["count"] += 1
        call_count["mach_values"].append(mach)
        return 1.5

    vehicle = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=counting_cd,
    )
    state = InitialState(
        altitude=80000.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
    )

    result = simulate(vehicle, state)

    # Integration should call Cd many times (one per RK45 sub-step)
    assert call_count["count"] > 100, (
        f"Cd callable was invoked only {call_count['count']} times; "
        f"expected hundreds of evaluations during integration."
    )

    # Mach values should span a wide range as the vehicle decelerates
    mach_range = max(call_count["mach_values"]) - min(call_count["mach_values"])
    assert mach_range > 5.0, (
        f"Mach range seen by Cd callable was {mach_range:.1f}; "
        f"expected at least 5 from entry hypersonic to subsonic."
    )


def test_primary_constructor_and_classmethod_are_equivalent():
    """Vehicle(...) and Vehicle.from_mass_area_cd(...) produce identical vehicles."""
    v1 = Vehicle(
        reference_area=0.8,
        mass=500.0,
        drag_coefficient=1.5,
        lift_to_drag_ratio=0.2,
        nose_radius=0.15,
    )
    v2 = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.2, nose_radius=0.15,
    )

    assert v1 == v2


def test_beta_property_with_constant_cd():
    """Vehicle.beta() returns m / (Cd * S) for constant Cd, any Mach."""
    v = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
    )
    expected_beta = 500.0 / (1.5 * 0.8)
    assert abs(v.beta(mach=10.0) - expected_beta) < 1e-6
    assert abs(v.beta(mach=2.0) - expected_beta) < 1e-6
    # Default Mach should also give the same value for constant Cd
    assert abs(v.beta() - expected_beta) < 1e-6


def test_beta_property_with_mach_dependent_cd():
    """Vehicle.beta() varies with Mach when Cd is Mach-dependent."""
    def cd_function(mach: float) -> float:
        return 1.0 if mach > 5.0 else 1.5

    v = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=cd_function,
    )
    beta_hypersonic = v.beta(mach=20.0)
    beta_subsonic = v.beta(mach=0.5)

    assert beta_hypersonic != beta_subsonic
    assert abs(beta_hypersonic - 500.0 / (1.0 * 0.8)) < 1e-6
    assert abs(beta_subsonic - 500.0 / (1.5 * 0.8)) < 1e-6