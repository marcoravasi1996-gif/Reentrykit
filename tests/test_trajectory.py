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
def nominal_entry_state():
    return InitialState(
        altitude=80_000.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
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
    """A drag-free vehicle launched vertically conserves mechanical energy.

    In vacuum, a ballistic object's total energy (kinetic + gravitational
    potential) is conserved. We verify this by computing total specific
    energy at the start and at a later point in the trajectory, and
    confirming they match.
    """
    vehicle = Vehicle(
        reference_area=0.8,
        mass=500.0,
        drag_coefficient=0.0,  # drag-free
        lift_to_drag_ratio=0.0,
        nose_radius=0.1,
    )
    initial_state = InitialState(
        altitude=80000.0,
        velocity=100.0,
        flight_path_angle=np.deg2rad(90.0),
    )
    result = simulate(vehicle, initial_state, max_time=30.0, dt_output=0.1)

    # Gravitational parameter for Earth (must match planet.py)
    mu = 3.986004418e14
    radius_earth = 6378137.0

    def specific_energy(altitude: float, velocity: float) -> float:
        """Specific mechanical energy (KE + gravitational PE) per unit mass."""
        r = radius_earth + altitude
        return 0.5 * velocity**2 - mu / r

    # Initial specific energy
    E_initial = specific_energy(initial_state.altitude, initial_state.velocity)

    # Check energy conservation at every output point.
    # Energy should be conserved to high precision (~1e-6 relative) in vacuum
    # with a properly integrated trajectory.
    for i in range(len(result.time)):
        E_i = specific_energy(result.altitude[i], abs(result.velocity[i]))
        relative_error = abs(E_i - E_initial) / abs(E_initial)
        assert relative_error < 1e-4, (
            f"Energy not conserved at t={result.time[i]:.2f}s: "
            f"E={E_i:.6e} vs E0={E_initial:.6e} (rel err {relative_error:.2e})"
        )

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
# ---------------------------------------------------------------------------
# Bank angle modulation
# ---------------------------------------------------------------------------


def test_zero_bank_matches_planar_flight(nominal_entry_state):
    """With bank_angle=0, bank-angle-aware code reproduces planar-flight results.

    Two vehicles with the same L/D should give identical trajectories whether
    bank_angle is left at default (0.0) or explicitly set to 0.0.
    """
    v_default = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.25,
    )
    v_explicit = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.25, bank_angle=0.0,
    )

    r1 = simulate(v_default, nominal_entry_state)
    r2 = simulate(v_explicit, nominal_entry_state)

    np.testing.assert_allclose(r1.velocity, r2.velocity, rtol=1e-12)
    np.testing.assert_allclose(r1.altitude, r2.altitude, rtol=1e-12)
    np.testing.assert_allclose(r1.flight_path_angle, r2.flight_path_angle, rtol=1e-12)
    np.testing.assert_allclose(r1.downrange, r2.downrange, rtol=1e-12)
    # With zero bank and zero initial heading, crossrange should stay zero
    assert np.max(np.abs(r1.crossrange)) < 1e-6
    assert np.max(np.abs(r2.crossrange)) < 1e-6


def test_bank_90_gives_pure_lateral_lift(nominal_entry_state):
    """With bank_angle = 90 degrees, lift is entirely lateral — no vertical
    component. The altitude-velocity profile should match a pure ballistic
    vehicle (since vertical lift is the only way L/D affects descent).
    """
    ballistic = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.0,
    )
    banked_90 = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.5, bank_angle=np.pi / 2,
    )

    r_ballistic = simulate(ballistic, nominal_entry_state)
    r_banked = simulate(banked_90, nominal_entry_state)

    # Peak deceleration and altitude trajectory should match ballistic
    # (because cos(90 deg) = 0 means zero vertical lift component)
    peak_ballistic = -np.gradient(r_ballistic.velocity, r_ballistic.time).min() / 9.80665
    peak_banked = -np.gradient(r_banked.velocity, r_banked.time).min() / 9.80665
    np.testing.assert_allclose(peak_banked, peak_ballistic, rtol=1e-3)

    # But heading should change due to lateral lift
    heading_change = np.abs(r_banked.heading[-1] - r_banked.heading[0])
    assert heading_change > 0.01, (
        f"90-deg bank should produce lateral heading change, got "
        f"{np.rad2deg(heading_change):.3f} deg"
    )

    # And cross-range should grow
    crossrange_final = np.abs(r_banked.crossrange[-1])
    assert crossrange_final > 1000.0, (
        f"90-deg bank should produce significant crossrange, got "
        f"{crossrange_final:.1f} m"
    )


def test_bank_180_steepens_descent(nominal_entry_state):
    """With bank_angle = 180 degrees, lift points vertically downward
    (cos(180) = -1). The vehicle should descend faster than ballistic."""
    ballistic = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.0,
    )
    inverted = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.5, bank_angle=np.pi,
    )

    r_ballistic = simulate(ballistic, nominal_entry_state)
    r_inverted = simulate(inverted, nominal_entry_state)

    # Inverted lift -> steeper descent -> reaches ground sooner
    assert r_inverted.time[-1] < r_ballistic.time[-1], (
        f"180-deg bank should reach ground faster: inverted {r_inverted.time[-1]:.1f}s "
        f"vs ballistic {r_ballistic.time[-1]:.1f}s"
    )

    # Peak deceleration should be higher (deeper into atmosphere at high V)
    peak_ballistic = -np.gradient(r_ballistic.velocity, r_ballistic.time).min() / 9.80665
    peak_inverted = -np.gradient(r_inverted.velocity, r_inverted.time).min() / 9.80665
    assert peak_inverted > peak_ballistic, (
        f"180-deg bank should produce higher peak g than ballistic: "
        f"inverted {peak_inverted:.2f}g vs ballistic {peak_ballistic:.2f}g"
    )


def test_opposite_bank_mirrors_trajectory(nominal_entry_state):
    """sigma = +90 deg and sigma = -90 deg should produce mirror-image
    trajectories: same altitude-velocity profile, opposite crossrange."""
    bank_positive = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.5, bank_angle=np.pi / 2,
    )
    bank_negative = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.5, bank_angle=-np.pi / 2,
    )

    r_pos = simulate(bank_positive, nominal_entry_state)
    r_neg = simulate(bank_negative, nominal_entry_state)

    # Velocity and altitude histories should be identical (bank direction
    # doesn't affect in-plane dynamics when initial heading is zero)
    np.testing.assert_allclose(r_pos.velocity, r_neg.velocity, rtol=1e-8)
    np.testing.assert_allclose(r_pos.altitude, r_neg.altitude, rtol=1e-8)

    # Crossrange should mirror (opposite signs)
    np.testing.assert_allclose(r_pos.crossrange, -r_neg.crossrange, rtol=1e-6, atol=1e-3)
    # Heading should mirror
    np.testing.assert_allclose(r_pos.heading, -r_neg.heading, rtol=1e-6, atol=1e-6)


def test_bank_angle_callable_invoked(nominal_entry_state):
    """A time-varying bank angle callable should be invoked during integration."""
    call_count = {"n": 0}

    def bank_schedule(t: float) -> float:
        call_count["n"] += 1
        return np.pi / 4 * np.sin(t / 50.0)  # oscillating bank

    vehicle = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.3, bank_angle=bank_schedule,
    )

    simulate(vehicle, nominal_entry_state)

    assert call_count["n"] > 100, (
        f"Bank-angle callable should be invoked many times during integration, "
        f"got {call_count['n']} calls"
    )


def test_initial_heading_changes_landing_location():
    """Different initial headings send a ballistic vehicle to different
    geographic endpoints, even though total ground-track distance is the same.

    With the Phase 2 lat/lon equations, downrange is measured along the
    initial heading direction — so a ballistic trajectory has zero
    crossrange by construction. The observable effect of initial heading
    is on final latitude/longitude.
    """
    vehicle = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
        lift_to_drag_ratio=0.0,  # ballistic
    )

    state_north = InitialState(
        altitude=80000.0, velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
        heading=0.0,  # due north
    )
    state_east = InitialState(
        altitude=80000.0, velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
        heading=np.pi / 2,  # due east
    )

    r_north = simulate(vehicle, state_north)
    r_east = simulate(vehicle, state_east)

    # Same total downrange regardless of heading (ballistic, spherical Earth)
    np.testing.assert_allclose(
        r_north.downrange[-1], r_east.downrange[-1], rtol=1e-3,
    )

    # Crossrange is near zero for ballistic flight (no lateral lift)
    assert abs(r_north.crossrange[-1]) < 100.0  # under 100 m
    assert abs(r_east.crossrange[-1]) < 100.0

    # Northbound trajectory ends with higher latitude than eastbound
    assert r_north.latitude[-1] > r_east.latitude[-1]

    # Eastbound trajectory ends with higher longitude than northbound
    assert r_east.longitude[-1] > r_north.longitude[-1]

# ---------------------------------------------------------------------------
# Rotating Earth (Coriolis and centrifugal effects)
# ---------------------------------------------------------------------------


def _ballistic_vehicle_at_latitude(latitude_deg: float, heading_deg: float):
    """Build a ballistic vehicle and entry state at the given latitude/heading."""
    vehicle = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
    )
    state = InitialState(
        altitude=80000.0,
        velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
        heading=np.deg2rad(heading_deg),
        latitude=np.deg2rad(latitude_deg),
        longitude=0.0,
    )
    return vehicle, state


def test_non_rotating_earth_reproduces_default_behavior():
    """Passing EARTH_NON_ROTATING explicitly matches the default planet
    (which is also EARTH_NON_ROTATING in Phase 2)."""
    from reentrykit.planet import EARTH_NON_ROTATING

    vehicle, state = _ballistic_vehicle_at_latitude(0.0, 90.0)

    r_default = simulate(vehicle, state)
    r_explicit = simulate(vehicle, state, planet=EARTH_NON_ROTATING)

    np.testing.assert_allclose(r_default.velocity, r_explicit.velocity, rtol=1e-12)
    np.testing.assert_allclose(r_default.altitude, r_explicit.altitude, rtol=1e-12)
    np.testing.assert_allclose(r_default.latitude, r_explicit.latitude, rtol=1e-12)


def test_rotating_earth_ballistic_eastward_has_reduced_effective_gravity():
    """A ballistic vehicle flying due east at the equator experiences a
    reduced effective gravity due to centrifugal + Coriolis pull-up.
    The trajectory reaches higher altitudes at each velocity than the
    non-rotating equivalent."""
    from reentrykit.planet import EARTH, EARTH_NON_ROTATING

    # Start at equator flying due east (where Coriolis pull-up is maximal
    # for eastward flight)
    vehicle, state = _ballistic_vehicle_at_latitude(0.0, 90.0)

    r_rotating = simulate(vehicle, state, planet=EARTH)
    r_static = simulate(vehicle, state, planet=EARTH_NON_ROTATING)

    # At similar velocity points, rotating-earth trajectory should be higher
    # (less gravity pulling down → descends slower)
    # Find a mid-trajectory velocity value
    v_target = 5000.0
    i_rot = int(np.argmin(np.abs(r_rotating.velocity - v_target)))
    i_stat = int(np.argmin(np.abs(r_static.velocity - v_target)))

    assert r_rotating.altitude[i_rot] > r_static.altitude[i_stat], (
        f"Rotating Earth (eastward) should give higher altitude at V=5000: "
        f"rot={r_rotating.altitude[i_rot]:.0f} m, "
        f"static={r_static.altitude[i_stat]:.0f} m"
    )


def test_rotating_earth_westward_vs_eastward_differs_at_mid_latitude():
    """At a mid-latitude, eastward flight experiences more centrifugal-like
    pull-up than westward (the vehicle is co-rotating vs counter-rotating
    with Earth). Descent profiles differ accordingly."""
    from reentrykit.planet import EARTH

    # Fly from 45 deg N, once eastward and once westward
    vehicle, state_east = _ballistic_vehicle_at_latitude(45.0, 90.0)
    _, state_west = _ballistic_vehicle_at_latitude(45.0, 270.0)

    r_east = simulate(vehicle, state_east, planet=EARTH)
    r_west = simulate(vehicle, state_west, planet=EARTH)

    # Compare peak g: eastward vehicle is co-rotating, sees reduced effective
    # gravity, descends more gradually, encounters peak g at different time
    peak_east = -np.gradient(r_east.velocity, r_east.time).min() / 9.80665
    peak_west = -np.gradient(r_west.velocity, r_west.time).min() / 9.80665

    # Expect a measurable difference (even if small) between eastward and westward
    relative_diff = abs(peak_east - peak_west) / peak_east
    assert relative_diff > 1e-4, (
        f"Expected rotating-Earth to produce measurable E/W asymmetry at "
        f"mid-latitudes, got peak_east={peak_east:.4f}, peak_west={peak_west:.4f}, "
        f"diff={relative_diff*100:.4f}%"
    )


def test_rotating_earth_coriolis_deflects_heading_northward_flight():
    """A ballistic vehicle flying due north from the equator experiences
    Coriolis heading drift. In the northern hemisphere, rightward Coriolis
    deflection rotates heading clockwise → toward the east.

    Heading is measured at peak deceleration rather than at ground impact,
    because the 3-DOF parameterization has a mathematical singularity in
    heading as the flight-path angle approaches ±90° (terminal vertical
    fall). This singularity is not a physics problem — heading is simply
    undefined for vertical motion — but it does mean our test has to
    sample heading while the trajectory is still well-behaved.
    """
    from reentrykit.planet import EARTH, EARTH_NON_ROTATING

    # Fly due north from the equator
    vehicle, state = _ballistic_vehicle_at_latitude(0.0, 0.0)

    r_rotating = simulate(vehicle, state, planet=EARTH)
    r_static = simulate(vehicle, state, planet=EARTH_NON_ROTATING)

    # Measure heading at peak deceleration (well-behaved, physically meaningful)
    dV_dt = np.gradient(r_rotating.velocity, r_rotating.time)
    i_peak = dV_dt.argmin()

    heading_at_peak = r_rotating.heading[i_peak]
    heading_static_at_peak = r_static.heading[i_peak]

    heading_drift = heading_at_peak - heading_static_at_peak
    # Physics estimate at peak deceleration (~30s into flight, latitude ~2 deg):
    # Coriolis rate ~ 2*Omega*sin(phi) ~ 5e-6 rad/s, integrated over ~30s
    # gives ~0.01 deg. Using 0.005 deg as threshold for clear detection
    # above numerical noise.
    assert heading_drift > np.deg2rad(0.005), (
        f"Expected Coriolis-induced heading drift > 0.005 deg at peak deceleration "
        f"for N-bound flight on rotating Earth, got "
        f"{np.rad2deg(heading_drift):.4f} deg"
    )

    # Also verify sign: northern-hemisphere Coriolis deflects rightward (east).
    # In our aerospace convention, eastward deflection means positive dpsi.
    assert heading_drift > 0, (
        f"Expected positive heading drift (eastward) for N-bound flight in "
        f"northern hemisphere, got {np.rad2deg(heading_drift):.4f} deg"
    )


def test_rotating_earth_stardust_peak_g_similar_to_non_rotating():
    """Stardust-class validation: peak g should be nearly unchanged by Earth
    rotation. The ballistic coefficient and entry conditions dominate; Coriolis
    and centrifugal are small perturbations over the short Stardust trajectory.

    This ensures Phase 2 doesn't silently break Stardust/Genesis validations.
    """
    from reentrykit.planet import EARTH, EARTH_NON_ROTATING

    # Stardust-like vehicle and entry
    vehicle = Vehicle.from_mass_area_cd(
        mass=45.8, reference_area=np.pi * (0.811 / 2) ** 2,
        drag_coefficient=1.0,
    )
    state = InitialState(
        altitude=125000.0,
        velocity=12600.0,
        flight_path_angle=np.deg2rad(-8.2),
        heading=np.deg2rad(90.0),  # due east
        latitude=np.deg2rad(0.0),
        longitude=0.0,
    )

    r_rotating = simulate(vehicle, state, planet=EARTH, max_time=500.0, dt_output=0.05)
    r_static = simulate(vehicle, state, planet=EARTH_NON_ROTATING, max_time=500.0, dt_output=0.05)

    peak_rot = -np.gradient(r_rotating.velocity, r_rotating.time).min() / 9.80665
    peak_stat = -np.gradient(r_static.velocity, r_static.time).min() / 9.80665

    # Peak g differs by at most a few percent (Stardust is short enough
    # that rotation effects don't accumulate dramatically)
    relative_diff = abs(peak_rot - peak_stat) / peak_stat
    assert relative_diff < 0.05, (
        f"Stardust peak g on rotating vs non-rotating Earth should differ by "
        f"<5%, got rot={peak_rot:.2f}, stat={peak_stat:.2f}, "
        f"diff={relative_diff*100:.2f}%"
    )