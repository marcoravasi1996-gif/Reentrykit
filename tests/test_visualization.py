"""Smoke tests for the visualization module.

Visualization correctness is hard to assert without rendering and
comparing images. These tests verify only that the functions execute
without errors and return correctly-typed objects.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from reentrykit.planet import EARTH_NON_ROTATING
from reentrykit.trajectory import InitialState, Vehicle, simulate
from reentrykit.visualization import (
    _latlon_to_ecef,
    _local_tangent_basis,
    _vbc_heading_to_enu,
    plot_trajectory_3d,
    plot_trajectory_summary,
)


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------


def test_latlon_to_ecef_at_equator_prime_meridian():
    """At lat=0, lon=0, ECEF should be (R, 0, 0)."""
    lat = np.array([0.0])
    lon = np.array([0.0])
    alt = np.array([0.0])

    x, y, z = _latlon_to_ecef(lat, lon, alt)

    assert x[0] == pytest.approx(6_378_137.0, rel=1e-9)
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert z[0] == pytest.approx(0.0, abs=1e-6)


def test_latlon_to_ecef_at_north_pole():
    """At lat=pi/2, lon=any, ECEF should be (0, 0, R)."""
    lat = np.array([np.pi / 2])
    lon = np.array([np.pi / 4])  # arbitrary
    alt = np.array([0.0])

    x, y, z = _latlon_to_ecef(lat, lon, alt)

    assert x[0] == pytest.approx(0.0, abs=1.0)
    assert y[0] == pytest.approx(0.0, abs=1.0)
    assert z[0] == pytest.approx(6_378_137.0, rel=1e-9)


def test_latlon_to_ecef_with_altitude():
    """Adding altitude should increase ECEF radial distance."""
    lat = np.array([0.0])
    lon = np.array([0.0])
    alt0 = np.array([0.0])
    alt1 = np.array([100_000.0])

    x0, _, _ = _latlon_to_ecef(lat, lon, alt0)
    x1, _, _ = _latlon_to_ecef(lat, lon, alt1)

    assert x1[0] - x0[0] == pytest.approx(100_000.0, rel=1e-9)


def test_local_tangent_basis_orthonormal():
    """East, north, up at any point should be orthonormal."""
    east, north, up = _local_tangent_basis(np.deg2rad(45), np.deg2rad(60))

    # Each is unit length
    assert np.linalg.norm(east) == pytest.approx(1.0, rel=1e-12)
    assert np.linalg.norm(north) == pytest.approx(1.0, rel=1e-12)
    assert np.linalg.norm(up) == pytest.approx(1.0, rel=1e-12)

    # Mutually orthogonal
    assert abs(np.dot(east, north)) < 1e-12
    assert abs(np.dot(east, up)) < 1e-12
    assert abs(np.dot(north, up)) < 1e-12


def test_vbc_heading_to_enu_due_east():
    """V-B-C heading=0, gamma=0 (horizontal, due east) -> (1, 0, 0)."""
    e, n, u = _vbc_heading_to_enu(heading=0.0, flight_path_angle=0.0)
    assert e == pytest.approx(1.0, rel=1e-12)
    assert n == pytest.approx(0.0, abs=1e-12)
    assert u == pytest.approx(0.0, abs=1e-12)


def test_vbc_heading_to_enu_due_north():
    """V-B-C heading=pi/2, gamma=0 -> (0, 1, 0) due north horizontal."""
    e, n, u = _vbc_heading_to_enu(heading=np.pi / 2, flight_path_angle=0.0)
    assert e == pytest.approx(0.0, abs=1e-12)
    assert n == pytest.approx(1.0, rel=1e-12)
    assert u == pytest.approx(0.0, abs=1e-12)


def test_vbc_heading_to_enu_descending():
    """Descending flight (gamma < 0) -> negative up component."""
    e, n, u = _vbc_heading_to_enu(heading=0.0, flight_path_angle=np.deg2rad(-10))
    assert u < 0.0
    # In horizontal plane, still due east
    assert e > 0


# ---------------------------------------------------------------------------
# Plot functions: smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_trajectory():
    """A simple trajectory for plot testing."""
    vehicle = Vehicle.from_mass_area_cd(
        mass=500.0, reference_area=0.8, drag_coefficient=1.5,
    )
    state = InitialState(
        altitude=80_000.0, velocity=7500.0,
        flight_path_angle=np.deg2rad(-5.0),
        heading=np.deg2rad(15.0),
        latitude=np.deg2rad(40.0),
        longitude=np.deg2rad(-120.0),
    )
    return state, simulate(vehicle, state, planet=EARTH_NON_ROTATING)


def test_plot_trajectory_3d_returns_figure(reference_trajectory):
    """plot_trajectory_3d returns a Plotly Figure without error."""
    state, result = reference_trajectory
    fig = plot_trajectory_3d(result, initial_state=state)
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_3d_contains_earth_and_trajectory(reference_trajectory):
    """Returned figure contains the Earth surface and the trajectory line."""
    state, result = reference_trajectory
    fig = plot_trajectory_3d(result, initial_state=state)

    trace_names = [t.name for t in fig.data]
    assert "Earth" in trace_names
    assert "Trajectory" in trace_names


def test_plot_trajectory_3d_without_initial_state(reference_trajectory):
    """plot_trajectory_3d works without an initial_state (no entry markers)."""
    _, result = reference_trajectory
    fig = plot_trajectory_3d(result, initial_state=None,
                              show_local_frame=False, mark_peak_g=False)
    assert isinstance(fig, go.Figure)


def test_plot_trajectory_3d_disables_optional_features(reference_trajectory):
    """Disabling features removes the corresponding traces."""
    state, result = reference_trajectory

    fig_full = plot_trajectory_3d(result, initial_state=state,
                                   show_local_frame=True, mark_peak_g=True)
    fig_minimal = plot_trajectory_3d(result, initial_state=state,
                                      show_local_frame=False, mark_peak_g=False)

    # Minimal version should have fewer traces
    assert len(fig_minimal.data) < len(fig_full.data)


def test_plot_trajectory_summary_returns_figure(reference_trajectory):
    """plot_trajectory_summary produces a 4-panel figure."""
    state, result = reference_trajectory
    fig = plot_trajectory_summary(result, initial_state=state)
    assert isinstance(fig, go.Figure)
    # Should have multiple traces from 4 panels
    assert len(fig.data) > 5


def test_plot_trajectory_summary_without_initial_state(reference_trajectory):
    """plot_trajectory_summary works without initial_state."""
    _, result = reference_trajectory
    fig = plot_trajectory_summary(result, initial_state=None)
    assert isinstance(fig, go.Figure)