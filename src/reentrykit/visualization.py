"""3D visualization of trajectory results using plotly.

Helpers to render a TrajectoryResult as an interactive 3D Earth + trajectory
view, or as a four-panel summary combining the 3D view with ground track,
altitude profile, and deceleration profile.

These tools are primarily diagnostic — useful for verifying that a
simulated trajectory goes where expected (given the V-B-C heading
convention used internally) and for detecting setup errors visually.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reentrykit.trajectory import InitialState, TrajectoryResult

# Standard Earth parameters (for visualization; match the EARTH planet model)
_EARTH_RADIUS_M = 6_378_137.0


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


def _latlon_to_ecef(
    latitude: np.ndarray,
    longitude: np.ndarray,
    altitude: np.ndarray,
    earth_radius: float = _EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical latitude/longitude/altitude to ECEF Cartesian [m]."""
    r = earth_radius + altitude
    x = r * np.cos(latitude) * np.cos(longitude)
    y = r * np.cos(latitude) * np.sin(longitude)
    z = r * np.sin(latitude)
    return x, y, z


def _local_tangent_basis(
    latitude: float, longitude: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local east/north/up unit vectors in ECEF at a given lat/lon.

    Returns (east, north, up) as 3-component numpy arrays.
    """
    sin_phi, cos_phi = np.sin(latitude), np.cos(latitude)
    sin_theta, cos_theta = np.sin(longitude), np.cos(longitude)

    # Local east: perpendicular to meridian, in equatorial plane
    east = np.array([-sin_theta, cos_theta, 0.0])
    # Local north: perpendicular to east, toward north pole
    north = np.array(
        [-sin_phi * cos_theta, -sin_phi * sin_theta, cos_phi]
    )
    # Local up: radial outward
    up = np.array([cos_phi * cos_theta, cos_phi * sin_theta, sin_phi])
    return east, north, up


def _vbc_heading_to_enu(
    heading: float, flight_path_angle: float
) -> tuple[float, float, float]:
    """Convert V-B-C (ψ from east CCW, γ from horizontal) to ENU unit vector.

    Returns (east_component, north_component, up_component) of the velocity
    unit vector in the local tangent frame.
    """
    east_c = np.cos(flight_path_angle) * np.cos(heading)
    north_c = np.cos(flight_path_angle) * np.sin(heading)
    up_c = np.sin(flight_path_angle)
    return east_c, north_c, up_c


# ---------------------------------------------------------------------------
# Earth sphere mesh
# ---------------------------------------------------------------------------


def _make_earth_mesh(
    n_lat: int = 36, n_lon: int = 72, earth_radius: float = _EARTH_RADIUS_M
) -> go.Surface:
    """Semi-transparent Earth sphere as a plotly Surface trace."""
    phi = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    theta = np.linspace(-np.pi, np.pi, n_lon)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")

    x = earth_radius * np.cos(phi_grid) * np.cos(theta_grid)
    y = earth_radius * np.cos(phi_grid) * np.sin(theta_grid)
    z = earth_radius * np.sin(phi_grid)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.3,
        colorscale=[[0, "lightblue"], [1, "lightblue"]],
        showscale=False,
        name="Earth",
        hoverinfo="skip",
        lighting=dict(ambient=0.8, diffuse=0.2, specular=0.0),
    )


def _make_meridian_grid(
    earth_radius: float = _EARTH_RADIUS_M,
) -> list[go.Scatter3d]:
    """Major meridian and parallel lines for orientation reference."""
    traces: list[go.Scatter3d] = []
    n = 50

    # Equator (lat = 0)
    theta = np.linspace(-np.pi, np.pi, n)
    x = earth_radius * np.cos(theta)
    y = earth_radius * np.sin(theta)
    z = np.zeros_like(theta)
    traces.append(
        go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="gray", width=1),
            name="Equator", hoverinfo="skip", showlegend=False,
        )
    )

    # Prime meridian (lon = 0)
    phi = np.linspace(-np.pi / 2, np.pi / 2, n)
    x = earth_radius * np.cos(phi)
    y = np.zeros_like(phi)
    z = earth_radius * np.sin(phi)
    traces.append(
        go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color="gray", width=1),
            name="Prime Meridian", hoverinfo="skip", showlegend=False,
        )
    )
    return traces


# ---------------------------------------------------------------------------
# Main trajectory 3D plot
# ---------------------------------------------------------------------------


def _trajectory_trace(
    result: TrajectoryResult, earth_radius: float = _EARTH_RADIUS_M
) -> go.Scatter3d:
    """Trajectory path as a 3D line colored by time."""
    x, y, z = _latlon_to_ecef(
        result.latitude, result.longitude, result.altitude, earth_radius
    )

    # Hover text: altitude, velocity, time
    hover_text = [
        f"t={t:.1f}s<br>alt={h/1000:.1f}km<br>V={v:.0f}m/s<br>"
        f"lat={np.rad2deg(lat):.2f}°<br>lon={np.rad2deg(lon):.2f}°"
        for t, h, v, lat, lon in zip(
            result.time, result.altitude, result.velocity,
            result.latitude, result.longitude,
        )
    ]

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=result.time, colorscale="Viridis", width=6,
                  colorbar=dict(title="Time [s]", x=1.05)),
        name="Trajectory",
        text=hover_text,
        hoverinfo="text",
    )


def _frame_arrows_at_entry(
    initial_state: InitialState,
    arrow_length_km: float = 400.0,
    earth_radius: float = _EARTH_RADIUS_M,
) -> list[go.Scatter3d]:
    """Draw local ENU frame + initial velocity at the entry point.

    - Red line: local east
    - Green line: local north
    - Blue line: local up
    - Yellow line: initial velocity direction (from V-B-C heading + FPA)
    """
    arrow_len_m = arrow_length_km * 1000.0

    x0, y0, z0 = _latlon_to_ecef(
        np.array([initial_state.latitude]),
        np.array([initial_state.longitude]),
        np.array([initial_state.altitude]),
        earth_radius,
    )
    origin = np.array([x0[0], y0[0], z0[0]])

    east, north, up = _local_tangent_basis(
        initial_state.latitude, initial_state.longitude
    )

    # Velocity direction in ENU, then rotate to ECEF
    v_e, v_n, v_u = _vbc_heading_to_enu(
        initial_state.heading, initial_state.flight_path_angle
    )
    v_ecef = v_e * east + v_n * north + v_u * up

    def _line(vec: np.ndarray, color: str, name: str) -> go.Scatter3d:
        tip = origin + vec * arrow_len_m
        return go.Scatter3d(
            x=[origin[0], tip[0]],
            y=[origin[1], tip[1]],
            z=[origin[2], tip[2]],
            mode="lines",
            line=dict(color=color, width=8),
            name=name,
            hoverinfo="name",
        )

    return [
        _line(east, "red", "Local East"),
        _line(north, "green", "Local North"),
        _line(up, "royalblue", "Local Up"),
        _line(v_ecef, "gold", "Initial Velocity"),
    ]


def _peak_g_markers(result: TrajectoryResult) -> list[go.Scatter3d]:
    """Mark peak-g locations with star markers."""
    dV_dt = np.gradient(result.velocity, result.time)
    g_load = -dV_dt / 9.80665

    # First peak (during first 40% of trajectory)
    n_half = max(len(result.time) // 2, 2)
    first_window = g_load[:n_half]
    i_first = int(first_window.argmax())

    # Second peak (in remainder)
    second_window_start = min(i_first + 100, len(result.time) - 1)
    second_window = g_load[second_window_start:]
    i_second = (
        second_window_start + int(second_window.argmax())
        if second_window.size > 0 else None
    )

    markers: list[go.Scatter3d] = []
    for idx, label, color in [
        (i_first, "1st Peak", "orange"),
        (i_second, "2nd Peak", "magenta"),
    ]:
        if idx is None or idx >= len(result.time):
            continue
        x, y, z = _latlon_to_ecef(
            np.array([result.latitude[idx]]),
            np.array([result.longitude[idx]]),
            np.array([result.altitude[idx]]),
        )
        markers.append(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond"),
                name=f"{label}: {g_load[idx]:.2f} g",
                hoverinfo="name",
            )
        )
    return markers


def plot_trajectory_3d(
    result: TrajectoryResult,
    initial_state: Optional[InitialState] = None,
    show_local_frame: bool = True,
    mark_peak_g: bool = True,
    planet_name: str = "Earth",
    title: Optional[str] = None,
) -> go.Figure:
    """Interactive 3D view of a trajectory around a spherical Earth.

    Parameters
    ----------
    result
        A `TrajectoryResult` from `simulate(...)`.
    initial_state
        If provided, draw the entry point and initial velocity vector.
    show_local_frame
        If True (and `initial_state` is provided), draw local ENU unit
        vectors at the entry point in red/green/blue and the initial
        velocity direction in gold.
    mark_peak_g
        If True, place markers at first and second peak deceleration.
    planet_name
        Label for the planet (used in plot title).
    title
        Override the default title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D figure; call `.show()` to display in a notebook.
    """
    traces: list[go.Surface | go.Scatter3d] = [_make_earth_mesh()]
    traces.extend(_make_meridian_grid())
    traces.append(_trajectory_trace(result))

    if initial_state is not None and show_local_frame:
        traces.extend(_frame_arrows_at_entry(initial_state))

    if mark_peak_g:
        traces.extend(_peak_g_markers(result))

    fig = go.Figure(data=traces)

    default_title = f"Trajectory around {planet_name}"
    if initial_state is not None:
        default_title += (
            f" — entry at {np.rad2deg(initial_state.latitude):.2f}°, "
            f"{np.rad2deg(initial_state.longitude):.2f}°, "
            f"ψ={np.rad2deg(initial_state.heading):.1f}°"
        )

    fig.update_layout(
        title=title or default_title,
        scene=dict(
            xaxis=dict(title="ECEF X [m]"),
            yaxis=dict(title="ECEF Y [m]"),
            zaxis=dict(title="ECEF Z [m]"),
            aspectmode="data",
        ),
        height=700,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Four-panel summary
# ---------------------------------------------------------------------------


def plot_trajectory_summary(
    result: TrajectoryResult,
    initial_state: Optional[InitialState] = None,
    planet_name: str = "Earth",
    title: Optional[str] = None,
) -> go.Figure:
    """Four-panel trajectory diagnostic: 3D view, ground track, altitude, g-load."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            f"3D Trajectory around {planet_name}",
            "Ground track (latitude vs longitude)",
            "Altitude vs time",
            "Deceleration vs time",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    # Panel 1: 3D view
    fig.add_trace(_make_earth_mesh(), row=1, col=1)
    for grid_trace in _make_meridian_grid():
        fig.add_trace(grid_trace, row=1, col=1)
    fig.add_trace(_trajectory_trace(result), row=1, col=1)
    if initial_state is not None:
        for arrow in _frame_arrows_at_entry(initial_state):
            fig.add_trace(arrow, row=1, col=1)
    for marker in _peak_g_markers(result):
        fig.add_trace(marker, row=1, col=1)

    # Panel 2: Ground track (lat vs lon)
    fig.add_trace(
        go.Scatter(
            x=np.rad2deg(result.longitude),
            y=np.rad2deg(result.latitude),
            mode="lines",
            line=dict(color="royalblue", width=2),
            name="Ground track",
            showlegend=False,
        ),
        row=1, col=2,
    )
    if initial_state is not None:
        fig.add_trace(
            go.Scatter(
                x=[np.rad2deg(initial_state.longitude)],
                y=[np.rad2deg(initial_state.latitude)],
                mode="markers",
                marker=dict(size=10, color="green", symbol="circle"),
                name="Entry",
                showlegend=False,
            ),
            row=1, col=2,
        )

    # Panel 3: Altitude vs time
    fig.add_trace(
        go.Scatter(
            x=result.time, y=result.altitude / 1000.0,
            mode="lines",
            line=dict(color="darkgreen", width=2),
            name="Altitude",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Panel 4: Deceleration vs time
    dV_dt = np.gradient(result.velocity, result.time)
    g_load = -dV_dt / 9.80665
    fig.add_trace(
        go.Scatter(
            x=result.time, y=g_load,
            mode="lines",
            line=dict(color="crimson", width=2),
            name="Deceleration",
            showlegend=False,
        ),
        row=2, col=2,
    )

    # Axis labels
    fig.update_xaxes(title_text="Longitude [deg]", row=1, col=2)
    fig.update_yaxes(title_text="Latitude [deg]", row=1, col=2)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Altitude [km]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Deceleration [g]", row=2, col=2)

    # 3D scene aspect
    fig.update_scenes(aspectmode="data", row=1, col=1)

    default_title = f"Trajectory Diagnostic — {planet_name}"
    if initial_state is not None:
        default_title += (
            f"  |  entry {np.rad2deg(initial_state.latitude):.1f}°, "
            f"{np.rad2deg(initial_state.longitude):.1f}°, "
            f"ψ={np.rad2deg(initial_state.heading):.1f}°"
        )

    fig.update_layout(
        title=title or default_title,
        height=900,
        showlegend=True,
    )
    return fig