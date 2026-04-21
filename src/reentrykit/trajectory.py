"""3-DOF point-mass reentry trajectory simulator.

Integrates the planar flight equations over a non-rotating Earth with
constant gravitational acceleration. Uses the US1976 atmosphere model
from :mod:`reentrykit.atmosphere` for density and speed of sound at each
integration step.

The vehicle is treated as a point mass with constant aerodynamic
coefficients (drag, optional lift-to-drag ratio). This is the standard
preliminary-design fidelity for reentry mission analysis.

References
----------
Allen, H.J. and Eggers, A.J. (1958). *A Study of the Motion and
Aerodynamic Heating of Ballistic Missiles Entering the Earth's
Atmosphere at High Supersonic Speeds*. NACA Report 1381.

Vinh, N.X., Busemann, A., and Culp, R.D. (1980). *Hypersonic and
Planetary Entry Flight Mechanics*. University of Michigan Press.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

from reentrykit.atmosphere import MAX_ALTITUDE, us1976

# Physical constants for trajectory dynamics
G0 = 9.80665  # standard gravitational acceleration [m/s^2]
EARTH_RADIUS = 6_378_137.0  # WGS-84 equatorial radius [m]

# Integration limits and defaults
_DEFAULT_MAX_TIME = 3600.0  # integration ceiling [s]
_MIN_ALTITUDE = 0.0  # trajectory terminates at this altitude [m]

class Vehicle(NamedTuple):
    """Aerodynamic and mass properties of a reentry vehicle.

    Treats the vehicle as a point mass with constant aerodynamic coefficients.
    Suitable for preliminary-design trajectory analysis.
    """

    mass: float  # [kg]
    reference_area: float  # [m^2]
    drag_coefficient: float  # [-]
    lift_to_drag_ratio: float = 0.0  # [-], zero for pure ballistic
    nose_radius: float = 0.1  # [m], used for downstream heating analysis


class InitialState(NamedTuple):
    """Initial state of the vehicle at the start of integration."""

    altitude: float  # [m], above mean sea level
    velocity: float  # [m/s], magnitude relative to the atmosphere
    flight_path_angle: float  # [rad], negative when descending
    downrange: float = 0.0  # [m], along-track ground distance


class TrajectoryResult(NamedTuple):
    """Time history of a reentry trajectory.

    All fields are NumPy arrays of the same length, indexed by time step.
    """

    time: np.ndarray  # [s]
    altitude: np.ndarray  # [m]
    velocity: np.ndarray  # [m/s]
    flight_path_angle: np.ndarray  # [rad]
    downrange: np.ndarray  # [m]
    mach: np.ndarray  # [-]
    dynamic_pressure: np.ndarray  # [Pa]
    density: np.ndarray  # [kg/m^3]
    termination_reason: str  # why integration stopped

def _derivatives(
    time: float,
    state: np.ndarray,
    vehicle: Vehicle,
) -> np.ndarray:
    """Compute the time derivatives of the state vector.

    The state vector is [V, gamma, h, s]:
    - V: velocity magnitude [m/s]
    - gamma: flight-path angle [rad], negative descending
    - h: altitude [m]
    - s: downrange distance [m]

    Called repeatedly by solve_ivp during integration.
    """
    velocity, flight_path_angle, altitude, _downrange = state

    # Atmospheric properties at current altitude
    atmo = us1976(altitude)
    density = atmo.density

    # Dynamic pressure
    q_dyn = 0.5 * density * velocity**2

    # Aerodynamic forces
    drag = q_dyn * vehicle.reference_area * vehicle.drag_coefficient
    lift = drag * vehicle.lift_to_drag_ratio

    # Gravity and the effective radial acceleration
    r = EARTH_RADIUS + altitude
    sin_gamma = np.sin(flight_path_angle)
    cos_gamma = np.cos(flight_path_angle)

    # Equations of motion
    dV_dt = -drag / vehicle.mass - G0 * sin_gamma
    dgamma_dt = lift / (vehicle.mass * velocity) - (G0 / velocity - velocity / r) * cos_gamma
    dh_dt = velocity * sin_gamma
    ds_dt = EARTH_RADIUS * velocity * cos_gamma / r

    return np.array([dV_dt, dgamma_dt, dh_dt, ds_dt])

def simulate(
    vehicle: Vehicle,
    initial_state: InitialState,
    max_time: float = _DEFAULT_MAX_TIME,
    dt_output: float = 1.0,
) -> TrajectoryResult:
    """Simulate a reentry trajectory.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle mass, area, and aerodynamic coefficients.
    initial_state : InitialState
        Starting altitude, velocity, flight-path angle, and downrange.
    max_time : float
        Maximum integration time [s]. Integration stops earlier on
        ground impact.
    dt_output : float
        Output time step [s]. Used to build the evenly-spaced time
        array for result sampling. Does not affect integration accuracy.

    Returns
    -------
    TrajectoryResult
        Time history of the trajectory with state variables and derived
        quantities at each output step.

    Raises
    ------
    ValueError
        If the initial altitude is outside the valid atmosphere range
        [0, 86000] m.
    """
    if initial_state.altitude < 0.0 or initial_state.altitude > MAX_ALTITUDE:
        raise ValueError(
            f"Initial altitude {initial_state.altitude} m is outside the "
            f"valid atmosphere range [0, {MAX_ALTITUDE}] m."
        )

    # Pack the initial state into the form solve_ivp expects
    y0 = np.array([
        initial_state.velocity,
        initial_state.flight_path_angle,
        initial_state.altitude,
        initial_state.downrange,
    ])

    # Event: stop integration when altitude reaches zero
    def ground_impact(time: float, state: np.ndarray, vehicle: Vehicle) -> float:
        return state[2] - _MIN_ALTITUDE  # altitude component

    ground_impact.terminal = True
    ground_impact.direction = -1  # only trigger on descending crossing

    # Evenly-spaced output times
    t_eval = np.arange(0.0, max_time + dt_output, dt_output)

    # Integrate
    solution = solve_ivp(
            fun=_derivatives,
            t_span=(0.0, max_time),
            y0=y0,
            args=(vehicle,),
            method="RK45",
            events=ground_impact,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.1,
    )

    # Extract time histories
    time = solution.t
    velocity = solution.y[0]
    flight_path_angle = solution.y[1]
    altitude = solution.y[2]
    downrange = solution.y[3]

    # Derived quantities at each time step
    densities = np.array([us1976(h).density for h in altitude])
    speeds_of_sound = np.array([us1976(h).speed_of_sound for h in altitude])
    mach = velocity / speeds_of_sound
    dynamic_pressure = 0.5 * densities * velocity**2

    # Determine why integration stopped
    if solution.t_events[0].size > 0:
        termination_reason = "Ground impact"
    elif solution.t[-1] >= max_time - 1e-6:
        termination_reason = "Max time reached"
    else:
        termination_reason = f"Integrator status: {solution.message}"

    return TrajectoryResult(
        time=time,
        altitude=altitude,
        velocity=velocity,
        flight_path_angle=flight_path_angle,
        downrange=downrange,
        mach=mach,
        dynamic_pressure=dynamic_pressure,
        density=densities,
        termination_reason=termination_reason,
    )
