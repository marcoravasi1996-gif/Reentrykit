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

from typing import Callable, NamedTuple, Union

# Type alias: L/D can be a constant float or a function of time [seconds]
LiftToDragSpec = Union[float, Callable[[float], float]]

# Type alias: Cd can be a constant float or a function of Mach number
DragSpec = Union[float, Callable[[float], float]]

import numpy as np
from scipy.integrate import solve_ivp

from reentrykit.atmosphere import MAX_ALTITUDE, MAX_EXTENDED_ALTITUDE, us1976

# Physical constants for trajectory dynamics
G0 = 9.80665  # standard gravitational acceleration [m/s^2]
EARTH_RADIUS = 6_378_137.0  # WGS-84 equatorial radius [m]

# Integration limits and defaults
_DEFAULT_MAX_TIME = 3600.0  # integration ceiling [s]
_MIN_ALTITUDE = 0.0  # trajectory terminates at this altitude [m]

class Vehicle(NamedTuple):
    """Aerodynamic and mass properties of a reentry vehicle.

    The vehicle is parameterized by its physical properties (mass, reference
    area, drag coefficient). The drag coefficient can be either constant or
    a callable of Mach number, allowing high-fidelity modeling of Cd variation
    across flow regimes (free-molecular, transitional, continuum hypersonic,
    transonic, subsonic).

    Lift is specified as a lift-to-drag ratio — constant or a callable of time.
    For ballistic flight, leave `lift_to_drag_ratio` at its default of 0.0.

    Ballistic coefficient β = m / (Cd · S) is no longer a stored attribute
    because it varies with Mach when Cd varies. To compute instantaneous β at
    a given Mach, use `vehicle.beta(mach)`.

    Nose radius and mass are carried through for downstream analyses
    (aerothermal heating, structural loads).
    """

    reference_area: float              # [m^2], vehicle frontal area
    mass: float                        # [kg]
    drag_coefficient: DragSpec = 1.0   # [-] or callable(mach) -> Cd
    lift_to_drag_ratio: LiftToDragSpec = 0.0  # [-] or callable(t_sec) -> L/D
    nose_radius: float = 0.1           # [m], for stagnation-point heating

    @classmethod
    def from_mass_area_cd(
        cls,
        mass: float,
        reference_area: float,
        drag_coefficient: DragSpec,
        lift_to_drag_ratio: LiftToDragSpec = 0.0,
        nose_radius: float = 0.1,
    ) -> "Vehicle":
        """Construct a Vehicle from mass, reference area, and drag coefficient.

        This classmethod preserves backward compatibility with code written
        when `ballistic_coefficient` was the primary parameter. Using the
        primary constructor `Vehicle(reference_area=..., mass=..., ...)` is
        equivalent.

        The `drag_coefficient` may be a constant or a callable of Mach number,
        enabling Mach-dependent Cd modeling. The `lift_to_drag_ratio` may be
        a constant or a callable of time [s].
        """
        return cls(
            reference_area=reference_area,
            mass=mass,
            drag_coefficient=drag_coefficient,
            lift_to_drag_ratio=lift_to_drag_ratio,
            nose_radius=nose_radius,
        )

    def beta(self, mach: float = 10.0) -> float:
        """Instantaneous ballistic coefficient [kg/m^2] at a given Mach.

        β = m / (Cd · S). For constant Cd, this value is constant.
        For Mach-dependent Cd, β varies with Mach.

        Default Mach of 10 gives a representative hypersonic value for
        reentry mission design.
        """
        cd = self.drag_coefficient
        cd_value = cd(mach) if callable(cd) else cd
        return self.mass / (cd_value * self.reference_area)
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

    # Atmospheric properties at current altitude. us1976() handles 0-200 km
    # via US1976 below 86 km and exponential extension above.
    atmo = us1976(altitude)
    density = atmo.density
    speed_of_sound = atmo.speed_of_sound

    # Mach number — used to evaluate Cd if it is Mach-dependent.
    # Guard against division by zero when velocity is near zero (end of flight).
    mach = velocity / speed_of_sound if velocity > 1.0 else 0.0

    # Evaluate Cd: constant float or callable(mach) -> Cd
    cd = vehicle.drag_coefficient
    cd_value = cd(mach) if callable(cd) else cd

    # Drag acceleration: a = (1/2) * rho * V^2 * Cd * S / m
    drag_accel = 0.5 * density * velocity**2 * cd_value * vehicle.reference_area / vehicle.mass

    # Evaluate L/D: constant float or callable(t_sec) -> L/D
    ld = vehicle.lift_to_drag_ratio
    ld_value = ld(time) if callable(ld) else ld
    lift_accel = drag_accel * ld_value
    # Gravity and the effective radial acceleration
    r = EARTH_RADIUS + altitude
    sin_gamma = np.sin(flight_path_angle)
    cos_gamma = np.cos(flight_path_angle)

    # Equations of motion
    dV_dt = -drag_accel - G0 * sin_gamma
    dgamma_dt = lift_accel / velocity - (G0 / velocity - velocity / r) * cos_gamma
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
    if initial_state.altitude < 0.0 or initial_state.altitude > MAX_EXTENDED_ALTITUDE:
        raise ValueError(
            f"Initial altitude {initial_state.altitude} m is outside the "
            f"valid atmosphere range [0, {MAX_EXTENDED_ALTITUDE}] m."
        )

    # Pack the initial state into the form solve_ivp expects
    y0 = np.array([
        initial_state.velocity,
        initial_state.flight_path_angle,
        initial_state.altitude,
        initial_state.downrange,
    ])

    # Event: stop integration when altitude reaches zero
    # Event: stop integration when altitude reaches zero
    def ground_impact(time: float, state: np.ndarray, vehicle: Vehicle) -> float:
        return state[2] - _MIN_ALTITUDE  # altitude component

    ground_impact.terminal = True
    ground_impact.direction = -1  # only trigger on descending crossing

    # Event: stop integration if the vehicle skips above the atmosphere ceiling.
    # This prevents the integrator from probing altitudes beyond the model's
    # valid range, and correctly terminates trajectories that skip out
    # permanently (e.g. excessive lift modulation without guidance).
    def skip_out(time: float, state: np.ndarray, vehicle: Vehicle) -> float:
        return MAX_EXTENDED_ALTITUDE - state[2]

    skip_out.terminal = True
    skip_out.direction = -1  # only trigger on ascending crossing

    # Evenly-spaced output times
    t_eval = np.arange(0.0, max_time + dt_output, dt_output)

    # Integrate
    solution = solve_ivp(
            fun=_derivatives,
            t_span=(0.0, max_time),
            y0=y0,
            args=(vehicle,),
            method="RK45",
            events=(ground_impact, skip_out),
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

    # Derived quantities at each time step. us1976() handles the full
    # 0-200 km range via an exponential extension above the validated 86 km ceiling.
    densities = np.array([us1976(h).density for h in altitude])
    speeds_of_sound = np.array([us1976(h).speed_of_sound for h in altitude])
    mach = velocity / speeds_of_sound
    dynamic_pressure = 0.5 * densities * velocity**2

    # Determine why integration stopped
    if solution.t_events[0].size > 0:
        termination_reason = "Ground impact"
    elif solution.t_events[1].size > 0:
        termination_reason = "Skip-out above extended atmosphere ceiling"
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
