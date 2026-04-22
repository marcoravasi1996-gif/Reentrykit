"""3-DOF point-mass reentry trajectory simulator with bank-angle modulation.

Integrates the motion of a reentry vehicle over a non-rotating spherical Earth
using the Vinh-Busemann-Culp equations of motion.

State vector: [V, gamma, psi, h, s, y]
    V      : atmosphere-relative velocity magnitude [m/s]
    gamma  : flight-path angle [rad], positive above horizontal
    psi    : heading angle [rad], positive = right of downrange axis
    h      : altitude above mean sea level [m]
    s      : downrange distance from entry point [m]
    y      : crossrange distance from entry plane [m]

Lift modulation is specified by two fields on the Vehicle:
    lift_to_drag_ratio : signed L/D, constant or callable(t_sec) -> L/D
    bank_angle         : rotation of lift vector out of vertical plane,
                         constant or callable(t_sec) -> sigma [rad]

At each integration step:
    vertical lift component = drag * (L/D) * cos(sigma)
    lateral  lift component = drag * (L/D) * sin(sigma)

When bank_angle is zero (default), the simulator reduces to planar flight
with (L/D) acting entirely in the vertical plane. This is fully backward
compatible with code that predates bank-angle modeling.

Bank angle sign convention: positive sigma = right wing down (aerospace),
rotating the lift vector clockwise when viewed from behind the vehicle.

References
----------
- Vinh, N.X., Busemann, A., and Culp, R.D. (1980). Hypersonic and Planetary
  Entry Flight Mechanics. University of Michigan Press. Chapter 3.
- Allen, H.J. and Eggers, A.J. (1958). NACA Report 1381.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from reentrykit.atmosphere import MAX_ALTITUDE, MAX_EXTENDED_ALTITUDE, us1976
from reentrykit.planet import EARTH, PlanetModel

# Type aliases: constant float or time-varying callable
LiftToDragSpec = Union[float, Callable[[float], float]]
DragSpec = Union[float, Callable[[float], float]]
BankAngleSpec = Union[float, Callable[[float], float]]

# Physical constants come from the PlanetModel passed to simulate().
# The legacy constant G0 is kept as an import for backward compatibility
# with code that computes Allen-Eggers predictions at sea level.
G0 = 9.80665              # [m/s^2], standard gravitational acceleration at Earth's surface

# Numerical guards
_MIN_ALTITUDE = 0.0   # [m], ground impact
_COS_GAMMA_FLOOR = 1e-4  # prevents division by zero at gamma = +/- 90 deg


class Vehicle(NamedTuple):
    """Aerodynamic and mass properties of a reentry vehicle.

    Lift is specified via two fields:
      - lift_to_drag_ratio: signed L/D, the ratio of lift magnitude to drag
      - bank_angle: rotation of the lift vector out of the vertical plane

    Both may be constants or callables of time [s]. When bank_angle is 0,
    lift acts entirely in the vertical (trajectory) plane — planar flight.
    Nonzero bank tilts the lift vector, adding a lateral component that
    curves the trajectory in heading.

    Drag coefficient may be constant or a callable of Mach number for
    variable-Cd modeling.
    """

    reference_area: float                    # [m^2]
    mass: float                              # [kg]
    drag_coefficient: DragSpec = 1.0         # [-] or callable(mach)
    lift_to_drag_ratio: LiftToDragSpec = 0.0 # [-] or callable(t_sec)
    bank_angle: BankAngleSpec = 0.0          # [rad] or callable(t_sec)
    nose_radius: float = 0.1                 # [m], stagnation-point heating

    @classmethod
    def from_mass_area_cd(
        cls,
        mass: float,
        reference_area: float,
        drag_coefficient: DragSpec,
        lift_to_drag_ratio: LiftToDragSpec = 0.0,
        bank_angle: BankAngleSpec = 0.0,
        nose_radius: float = 0.1,
    ) -> "Vehicle":
        """Convenience constructor accepting the same arguments as the
        primary constructor. Preserved for backward compatibility."""
        return cls(
            reference_area=reference_area,
            mass=mass,
            drag_coefficient=drag_coefficient,
            lift_to_drag_ratio=lift_to_drag_ratio,
            bank_angle=bank_angle,
            nose_radius=nose_radius,
        )

    def beta(self, mach: float = 10.0) -> float:
        """Instantaneous ballistic coefficient [kg/m^2] at the given Mach."""
        cd = self.drag_coefficient
        cd_value = cd(mach) if callable(cd) else cd
        return self.mass / (cd_value * self.reference_area)


class InitialState(NamedTuple):
    """Initial conditions at the entry interface."""

    altitude: float                # [m]
    velocity: float                # [m/s], atmosphere-relative
    flight_path_angle: float       # [rad], negative for descent
    heading: float = 0.0           # [rad], 0 = along positive downrange axis
    downrange: float = 0.0         # [m]
    crossrange: float = 0.0        # [m]


class TrajectoryResult(NamedTuple):
    """Time history of the trajectory and derived aerodynamic quantities."""

    time: np.ndarray                 # [s]
    altitude: np.ndarray             # [m]
    velocity: np.ndarray             # [m/s]
    flight_path_angle: np.ndarray    # [rad]
    heading: np.ndarray              # [rad]
    downrange: np.ndarray            # [m]
    crossrange: np.ndarray           # [m]
    mach: np.ndarray                 # [-]
    dynamic_pressure: np.ndarray     # [Pa]
    density: np.ndarray              # [kg/m^3]
    termination_reason: str


def _derivatives(
    time: float,
    state: np.ndarray,
    vehicle: Vehicle,
    planet: PlanetModel,
) -> list[float]:
    """Equations of motion for 3-DOF point-mass flight with bank angle.

    State: [V, gamma, psi, h, s, y]
    Reduces to planar flight when bank_angle is zero and initial heading is zero.
    """
    velocity, flight_path_angle, heading, altitude, _downrange, _crossrange = state

    # Atmosphere from the planet model
    atmo = planet.atmosphere(altitude)
    density = atmo.density
    speed_of_sound = atmo.speed_of_sound
    # Mach for Cd lookup
    mach = velocity / speed_of_sound if velocity > 1.0 else 0.0

    # Drag coefficient: constant or Mach-dependent
    cd = vehicle.drag_coefficient
    cd_value = cd(mach) if callable(cd) else cd

    # Drag acceleration: a = (1/2) rho V^2 Cd S / m
    drag_accel = 0.5 * density * velocity**2 * cd_value * vehicle.reference_area / vehicle.mass

    # Lift: L/D and bank angle, each either constant or callable
    ld = vehicle.lift_to_drag_ratio
    ld_value = ld(time) if callable(ld) else ld
    sigma = vehicle.bank_angle
    sigma_value = sigma(time) if callable(sigma) else sigma

    # Decompose lift into vertical (in-plane) and lateral (out-of-plane) components
    cos_sigma = np.cos(sigma_value)
    sin_sigma = np.sin(sigma_value)
    vertical_lift_accel = drag_accel * ld_value * cos_sigma
    lateral_lift_accel = drag_accel * ld_value * sin_sigma

    # Geometry and altitude-dependent gravity
    r = planet.radius + altitude
    g = planet.gravity(altitude)
    sin_gamma = np.sin(flight_path_angle)
    cos_gamma = np.cos(flight_path_angle)
    sin_psi = np.sin(heading)
    cos_psi = np.cos(heading)

    # Guard against gamma singularity at +/- 90 deg (never occurs for reentry)
    cos_gamma_safe = np.sign(cos_gamma) * max(abs(cos_gamma), _COS_GAMMA_FLOOR)

    # Equations of motion (Vinh-Busemann-Culp, non-rotating planet)
    # Note: g is altitude-dependent (g = mu / r^2), not the sea-level constant.
    # Rotating-planet Coriolis and centrifugal terms will be added in Phase 2.
    dV_dt = -drag_accel - g * sin_gamma
    dgamma_dt = (
        vertical_lift_accel / velocity
        - (g / velocity - velocity / r) * cos_gamma
    )
    dpsi_dt = lateral_lift_accel / (velocity * cos_gamma_safe)
    dh_dt = velocity * sin_gamma
    ds_dt = planet.radius * velocity * cos_gamma * cos_psi / r
    dy_dt = planet.radius * velocity * cos_gamma * sin_psi / r

    return [dV_dt, dgamma_dt, dpsi_dt, dh_dt, ds_dt, dy_dt]


def simulate(
    vehicle: Vehicle,
    initial_state: InitialState,
    planet: PlanetModel = EARTH,
    max_time: float = 1000.0,
    dt_output: float = 0.1,
) -> TrajectoryResult:
    """Integrate the reentry trajectory from entry interface to termination.

    Terminates on ground impact (h = 0) or skip-out above the 200 km
    extended atmosphere ceiling.
    """
    if initial_state.altitude < 0.0 or initial_state.altitude > planet.max_atmosphere_altitude:
        raise ValueError(
            f"Initial altitude {initial_state.altitude} m is outside the "
            f"valid atmosphere range [0, {planet.max_atmosphere_altitude}] m for {planet.name}."
        )
    if initial_state.velocity <= 0.0:
        raise ValueError(f"Initial velocity must be positive, got {initial_state.velocity} m/s.")

    # 6-element state vector
    y0 = [
        initial_state.velocity,
        initial_state.flight_path_angle,
        initial_state.heading,
        initial_state.altitude,
        initial_state.downrange,
        initial_state.crossrange,
    ]

    t_eval = np.arange(0.0, max_time, dt_output)

    def ground_impact(time: float, state: np.ndarray, vehicle: Vehicle, planet: PlanetModel) -> float:
        return state[3] - _MIN_ALTITUDE

    ground_impact.terminal = True
    ground_impact.direction = -1

    def skip_out(time: float, state: np.ndarray, vehicle: Vehicle, planet: PlanetModel) -> float:
        return planet.max_atmosphere_altitude - state[3]

    skip_out.terminal = True
    skip_out.direction = -1

    solution = solve_ivp(
        fun=_derivatives,
        t_span=(0.0, max_time),
        y0=y0,
        args=(vehicle, planet),
        method="RK45",
        events=(ground_impact, skip_out),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.1,
    )

    if solution.t_events[0].size > 0:
        termination_reason = "Ground impact"
    elif solution.t_events[1].size > 0:
        termination_reason = "Skip-out above extended atmosphere ceiling"
    elif solution.t[-1] >= max_time - 1e-6:
        termination_reason = "Max time reached"
    else:
        termination_reason = f"Integrator status: {solution.message}"

    time = solution.t
    velocity = solution.y[0]
    flight_path_angle = solution.y[1]
    heading = solution.y[2]
    altitude = solution.y[3]
    downrange = solution.y[4]
    crossrange = solution.y[5]

    densities = np.array([planet.atmosphere(h).density for h in altitude])
    speeds_of_sound = np.array([planet.atmosphere(h).speed_of_sound for h in altitude])
    mach = velocity / speeds_of_sound
    dynamic_pressure = 0.5 * densities * velocity**2

    return TrajectoryResult(
        time=time,
        altitude=altitude,
        velocity=velocity,
        flight_path_angle=flight_path_angle,
        heading=heading,
        downrange=downrange,
        crossrange=crossrange,
        mach=mach,
        dynamic_pressure=dynamic_pressure,
        density=densities,
        termination_reason=termination_reason,
    )
