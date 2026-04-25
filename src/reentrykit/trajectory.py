"""3-DOF point-mass reentry trajectory simulator with bank-angle modulation.

Integrates the motion of a reentry vehicle over a planet (rotating or
non-rotating) using the Vinh-Busemann-Culp equations of motion.

State vector: [V, gamma, psi, h, phi, theta]
    V      : atmosphere-relative velocity magnitude [m/s]
    gamma  : flight-path angle [rad], positive above horizontal
    psi    : heading angle [rad], V-B-C convention (from east, CCW)
    h      : altitude above mean sea level [m]
    phi    : latitude [rad], positive north
    theta  : longitude [rad], positive east

Downrange and crossrange are computed post-hoc from latitude/longitude
as great-circle distances projected onto the entry heading.

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

from reentrykit.planet import EARTH_NON_ROTATING, PlanetModel

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
        """Convenience constructor with mass-first argument order; otherwise
        identical to the default Vehicle constructor."""
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
    """Initial conditions at the entry interface.

    Heading convention: psi is measured from the local parallel of
    latitude (east) to the horizontal projection of velocity, positive
    counterclockwise (toward north). This follows Vinh, Busemann, and
    Culp (1980), "Hypersonic and Planetary Entry Flight Mechanics,"
    §2-5.

        psi = 0      -> heading due east
        psi = pi/2   -> heading due north
        psi = pi     -> heading due west
        psi = 3pi/2  -> heading due south

    Bank angle sigma (on the Vehicle) also follows V-B-C: positive CCW
    (opposite of aerospace standard). Users providing bank schedules
    from aerospace sources must negate them at input.
    """

    altitude: float                # [m]
    velocity: float                # [m/s], atmosphere-relative at entry
    flight_path_angle: float       # [rad], negative for descent
    heading: float = 0.0           # [rad], V-B-C convention (from east, CCW)
    latitude: float = 0.0          # [rad], positive north
    longitude: float = 0.0         # [rad], positive east


class TrajectoryResult(NamedTuple):
    """Time history of the trajectory and derived aerodynamic quantities.

    Position is reported as latitude/longitude (the simulation's native
    state variables) plus convenience downrange/crossrange arrays
    computed as great-circle distances from the entry point.

    Heading follows Vinh-Busemann-Culp convention: psi = 0 is east,
    psi = pi/2 is north, psi positive counterclockwise.
    """

    time: np.ndarray                 # [s]
    altitude: np.ndarray             # [m]
    velocity: np.ndarray             # [m/s]
    flight_path_angle: np.ndarray    # [rad]
    heading: np.ndarray              # [rad], V-B-C convention (from east, CCW)
    latitude: np.ndarray             # [rad]
    longitude: np.ndarray            # [rad]
    downrange: np.ndarray            # [m], great-circle distance from entry
    crossrange: np.ndarray           # [m], signed cross-track distance from entry heading
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
    """Equations of motion for 3-DOF point-mass flight with bank angle,
    atmosphere-relative velocity in the planet-fixed rotating frame.

    Follows Vinh, Busemann, and Culp (1980), "Hypersonic and Planetary
    Entry Flight Mechanics," §2-5, equations (2-44) through (2-49).
    Conventions match V-B-C exactly.

    State vector: [V, gamma, psi, h, phi, theta]
        V     : atmosphere-relative velocity magnitude [m/s]
        gamma : flight-path angle [rad], positive above horizontal
        psi   : heading [rad], measured from local parallel of latitude
                (east) to the horizontal projection of V. Positive when
                rotating counterclockwise (from east toward north).
                So psi = 0 is due east, psi = pi/2 is due north,
                psi = pi is due west.
        h     : altitude above mean sea level [m]
        phi   : latitude [rad], positive north
        theta : longitude [rad], positive east

    Bank angle sigma: positive when rotating counterclockwise (V-B-C
    convention). This is opposite to standard aerospace convention;
    users supplying sigma from aerospace literature (e.g., Apollo
    bank schedules) must negate it at input.

    When Omega = 0 (as in EARTH_NON_ROTATING), all rotation-related
    terms vanish and the equations reduce to the non-rotating
    formulation.
    """
    velocity, flight_path_angle, heading, altitude, latitude, _longitude = state

    # Atmosphere from the planet model
    atmo = planet.atmosphere(altitude)
    density = atmo.density
    speed_of_sound = atmo.speed_of_sound

    # Mach for Cd lookup
    mach = velocity / speed_of_sound if velocity > 1.0 else 0.0

    # Drag coefficient
    cd = vehicle.drag_coefficient
    cd_value = cd(mach) if callable(cd) else cd

    # Drag acceleration (magnitude, positive along -V direction)
    drag_accel = 0.5 * density * velocity**2 * cd_value * vehicle.reference_area / vehicle.mass

    # Lift and bank angle (V-B-C sigma convention: positive CCW)
    ld = vehicle.lift_to_drag_ratio
    ld_value = ld(time) if callable(ld) else ld
    sigma = vehicle.bank_angle
    sigma_value = sigma(time) if callable(sigma) else sigma

    cos_sigma = np.cos(sigma_value)
    sin_sigma = np.sin(sigma_value)

    # Geometry and planet properties
    r = planet.radius + altitude
    g = planet.gravity(altitude)
    omega = planet.rotation_rate

    # Pre-compute trig values
    sin_gamma = np.sin(flight_path_angle)
    cos_gamma = np.cos(flight_path_angle)
    sin_psi = np.sin(heading)
    cos_psi = np.cos(heading)
    sin_phi = np.sin(latitude)
    cos_phi = np.cos(latitude)

    # Guard cos(gamma) against zero (vertical flight singularity)
    cos_gamma_safe = np.sign(cos_gamma) * max(abs(cos_gamma), _COS_GAMMA_FLOOR)

    # --- V-B-C Equations (2-44) through (2-49), atmosphere-relative velocity ---

    # (2-44) dV/dt
    dV_dt = (
        -drag_accel
        - g * sin_gamma
        + omega**2 * r * cos_phi * (
            sin_gamma * cos_phi
            - cos_gamma * sin_phi * sin_psi
        )
    )

    # (2-45) dgamma/dt
    dgamma_dt = (
        (drag_accel * ld_value * cos_sigma) / velocity
        - (g / velocity - velocity / r) * cos_gamma
        + 2.0 * omega * cos_phi * cos_psi
        + (omega**2 * r * cos_phi / velocity) * (
            cos_gamma * cos_phi
            + sin_gamma * sin_phi * sin_psi
        )
    )

    # (2-46) dpsi/dt
    dpsi_dt = (
        (drag_accel * ld_value * sin_sigma) / (velocity * cos_gamma_safe)
        - velocity * cos_gamma * cos_psi * np.tan(latitude) / r
        + 2.0 * omega * (
            np.tan(flight_path_angle) * cos_phi * sin_psi
            - sin_phi
        )
        - omega**2 * r * sin_phi * cos_phi * cos_psi / (velocity * cos_gamma_safe)
    )

    # (2-47) dh/dt (altitude)
    dh_dt = velocity * sin_gamma

    # (2-48) dphi/dt (latitude)
    dphi_dt = velocity * cos_gamma * sin_psi / r

    # (2-49) dtheta/dt (longitude)
    dtheta_dt = velocity * cos_gamma * cos_psi / (r * cos_phi)

    return [dV_dt, dgamma_dt, dpsi_dt, dh_dt, dphi_dt, dtheta_dt]

def _compute_downrange_crossrange(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    entry_latitude: float,
    entry_longitude: float,
    entry_heading: float,
    planet_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute great-circle downrange and crossrange from entry point.

    Downrange is the component of great-circle distance along the initial
    heading direction. Crossrange is the perpendicular component (signed:
    positive = to the right of initial heading).
    """
    # Great-circle distance from entry point to each (lat, lon)
    dlat = latitudes - entry_latitude
    dlon = longitudes - entry_longitude

    # Spherical trig: distance and bearing from entry to current position
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(entry_latitude) * np.cos(latitudes) * np.sin(dlon / 2.0) ** 2
    )
    # Clamp for numerical safety (arcsin/arctan2 arguments)
    a_clamped = np.clip(a, 0.0, 1.0)
    great_circle_distance = 2.0 * planet_radius * np.arcsin(np.sqrt(a_clamped))

    # Bearing (azimuth) from entry point, measured from north clockwise
    y_comp = np.sin(dlon) * np.cos(latitudes)
    x_comp = (
        np.cos(entry_latitude) * np.sin(latitudes)
        - np.sin(entry_latitude) * np.cos(latitudes) * np.cos(dlon)
    )
    bearing_from_entry = np.arctan2(y_comp, x_comp)

    # Project great-circle distance onto heading direction:
    # downrange = d * cos(bearing - heading)
    # crossrange = d * sin(bearing - heading) [positive = right of heading]
    # Convert V-B-C entry_heading (from east, CCW) to standard navigation
    # bearing (from north, CW) before taking the difference. The spherical-
    # trig bearing computation above is in the navigation convention
    # (bearing from north, clockwise), independent of our ψ convention.
    entry_bearing_from_north = np.pi / 2.0 - entry_heading
    angle_offset = bearing_from_entry - entry_bearing_from_north
    downrange = great_circle_distance * np.cos(angle_offset)
    crossrange = great_circle_distance * np.sin(angle_offset)

    return downrange, crossrange

def simulate(
    vehicle: Vehicle,
    initial_state: InitialState,
    planet: PlanetModel = EARTH_NON_ROTATING,
    max_time: float = 1000.0,
    dt_output: float = 0.1,
) -> TrajectoryResult:
    """Integrate the reentry trajectory from entry interface to termination.

    Terminates on ground impact (h = 0) or skip-out above the planet's
    maximum atmosphere altitude.
    """
    if initial_state.altitude < 0.0 or initial_state.altitude > planet.max_atmosphere_altitude:
        raise ValueError(
            f"Initial altitude {initial_state.altitude} m is outside the "
            f"valid atmosphere range [0, {planet.max_atmosphere_altitude}] m for {planet.name}."
        )
    if initial_state.velocity <= 0.0:
        raise ValueError(f"Initial velocity must be positive, got {initial_state.velocity} m/s.")

    # 6-element state vector: [V, gamma, psi, h, phi, theta]
    y0 = [
        initial_state.velocity,
        initial_state.flight_path_angle,
        initial_state.heading,
        initial_state.altitude,
        initial_state.latitude,
        initial_state.longitude,
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
    latitude = solution.y[4]
    longitude = solution.y[5]

    densities = np.array([planet.atmosphere(h).density for h in altitude])
    speeds_of_sound = np.array([planet.atmosphere(h).speed_of_sound for h in altitude])
    mach = velocity / speeds_of_sound
    dynamic_pressure = 0.5 * densities * velocity**2

    # Convenience: compute downrange (great-circle from entry) and crossrange
    # (perpendicular deviation from initial heading great circle)
    downrange, crossrange = _compute_downrange_crossrange(
        latitude, longitude,
        initial_state.latitude, initial_state.longitude, initial_state.heading,
        planet.radius,
    )

    return TrajectoryResult(
        time=time,
        altitude=altitude,
        velocity=velocity,
        flight_path_angle=flight_path_angle,
        heading=heading,
        latitude=latitude,
        longitude=longitude,
        downrange=downrange,
        crossrange=crossrange,
        mach=mach,
        dynamic_pressure=dynamic_pressure,
        density=densities,
        termination_reason=termination_reason,
    )
