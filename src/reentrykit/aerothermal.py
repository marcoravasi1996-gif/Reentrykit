"""Aerothermal heating for atmospheric entry vehicles.

Stagnation-point convective heating using the Sutton-Graves correlation:

    q_cw = K · sqrt(rho / R_N) · V^3

where:
    q_cw  = stagnation-point convective heat flux [W/m²]
    K     = Sutton-Graves constant (atmosphere-specific) [W·s³·kg^-0.5·m^-1.5]
    rho   = freestream density [kg/m³]
    R_N   = effective nose radius of the blunt body [m]
    V     = freestream velocity [m/s]

This is a first-order analytical correlation for convective heating at the
stagnation point of a blunt body. It does not account for radiative heating
(significant above ~10 km/s), boundary-layer transition, or heating at
locations other than the stagnation point.

Reference:
    Sutton, K., and Graves, R. A. (1971). "A General Stagnation-Point
    Convective-Heating Equation for Arbitrary Gas Mixtures."
    NASA Technical Report R-376.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from reentrykit.trajectory import TrajectoryResult


# Sutton-Graves constant for Earth air (SI units).
# Value from the original Sutton-Graves 1971 reference.
SUTTON_GRAVES_K_EARTH = 1.7415e-4  # [W · s^3 · kg^-0.5 · m^-1.5]


class HeatingResult(NamedTuple):
    """Stagnation-point heating history for a trajectory.

    Attributes
    ----------
    time : np.ndarray
        Time array [s], same as the trajectory's time array.
    heat_flux : np.ndarray
        Stagnation-point convective heat flux [W/m²] at each time step.
    heat_load : np.ndarray
        Integrated heat load [J/m²] at each time step. heat_load[-1] is
        the total heat load delivered to the stagnation point.
    peak_heat_flux : float
        Maximum stagnation-point heat flux over the trajectory [W/m²].
    peak_heat_flux_time : float
        Time [s] at which peak heat flux occurs.
    peak_heat_flux_altitude : float
        Altitude [m] at which peak heat flux occurs.
    total_heat_load : float
        Total integrated heat load over the full trajectory [J/m²].
    """

    time: np.ndarray
    heat_flux: np.ndarray
    heat_load: np.ndarray
    peak_heat_flux: float
    peak_heat_flux_time: float
    peak_heat_flux_altitude: float
    total_heat_load: float


def sutton_graves_heat_flux(
    density: float,
    velocity: float,
    nose_radius: float,
    constant: float = SUTTON_GRAVES_K_EARTH,
) -> float:
    """Stagnation-point convective heat flux via Sutton-Graves correlation.

    Parameters
    ----------
    density : float
        Freestream density [kg/m³]. Must be non-negative.
    velocity : float
        Freestream velocity magnitude [m/s]. Must be non-negative.
    nose_radius : float
        Effective nose radius of the blunt body [m]. Must be positive.
    constant : float, optional
        Sutton-Graves constant for the atmosphere. Default is the Earth
        air value (SUTTON_GRAVES_K_EARTH = 1.7415e-4).

    Returns
    -------
    float
        Stagnation-point convective heat flux [W/m²].

    Raises
    ------
    ValueError
        If nose_radius is not positive, or if density/velocity are negative.
    """
    if nose_radius <= 0.0:
        raise ValueError(
            f"nose_radius must be positive, got {nose_radius}"
        )
    if density < 0.0:
        raise ValueError(f"density must be non-negative, got {density}")
    if velocity < 0.0:
        raise ValueError(f"velocity must be non-negative, got {velocity}")

    return constant * np.sqrt(density / nose_radius) * velocity ** 3


def heating_history(
    result: TrajectoryResult,
    nose_radius: float,
    constant: float = SUTTON_GRAVES_K_EARTH,
) -> HeatingResult:
    """Compute stagnation-point heating history from a trajectory result.

    Parameters
    ----------
    result : TrajectoryResult
        Simulator output containing time, density, and velocity arrays.
    nose_radius : float
        Effective nose radius of the blunt body [m].
    constant : float, optional
        Sutton-Graves constant. Default is Earth air.

    Returns
    -------
    HeatingResult
        Named tuple containing time, heat_flux, heat_load arrays and
        summary peak/total values.
    """
    if nose_radius <= 0.0:
        raise ValueError(
            f"nose_radius must be positive, got {nose_radius}"
        )

    # Heat flux at each time step (vectorized Sutton-Graves)
    q_dot = constant * np.sqrt(result.density / nose_radius) * result.velocity ** 3

    # Integrated heat load (trapezoidal rule)
    q_integrated = np.zeros_like(q_dot)
    if len(q_dot) > 1:
        q_integrated[1:] = np.cumsum(
            0.5 * (q_dot[:-1] + q_dot[1:]) * np.diff(result.time)
        )

    # Summary statistics
    i_peak = int(q_dot.argmax())

    return HeatingResult(
        time=result.time,
        heat_flux=q_dot,
        heat_load=q_integrated,
        peak_heat_flux=float(q_dot[i_peak]),
        peak_heat_flux_time=float(result.time[i_peak]),
        peak_heat_flux_altitude=float(result.altitude[i_peak]),
        total_heat_load=float(q_integrated[-1]),
    )