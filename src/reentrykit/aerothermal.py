"""Aerothermal heating for atmospheric entry vehicles.

This module provides stagnation-point heating calculations for blunt-body
entry vehicles. Two heating mechanisms are implemented:

1. **Convective heating** via the Sutton-Graves correlation (Sutton & Graves
   1971). Valid across all entry velocities. Typical accuracy ~10-15%.

2. **Radiative heating** via a Tauber-Sutton-style correlation (Tauber &
   Sutton 1991) for Earth atmosphere. Significant above ~10 km/s; dominant
   above ~12 km/s for capsules with R_N > 0.3 m. Typical accuracy ~30%.

The Tauber-Sutton correlation has explicit validity ranges:
  - 10,000 m/s ≤ V ≤ 16,000 m/s
  - 6.66e-5 kg/m³ ≤ ρ ≤ 6.31e-4 kg/m³

Outside these ranges, the radiative term returns 0.0 (conservative for
trajectory integration; surrounding regions contribute negligibly to peak
heating in practice).

Usage
-----
Point calculations:

    q_conv = sutton_graves_heat_flux(rho, V, R_N)
    q_rad  = tauber_sutton_heat_flux(rho, V, R_N)

Trajectory-level:

    heat = heating_history(trajectory, nose_radius=0.5)
    print(heat.peak_total_flux)
    print(heat.peak_convective_flux)
    print(heat.peak_radiative_flux)

References
----------
Sutton, K., & Graves, R. A. (1971). "A General Stagnation-Point
Convective-Heating Equation for Arbitrary Gas Mixtures." NASA TR R-376.

Tauber, M. E., & Sutton, K. (1991). "Stagnation-Point Radiative Heating
Relations for Earth and Mars Entries." Journal of Spacecraft and
Rockets, 28(1), 40-42.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from reentrykit.trajectory import TrajectoryResult


# =========================================================================
# Convective heating (Sutton-Graves)
# =========================================================================

# Sutton-Graves constant for Earth air, SI units.
SUTTON_GRAVES_K_EARTH = 1.7415e-4  # [W · s^3 · kg^-0.5 · m^-1.5]


# =========================================================================
# Radiative heating (Tauber-Sutton, Earth)
# =========================================================================

# Validity range from source.
TAUBER_SUTTON_V_MIN = 10_000.0   # [m/s]
TAUBER_SUTTON_V_MAX = 16_000.0   # [m/s]
TAUBER_SUTTON_RHO_MIN = 6.66e-5  # [kg/m^3]
TAUBER_SUTTON_RHO_MAX = 6.31e-4  # [kg/m^3]

# Tabulated f(V) for Earth atmosphere from Tauber-Sutton 1991.
# Velocity in m/s; f(V) dimensionless within the formula's unit system.
_TAUBER_SUTTON_V_TABLE = np.array([
    10_000.0, 10_250.0, 10_500.0, 10_750.0, 11_000.0, 11_500.0,
    12_000.0, 12_500.0, 13_000.0, 13_500.0, 14_000.0, 14_500.0,
    15_000.0, 15_500.0, 16_000.0,
])
_TAUBER_SUTTON_F_TABLE = np.array([
      35.0,    55.0,    81.0,   115.0,   151.0,   238.0,
     359.0,   495.0,   660.0,   850.0,  1065.0,  1313.0,
    1550.0,  1780.0,  2040.0,
])

# Coefficients in the formula
#   q_rad [W/cm^2] = C * R_N^a * rho^b * f(V)
# where
#   a = 1.072e6 * V^(-1.88) * rho^(-0.325)
# and a is clamped depending on R_N (see _tauber_sutton_a_exponent).
_TAUBER_SUTTON_C = 4.736e4   # in W/cm^2 with rho in kg/m^3, R_N in m
_TAUBER_SUTTON_B = 1.22      # density exponent

# Conversion: formula yields W/cm^2; module convention is SI (W/m^2).
_WCM2_TO_WM2 = 1.0e4


# =========================================================================
# Heat-flux point functions
# =========================================================================


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
        Freestream density [kg/m^3]. Must be non-negative.
    velocity : float
        Freestream velocity magnitude [m/s]. Must be non-negative.
    nose_radius : float
        Effective nose radius [m]. Must be positive.
    constant : float, optional
        Sutton-Graves constant for the atmosphere.

    Returns
    -------
    float
        Convective heat flux [W/m^2].
    """
    if nose_radius <= 0.0:
        raise ValueError(f"nose_radius must be positive, got {nose_radius}")
    if density < 0.0:
        raise ValueError(f"density must be non-negative, got {density}")
    if velocity < 0.0:
        raise ValueError(f"velocity must be non-negative, got {velocity}")

    return constant * np.sqrt(density / nose_radius) * velocity ** 3


def _tauber_sutton_a_exponent(
    velocity: float, density: float, nose_radius: float,
) -> float:
    """Compute the R_N exponent `a` in Tauber-Sutton with R_N-dependent clamp.

    a = 1.072e6 * V^(-1.88) * rho^(-0.325)
    Clamped to:
      - a (no clamp)            for R_N < 1 m
      - min(a, 0.6)             for 1 <= R_N < 2 m
      - min(a, 0.5)             for 2 <= R_N <= 3 m
    Raises ValueError if R_N > 3 m (outside source validity).
    """
    if nose_radius > 3.0:
        raise ValueError(
            f"nose_radius {nose_radius} m exceeds Tauber-Sutton validity "
            f"(R_N <= 3 m)"
        )

    a = 1.072e6 * velocity ** (-1.88) * density ** (-0.325)

    if 1.0 <= nose_radius < 2.0:
        a = min(a, 0.6)
    elif 2.0 <= nose_radius <= 3.0:
        a = min(a, 0.5)

    return a


def tauber_sutton_heat_flux(
    density: float,
    velocity: float,
    nose_radius: float,
) -> float:
    """Stagnation-point radiative heat flux via Tauber-Sutton (Earth).

    q_rad [W/cm^2] = C * R_N^a * rho^1.22 * f(V),
    where a is itself a function of V and rho (see source).

    Returns 0.0 if outside the validity range:
      10_000 <= V <= 16_000 m/s
      6.66e-5 <= rho <= 6.31e-4 kg/m^3

    Returned value is converted to SI: W/m^2.

    Parameters
    ----------
    density : float
        Freestream density [kg/m^3]. Must be non-negative.
    velocity : float
        Freestream velocity magnitude [m/s]. Must be non-negative.
    nose_radius : float
        Effective nose radius [m]. Must be positive and <= 3 m.

    Returns
    -------
    float
        Radiative heat flux [W/m^2].
    """
    if nose_radius <= 0.0:
        raise ValueError(f"nose_radius must be positive, got {nose_radius}")
    if density < 0.0:
        raise ValueError(f"density must be non-negative, got {density}")
    if velocity < 0.0:
        raise ValueError(f"velocity must be non-negative, got {velocity}")

    # Outside V validity -> no radiation contribution
    if velocity < TAUBER_SUTTON_V_MIN or velocity > TAUBER_SUTTON_V_MAX:
        return 0.0

    # Outside rho validity -> no radiation contribution
    if density < TAUBER_SUTTON_RHO_MIN or density > TAUBER_SUTTON_RHO_MAX:
        return 0.0

    a = _tauber_sutton_a_exponent(velocity, density, nose_radius)

    f_v = float(np.interp(
        velocity, _TAUBER_SUTTON_V_TABLE, _TAUBER_SUTTON_F_TABLE
    ))

    q_wcm2 = (
        _TAUBER_SUTTON_C
        * (nose_radius ** a)
        * (density ** _TAUBER_SUTTON_B)
        * f_v
    )

    return q_wcm2 * _WCM2_TO_WM2


# =========================================================================
# Trajectory-level heating history
# =========================================================================


class HeatingResult(NamedTuple):
    """Stagnation-point heating history for a trajectory.

    Includes convective (Sutton-Graves) and radiative (Tauber-Sutton)
    components plus their sum.

    Attributes
    ----------
    time : np.ndarray
        Time [s], same as trajectory time array.
    convective_flux : np.ndarray
        Stagnation-point convective heat flux [W/m^2].
    radiative_flux : np.ndarray
        Stagnation-point radiative heat flux [W/m^2].
    total_flux : np.ndarray
        convective_flux + radiative_flux [W/m^2].
    convective_load : np.ndarray
        Cumulative convective heat load [J/m^2].
    radiative_load : np.ndarray
        Cumulative radiative heat load [J/m^2].
    total_load : np.ndarray
        Cumulative total heat load [J/m^2].
    peak_convective_flux : float
        Peak convective flux over the trajectory [W/m^2].
    peak_radiative_flux : float
        Peak radiative flux over the trajectory [W/m^2].
    peak_total_flux : float
        Peak total flux over the trajectory [W/m^2].
    peak_total_flux_time : float
        Time at which peak total flux occurs [s].
    peak_total_flux_altitude : float
        Altitude at which peak total flux occurs [m].
    total_convective_load : float
        Integrated convective heat load over full trajectory [J/m^2].
    total_radiative_load : float
        Integrated radiative heat load over full trajectory [J/m^2].
    total_integrated_load : float
        Integrated total heat load over full trajectory [J/m^2].

    Backward-compatibility attributes (for code expecting the old API):
    heat_flux : alias for convective_flux
    heat_load : alias for convective_load
    peak_heat_flux : alias for peak_convective_flux
    peak_heat_flux_time : alias for peak_total_flux_time at peak conv loc
    peak_heat_flux_altitude : alias for peak_convective altitude
    total_heat_load : alias for total_convective_load
    """

    time: np.ndarray
    convective_flux: np.ndarray
    radiative_flux: np.ndarray
    total_flux: np.ndarray
    convective_load: np.ndarray
    radiative_load: np.ndarray
    total_load: np.ndarray
    peak_convective_flux: float
    peak_radiative_flux: float
    peak_total_flux: float
    peak_total_flux_time: float
    peak_total_flux_altitude: float
    total_convective_load: float
    total_radiative_load: float
    total_integrated_load: float

    # ----- Backward-compatibility aliases -----
    @property
    def heat_flux(self) -> np.ndarray:
        return self.convective_flux

    @property
    def heat_load(self) -> np.ndarray:
        return self.convective_load

    @property
    def peak_heat_flux(self) -> float:
        return self.peak_convective_flux

    @property
    def peak_heat_flux_time(self) -> float:
        # Time of peak convective flux (the original meaning)
        i = int(np.argmax(self.convective_flux))
        return float(self.time[i])

    @property
    def peak_heat_flux_altitude(self) -> float:
        # Returns NaN: altitude not stored in this NamedTuple. Preserve
        # backward-compat callers that don't use altitude (most don't).
        return float("nan")

    @property
    def total_heat_load(self) -> float:
        return self.total_convective_load


def _integrate(flux: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration of flux over time."""
    integrated = np.zeros_like(flux)
    if len(flux) > 1:
        integrated[1:] = np.cumsum(
            0.5 * (flux[:-1] + flux[1:]) * np.diff(time)
        )
    return integrated


def heating_history(
    result: TrajectoryResult,
    nose_radius: float,
    sutton_graves_constant: float = SUTTON_GRAVES_K_EARTH,
) -> HeatingResult:
    """Compute full stagnation-point heating history from a trajectory.

    Includes convective (Sutton-Graves) and radiative (Tauber-Sutton)
    components.

    Parameters
    ----------
    result : TrajectoryResult
        Simulator output with time, density, altitude, and velocity.
    nose_radius : float
        Effective nose radius [m].
    sutton_graves_constant : float, optional
        Convective heating constant.

    Returns
    -------
    HeatingResult
    """
    if nose_radius <= 0.0:
        raise ValueError(f"nose_radius must be positive, got {nose_radius}")

    # --- Convective (vectorized) -----------------------------------------
    q_conv = (
        sutton_graves_constant
        * np.sqrt(result.density / nose_radius)
        * result.velocity ** 3
    )

    # --- Radiative (vectorized within validity ranges) ------------------
    in_v_range = (
        (result.velocity >= TAUBER_SUTTON_V_MIN)
        & (result.velocity <= TAUBER_SUTTON_V_MAX)
    )
    in_rho_range = (
        (result.density >= TAUBER_SUTTON_RHO_MIN)
        & (result.density <= TAUBER_SUTTON_RHO_MAX)
    )
    in_range = in_v_range & in_rho_range

    q_rad = np.zeros_like(q_conv)
    if in_range.any():
        # Compute a element-wise where in-range
        a_exp = (
            1.072e6
            * result.velocity[in_range] ** (-1.88)
            * result.density[in_range] ** (-0.325)
        )
        if 1.0 <= nose_radius < 2.0:
            a_exp = np.minimum(a_exp, 0.6)
        elif 2.0 <= nose_radius <= 3.0:
            a_exp = np.minimum(a_exp, 0.5)
        elif nose_radius > 3.0:
            raise ValueError(
                f"nose_radius {nose_radius} m exceeds Tauber-Sutton "
                f"validity (R_N <= 3 m)"
            )

        f_v = np.interp(
            result.velocity[in_range],
            _TAUBER_SUTTON_V_TABLE,
            _TAUBER_SUTTON_F_TABLE,
        )

        q_rad_in_range_wcm2 = (
            _TAUBER_SUTTON_C
            * (nose_radius ** a_exp)
            * (result.density[in_range] ** _TAUBER_SUTTON_B)
            * f_v
        )
        q_rad[in_range] = q_rad_in_range_wcm2 * _WCM2_TO_WM2

    q_total = q_conv + q_rad

    # --- Cumulative loads ------------------------------------------------
    q_conv_load = _integrate(q_conv, result.time)
    q_rad_load = _integrate(q_rad, result.time)
    q_total_load = _integrate(q_total, result.time)

    # --- Summary statistics ---------------------------------------------
    i_peak = int(q_total.argmax())

    return HeatingResult(
        time=result.time,
        convective_flux=q_conv,
        radiative_flux=q_rad,
        total_flux=q_total,
        convective_load=q_conv_load,
        radiative_load=q_rad_load,
        total_load=q_total_load,
        peak_convective_flux=float(q_conv.max()),
        peak_radiative_flux=float(q_rad.max()),
        peak_total_flux=float(q_total[i_peak]),
        peak_total_flux_time=float(result.time[i_peak]),
        peak_total_flux_altitude=float(result.altitude[i_peak]),
        total_convective_load=float(q_conv_load[-1]),
        total_radiative_load=float(q_rad_load[-1]),
        total_integrated_load=float(q_total_load[-1]),
    )