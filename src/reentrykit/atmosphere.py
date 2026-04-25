"""US Standard Atmosphere 1976 model, 0-86 km regime.

Implements the piecewise hydrostatic atmospheric model defined in NOAA/NASA/USAF
(1976), *U.S. Standard Atmosphere, 1976*, NASA-TM-X-74335, covering the well-mixed
regime from sea level to 86 km geopotential altitude.

The atmosphere is modeled as seven stacked layers, each with a base temperature
and constant lapse rate. Pressure is integrated hydrostatically from the sea-level
anchor of 101325 Pa. Density follows from the ideal gas law with constant molar
mass M = 0.0289644 kg/mol, and speed of sound from the standard perfect-gas
relation with gamma = 1.4.

Above 86 km the atmosphere enters the diffusive-separation regime with
altitude-dependent composition and non-analytic temperature profiles.
Faithful modeling requires NRLMSISE-00 or the full US1976 upper-atmosphere
formulation (both outside the scope of this module).

For convenience in reentry trajectory analysis — where vehicles may enter
above 86 km but only briefly pass through near-vacuum before reaching the
validated regime — this module provides a simple exponential extension
from 86 km to 500 km. The extension uses a fixed scale height (7 km)
anchored at the US1976 density at 86 km, continuous in value but not in
derivative. Temperature and speed of sound in the extension are held at
their 86 km values.

This is a numerical convenience for trajectory integration, not a
physically valid atmospheric model above ~120 km. Above 100-120 km the
real atmosphere undergoes diffusive separation with composition varying
strongly with altitude and solar activity (atomic oxygen dominant by
~150 km). For applications requiring physically meaningful upper-
atmosphere data, use NRLMSISE-00 or the full US1976 upper-atmosphere
formulation (both outside the scope of this module).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Physical constants from US Standard Atmosphere 1976
G0 = 9.80665  # standard gravitational acceleration [m/s^2]
M_AIR = 0.0289644  # molar mass of dry air [kg/mol]
R_STAR = 8.31432  # universal gas constant, US1976 value [J/(mol*K)]
GAMMA = 1.4  # ratio of specific heats for air [-]

# Derived constant used repeatedly in pressure calculations
GMR = G0 * M_AIR / R_STAR  # [K/m]

# US1976 atmospheric layers from 0 to 86 km geopotential altitude.
# Each row: [base_altitude (m), base_temperature (K), lapse_rate (K/m)]
_LAYER_BASE_ALTITUDES = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
_LAYER_BASE_TEMPERATURES = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65])
_LAYER_LAPSE_RATES = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])

# Sea-level pressure anchor [Pa]
_P0 = 101325.0

# Upper bound of validity for this implementation [m]
# Upper bound of the US1976 validated regime [m]
MAX_ALTITUDE = 86000.0

# Upper bound of the exponential extension [m]. Above this, callers must
# model the atmosphere themselves or use a higher-fidelity model.
MAX_EXTENDED_ALTITUDE = 500_000.0   # 500 km, numerical extension only (not physically valid above ~120 km)
# Scale height used for the 86-200 km exponential extension [m].
# Chosen as a round value close to Tetzman (2010, M.S. Thesis, University
# of Minnesota) H = 6.93 km, which itself is an altitude-averaged value
# calibrated for Apollo-class reentry trajectories.
_UPPER_SCALE_HEIGHT = 7000.0

def _compute_base_pressures() -> np.ndarray:
    """Compute pressure at each layer base by integrating upward from sea level."""
    n_layers = len(_LAYER_BASE_ALTITUDES)
    pressures = np.zeros(n_layers)
    pressures[0] = _P0

    for i in range(n_layers - 1):
        h_base = _LAYER_BASE_ALTITUDES[i]
        h_top = _LAYER_BASE_ALTITUDES[i + 1]
        t_base = _LAYER_BASE_TEMPERATURES[i]
        lapse = _LAYER_LAPSE_RATES[i]
        p_base = pressures[i]

        if lapse == 0.0:
            # Isothermal layer
            pressures[i + 1] = p_base * np.exp(-GMR * (h_top - h_base) / t_base)
        else:
            # Layer with temperature gradient
            t_top = t_base + lapse * (h_top - h_base)
            pressures[i + 1] = p_base * (t_top / t_base) ** (-GMR / lapse)

    return pressures


_LAYER_BASE_PRESSURES = _compute_base_pressures()

class AtmosphereState(NamedTuple):
    """Atmospheric state at a specific altitude."""

    temperature: float  # [K]
    pressure: float  # [Pa]
    density: float  # [kg/m^3]
    speed_of_sound: float  # [m/s]

def _us1976_below_ceiling(altitude: float) -> AtmosphereState:
    """Compute atmospheric properties strictly within the 0-86 km validated range.

    Internal helper used to both serve requests below the ceiling and to anchor
    the exponential extension above it.
    """
    # Find which layer the altitude falls in
    layer_index = int(np.searchsorted(_LAYER_BASE_ALTITUDES, altitude, side="right") - 1)

    # Look up the layer properties
    h_base = _LAYER_BASE_ALTITUDES[layer_index]
    t_base = _LAYER_BASE_TEMPERATURES[layer_index]
    lapse = _LAYER_LAPSE_RATES[layer_index]
    p_base = _LAYER_BASE_PRESSURES[layer_index]

    # Temperature: linear in altitude within the layer
    temperature = t_base + lapse * (altitude - h_base)

    # Pressure: two cases depending on lapse rate
    if lapse == 0.0:
        pressure = p_base * np.exp(-GMR * (altitude - h_base) / t_base)
    else:
        pressure = p_base * (temperature / t_base) ** (-GMR / lapse)

    # Density from ideal gas law
    density = pressure * M_AIR / (R_STAR * temperature)

    # Speed of sound for an ideal gas
    speed_of_sound = np.sqrt(GAMMA * R_STAR * temperature / M_AIR)

    return AtmosphereState(
        temperature=float(temperature),
        pressure=float(pressure),
        density=float(density),
        speed_of_sound=float(speed_of_sound),
    )


# State at the ceiling, pre-computed once at import time. Used as the anchor
# for the 86-200 km exponential extension.
_STATE_AT_CEILING = _us1976_below_ceiling(MAX_ALTITUDE)


def us1976(altitude: float) -> AtmosphereState:
    """Compute atmospheric properties at a given geopotential altitude.

    Below 86 km, returns US1976-standard values (temperature, pressure,
    density, speed of sound) from the layered hydrostatic model.

    Between 86 km and 500 km, returns an exponential extension anchored at
    the 86 km US1976 state: density decays as exp(-(h - 86 km) / H) with
    H = 7 km, pressure decays in the same ratio, while temperature and
    speed of sound are held at their 86 km values. This is a numerical
    convenience for trajectory integration; it is not a physically valid
    atmospheric model above ~120 km — see module docstring.

    Parameters
    ----------
    altitude : float
        Geopotential altitude above mean sea level [m]. Must be in the range
        [0, 500000].

    Returns
    -------
    AtmosphereState
        Temperature [K], pressure [Pa], density [kg/m^3], and speed of
        sound [m/s] at the requested altitude.

    Raises
    ------
    ValueError
        If altitude is outside the valid range [0, 500000] m.

    References
    ----------
    NOAA/NASA/USAF (1976). *U.S. Standard Atmosphere, 1976*.
    NASA-TM-X-74335.
    """
    # Tolerate tiny numerical overshoots from ODE integrators.
    _TOL = 100.0
    if altitude < -_TOL or altitude > MAX_EXTENDED_ALTITUDE + _TOL:
        raise ValueError(
            f"Altitude {altitude} m is outside the valid range "
            f"[0, {MAX_EXTENDED_ALTITUDE}] m."
        )
    altitude = float(np.clip(altitude, 0.0, MAX_EXTENDED_ALTITUDE))

    if altitude <= MAX_ALTITUDE:
        return _us1976_below_ceiling(altitude)

    # Exponential extension above 86 km, continuous in density/pressure with
    # US1976 at the ceiling. Temperature and speed of sound are frozen at the
    # ceiling value — an approximation that is acceptable because this regime
    # is near-vacuum for trajectory purposes.
    decay = np.exp(-(altitude - MAX_ALTITUDE) / _UPPER_SCALE_HEIGHT)
    density = _STATE_AT_CEILING.density * decay
    pressure = _STATE_AT_CEILING.pressure * decay
    return AtmosphereState(
        temperature=_STATE_AT_CEILING.temperature,
        pressure=float(pressure),
        density=float(density),
        speed_of_sound=_STATE_AT_CEILING.speed_of_sound,
    )