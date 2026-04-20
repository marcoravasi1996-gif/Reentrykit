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
altitude-dependent composition and non-analytic temperature profiles. For
altitudes above 86 km, consider NRLMSISE-00 or the full US1976 upper-atmosphere
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
MAX_ALTITUDE = 86000.0

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

def us1976(altitude: float) -> AtmosphereState:
    """Compute atmospheric properties at a given geopotential altitude.

    Parameters
    ----------
    altitude : float
        Geopotential altitude above mean sea level [m]. Must be in the range
        [0, 86000].

    Returns
    -------
    AtmosphereState
        Temperature [K], pressure [Pa], density [kg/m^3], and speed of
        sound [m/s] at the requested altitude.

    Raises
    ------
    ValueError
        If altitude is outside the valid range [0, 86000] m.

    References
    ----------
    NOAA/NASA/USAF (1976). *U.S. Standard Atmosphere, 1976*.
    NASA-TM-X-74335.
    """
    if altitude < 0.0 or altitude > MAX_ALTITUDE:
        raise ValueError(
            f"Altitude {altitude} m is outside the valid range [0, {MAX_ALTITUDE}] m."
        )

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
