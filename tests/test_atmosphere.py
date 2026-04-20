"""Tests for the US Standard Atmosphere 1976 module.

Reference values are from the official US Standard Atmosphere 1976 tables
(NASA-TM-X-74335) as reproduced in multiple published sources. We validate
at representative altitudes spanning the full 0-86 km range, covering all
seven atmospheric layers.
"""

from __future__ import annotations

import pytest

from reentrykit.atmosphere import AtmosphereState, MAX_ALTITUDE, us1976

# Reference values from the US Standard Atmosphere 1976 tables.
# Each entry: (altitude_m, temperature_K, pressure_Pa, density_kg_m3)
# Values reproduced from NASA-TM-X-74335 and cross-checked against
# standard atmosphere calculators.
US1976_REFERENCE_DATA = [
    # (altitude, T, p, rho)
    (0.0, 288.150, 101325.0, 1.22500),
    (5000.0, 255.650, 54019.9, 0.73612),
    (11000.0, 216.650, 22632.1, 0.36391),
    (15000.0, 216.650, 12044.6, 0.19367),
    (20000.0, 216.650, 5474.89, 0.08803),
    (25000.0, 221.552, 2511.02, 0.03947),
    (32000.0, 228.650, 868.019, 0.01322),
    (40000.0, 251.050, 277.522, 0.003851),
    (47000.0, 270.650, 110.906, 0.001427),
    (50000.0, 270.650, 75.9448, 0.0009775),
    (71000.0, 214.650, 3.95642, 6.421e-5),
    (80000.0, 196.650, 0.88628, 1.570e-5),
]

# Tolerance for validation: 0.1% relative error
RTOL = 1e-3

def test_sea_level_conditions():
    """Sea level values match the textbook standard atmosphere."""
    state = us1976(0.0)

    assert state.temperature == pytest.approx(288.15, rel=RTOL)
    assert state.pressure == pytest.approx(101325.0, rel=RTOL)
    assert state.density == pytest.approx(1.225, rel=RTOL)
    assert state.speed_of_sound == pytest.approx(340.3, rel=RTOL)

@pytest.mark.parametrize(("altitude", "temperature", "pressure", "density"), US1976_REFERENCE_DATA)
def test_matches_reference_table(
    altitude: float,
    temperature: float,
    pressure: float,
    density: float,
):
    """Model output matches the published US1976 reference table."""
    state = us1976(altitude)

    assert state.temperature == pytest.approx(temperature, rel=RTOL)
    assert state.pressure == pytest.approx(pressure, rel=RTOL)
    assert state.density == pytest.approx(density, rel=RTOL)
