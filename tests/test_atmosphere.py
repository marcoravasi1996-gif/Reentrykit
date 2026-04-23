"""Tests for the US Standard Atmosphere 1976 module.

Reference values are from the official US Standard Atmosphere 1976 tables
(NASA-TM-X-74335) as reproduced in multiple published sources. We validate
at representative altitudes spanning the full 0-86 km range, covering all
seven atmospheric layers.
"""

from __future__ import annotations
import numpy as np
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
# ---------------------------------------------------------------------------
# Exponential extension above 86 km
# ---------------------------------------------------------------------------


def test_extension_continuous_at_ceiling():
    """Density and pressure are continuous at the 86 km transition."""
    state_below = us1976(MAX_ALTITUDE - 1.0)
    state_above = us1976(MAX_ALTITUDE + 1.0)

    np.testing.assert_allclose(state_above.density, state_below.density, rtol=1e-3)
    np.testing.assert_allclose(state_above.pressure, state_below.pressure, rtol=1e-3)


def test_extension_density_decays_exponentially():
    """Density in the extension region follows exp(-h/H) with H = 7 km."""
    rho_86 = us1976(MAX_ALTITUDE).density
    rho_93 = us1976(MAX_ALTITUDE + 7000.0).density

    ratio = rho_86 / rho_93
    assert abs(ratio - np.e) / np.e < 0.01, (
        f"Density drop over one scale height: {ratio:.3f}, expected {np.e:.3f}"
    )


def test_extension_temperature_frozen():
    """Temperature is held at the 86 km value throughout the extension."""
    t_86 = us1976(MAX_ALTITUDE).temperature
    assert us1976(100_000.0).temperature == t_86
    assert us1976(150_000.0).temperature == t_86
    assert us1976(200_000.0).temperature == t_86


def test_extension_valid_at_200km_upper_bound():
    """us1976 accepts altitudes up to the extended ceiling."""
    state = us1976(200_000.0)
    assert state.density > 0.0
    assert state.pressure > 0.0


def test_extension_rejects_above_extended_ceiling():
    """us1976 rejects altitudes above 200 km."""
    with pytest.raises(ValueError, match="outside the valid range"):
        us1976(600_000.0)


def test_extension_density_much_lower_than_at_ceiling():
    """At 200 km, density is dramatically lower than at 86 km (vacuum regime)."""
    rho_86 = us1976(MAX_ALTITUDE).density
    rho_200 = us1976(200_000.0).density

    ratio = rho_200 / rho_86
    assert ratio < 1e-6, f"Density ratio 200km/86km = {ratio:.2e}, expected <1e-6"
    assert ratio > 0.0, "Density must remain strictly positive"