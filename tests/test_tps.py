"""Unit tests for the TPS sizing module."""

from __future__ import annotations

import numpy as np
import pytest

from reentrykit.aerothermal import HeatingResult, heating_history
from reentrykit.planet import EARTH_NON_ROTATING
from reentrykit.tps import (
    AVCOAT,
    CARBON_PHENOLIC,
    PICA,
    TPSMaterial,
    TPSSizingResult,
    size_tps,
    transient_bondline_temperature,
)
from reentrykit.trajectory import InitialState, Vehicle, simulate


# ---------------------------------------------------------------------------
# Material definitions
# ---------------------------------------------------------------------------


def test_pica_is_low_density_ablator():
    """PICA should have density around 270 kg/m³."""
    assert 200 < PICA.density < 350
    assert PICA.max_bondline_temperature == 523.0


def test_avcoat_is_denser_than_pica():
    """AVCOAT is a legacy denser ablator than PICA."""
    assert AVCOAT.density > PICA.density


def test_material_properties_are_positive():
    """All thermal properties must be positive for physical materials."""
    for mat in [PICA, AVCOAT]:
        assert mat.density > 0
        assert mat.thermal_conductivity > 0
        assert mat.specific_heat > 0
        assert mat.max_surface_temperature > 273
        assert mat.max_bondline_temperature > 273


# ---------------------------------------------------------------------------
# Transient heat solver: invariants
# ---------------------------------------------------------------------------


def test_zero_heat_flux_gives_no_heating():
    """With zero applied flux, bondline should stay at initial temperature."""
    t = np.linspace(0, 100, 101)
    q = np.zeros_like(t)

    _, T_surf, T_bl = transient_bondline_temperature(
        thickness=0.050, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=300.0, n_nodes=30,
    )

    # All temperatures should stay near 300 K
    np.testing.assert_allclose(T_bl, 300.0, atol=0.01)
    np.testing.assert_allclose(T_surf, 300.0, atol=0.01)


def test_thicker_slab_gives_cooler_bondline():
    """Doubling thickness should reduce peak bondline temperature."""
    t = np.linspace(0, 150, 301)
    # Constant heat flux profile for simplicity
    q = np.where((t > 20) & (t < 80), 1e6, 0.0)   # 1 MW/m² pulse

    _, _, T_bl_thin = transient_bondline_temperature(
        thickness=0.010, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=30,
    )
    _, _, T_bl_thick = transient_bondline_temperature(
        thickness=0.030, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=30,
    )

    assert T_bl_thin.max() > T_bl_thick.max()


def test_heat_flux_causes_surface_temperature_rise():
    """Positive heat flux should raise front-face temperature."""
    t = np.linspace(0, 100, 201)
    q = np.where((t > 20) & (t < 50), 2e6, 0.0)

    _, T_surf, _ = transient_bondline_temperature(
        thickness=0.030, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=30,
    )

    assert T_surf.max() > 293.0
    assert T_surf.max() > 400.0   # some meaningful heating


def test_transient_solver_rejects_zero_thickness():
    t = np.linspace(0, 10, 11)
    q = np.zeros_like(t)
    with pytest.raises(ValueError, match="thickness"):
        transient_bondline_temperature(
            thickness=0.0, heat_flux_time=t, heat_flux=q, material=PICA,
        )


def test_transient_solver_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="same length"):
        transient_bondline_temperature(
            thickness=0.01,
            heat_flux_time=np.array([0, 1, 2]),
            heat_flux=np.array([0, 1]),
            material=PICA,
        )


# ---------------------------------------------------------------------------
# Sizing procedure: Stardust validation
# ---------------------------------------------------------------------------


@pytest.fixture
def stardust_heating():
    """Compute heating history for a Stardust-class trajectory."""
    vehicle = Vehicle.from_mass_area_cd(
        mass=45.8,
        reference_area=np.pi * (0.811 / 2) ** 2,
        drag_coefficient=1.0,
        lift_to_drag_ratio=0.0,
        nose_radius=0.2202,
    )
    state = InitialState(
        altitude=125_000.0,
        velocity=12_300.0,
        flight_path_angle=np.deg2rad(-8.2),
        heading=np.deg2rad(15.0),
        latitude=np.deg2rad(41.0),
        longitude=np.deg2rad(-128.0),
    )
    result = simulate(vehicle, state, planet=EARTH_NON_ROTATING,
                     max_time=500.0, dt_output=0.1)
    return heating_history(result, nose_radius=0.2202)


def test_size_tps_returns_sizing_result(stardust_heating):
    result = size_tps(stardust_heating, PICA)
    assert isinstance(result, TPSSizingResult)
    assert result.required_thickness > 0
    assert result.peak_bondline_temperature <= PICA.max_bondline_temperature + 1.0


def test_stardust_pica_thickness_order_of_magnitude(stardust_heating):
    """Stardust PICA thickness (Level 1 no-ablation sizing).

    Actual Stardust heatshield: 58 mm PICA. Our Level 1 sizing computes
    the minimum thickness for thermal insulation assuming no surface mass
    loss. Real ablator thickness is typically 30-50% greater to account
    for recession and design margin.

    Expected Level 1 result: 20-50 mm range (bracketing our insulation-
    only prediction; below the 58 mm flown thickness because ablation
    is unmodeled).
    """
    result = size_tps(stardust_heating, PICA)

    assert 0.020 < result.required_thickness < 0.080, (
        f"Stardust PICA thickness {result.required_thickness*1000:.1f} mm "
        f"outside 20-80 mm expected Level 1 range"
    )


def test_size_tps_rejects_too_thin_minimum(stardust_heating):
    """If min thickness already keeps bondline cool, sizing is invalid."""
    # Extremely large minimum that already passes
    with pytest.raises(ValueError, match="Minimum thickness"):
        size_tps(stardust_heating, PICA,
                thickness_min=0.500, thickness_max=1.000)


def test_size_tps_rejects_too_thin_maximum(stardust_heating):
    """If max thickness is insufficient, sizing fails."""
    with pytest.raises(ValueError, match="Maximum thickness"):
        size_tps(stardust_heating, PICA,
                thickness_min=0.001, thickness_max=0.002)
        
def test_carbon_phenolic_is_denser_than_avcoat():
    """Carbon-phenolic should be denser than AVCOAT (legacy heavy ablator hierarchy)."""
    assert CARBON_PHENOLIC.density > AVCOAT.density


def test_carbon_phenolic_in_solver():
    """The solver runs cleanly with carbon-phenolic material."""
    t = np.linspace(0, 100, 201)
    q = np.where((t > 20) & (t < 50), 2e6, 0.0)

    _, T_surf, T_bl = transient_bondline_temperature(
        thickness=0.030, heat_flux_time=t, heat_flux=q,
        material=CARBON_PHENOLIC,
        initial_temperature=293.0, n_nodes=30,
    )

    assert T_surf.max() > 293.0
    assert T_bl.max() < CARBON_PHENOLIC.max_bondline_temperature

def test_return_full_field_returns_temperature_field():
    """return_full_field=True returns spatial T(x,t) instead of surface/bondline."""
    t = np.linspace(0, 100, 101)
    q = np.where((t > 20) & (t < 50), 1e6, 0.0)

    n_nodes = 30
    time, x, T_field = transient_bondline_temperature(
        thickness=0.030, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=n_nodes,
        return_full_field=True,
    )

    # Shape checks
    assert T_field.shape == (len(time), n_nodes)
    assert x.shape == (n_nodes,)
    assert x[0] == 0.0
    assert x[-1] == pytest.approx(0.030)

    # Front face hotter than back face during heating
    i_during_heat = int(len(time) * 0.4)  # roughly during heating
    assert T_field[i_during_heat, 0] > T_field[i_during_heat, -1]


def test_return_full_field_consistency_with_default():
    """T_field[:, 0] equals surface_temp from default mode; T_field[:, -1] equals bondline_temp."""
    t = np.linspace(0, 100, 201)
    q = np.where((t > 20) & (t < 50), 2e6, 0.0)

    n_nodes = 40
    # Default mode
    time1, T_surf, T_bl = transient_bondline_temperature(
        thickness=0.050, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=n_nodes,
        return_full_field=False,
    )

    # Full-field mode
    time2, x, T_field = transient_bondline_temperature(
        thickness=0.050, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=n_nodes,
        return_full_field=True,
    )

    np.testing.assert_allclose(time1, time2, rtol=1e-12)
    np.testing.assert_allclose(T_field[:, 0], T_surf, rtol=1e-12)
    np.testing.assert_allclose(T_field[:, -1], T_bl, rtol=1e-12)

def test_lower_emissivity_gives_hotter_surface():
    """Lower surface emissivity reduces reradiation, raising surface temperature."""
    t = np.linspace(0, 100, 201)
    q = np.where((t > 20) & (t < 50), 5e6, 0.0)  # high heat flux

    # High emissivity (efficient radiator)
    _, T_surf_hi, _ = transient_bondline_temperature(
        thickness=0.030, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=30,
        surface_emissivity=0.95,
    )

    # Low emissivity (poor radiator)
    _, T_surf_lo, _ = transient_bondline_temperature(
        thickness=0.030, heat_flux_time=t, heat_flux=q, material=PICA,
        initial_temperature=293.0, n_nodes=30,
        surface_emissivity=0.30,
    )

    assert T_surf_lo.max() > T_surf_hi.max()


def test_emissivity_outside_valid_range_rejected():
    """Emissivity outside (0, 1] should raise."""
    t = np.linspace(0, 10, 11)
    q = np.zeros_like(t)

    with pytest.raises(ValueError, match="emissivity"):
        transient_bondline_temperature(
            thickness=0.01, heat_flux_time=t, heat_flux=q, material=PICA,
            surface_emissivity=0.0,
        )
    with pytest.raises(ValueError, match="emissivity"):
        transient_bondline_temperature(
            thickness=0.01, heat_flux_time=t, heat_flux=q, material=PICA,
            surface_emissivity=1.5,
        )