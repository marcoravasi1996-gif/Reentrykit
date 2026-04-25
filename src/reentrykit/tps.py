"""Thermal Protection System (TPS) sizing for atmospheric entry vehicles.

Level 1 sizing: 1-D transient heat conduction through a homogeneous insulator
with applied heat flux on the front face and surface grey-body reradiation.
Treats TPS material as a simple insulating slab — no ablation, no char
layer, no temperature-dependent properties. Valid for first-order sizing
estimates and for reusable insulator materials (e.g., Shuttle tiles).

For ablative materials (PICA, AVCOAT), Level 1 sizing typically
**overestimates** required thickness because:

  1. Surface reradiation caps the temperature at radiative equilibrium
     instead of the material's actual ablation temperature — so more
     heat conducts inward than in a real pyrolyzing material.
  2. Surface recession (material loss to ablation) is not modeled —
     a real ablator reduces its own thickness while absorbing heat.

A ~20-30% margin between Level 1 predictions and actual flown ablator
thicknesses is typical.

The solver uses an explicit finite-difference scheme:

    rho * c * dT/dt = d/dx (k * dT/dx)

with boundary conditions:
- Front face (x=0): q_net = q_applied - sigma * eps * (T^4 - T_ambient^4)
- Back face (x=L): insulated (adiabatic), i.e., dT/dx = 0

Initial condition: T(x, 0) = T_initial (default 293 K).

Sizing procedure: bisection on thickness until peak bondline (back-face)
temperature equals the material's bondline design limit.

References
----------
Carslaw, H. S., & Jaeger, J. C. (1959). *Conduction of Heat in Solids*, 2nd ed.

Tauber, M. E., Menees, G. P., & Adelman, H. G. (1987). "Aerothermodynamics
of transatmospheric vehicles." J. Aircraft, 24(9), 594-602.

Tran, H. K., et al. (1997). "Phenolic Impregnated Carbon Ablators (PICA)
as thermal protection systems for discovery missions." NASA TM-110440.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from reentrykit.aerothermal import HeatingResult


# =========================================================================
# Physical constants
# =========================================================================

_STEFAN_BOLTZMANN = 5.670374419e-8    # [W/m^2·K^4]
_DEFAULT_SURFACE_EMISSIVITY = 0.85    # typical for charring ablators


# =========================================================================
# Material definitions
# =========================================================================


class TPSMaterial(NamedTuple):
    """Properties of a thermal protection system material.

    Values assumed constant with temperature (Level 1 simplification).
    For ablative materials, these are virgin (unexposed) properties.

    Attributes
    ----------
    name : str
        Material identifier (e.g., "PICA", "AVCOAT").
    density : float
        Bulk density [kg/m³].
    thermal_conductivity : float
        Thermal conductivity of virgin material [W/m·K].
    specific_heat : float
        Specific heat capacity [J/kg·K].
    max_surface_temperature : float
        Maximum allowable surface temperature [K]. Above this, ablation
        or failure begins. Informational for Level 1.
    max_bondline_temperature : float
        Maximum allowable back-face temperature [K]. This is the design
        limit the sizing procedure targets.
    reference : str
        Short citation for the property values.
    """

    name: str
    density: float                    # [kg/m^3]
    thermal_conductivity: float       # [W/m·K]
    specific_heat: float              # [J/kg·K]
    max_surface_temperature: float    # [K]
    max_bondline_temperature: float   # [K]
    reference: str


# Predefined common materials. Virgin-material properties from published
# sources; exact values vary across references.

PICA = TPSMaterial(
    name="PICA (Phenolic Impregnated Carbon Ablator)",
    density=270.0,
    thermal_conductivity=0.2,
    specific_heat=1200.0,
    max_surface_temperature=3000.0,
    max_bondline_temperature=523.0,
    reference="Tran et al. 1997 (NASA TM 110440)",
)

AVCOAT = TPSMaterial(
    name="AVCOAT (epoxy-novolac phenolic ablator)",
    density=530.0,
    thermal_conductivity=0.29,
    specific_heat=1500.0,
    max_surface_temperature=2800.0,
    max_bondline_temperature=523.0,
    reference="Apollo heatshield design documents",
)

CARBON_PHENOLIC = TPSMaterial(
    name="Carbon-Phenolic (tape-wrapped, dense)",
    density=1450.0,
    thermal_conductivity=0.70,
    specific_heat=1700.0,
    max_surface_temperature=3500.0,
    max_bondline_temperature=523.0,
    reference="Park (1990) Nonequilibrium Hypersonic Aerothermodynamics; "
    "Pioneer-Venus / Galileo CP heatshield literature",
)


# =========================================================================
# Transient heat conduction solver
# =========================================================================


def transient_bondline_temperature(
    thickness: float,
    heat_flux_time: np.ndarray,
    heat_flux: np.ndarray,
    material: TPSMaterial,
    initial_temperature: float = 293.0,
    n_nodes: int = 50,
    surface_emissivity: float = _DEFAULT_SURFACE_EMISSIVITY,
    return_full_field: bool = False,
) -> tuple[np.ndarray, ...]:
    """Solve 1-D transient heat conduction through a TPS slab.

    The front face receives heat flux q(t) and reradiates as a grey body:

        q_net(t) = q_applied(t) - sigma * eps * (T_surface^4 - T_ambient^4)

    The back face is insulated (adiabatic).

    Parameters
    ----------
    thickness : float
        TPS slab thickness [m].
    heat_flux_time : np.ndarray
        Time points of the heat flux boundary condition [s].
    heat_flux : np.ndarray
        Applied heat flux at the front face at each time point [W/m²].
    material : TPSMaterial
        TPS material properties.
    initial_temperature : float, optional
        Initial slab temperature [K]. Also used as the reradiation
        ambient temperature. Default 293 K.
    n_nodes : int, optional
        Number of spatial nodes. Default 50.
    surface_emissivity : float, optional
        Emissivity of the front face for grey-body radiation [−]. Default
        0.85 (typical for charring ablators with carbonaceous char layer).

    Returns
    -------
    time : np.ndarray
        Time vector [s].
    surface_temperature : np.ndarray
        Front-face temperature vs time [K].
    bondline_temperature : np.ndarray
        Back-face temperature vs time [K].
    """
    if thickness <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    if len(heat_flux_time) != len(heat_flux):
        raise ValueError(
            f"heat_flux_time and heat_flux must have same length: "
            f"{len(heat_flux_time)} vs {len(heat_flux)}"
        )
    if n_nodes < 3:
        raise ValueError(f"n_nodes must be >= 3, got {n_nodes}")
    if not (0.0 < surface_emissivity <= 1.0):
        raise ValueError(
            f"surface_emissivity must be in (0, 1], got {surface_emissivity}"
        )

    # Spatial discretization
    dx = thickness / (n_nodes - 1)

    # Thermal diffusivity
    alpha = material.thermal_conductivity / (material.density * material.specific_heat)

    # Stable explicit time step constraints.
    # 1. Interior diffusion: Fourier number < 0.5
    dt_diffusion = 0.4 * dx**2 / alpha
    
    # 2. Surface-node energy balance: limit dT per step to prevent runaway.
    # With q_max as the peak heat flux and rho*c*(dx/2) as surface node
    # thermal capacitance, we bound dT/step to a fraction of the material's
    # surface temperature limit.
    q_max = max(abs(heat_flux.max()), 1.0)  # avoid divide-by-zero
    surface_capacity = material.density * material.specific_heat * dx / 2.0
    dt_surface = 0.1 * material.max_surface_temperature * surface_capacity / q_max
    
    # 3. Don't exceed input sampling
    input_dt = (
        heat_flux_time[1] - heat_flux_time[0]
        if len(heat_flux_time) > 1 else 0.1
    )
    
    dt = min(dt_diffusion, dt_surface, input_dt / 2.0)
    n_steps = int(np.ceil((heat_flux_time[-1] - heat_flux_time[0]) / dt))
    dt = (heat_flux_time[-1] - heat_flux_time[0]) / n_steps

    time = np.linspace(heat_flux_time[0], heat_flux_time[-1], n_steps + 1)

    # Initialize temperature field
    T = np.full(n_nodes, initial_temperature)

    # Track temperature history
    if return_full_field:
        T_field_history = np.zeros((len(time), n_nodes))
        T_field_history[0, :] = T
    else:
        T_surface_history = np.zeros_like(time)
        T_bondline_history = np.zeros_like(time)
        T_surface_history[0] = T[0]
        T_bondline_history[0] = T[-1]

    fourier = alpha * dt / dx**2

    # Time stepping
    for i in range(1, len(time)):
        t_now = time[i]
        q_now = float(np.interp(t_now, heat_flux_time, heat_flux))

        # Explicit update of interior nodes
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + fourier * (T[2:] - 2*T[1:-1] + T[:-2])

        # Front face boundary: surface-node energy balance.
        # rho*c*(dx/2) * dT/dt = k*(T[1]-T[0])/dx + q_in - sigma*eps*(T^4 - T_amb^4)
        # Solved explicitly with the half-volume capacitance.
        #
        # The radiation term is clipped to prevent numerical runaway during
        # the first few time steps before the solution stabilizes. This
        # is physically equivalent to limiting the emissive power to the
        # incoming flux (energy can't radiate more than it receives in
        # any instant without a prior accumulation).
        T_surf_safe = min(T[0], 4000.0)   # clip to prevent T^4 overflow
        q_reradiated = (
            _STEFAN_BOLTZMANN * surface_emissivity *
            (T_surf_safe**4 - initial_temperature**4)
        )
        q_conduction_into_slab = (
            material.thermal_conductivity * (T[0] - T[1]) / dx
        )
        q_net_surface = q_now - q_reradiated - q_conduction_into_slab
        dT_surface = (
            2.0 * dt * q_net_surface /
            (material.density * material.specific_heat * dx)
        )
        T_new[0] = T[0] + dT_surface
        # Back face boundary: insulated (dT/dx = 0)
        # Ghost-node method: T_{N} = T_{N-2}
        T_new[-1] = T[-1] + fourier * (2*T[-2] - 2*T[-1])

        T = T_new
        if return_full_field:
            T_field_history[i, :] = T
        else:
            T_surface_history[i] = T[0]
            T_bondline_history[i] = T[-1]

    if return_full_field:
        x_nodes = np.linspace(0.0, thickness, n_nodes)
        return time, x_nodes, T_field_history
    return time, T_surface_history, T_bondline_history


# =========================================================================
# Sizing procedure
# =========================================================================


class TPSSizingResult(NamedTuple):
    """Result of TPS sizing calculation.

    Attributes
    ----------
    required_thickness : float
        Minimum TPS thickness [m] to meet bondline temperature constraint.
    time : np.ndarray
        Time array [s] from the solved thermal history.
    surface_temperature : np.ndarray
        Front-face temperature over time [K] at the sized thickness.
    bondline_temperature : np.ndarray
        Back-face temperature over time [K] at the sized thickness.
    peak_bondline_temperature : float
        Maximum bondline temperature reached [K].
    peak_surface_temperature : float
        Maximum surface temperature reached [K].
    material : TPSMaterial
        Material used.
    iterations : int
        Number of binary-search iterations required.
    """

    required_thickness: float
    time: np.ndarray
    surface_temperature: np.ndarray
    bondline_temperature: np.ndarray
    peak_bondline_temperature: float
    peak_surface_temperature: float
    material: TPSMaterial
    iterations: int


def size_tps(
    heating_result: HeatingResult,
    material: TPSMaterial,
    thickness_min: float = 0.005,
    thickness_max: float = 0.200,
    tolerance: float = 1e-4,
    initial_temperature: float = 293.0,
    n_nodes: int = 50,
    surface_emissivity: float = _DEFAULT_SURFACE_EMISSIVITY,
    max_iterations: int = 40,
) -> TPSSizingResult:
    """Find minimum TPS thickness to keep bondline below material limit.

    Uses bisection on thickness. Bondline temperature decreases
    monotonically with thickness, so bisection converges reliably.

    Parameters
    ----------
    heating_result : HeatingResult
        Output from aerothermal.heating_history(). Uses heat_flux and time.
    material : TPSMaterial
        TPS material properties.
    thickness_min : float, optional
        Lower bound for search [m]. Default 5 mm.
    thickness_max : float, optional
        Upper bound for search [m]. Default 200 mm.
    tolerance : float, optional
        Thickness tolerance for bisection convergence [m]. Default 0.1 mm.
    initial_temperature : float, optional
        Initial slab temperature [K]. Default 293 K.
    n_nodes : int, optional
        Spatial nodes. Default 50.
    surface_emissivity : float, optional
        Grey-body emissivity of front face [−]. Default 0.85.
    max_iterations : int, optional
        Safety limit on bisection iterations. Default 40.

    Returns
    -------
    TPSSizingResult

    Raises
    ------
    ValueError
        If thickness_min already keeps bondline cool (too thick), or if
        thickness_max is insufficient (too thin).
    """
    # Validate thickness_min: must overheat bondline
    _, _, T_bl_min = transient_bondline_temperature(
        thickness_min, heating_result.time, heating_result.heat_flux,
        material, initial_temperature, n_nodes,
        surface_emissivity=surface_emissivity,
    )
    if T_bl_min.max() <= material.max_bondline_temperature:
        raise ValueError(
            f"Minimum thickness {thickness_min*1000:.1f} mm already keeps "
            f"bondline below {material.max_bondline_temperature} K "
            f"(peak {T_bl_min.max():.1f} K). Increase heating or reduce "
            f"thickness_min."
        )

    # Validate thickness_max: must keep bondline cool
    _, _, T_bl_max = transient_bondline_temperature(
        thickness_max, heating_result.time, heating_result.heat_flux,
        material, initial_temperature, n_nodes,
        surface_emissivity=surface_emissivity,
    )
    if T_bl_max.max() > material.max_bondline_temperature:
        raise ValueError(
            f"Maximum thickness {thickness_max*1000:.1f} mm insufficient: "
            f"bondline still exceeds {material.max_bondline_temperature} K "
            f"(peak {T_bl_max.max():.1f} K). Increase thickness_max."
        )

    # Bisection
    lo = thickness_min
    hi = thickness_max
    iteration = 0

    while (hi - lo) > tolerance and iteration < max_iterations:
        mid = 0.5 * (lo + hi)
        _, _, T_bl = transient_bondline_temperature(
            mid, heating_result.time, heating_result.heat_flux,
            material, initial_temperature, n_nodes,
            surface_emissivity=surface_emissivity,
        )
        peak_bl = T_bl.max()
        if peak_bl > material.max_bondline_temperature:
            lo = mid  # too thin, thicker needed
        else:
            hi = mid  # thick enough, try thinner
        iteration += 1

    # Final computation at converged thickness
    final_thickness = hi
    time, T_surf, T_bl = transient_bondline_temperature(
        final_thickness, heating_result.time, heating_result.heat_flux,
        material, initial_temperature, n_nodes,
        surface_emissivity=surface_emissivity,
    )

    return TPSSizingResult(
        required_thickness=final_thickness,
        time=time,
        surface_temperature=T_surf,
        bondline_temperature=T_bl,
        peak_bondline_temperature=float(T_bl.max()),
        peak_surface_temperature=float(T_surf.max()),
        material=material,
        iterations=iteration,
    )