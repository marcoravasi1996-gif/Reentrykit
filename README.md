# reentrykit

**Validated Python tools for atmospheric reentry trajectory analysis, aerothermal heating, and Level 1 thermal protection system (TPS) sizing.**

A focused engineering simulator that reproduces published flight reconstructions of three NASA missions to within engineering accuracy:

| Mission | Peak deceleration (flight) | Reference | Our result | Error |
|---|---|---|---|---|
| **Apollo 4** (AS-501) | 7.26 g | Hilje 1969, NASA TN D-5399 | 7.42 g (with bank-angle modulation) | **+2.2%** |
| **Stardust** SRC | 33.4 g | Desai et al. 2008 | 33.21 g | **−0.6%** |
| **Genesis** SRC | 27.0 g | Desai & Lyons 2008 | 28.12 / 25.83 g (brackets) | **±4.3%** |

All validations include rotating-Earth physics (Coriolis + centrifugal), Sutton-Graves convective and Tauber-Sutton radiative stagnation-point heating, and bondline-temperature-constrained TPS thickness sizing for PICA and carbon-phenolic ablators.

## What's Inside

```
src/reentrykit/
├── atmosphere.py     # US1976 0–86 km + exponential extension to 500 km
├── planet.py         # WGS-84 Earth (rotating + non-rotating variants)
├── trajectory.py     # 3-DOF Vinh-Busemann-Culp equations of motion
├── aerothermal.py    # Sutton-Graves convective + Tauber-Sutton radiative
├── tps.py            # Level 1 TPS sizing with PICA, AVCOAT, carbon-phenolic
└── visualization.py  # 3D trajectory rendering (Plotly + matplotlib)
```

127 unit and validation tests covering every module. Trajectory equations follow Vinh, Busemann & Culp (1980), *Hypersonic and Planetary Entry Flight Mechanics*, eqs. (2-44) through (2-49).

## Quick Start

```bash
git clone https://github.com/marcoravasi1996-gif/reentrykit.git
cd reentrykit
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev,notebooks]"
pytest -v                        # confirm 127 tests pass
```

## Example: Simulate a ballistic Stardust entry

```python
import numpy as np
from reentrykit.trajectory import Vehicle, InitialState, simulate
from reentrykit.planet import EARTH_NON_ROTATING

vehicle = Vehicle.from_mass_area_cd(
    mass=45.8, reference_area=0.517,
    drag_coefficient=1.0, lift_to_drag_ratio=0.0,
    nose_radius=0.2202,
)

entry = InitialState(
    altitude=125_000.0,                   # 125 km
    velocity=12_300.0,                    # m/s, atmosphere-relative
    flight_path_angle=np.deg2rad(-8.2),
    heading=np.deg2rad(15.0),             # V-B-C convention: ψ from east CCW
    latitude=np.deg2rad(41.0),
    longitude=np.deg2rad(-128.0),
)

result = simulate(vehicle, entry, planet=EARTH_NON_ROTATING)

dV_dt = np.gradient(result.velocity, result.time)
peak_g = -dV_dt.min() / 9.80665
print(f"Peak deceleration: {peak_g:.2f} g")    # 33.21 g
```

For TPS sizing, heating, and rotating-Earth examples, see the validation notebooks below.

## Validation Notebooks

The `notebooks/` directory contains the full validation campaign:

| Notebook | Topic |
|---|---|
| `01_atmosphere.ipynb` | US1976 + exponential extension demonstration |
| `02_trajectory.ipynb` | Allen-Eggers analytical vs. simulator |
| `03_apollo_4_validation.ipynb` | Apollo 4 against Hilje (1969) flight reconstruction |
| `04_stardust_validation.ipynb` | Stardust against Desai (2008) |
| `05_genesis_validation.ipynb` | Genesis against Desai & Lyons (2008) |
| `06_rotating_earth_validation.ipynb` | Coriolis rate, energy conservation, orbit stability |
| `07_visualization_demo.ipynb` | V-B-C heading convention + rotating-Earth physics |
| `08_tps_sizing_stardust.ipynb` | PICA Level 1 sizing for Stardust (36 mm vs flown 58 mm) |
| `09_tps_sizing_genesis.ipynb` | PICA vs carbon-phenolic mass-thickness trade |

## Key Conventions

**Heading angle (Vinh-Busemann-Culp, eq. 2-44):** ψ measured CCW from east. ψ=0 → east, ψ=π/2 → north, ψ=π → west. Aerospace azimuth (CW from north) requires conversion: `psi_vbc = pi/2 − azimuth_aerospace`.

**Bank angle:** σ positive CCW (right wing up looking along velocity vector). Aerospace bank schedules (positive CW = right wing down) require sign negation at input.

**State vector:** `[V, γ, ψ, h, φ, θ]` for atmosphere-relative speed, flight-path angle, heading, altitude, latitude, longitude.

## Scope and Limitations

This is a **Level 1 simulator** for preliminary design and validation. It does not include:

- Ablation mass loss or recession (TPS sizing under-predicts ablator thickness by 30–50%, by design)
- Finite-rate chemistry or non-equilibrium thermochemistry
- 6-DOF dynamics or attitude propagation
- Mach-dependent drag coefficients are supported via callable, but no aero database is bundled
- Closed-loop guidance (replays digitized commands, does not solve the guidance problem)

For production-grade entry analysis, tools like POST-II, NASA's Tetra, or DSMC-coupled aerothermal codes are appropriate. `reentrykit` targets the design-space exploration and validation-against-flight-data niche.

## References

The simulator's equations and correlations come from:

- Vinh, N. X., Busemann, A., Culp, R. D. (1980). *Hypersonic and Planetary Entry Flight Mechanics*. University of Michigan Press.
- Sutton, K., & Graves, R. A. (1971). *A General Stagnation-Point Convective Heating Equation for Arbitrary Gas Mixtures*. NASA TR R-376.
- Tauber, M. E., & Sutton, K. (1991). *Stagnation-Point Radiative Heating Relations for Earth and Mars Entries*. JSR 28(1), 40–42.
- Tran, H. K., et al. (1997). *Phenolic Impregnated Carbon Ablators (PICA) as Thermal Protection Systems for Discovery Missions*. NASA TM-110440.
- Hilje, E. R. (1969). *Entry Aerodynamics at Lunar Return Conditions Obtained from the Flight of Apollo 4 (AS-501)*. NASA TN D-5399.
- Desai, P. N., Lyons, D. T., Tooley, J., Kangas, J. (2008). *Entry, Descent, and Landing Operations Analysis for the Stardust Entry Capsule*. JSR 45(6), 1262–1268.
- Desai, P. N., & Lyons, D. T. (2008). *Entry, Descent, and Landing Operations Analysis for the Genesis Entry Capsule*. JSR 45(1), 27–32.

## License

MIT. See [LICENSE](LICENSE).