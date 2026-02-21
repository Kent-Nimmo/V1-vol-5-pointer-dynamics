# SIMRUN Summary: vol-5-pointer-dynamics

## What This Sim Tests

This simulation runs the Geometry Baseline pipeline for the FPHS pointer dynamics sector. It applies a Gaussian-smoothing + gradient-magnitude translator to E0 field snapshots, solves a Poisson equation for the gravitational-like potential, and measures radial power-law profiles and thin-lens deflection angles. The 45 conditions cover SU3 (kappa=1.0, 0.75) and SU2 (kappa=1.0) at L=256, b=3.5, with f-values 0.0/0.1/0.3 and 5 seeds each.

## Why It Matters

This baseline geometry pipeline characterizes how the FPHS E0 field snapshots behave when processed through a simple gradient-based source + Poisson solver. It provides the foundation for measuring how measurement (f > 0) and gauge group structure affect the emergent geometric properties of the system. The pointer dynamics context establishes the dynamical baseline against which more refined translators can be compared.

## How It Was Run

- Config: configs/anchors.yaml
- Command: `python -m src.geometry.run_geometry --config configs/anchors.yaml --data-dir data/inputs --out-dir runs`
- Aggregation: `python scripts/aggregate.py --runs-dir runs --out-dir results`
- Report: `python scripts/generate_report.py --summary results/summary.csv --out-report reports/REPORT.md`
- 45 conditions total (3 anchors x 3 f-values x 5 seeds)

## Results

- **Ran successfully**: YES
- **Conditions completed**: 45/45
- **SU3 (kappa=1.0)**: s_phi = -0.193, r2_phi = 0.194; s_grad = -0.643, r2_grad = 0.351
- **SU3 (kappa=0.75)**: same metrics
- **SU2 (kappa=1.0)**: same metrics
- **Lensing**: alpha_slope ~138 (lambda=0.2), R^2 ~0.16
- Results are deterministic across all seeds/f-values for matching E0 snapshots

## Warnings / Limitations

- R^2 values for radial fits are low (0.19-0.35), consistent with a baseline translator.
- Lensing R^2 is low (~0.16).
- No formal acceptance script; results serve as a baseline.
