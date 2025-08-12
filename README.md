# Geometry Baseline for FPHS (Baseline A)

This repository implements the **translator baseline** (also called **Baseline A**) for the geometry sector of the **Fractal Pivot Hypersurface (FPHS)** stack.  Its goal is to provide a clean, reproducible pipeline for constructing a "geometry potential" from FPHS field snapshots and characterising its large–scale behaviour via radial and lensing fits.

> **Why another repo?**  The pointer‐dynamics simulation (in `pointer‑dynamics‑fphs`) explores how quantum packets become classical under a static potential derived from FPHS.  This repo focuses on a complementary question: how the **geometry translator** maps a gauge‐theory field (the "electric field" snapshot `E0`) to an effective potential \(\Phi\) that controls gravitational/optical behaviour.  The pipeline here is general and can run on **real theory data** as soon as it is available.

## What this repository does

1. **Loads FPHS field snapshots.**  Each snapshot contains a 2‑D array `E0` (size \(L\times L\)) storing the electric field on a lattice for a given gauge, coupling `b`, momentum scale `\kappa`, measurement fraction `f` and random seed.  Real snapshots should be provided as `.npz` files under `data/inputs`, with metadata embedded in the filename or accompanying JSON.  For development and testing, the code can generate synthetic fields (white noise) as placeholders – but **these are *not* used for results**.
2. **Applies the baseline translator (gradient‑as‑mass).**  For each snapshot it:
   - Smooths `E0` with a Gaussian filter of width \(\ell=4\) lattice units.
   - Computes the gradient magnitude of the smoothed field, then normalises it by its root‑mean‑square (RMS) to obtain a dimensionless envelope \(N(|\nabla E_0|)\).  No background removal is applied.
   - Treats this envelope as a mass sheet in three dimensions and solves the Poisson equation using aperiodic convolution with kernel \(G_\varepsilon(r) = 1/(r^2 + \varepsilon^2)\) with \(\varepsilon = 0.5\ell\).  Zero padding by a factor of 4 along each axis ensures the convolution is effectively aperiodic.  The result is the **geometry potential** \(\Phi\) and its gradient field \(|\nabla\Phi|\).
   - Optionally creates a **normalised copy** of \(|\nabla\Phi|\) by scaling it so that the median value equals 1.  This keeps slope fits from being skewed by the overall amplitude.
3. **Computes radial profiles and fits.**  \(\Phi\) and \(|\nabla\Phi|\) are radially binned into 36 logarithmic rings from \(r=3\ell\) to \(0.3L\).  Rings with fewer than 50 samples are discarded.  Weighted least‑squares (WLS) regressions in log–log space provide slopes \(s_\Phi\) and \(s_{\nabla\Phi}\) and their coefficient of determination \(R^2\).  The raw and normalised fields can be fitted separately.
4. **Performs thin‑lens optics.**  Treats the geometry potential as an index perturbation \(n(x) = 1 + \lambda\,E_{\mathrm{env}}(x)\) and computes the deflection angles \(\alpha(b)\) for 32 log‑spaced impact parameters \(b\in[8,96]\) (for \(L=256\)).  A simple line integral of the transverse gradient yields the deflection.  A WLS fit of \(\alpha(b)\) versus \(1/b\) on the central 60 % of the points yields a slope and \(R^2\).
5. **Stores results.**  For each condition (gauge, `L`, `b`, `\kappa`, `f`, seed, smoothing length `\ell`, refractive index strength `\lambda`) it writes:
   - A JSON file summarising slopes, fit windows and normalisation flags.
   - CSV files containing the radial profiles and lensing tables.
   - Plots (PNG) showing the radial profiles and lensing fits.
6. **Aggregates runs.**  Scripts under `scripts/` collect per–condition JSON results into a summary CSV and produce a Markdown report.  Everything is reproducible via `scripts/run_all.sh` once the inputs are available.

## How to run

1. **Install dependencies.**  A Conda environment specification is provided under `env/environment.yml`; a minimal `requirements.txt` is also included.  Use either:

   ```bash
   # Using conda
   conda env create -f env/environment.yml
   conda activate fphs-geom

   # Using pip
   python3 -m venv venv
   source venv/bin/activate
   pip install -r env/requirements.txt
   ```

2. **Prepare inputs.**  Place real theory snapshots as `.npz` files into `data/inputs`.  Each file should have an `E0` array of shape `L x L`.  File names should encode the gauge (`SU2` or `SU3`), `L`, `b`, `kappa`, `f` and `seed` – for example: `E0_SU3_L256_b3.5_k1.0_f0.10_seed2.npz`.

3. **Run the pipeline.**  Execute the convenience script which calibrates nothing and simply processes all inputs defined in `configs/anchors.yaml`:

   ```bash
   ./scripts/run_all.sh
   ```

   The script loads each snapshot, applies the translator, computes radial and lensing fits, and writes outputs under `runs/` and `results/`.  It then aggregates all JSON results into `results/summary.csv` and produces a report in `reports/REPORT.md`.

4. **View results.**  Inspect the `results/` directory for summary CSVs and plots.  The `reports/` folder contains a high‑level report summarising slopes, lensing fits and QC flags.

## Important

*This repository intentionally does not contain the real FPHS data.*  Real snapshots and kernels must be added by the user.  The code can be tested with synthetic fields by setting the environment variable `FPHS_GEOM_USE_PLACEHOLDER=1`.  When this variable is set, the pipeline generates Gaussian random fields of the appropriate size in place of missing inputs.  These placeholders are for testing code functionality only and must **never be used for scientific conclusions**.

For questions or bug reports, please contact the maintainers.