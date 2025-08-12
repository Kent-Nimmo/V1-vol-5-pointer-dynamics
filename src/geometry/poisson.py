"""Poisson solver for the geometry baseline.

This module provides a function to compute the geometry potential ``Φ`` and its
gradient magnitude from the envelope field ``E_env``.  It uses an
aperiodic convolution with the kernel ``G_ε(r) = 1/(r^2 + ε^2)``.  To
achieve aperiodicity, the input field is zero‑padded by a factor
``pad_factor`` along each axis before performing an FFT‐based convolution.

The implementation is careful to avoid dividing by zero at the origin and
returns both the potential and the gradient magnitude on the original
lattice.  Double precision is used throughout.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2
from .translate import gradient_magnitude

def compute_phi(env: np.ndarray, ell: float, pad_factor: int = 4, epsilon_factor: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Compute the geometry potential ``Φ`` and its gradient from an envelope.

    Parameters
    ----------
    env : np.ndarray
        Envelope field ``E_env`` of shape ``(L, L)``.  Must be real‐valued.
    ell : float
        Smoothing length ``ℓ`` used to build the envelope.  Determines
        ``ε`` as ``ε = epsilon_factor * ℓ``.
    pad_factor : int, optional
        Factor by which to zero‑pad the grid along each axis.  Must be >= 1.
        Default is 4.
    epsilon_factor : float, optional
        Multiplier used to compute ``ε`` from ``ℓ``.  Default is 0.5.

    Returns
    -------
    phi : np.ndarray
        Geometry potential ``Φ`` on the original ``L x L`` grid.
    grad_phi_mag : np.ndarray
        Gradient magnitude ``|∇Φ|`` on the original grid.
    """
    L = env.shape[0]
    assert env.shape[0] == env.shape[1], "env must be square"
    # Determine epsilon and padded size.
    eps = epsilon_factor * float(ell)
    pad = int(pad_factor)
    if pad < 1:
        pad = 1
    # Compute padded grid size.  Use next power of two for speed if desired.
    N = L * pad
    # Zero‑pad the envelope to shape (N, N) by placing env in the centre.
    padded = np.zeros((N, N), dtype=np.float64)
    # Insert env into the top‑left corner of padded array.  Zero padding works because convolution is linear and padded region outside is zero.
    padded[:L, :L] = env.astype(np.float64)
    # Build the convolution kernel G(r) = 1/(r^2 + eps^2) on the padded grid.
    # Construct coordinates: range from 0 to N-1.  Distances wrap around for FFT; to obtain aperiodic result we rely on zero padding.
    yy, xx = np.ogrid[:N, :N]
    # Shift origin to avoid singularity at (0,0).  Distances measured from (0,0).
    rr2 = xx**2 + yy**2
    kernel = 1.0 / (rr2 + eps * eps)
    # Avoid the r=0 singularity by explicitly setting kernel[0,0] to 0.  The constant term does not affect gradients.
    kernel[0, 0] = 0.0
    # Perform convolution via FFT: note that real input yields real output after taking real part.
    conv_fft = ifft2(fft2(padded) * fft2(kernel)).real
    # Extract the central LxL region corresponding to the original grid.
    phi = conv_fft[:L, :L]
    # Compute the gradient magnitude of the potential.
    grad_phi_mag = gradient_magnitude(phi)
    return phi, grad_phi_mag