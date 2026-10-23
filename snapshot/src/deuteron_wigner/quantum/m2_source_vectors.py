"""Exact canonical sources acting on arbitrary retained columns.

The probe axis is shared across helicity blocks. It is not a selected basis,
source-image truncation or physical state choice. Sparse retained isometries
are contracted BEFORE any raw Q-by-probe image is allocated.
"""
from dataclasses import dataclass

import numpy as np

from .m2_coupled_longitudinal_jacobi import longitudinal_basis, oscillator_shell


@dataclass(frozen=True)
class SourceImage:
    retained_pilot: object
    space: object
    coupling_power: int
    values: np.ndarray
    components: dict


def validate_source_vectors(pilot, vectors):
    v = np.asarray(vectors, complex)
    if (v.ndim != 2 or v.shape[0] != len(pilot.matrix) or v.shape[1] == 0
            or not np.all(np.isfinite(v))):
        raise ValueError("nonempty finite columns on the actual retained pilot required")
    return v


def source_tensor(pilot, sector, vectors):
    v = validate_source_vectors(pilot, vectors)
    count = v.shape[1]
    if sector == "qqq":
        u = pilot.bases[0]
        shape = (len(longitudinal_basis(sector, pilot.total_k2)),
                 len(oscillator_shell(2, pilot.cutoff)), 1, 64, count)
        return np.asarray(u@v[:u.shape[1]]).reshape(shape)
    if sector != "qqqg":
        raise ValueError("explicit canonical qqq or qqqg parent required")
    shape = (len(longitudinal_basis(sector, pilot.total_k2)),
             len(oscillator_shell(3, pilot.cutoff)), 2, 64, 2, count)
    result = np.zeros(shape, complex)
    start = pilot.bases[0].shape[1]
    for hg, u in enumerate(pilot.bases[1:]):
        stop = start+u.shape[1]
        result[:, :, :, :, hg, :] = np.asarray(u@v[start:stop]).reshape(shape[:4]+(count,))
        start = stop
    if start != len(v):
        raise ValueError("retained helicity blocks do not exhaust the actual pilot")
    return result


def image_symmetry_tolerance(image, *, vector_action):
    # Legacy basis-column checks are unchanged. Vector actions accept
    # arbitrary magnitudes; this scales a validation bound, never support.
    return 3e-11*max(1., float(np.linalg.norm(image))) if vector_action else 3e-11
