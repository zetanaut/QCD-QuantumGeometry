"""Named common raw probe families on different longitudinal grids.

Three separately retained seed families live in qqq, qqqg(hg=-1), and
qqqg(hg=+1). They are diagnostics, not fitted or selected physical states.
Different K matrices are never aligned by their size or compact column id.
"""
from dataclasses import dataclass
import numpy as np
from .m2_coupled_longitudinal_jacobi import longitudinal_basis, oscillator_shell
from .m2_fourth_order_h2 import _norm_inverse_sqrt


@dataclass(frozen=True)
class LongitudinalProbeFrame:
    pilot: object
    vectors: np.ndarray
    longitudinal_power: float
    labels: tuple


def longitudinal_probe_frame(pilot, *, longitudinal_power):
    """Project the same occupation-label seeds and prod(x_i)^power on each grid.

    Every transverse coordinate is in the global-Jacobi ground at the
    caller's common b. The positive-mode composition grid has uniform cell
    weight; its common sector factor cancels in each seed's normalization.
    The exact color/isospin/S3/Jz projector is the pilot's raw isometry.
    No phase is taken from a compact eigenvector column and no small
    amplitude is discarded. Three sector/helicity families stay separate.
    """
    power = float(longitudinal_power)
    if isinstance(longitudinal_power, (bool, np.bool_)) or not np.isfinite(power) or power < 0:
        raise ValueError('explicit finite nonnegative longitudinal probe power required')
    if pilot.target != 'PROTON' or pilot.jz_twice != 1:
        raise ValueError('this named diagnostic is proton Jz=+1/2, not an implicit state choice')
    # Canonical quark label: 0=u down, 1=u up, 2=d down, 3=d up.
    # Each qqqg seed uses fixed raw color-octet coordinate 1,
    # followed by the ACTUAL joint sign/isospin projector, not a factorized
    # color-singlet assumption. The gluon helicity selects its pilot block.
    # build_coupled_pilot fixes its gluon-block order to (-1,+1).
    # Coordinate 0 has zero projected support for these symmetric orbital
    # and equal-spin seeds. Normalizing its roundoff residue would create
    # an arbitrary, compact-basis-dependent vector. Coordinate 1 is an
    # explicitly named raw seed, not selected from compact eigenvectors.
    specifications = (('qqq', 2, 1, 0, (0, 1, 3)),
                      ('qqqg', 3, 2, 1, (1, 1, 3)),
                      ('qqqg', 3, 2, 1, (0, 0, 3)))
    result = np.zeros((len(pilot.matrix), 3), complex)
    offset = 0
    for column, ((sector, d, colors, color, spins), basis) in enumerate(zip(specifications, pilot.bases)):
        states = longitudinal_basis(sector, pilot.total_k2)
        shell = oscillator_shell(d, pilot.cutoff)
        zero = shell.index((0,)*(2*d))
        amplitudes = np.array([np.prod(np.array(s, float)/pilot.total_k2)**power for s in states])
        raw = np.zeros((len(states), len(shell), colors, 64), complex)
        raw[:, zero, color, np.ravel_multi_index(spins, (4, 4, 4))] = amplitudes
        compact = np.asarray(basis.conj().T@raw.reshape(-1)).reshape(-1)
        norm = float(np.linalg.norm(compact))
        # Reject an ill-conditioned probe definition; never trim matrix
        # amplitudes or promote numerical noise into a normalized state.
        if not np.isfinite(norm) or norm <= 64*np.finfo(float).eps*np.linalg.norm(raw):
            raise ArithmeticError('the named physical-label probe has no numerically resolved projected support')
        result[offset:offset+basis.shape[1], column] = compact/norm
        offset += basis.shape[1]
    if offset != len(pilot.matrix):
        raise ValueError('the explicit three pilot blocks must exhaust retained coordinates')
    np.testing.assert_allclose(result.conj().T@result, np.eye(3), atol=3e-14, rtol=0.)
    return LongitudinalProbeFrame(pilot, result, power, ('qqq', 'qqqg_minus', 'qqqg_plus'))


def probe_current(raw_current, outgoing, incoming, frame):
    """Compress raw current, then normalize the full THREE-probe Gram matrices.

    G_a=Udag N_a U and j_probe=G_f^-1/2 Udag j_raw U G_i^-1/2.
    This is the same result as the full normalized current pulled back
    by T_a=N_a^1/2 U G_a^-1/2, not by U alone. Inter-family overlaps in
    the H2 norm are retained. The caller selects common coupling order
    before these norm maps. No physical calibration or state is selected.
    """
    if outgoing.owner.pilot is not frame.pilot or incoming.owner.pilot is not frame.pilot:
        raise ValueError('both snapshots must bind the exact named probe pilot')
    j = np.asarray(raw_current, complex)
    n = len(frame.pilot.matrix)
    if j.shape[-2:] != (n, n) or j.ndim not in (2, 3) or not np.all(np.isfinite(j)):
        raise ValueError('finite raw current matrices on the declared pilot required')
    u = frame.vectors
    norms = tuple(u.conj().T@s.metric@u for s in (outgoing, incoming))
    wf, wi = (_norm_inverse_sqrt(g) for g in norms)
    compressed = np.einsum('ab,...bc,cd->...ad', u.conj().T, j, u, optimize=True)
    normalized = np.einsum('ab,...bc,cd->...ad', wf, compressed, wi, optimize=True)
    return compressed, normalized, norms
