"""Canonical transverse-parity phase map and spin-half good-current probes.

Only local J+ at Delta+=0 is constructed. Dirac/Pauli extraction is an
explicit current-structure diagnostic, not full Lorentz/current closure,
physical form factors or a calibrated nucleon state.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from deuteron_wigner.bridge.modes.core import GAMMA, GAMMA_PLUS
from .m2_canonical_emission import lf_spinor
from .m2_constrained_propagators import slash
from .m2_coupled_longitudinal_jacobi import longitudinal_basis, oscillator_shell
from .m2_spin_flavor_s3_completion import SPIN_FLAVOR_BASIS


def transverse_parity_map(outgoing, incoming):
    """Px:(x,y,z)->(-x,y,z), retaining x+ and reversing Jz.

    For the canonical quark spinor S(Px)=-i Sigma_x gamma0:
    S u_h(p)=-i u_-h(Px p). Three quarks give i, each gluon has
    epsilon_h -> epsilon_-h, and each circular oscillator swaps +/- with
    phase (-1)^N. Thus Px²=-I in the odd-fermion retained space.
    No arbitrary eigensolver phase is used to align the two Jz fibers.
    """
    for name in ("total_k2","cutoff","target","width_GeV","quark_mass_GeV",
                 "gluon_mass_GeV","coupling"):
        if getattr(outgoing,name)!=getattr(incoming,name):
            raise ValueError("parity endpoints disagree on "+name)
    if outgoing.jz_twice != -incoming.jz_twice:
        raise ValueError("opposite Jz endpoints required")
    sf_lookup = {s:i for i,s in enumerate(SPIN_FLAVOR_BASIS)}
    sf_rows = [sf_lookup[tuple(x^1 for x in s)] for s in SPIN_FLAVOR_BASIS]
    spin = sparse.csr_matrix((np.ones(64), (sf_rows, np.arange(64))), shape=(64,64))
    offsets_i = np.r_[0,np.cumsum([u.shape[1] for u in incoming.bases])]
    offsets_f = np.r_[0,np.cumsum([u.shape[1] for u in outgoing.bases])]
    result = np.zeros((len(outgoing.matrix),len(incoming.matrix)),complex)
    for si,sector in enumerate(("qqq","qqqg","qqqg")):
        target = 0 if si==0 else 3-si
        uf,ui = outgoing.bases[target],incoming.bases[si]
        d = 2 if sector=="qqq" else 3
        nc = d-1
        shell = oscillator_shell(d,incoming.cutoff)
        lookup = {s:i for i,s in enumerate(shell)}
        rows = [lookup[s[d:]+s[:d]] for s in shell]
        phases = [(-1)**sum(s) for s in shell]
        orbital = sparse.csr_matrix((phases,(rows,np.arange(len(shell)))),shape=(len(shell),len(shell)))
        nl = len(longitudinal_basis(sector,incoming.total_k2))
        op = 1j*sparse.kron(sparse.eye(nl),sparse.kron(orbital,sparse.kron(sparse.eye(nc),spin)),format="csr")
        image = op@ui
        block = uf.getH()@image
        if sparse.linalg.norm(image-uf@block)>3e-10:
            raise ArithmeticError("transverse parity leaves the target symmetry fiber")
        result[offsets_f[target]:offsets_f[target+1],offsets_i[si]:offsets_i[si+1]] = block.toarray()
    if np.linalg.norm(result.conj().T@result-np.eye(len(incoming.matrix)))>3e-10:
        raise ArithmeticError("transverse parity is not unitary between retained fibers")
    return result


def parity_aligned_partner(outgoing, incoming, state, *, intrinsic_parity):
    """Spin-half phase convention: Px|up>=-i eta |down>.

    The positive-parity nucleon choice eta=+1 must be supplied explicitly.
    This aligns a caller-chosen state, not an automatic production state.
    """
    if (incoming.jz_twice!=1 or outgoing.jz_twice!=-1
            or isinstance(intrinsic_parity,(bool,np.bool_)) or intrinsic_parity not in (-1,1)):
        raise ValueError("spin-half up/down endpoints and explicit intrinsic parity required")
    v = np.asarray(state,complex)
    if v.shape!=(len(incoming.matrix),) or not np.all(np.isfinite(v)):
        raise ValueError("finite incoming state vector required")
    return 1j*intrinsic_parity*transverse_parity_map(outgoing,incoming)@v


def dirac_pauli_good_structures(transfer, *, mass_GeV, total_pplus):
    """Independent external-spinor tensors of gamma+ and i sigma+nu q_nu/2M.

    Symmetric frame p_T,in=-Delta/2, p_T,out=Delta/2, p+ fixed.
    Both rows and columns use helicity order (+1/2,-1/2).
    The common 2P+ is stripped. No microscopic current model is assumed.
    """
    delta = np.asarray(transfer,float)
    mass,pp = float(mass_GeV),float(total_pplus)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)) or not np.all(np.isfinite([mass,pp])) or min(mass,pp)<=0:
        raise ValueError("finite transverse transfer and positive mass/P+ required")
    qs = slash([0.,delta[0],delta[1],0.])
    pauli = -(GAMMA_PLUS@qs-qs@GAMMA_PLUS)/(4*mass)
    bra = [lf_spinor(pp,delta[0]/2,delta[1]/2,mass,h) for h in (1,-1)]
    ket = [lf_spinor(pp,-delta[0]/2,-delta[1]/2,mass,h) for h in (1,-1)]
    return tuple(np.array([[u.conj()@GAMMA[0]@op@v/(2*pp) for v in ket] for u in bra])
                 for op in (GAMMA_PLUS,pauli))


@dataclass(frozen=True)
class GoodFormFactorDiagnostic:
    dirac: complex
    pauli: complex
    residual_matrix: np.ndarray


def good_form_factor_diagnostic(current_matrix, transfer, *, mass_GeV, total_pplus):
    """Resolve two covariant structures without forcing any residual to zero.

    Positive nonzero Q² required; a zero-transfer Pauli value is a limit,
    not extractable by dividing a zero spin-flip amplitude by zero.
    """
    value = np.asarray(current_matrix,complex)
    structures = dirac_pauli_good_structures(transfer,mass_GeV=mass_GeV,total_pplus=total_pplus)
    if value.shape!=(2,2) or not np.all(np.isfinite(value)):
        raise ValueError("finite 2x2 helicity current required")
    if np.linalg.norm(structures[1])==0:
        raise ValueError("nonzero transfer required to resolve the Pauli structure")
    axes = np.column_stack([a.ravel() for a in structures])
    coefficients = np.linalg.lstsq(axes,value.ravel(),rcond=None)[0]
    residual = value-sum(c*a for c,a in zip(coefficients,structures))
    return GoodFormFactorDiagnostic(complex(coefficients[0]),complex(coefficients[1]),residual)


def excluded_parent_density_between(outgoing, incoming, transfer, *, charge):
    """Actual source-reachable free-Q parent density between helicities.

    Retained and excluded blocks keep their own explicit symmetry-coordinate
    maps. Distinct sector orderings are resolved by labels, never positions.
    Q cutoffs must coincide at these endpoints; cutoff transport is separate.
    """
    from .m2_canonical_good_current import good_current_density_between
    from .m2_excluded_fock_matching import intermediate_density_action_between
    retained = good_current_density_between(outgoing.pilot,incoming.pilot,transfer,charge=charge)
    if {s.source.space.sector for s in outgoing.sectors}!={s.source.space.sector for s in incoming.sectors}:
        raise ValueError("the same explicit excluded sectors are required")
    result=np.zeros((len(outgoing.matrix),len(incoming.matrix)),complex)
    result[:retained.shape[0],:retained.shape[1]]=retained
    offsets_in={}; start=len(incoming.pilot.matrix)
    for sector in incoming.sectors:
        n=len(sector.energies_GeV2)
        offsets_in[sector.source.space.sector]=(slice(start,start+n),sector)
        start+=n
    start=len(outgoing.pilot.matrix)
    for target in outgoing.sectors:
        count=len(target.energies_GeV2)
        columns,source=offsets_in[target.source.space.sector]
        raw=source.lift(np.eye(len(source.energies_GeV2)))
        image=intermediate_density_action_between(target.source.space,source.source.space,
                                                  raw,transfer,b_GeV=outgoing.pilot.width_GeV,charge=charge)
        result[start:start+count,columns]=target.project(image)
        start+=count
    return result


def matched_good_current_between(outgoing,incoming,energy_out,energy_in,transfer,*,charge):
    """Omega_out† rho_parent Omega_in, including both full excluded spin fibers."""
    op=excluded_parent_density_between(outgoing,incoming,transfer,charge=charge)
    return (outgoing.reduction.wave_operator(energy_out).conj().T
            @op@incoming.reduction.wave_operator(energy_in))


def normalized_matched_good_current_between(outgoing,incoming,energy_out,energy_in,transfer,*,charge):
    """N_out^-1/2 O_eff N_in^-1/2 for Euclidean quantum-register states."""
    from .m2_matched_h2_reduction import _inverse_sqrt
    left=_inverse_sqrt(outgoing.reduction.norm_metric(energy_out))
    right=_inverse_sqrt(incoming.reduction.norm_metric(energy_in))
    return left@matched_good_current_between(outgoing,incoming,energy_out,energy_in,transfer,charge=charge)@right
