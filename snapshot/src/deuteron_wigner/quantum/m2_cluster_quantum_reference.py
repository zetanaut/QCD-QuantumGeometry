"""Explicit finite H2 reference for quantum readout; no physical activation.

Raw g^0..g^4 current coefficients are normalized with BOTH full metrics.
Reference energies are linearization anchors, not self-consistent masses.
The old M2o and bare-current quantum contracts are deliberately unchanged.
"""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .m2_gaussian_cluster_h2 import (
    GaussianClusterH2, GaussianClusterSnapshot, build_cluster_pilot)
from .m2_gaussian_cluster_four_current import ClusterH2FourCurrent
from .m2_gaussian_cluster_h2_current import _endpoints
from .m2_smooth_longitudinal_projector import SMOOTH_PRESCRIPTION
from .m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
from .m2_total_isospin import require_half_isospin_H2_snapshot
from .m2_gaussian_field_regulator import GaussianClusterRegulator
from .m2_gaussian_reference_shape import GaussianReferenceShape
from .m2_gaussian_cluster_wick import GaussianClusterWick
from .m2_canonical_good_current import good_current_density_between, parton_charge


COMPONENTS = ("plus", "minus", "x", "y")


def _readonly(value):
    result = np.array(value, dtype=complex, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("finite matrix required")
    result.setflags(write=False)
    return result


def hamiltonian_coefficients(snapshot):
    """H_eff coefficients from the already bound H2 wave recurrence."""
    m, p = snapshot.owner, snapshot.owner.pilot
    a, b, c = snapshot.five_wave_coefficients
    v, w = m.qq.sources_by_power[1], m.qq.sources_by_power[2]
    h = {0:p.h0, 1:p.emission_per_gs,
         2:p.cross_pairs_per_gs2+p.instantaneous_fermion_per_gs2+p.wick_mass_per_gs2}
    h[2] = h[2]+sum(v[n].conj().T@a[n] for n in v)
    h[3] = sum(v[n].conj().T@b[n]+w[n].conj().T@a[n] for n in v)
    h[4] = sum(v[n].conj().T@c[n]+w[n].conj().T@b[n] for n in v)
    h[4] += sum(m.six.W[n].conj().T@x for n,x in snapshot.six_wave_coefficient.items())
    return h


def _close(actual, expected, label, *, atol=5e-13):
    if np.shape(actual) != np.shape(expected) or not np.allclose(
            actual, expected, rtol=3e-12, atol=atol):
        raise ValueError("reference mismatch: "+label)


@dataclass(frozen=True)
class ClusterQuantumReference:
    outgoing: GaussianClusterSnapshot
    incoming: GaussianClusterSnapshot
    coefficients: np.ndarray
    metadata: object
    ward_residuals: tuple
    physical = False

    @property
    def raw_current(self):
        return sum(self.incoming.owner.pilot.coupling**n*c
                   for n,c in enumerate(self.coefficients))

    @property
    def normalized_coefficients(self):
        return np.einsum("ab,nmbc,cd->nmad", self.outgoing.whitener,
                         self.coefficients, self.incoming.whitener, optimize=True)

    @property
    def operators(self):
        """Twenty transported raw coefficients and four resummed totals."""
        values = self.normalized_coefficients
        result = {f"g{n}_{mu}":values[n,j] for n in range(5)
                  for j,mu in enumerate(COMPONENTS)}
        total = np.einsum("ab,mbc,cd->mad", self.outgoing.whitener,
                          self.raw_current, self.incoming.whitener, optimize=True)
        result.update({"total_"+mu:total[j] for j,mu in enumerate(COMPONENTS)})
        return result


def bind_cluster_reference(outgoing, incoming, coefficients, *, transfer_GeV,
        mean_total_transverse_GeV, charge, loop_flavor_charges, prescription,
        quadrature_order, smooth_prescription, field_envelope_prescription,
        isospin_prescription):
    """Bind actual snapshots, domain and conventions; never infer a physical current.

    Caller supplies source-derived coefficients, either from the full assembly
    or a separately validated saved source export. Ward closure alone does
    not authenticate transverse or minus physics.
    """
    d, mean, order = _endpoints(outgoing, incoming, transfer_GeV,
        mean_total_transverse_GeV, prescription, quadrature_order)
    if (smooth_prescription != SMOOTH_PRESCRIPTION
            or field_envelope_prescription != SPECTRAL_PRESCRIPTION):
        raise ValueError("explicit smooth and gapped spectral prescriptions required")
    expected = parton_charge(np.array([0,2]), "q", charge)
    if not np.array_equal(loop_flavor_charges, expected):
        raise ValueError("loop charges must match both physical quark flavors")
    for s in (outgoing,incoming):
        require_half_isospin_H2_snapshot(s, prescription=isospin_prescription)
        eig, vec = np.linalg.eigh(s.metric)
        if eig[0] <= 0:
            raise ValueError("positive endpoint metric required")
        w = (vec/np.sqrt(eig))@vec.conj().T
        _close(s.whitener, w, "endpoint whitener")
        _close(s.matrix, w@(s.effective_hamiltonian+s.energy_GeV2*(
            s.metric-np.eye(len(w))))@w, "linearized solver Hamiltonian")
    c = _readonly(coefficients)
    shape = (5,4,len(outgoing.matrix),len(incoming.matrix))
    if c.shape != shape:
        raise ValueError("all five raw orders and four endpoint-shaped components required")
    pf, pi = outgoing.owner.pilot, incoming.owner.pilot
    rho = good_current_density_between(pf,pi,d,charge=charge)
    hf, hi = map(hamiltonian_coefficients,(outgoing,incoming))
    residuals = []
    for n in range(5):
        expected = hf[n]@rho-rho@hi[n]
        if n:
            expected += (outgoing.energy_GeV2-incoming.energy_GeV2)*c[n,0]
        # At nonzero mean momentum the kinematic plus contribution must stay.
        residue = (2*pi.total_pplus_GeV*np.einsum("j,jab->ab",d,c[n,2:])
                   -2*np.dot(mean,d)*c[n,0]-expected)
        _close(residue, np.zeros_like(residue), "raw Ward order "+str(n))
        residuals.append(float(np.linalg.norm(residue)))
    for s,h in ((outgoing,hf),(incoming,hi)):
        _close(sum(pi.coupling**n*h[n] for n in range(5)),
               s.effective_hamiltonian, "Hamiltonian order sum")
    metadata = dict(target=pi.target, jz_out_twice=pf.jz_twice,
        jz_in_twice=pi.jz_twice, total_k2=pi.total_k2, retained_cutoff=pi.cutoff,
        energy_out_GeV2=outgoing.energy_GeV2, energy_in_GeV2=incoming.energy_GeV2,
        transfer_GeV=tuple(d), mean_total_transverse_GeV=tuple(mean),
        charge=charge, loop_flavor_charges=tuple(loop_flavor_charges),
        physical_leg_prescription=prescription, quadrature_order=order,
        smooth_prescription=smooth_prescription, field_envelope_prescription=field_envelope_prescription,
        isospin_prescription=isospin_prescription, component_order=COMPONENTS,
        normalization="Jmu/(2Pplus); raw order g4 before both full norm maps",
        physical=False)
    return ClusterQuantumReference(outgoing,incoming,c,MappingProxyType(metadata),tuple(residuals))


def reference_from_current(current):
    if not isinstance(current,ClusterH2FourCurrent):
        raise TypeError("full explicitly bound ClusterH2FourCurrent required")
    p = current.physical_numerator
    return bind_cluster_reference(p.outgoing,p.incoming,
        np.stack([current.coefficients[n] for n in range(5)]),
        transfer_GeV=p.transfer_GeV, mean_total_transverse_GeV=p.mean_total_transverse_GeV,
        charge=p.charge, loop_flavor_charges=p.loop_flavor_charges,
        prescription=p.prescription, quadrature_order=p.quadrature_order,
        smooth_prescription=SMOOTH_PRESCRIPTION, field_envelope_prescription=SPECTRAL_PRESCRIPTION,
        isospin_prescription=current.isospin_prescription)


def load_saved_cluster_reference(report_path, *, charge, loop_flavor_charges,
        mean_total_transverse_GeV, quadrature_order):
    """Rebuild source-owned H/N/bases before binding an existing eleven-array export.

    The original benchmark's report omits mean momentum, U(1) and quadrature:
    these must be supplied explicitly from its documented source, not guessed.
    This loader supports that benchmark export, not arbitrary external files.
    """
    path = Path(report_path)
    info = json.loads(path.read_text())
    params = info["parameters"]
    matrix_file = Path(info["full_matrix_file"])
    if matrix_file.name != str(matrix_file) or matrix_file.suffix != ".npz":
        raise ValueError("adjacent NPZ basename required")
    if (info["component_order"] != list(COMPONENTS)
            or info["current_normalization"] != "Jmu/(2Pplus); order g4 selected before both norm maps"
            or info["physical"] is not False
            or info["finite_conditions_are_uncalibrated_test_values"] is not True):
        raise ValueError("explicit nonphysical finite-model export and normalization required")
    # This legacy export has a source-defined same-helicity, electromagnetic,
    # zero-mean benchmark. Different prescriptions need a live full assembly.
    if (charge != "electromagnetic" or tuple(loop_flavor_charges) != (2/3,-1/3)
            or tuple(mean_total_transverse_GeV) != (0.,0.) or quadrature_order != 8):
        raise ValueError("saved benchmark uses explicit electromagnetic/zero-mean/order-eight controls")
    shape = (GaussianReferenceShape(base_width_GeV=params["width"],
        reference_width_GeV=params["reference_width"],degree=params["shape_degree"])
        if "shape_degree" in params else None)
    reg = GaussianClusterRegulator(width_GeV=params["width"],boson_modes=params["nb"],
        fermion_modes=params["nf"],reference_shape=shape)
    wick = GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=params["muq"],finite_gluon_mass2_per_gs2_GeV2=params["mug"])
    pilot = build_cluster_pilot(total_k2=info["total_k2"],cutoff=info["retained_cutoff"],
        target=params["target"],jz_twice=info["jz_twice"],b_GeV=info["b_GeV"],
        quark_mass_GeV=info["quark_mass_GeV"],gluon_mass_GeV=info["gluon_mass_GeV"],
        coupling=info["coupling"],regulator=reg,total_pplus_GeV=info["total_pplus_GeV"],wick_operators=wick)
    model = GaussianClusterH2(pilot,excluded_cutoff=params["q5"],six_cutoff=params["q6"])
    f,i = model.snapshot(info["energy_out_GeV2"]),model.snapshot(info["energy_in_GeV2"])
    with np.load(path.parent/matrix_file,allow_pickle=False) as archive:
        a = {k:archive[k] for k in archive.files}
    expected = {"H_out","H_in","N_out","N_in","raw_current","normalized_current"}
    expected.update("current_power_"+str(n) for n in range(5))
    if set(a) != expected:
        raise ValueError("exactly the eleven source-export arrays required")
    for name,s in (("out",f),("in",i)):
        _close(a["H_"+name],s.effective_hamiltonian,"saved H "+name,atol=3e-14)
        _close(a["N_"+name],s.metric,"saved N "+name,atol=3e-15)
    reference = bind_cluster_reference(f,i,np.stack([a["current_power_"+str(n)] for n in range(5)]),
        transfer_GeV=info["transfer_GeV"],mean_total_transverse_GeV=mean_total_transverse_GeV,
        charge=charge,loop_flavor_charges=loop_flavor_charges,
        prescription=info["physical_leg_prescription"],quadrature_order=quadrature_order,
        smooth_prescription=info["smooth_prescription"],
        field_envelope_prescription=info["field_envelope_prescription"],
        isospin_prescription=info["isospin_prescription"])
    _close(reference.raw_current,a["raw_current"],"saved raw sum",atol=3e-14)
    _close(np.stack([reference.operators["total_"+mu] for mu in COMPONENTS]),
           a["normalized_current"],"saved both-norm current",atol=5e-14)
    return reference
