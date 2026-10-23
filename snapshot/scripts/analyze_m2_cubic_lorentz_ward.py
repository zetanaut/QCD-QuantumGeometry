"""Replay the cubic source Ward identity, regulator drift and labelled Fock spin piece."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from deuteron_wigner.quantum.m2_cubic_lorentz_ward import (
    emission_minus_vertex, emission_regular_ward, source_regulator_lorentz,
    cluster_quadratic_lorentz, apply_retained_fermion_spin,
)
from deuteron_wigner.quantum.m2_coupled_pilot import symmetry_basis
from deuteron_wigner.quantum.m2_canonical_cubic_spin import (
    canonical_emission_density, canonical_density_contact,
)
from deuteron_wigner.quantum.m2_cubic_lorentz_ward import emission_lorentz_contact
from deuteron_wigner.quantum.m2_cubic_spin_channels import (
    FERMION_CHANNELS, fermionic_cubic_density, fermionic_cubic_contact,
    fermionic_gauge_contact, gluonic_cubic_density, gluonic_cubic_contact,
    gluonic_gauge_contact, cubic_regular_ward,
)
from deuteron_wigner.quantum.m2_excluded_fock_vertices import three_gluon_numerator
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape


def flow(p, mass, axis, t):
    """Exact null Lorentz rotation, independently of the generator module."""
    q = np.array(p, float)
    pm = (mass*mass+q[1:]@q[1:])/(2*q[0])
    q[0] += t*q[axis+1]+t*t*pm/2
    q[axis+1] += t*pm
    return q


def derivative(f, h=1e-4):
    return (-f(2*h)+8*f(h)-8*f(-h)+f(-2*h))/(12*h)


def analyze_all_bare_channels(p):
    """Replay every cubic physical helicity, including both massive channels.

    Summarize maxima per channel; exact grid and parameters are retained here
    and in the result. No random sampling or fitted tolerance is involved.
    """
    rows = []
    for channel in (*FERMION_CHANNELS, "g_to_gg"):
        gluonic = channel == "g_to_gg"
        hs = list(product((-1, 1), repeat=3)) if gluonic else (-1, 1)
        masses_grid = (0.,) if gluonic else (0., .3)
        row = {"channel": channel, "mass_grid_GeV": list(masses_grid), "cases": 0,
            "bare_Ward_max_GeV_to_minus_half": 0., "product_rule_max_GeV_to_minus_half": 0.,
            "canonical_contact_max_GeV_to_minus_three_halves": 0.,
            "density_derivative_max_GeV_to_minus_three_halves": 0.,
            "smooth_regulator_defect_max_GeV_to_minus_half": 0.}
        for mass, h, axis, degree in product(masses_grid, hs, (0, 1), (None, 0, 1, 2)):
            masses = (mass, mass, 0.) if channel == "g_to_pair" else (mass, 0., mass)
            shape = None if degree is None else GaussianReferenceShape(
                base_width_GeV=.8, reference_width_GeV=1.2, degree=degree)
            reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8, fermion_modes=8, reference_shape=shape)
            def bare(q):
                return (three_gluon_numerator(*q, *h)/np.sqrt(8*np.prod(q[:, 0])) if gluonic else
                        fermionic_cubic_density(channel, q, mass, h)["Dirac"])
            def moved(t, regulated):
                q = np.array([flow(leg, m, axis, t) for leg, m in zip(p, masses)])
                r = source_regulator_lorentz(q, masses, "F"+str(axis+1), reg)[0] if regulated else 1.
                return r*bare(q)
            w0 = cubic_regular_ward(channel, p, mass, h, axis, derivative(lambda t: moved(t, False)))
            wr = cubic_regular_ward(channel, p, mass, h, axis, derivative(lambda t: moved(t, True)), regulator=reg)
            contact = (gluonic_cubic_contact(p, h, axis) if gluonic else
                       fermionic_cubic_contact(channel, p, mass, h, axis))
            z = gluonic_gauge_contact(p, h, axis) if gluonic else fermionic_gauge_contact(channel, p, h, axis)
            def density(t):
                q = p.copy()
                q[2, axis+1] -= t
                d = gluonic_cubic_density(q, h) if gluonic else fermionic_cubic_density(channel, q, mass, h)
                return d["gluon_density_difference"]+(0 if gluonic else d["fermion_density_difference"])
            expected_derivative = contact["gluon_density_derivative"]+(0 if gluonic else contact["fermion_density_derivative"])
            errors = {
                "bare_Ward_max_GeV_to_minus_half": float(np.linalg.norm(w0["residual"])),
                "product_rule_max_GeV_to_minus_half": float(np.linalg.norm(wr["residual"]-wr["smooth_regulator_defect"])),
                "canonical_contact_max_GeV_to_minus_three_halves": float(np.linalg.norm(contact["combined_contact"]-z)),
                "density_derivative_max_GeV_to_minus_three_halves": float(np.linalg.norm(derivative(density)-expected_derivative))}
            if max(errors.values()) > 1e-9:
                raise ArithmeticError("independent cubic channel identity failed: "+channel)
            errors["smooth_regulator_defect_max_GeV_to_minus_half"] = float(np.linalg.norm(wr["smooth_regulator_defect"]))
            for key, value in errors.items():
                row[key] = max(row[key], value)
            row["cases"] += 1
        rows.append(row)
    return {"channels": rows, "physical_momenta_GeV": p.tolist(),
            "shape_degrees": [None, 0, 1, 2], "regulator_width_GeV": .8,
            "reference_width_GeV": 1.2, "all_physical_helicities": True,
            "transverse_axes": [1, 2], "finite_domain_or_H2_spin_completion": False}


def analyze():
    pf, k = np.array([.7, .13, -.2]), np.array([.4, .04, .07])
    pi = pf+k
    ward, canonical = [], []
    for mass in (0., .3):
        for degree in (None, 0, 1, 2):
            shape = None if degree is None else GaussianReferenceShape(
                base_width_GeV=.8, reference_width_GeV=1.2, degree=degree)
            reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8,
                                           fermion_modes=8, reference_shape=shape)
            for hg in (-1, 1):
                for axis in (0, 1):
                    def vertex(t, regulated):
                        f, g, i = [flow(p, m, axis, t) for p, m in zip((pf, k, pi), (mass, 0., mass))]
                        c = emission_minus_vertex(f, g, i, mass, hg)
                        r = source_regulator_lorentz([f, g, i], [mass, 0., mass], "F"+str(axis+1), reg)[0]
                        return c*r if regulated else c
                    bare = emission_regular_ward(pf, k, pi, mass, hg, axis, derivative(lambda t: vertex(t, False)))
                    smooth = emission_regular_ward(pf, k, pi, mass, hg, axis,
                                                   derivative(lambda t: vertex(t, True)), regulator=reg)
                    residual = np.linalg.norm(smooth["residual"]-smooth["smooth_regulator_defect"])
                    bare_residual = np.linalg.norm(bare["residual"])
                    if max(residual, bare_residual) > 1e-9:
                        raise ArithmeticError("independent cubic Lorentz/product-rule identity failed")
                    quad = cluster_quadratic_lorentz([pf, k, pi], [mass, 0., mass], "F"+str(axis+1))
                    if degree is None:
                        reduced = canonical_density_contact(pf, k, pi, mass, hg, axis)
                        density = canonical_emission_density(pf, k, pi, mass, hg)
                        axis_vector = np.eye(3)[axis+1]
                        numerical = derivative(lambda t: sum(canonical_emission_density(
                            pf, k, pi-t*axis_vector, mass, hg)[part+"_density_difference"]
                            for part in ("fermion", "gluon")))
                        expected_derivative = reduced["fermion_density_derivative"]+reduced["gluon_density_derivative"]
                        identity = np.linalg.norm(reduced["combined_contact"]-emission_lorentz_contact(k, hg, axis))
                        derivative_error = np.linalg.norm(numerical-expected_derivative)
                        if max(identity, derivative_error) > 1e-10:
                            raise ArithmeticError("canonical Noether/coordinate-density reduction failed")
                        canonical.append({"mass_GeV": mass, "gluon_helicity": hg, "axis": axis+1,
                            "on_conservation_density_difference_norm_GeV_to_minus_half": float(np.linalg.norm(density["canonical"]-density["Dirac"])),
                            "density_derivative_residual_norm_GeV_to_minus_three_halves": float(derivative_error),
                            "combined_Noether_contact_residual_norm_GeV_to_minus_three_halves": float(identity),
                            "fermion_improvement_cancellation_norm_GeV_to_minus_three_halves": float(np.linalg.norm(reduced["fermion_improved_contact"]))})
                    ward.append({"mass_GeV": mass, "shape_degree": degree, "gluon_helicity": hg,
                        "axis": axis+1, "gap_GeV": bare["gap_GeV"],
                        "bare_Ward_residual_norm_GeV_to_minus_half": float(bare_residual),
                        "regulated_Ward_residual_norm_GeV_to_minus_half": float(np.linalg.norm(smooth["residual"])),
                        "smooth_regulator_defect_norm_GeV_to_minus_half": float(np.linalg.norm(smooth["smooth_regulator_defect"])),
                        "product_rule_residual_norm_GeV_to_minus_half": float(residual),
                        **quad})
    fock = []
    reg = GaussianClusterRegulator(width_GeV=.8, boson_modes=8, fermion_modes=8)
    for total, cutoff in ((9, 0), (7, 1)):
        for target in ("PROTON", "NEUTRON"):
            for jz in (-1, 1):
                for hg in (-1, 1):
                    inc = symmetry_basis("qqq", total, cutoff, target, jz, 0).toarray()
                    out = symmetry_basis("qqqg", total, cutoff, target, jz+2*hg, hg).toarray()
                    images = [np.column_stack([apply_retained_fermion_spin(
                        u.reshape(-1, 1, 64), total, cutoff, .3, .3, hg,
                        axis=a, regulator=reg, total_pplus_GeV=1.9).ravel() for u in inc.T]) for a in (0, 1)]
                    leakage = max(np.linalg.norm(v-out@(out.conj().T@v)) for v in images)
                    vector_residual = np.linalg.norm(hg*images[0]-1j*images[1])
                    if max(leakage, vector_residual) > 1e-12:
                        raise ArithmeticError("labelled fermion Noether piece violates its symmetry weight")
                    fock.append({"K2": total, "oscillator_cutoff": cutoff, "target": target,
                        "incoming_Jz_twice": jz, "outgoing_Jz_twice": jz+2*hg, "gluon_helicity": hg,
                        "incoming_rank": inc.shape[1], "outgoing_rank": out.shape[1],
                        "F1_fermion_piece_Frobenius_norm_per_gs": float(np.linalg.norm(images[0])),
                        "raw_sign_isospin_weight_leakage_norm": float(leakage),
                        "Jz_vector_identity_residual_norm": float(vector_residual)})
    return {"scope": "all four bare cubic channels: canonical Noether/coordinate-density reduction and Lorentz Ward identity; smooth source-regulator drift; actual retained FERMION spin-density piece only, not the full projected generator",
        "parameters": {"outgoing_quark_pplus_px_py_GeV": pf.tolist(), "gluon_pplus_px_py_GeV": k.tolist(),
            "incoming_quark_pplus_px_py_GeV": pi.tolist(), "regulator_width_GeV": .8,
            "shape_reference_width_GeV": 1.2, "field_modes_each": 8,
            "fock_total_pplus_GeV": 1.9, "fock_b_GeV": .3, "fock_mass_GeV": .3},
        "continuum_Ward_cases": ward, "canonical_Noether_cases": canonical, "projected_fermion_spin_cases": fock,
        "all_bare_cubic_channels": analyze_all_bare_channels(np.array([pf, k, pi])),
        "regulator_kernel_changed": False, "complete_interacting_spin_action": False,
        "current_covariance_repaired": False, "physical_prediction": False,
        "next": "derive the quartic/common-regulator/Wick/domain partners and higher-order Lorentz commutators, then complete actual Fock/H2/norm spin transport and multiplet/current controls"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(analyze(), indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
