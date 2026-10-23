"""Joint scalar calibration / withheld vector comparison for a bounded action."""
import argparse
import json
from pathlib import Path
import numpy as np

from deuteron_wigner.quantum.common_action_pilot import calibrated_action, scalar_elimination, minimal_vector_current
from deuteron_wigner.quantum.canonical_lf_fields import slash
from deuteron_wigner.quantum.chiral_soft_matching import G5


def evaluate():
    # This imports published summary targets, not raw correlated lattice
    # ensembles or a multi-dataset precision likelihood.
    target = dict(pion_mass_GeV=.1348, pion_mass_error_GeV=.0003,
                  mass_convention="Isospin-limit point used in arXiv:2503.20687, section VI; not charged physical mass",
                  decay_constant_GeV=.0922,
                  decay_constant_convention="Explicit illustrative isovector tau/2 normalization; not a newly extracted precision datum",
                  scalar_radius_squared_fm2=.515, scalar_stat_fm2=.028,
                  scalar_sys_fm2=.032, scalar_total_fm2=.043,
                  scalar_source="https://arxiv.org/abs/2503.20687",
                  scalar_operator="u-bar u + d-bar d; Nf=2+1 lattice result, normalized scalar slope; strange effects not explicitly resolved by this two-flavor pilot",
                  withheld_vector_radius_fm=.659, withheld_vector_error_fm=.004,
                  vector_source="https://pdglive.lbl.gov/DataBlock.action?node=S008CR",
                  matching_window_GeV2=[0., .1],
                  likelihood="No cross-dataset covariance, absolute scalar normalization, precision QED/isospin matching, or statistical exclusion significance claimed")
    cases = []
    for angle, ratio, coupling in [(.35, .8, .1), (.6, 1.4, .12), (1., .8, .2)]:
        a = calibrated_action(.1348, .0922, .515, .004, angle, ratio, coupling, [3., 1.])
        s = a.solve(.004); v, e = s["vacuum"], s["pion_modes"][:, 0]
        hp, hs = a.matrices(v)
        f0 = a.scalar_form_factor(v, e, 0.)
        qs = [0., .02, .05, .1]
        ff = [a.scalar_form_factor(v, e, q)/f0 for q in qs]
        dmass = 2e-5
        mh = lambda m: a.solve(m)["pion_mass_squared"][0]
        fh = (-mh(.004+2*dmass)+8*mh(.004+dmass)-8*mh(.004-dmass)+mh(.004-2*dmass))/(12*dmass)
        source = a.scalar_vertex(v, e)
        eliminated = [scalar_elimination(hs, a.scalar_source, source, -q) for q in qs]
        full = [a.scalar_form_factor(v, e, q) for q in qs]
        neg = [a.scalar_source[0]*source[0]/((hs[0, 0]+q)-hs[0, 1]**2/(hs[1, 1]+q)) for q in qs]
        current = []
        for q2 in [.02, .05, .1]:
            q = np.sqrt(q2); energy = np.sqrt(s["pion_mass_squared"][0]+q2/4)
            p0, p1 = np.array([energy, -q/2, 0., 0.]), np.array([energy, q/2, 0., 0.])
            j = minimal_vector_current(p1, p0, e, e)
            current.append(float(j[0]/(2*energy)))
        p0, p1 = np.array([.2, .02, .07, .03]), np.array([.21, .32, .15, -.1])
        axial, p = a.quark_sources(v, .004, p1, p0); q = p1-p0
        lhs = q[0]*axial[0]-sum(q[i]*axial[i] for i in [1, 2, 3])
        rhs = (slash(p1)-s["quark_mass"]*np.eye(4))@G5+G5@(slash(p0)-s["quark_mass"]*np.eye(4))+p
        chiral = a.solve(0.)
        cases.append(dict(angle=angle, quartic_ratio=ratio, coupling_GeV2=coupling,
            quartic=a.quartic.tolist(), quartic_over_16pi2=(a.quartic/(16*np.pi**2)).tolist(),
            radial_squared_GeV2=a.radial_squared.tolist(), scalar_source_GeV2=a.scalar_source.tolist(),
            yukawa=a.yukawa.tolist(), mass_source_GeV=.004,
            source_scheme="Unmatched exploratory mass/source coordinate; not assigned MSbar by assumption",
            vacuum_GeV=v.tolist(), pion_mass_squared_GeV2=s["pion_mass_squared"].tolist(),
            scalar_mass_squared_GeV2=s["scalar_mass_squared"].tolist(), decay_constants_GeV=s["decay_constants"].tolist(),
            pseudoscalar_residues_GeV2=s["pseudoscalar_residues"].tolist(), tree_quark_mass_GeV=s["quark_mass"],
            gap_norm_GeV3=s["gap_norm"], chiral_pion_mass_squared_GeV2=float(chiral["pion_mass_squared"][0]),
            scalar_F0_GeV=f0, scalar_radius_squared_fm2=a.scalar_radius_squared(v, e),
            scalar_Q2_GeV2=qs, normalized_scalar_F=ff, scalar_Feynman_Hellmann_error_GeV=abs(fh-f0),
            exact_elimination_max_error_GeV=float(np.max(abs(np.array(eliminated)-full))),
            frozen_sources_relative_error=float(np.linalg.norm(np.array(neg)-full)/np.linalg.norm(full)),
            quark_axial_Ward_max_error_GeV=float(np.max(abs(lhs-rhs))),
            pion_vector_F=current, vector_radius_squared_fm2=0.,
            vector_zero_is="Derived exact tree result for local potential + constant kinetic metric; missing physical interactions are not set to zero"))
    sensitivity = []
    for radius in [.515-.043, .515+.043]:
        a = calibrated_action(.1348, .0922, radius, .004, .6, 1.4, .12, [3., 1.])
        s = a.solve(.004)
        sensitivity.append(dict(target_scalar_radius_squared_fm2=radius, quartic=a.quartic.tolist(),
                                scalar_mass_squared_GeV2=s["scalar_mass_squared"].tolist()))
    return dict(scope="Local coupled linear-sigma/quark-meson tree candidate. No new composite-q-qbar bound pion, quantum quark self-energy, production QCD matching or nucleon prediction.",
        targets=target, cases=cases, scalar_one_sigma_summary_sensitivity=sensitivity,
        decision="Do not promote or expand this potential-only tree candidate for joint physics. Preserve its common-action/source checks; next qualify an independent transverse, chirally/gauge-invariant source interaction (or its matched dynamical origin) within the shared action. No loop-repair program or independent observable head.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path); args = parser.parse_args()
    rendered = json.dumps(evaluate(), indent=2, allow_nan=False)+"\n"
    if args.output: args.output.write_text(rendered)
    print(rendered)
