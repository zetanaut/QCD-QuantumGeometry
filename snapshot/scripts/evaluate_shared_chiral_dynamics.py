"""Replay one self-consistent nonproduction chiral determinant experiment."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from deuteron_wigner.quantum.canonical_lf_fields import GAMMA, slash
from deuteron_wigner.quantum.chiral_soft_matching import G5
from deuteron_wigner.quantum.nonlocal_source import ETA
from deuteron_wigner.quantum.shared_chiral_dynamics import (
    CovariantPV, RadialInteraction, SharedChiralAction,
)


def conditional_calibration(cutoff, target_mass=.14, target_f=.0924):
    """Illustrative two-target inverse problem, not authenticated data fitting.

    Select the declared small-M bracket, not a uniqueness claim over all
    possible branches. Solve the decay-residue equation and reconstruct a,m
    from pole/gap; then independently re-solve the forward equations.
    """
    r, s = CovariantPV(cutoff), target_mass**2
    def decay(mass):
        i2 = r.bubble(mass,s)
        z = 12*(i2+s*r.bubble(mass,s,derivative=True))
        return 12*mass*i2/np.sqrt(z)
    bracket = [target_mass/2+.001, .6*cutoff]
    mass = brentq(lambda m: decay(m)-target_f, *bracket, xtol=1e-14)
    a = 24*r.i1(mass*mass)+12*s*r.bubble(mass,s)
    current_mass = mass*12*s*r.bubble(mass,s)/a
    action = SharedChiralAction(r,RadialInteraction(a),current_mass,3)
    forward = action.solve_pion(action.solve_mass(2.))
    return dict(cutoff_GeV=cutoff, a_GeV2=a, G_GeV_minus2=1/(2*a),
                current_mass_GeV=current_mass, inverse_mass_bracket_GeV=bracket,
                forward=asdict(forward), target_mass_GeV=target_mass,
                target_decay_constant_GeV=target_f)


def evaluate():
    action = SharedChiralAction(CovariantPV(.8),RadialInteraction(.1),.004,3)
    mass = action.solve_mass(2.)
    pion = action.solve_pion(mass)
    chiral = replace(action,current_mass=0.)
    m0 = chiral.solve_mass(2.)
    p0 = chiral.solve_pion(m0)
    maxima = dict(gap_GeV3=0., pole_GeV2=0., PCAC_GeV3=0.,
                  direct_tadpole_GeV2=0., direct_bubble=0.,
                  independent_norm_relative=0., offshell_source_GeV3=0.,
                  quark_Ward_GeV=0., boosted_vertex=0.,
                  boosted_weighted_pseudoscalar_GeV=0.)
    gmors = []
    for current in [0., .00001, .0001, .001, .004, .008]:
        a = replace(action,current_mass=current)
        m = a.solve_mass(2.)
        p = a.solve_pion(m)
        maxima['gap_GeV3'] = max(maxima['gap_GeV3'],abs(a.gap(m)))
        maxima['pole_GeV2'] = max(maxima['pole_GeV2'],abs(a.pion_inverse(m,p.mass_squared)))
        maxima['PCAC_GeV3'] = max(maxima['PCAC_GeV3'],abs(p.mass_squared*p.decay_constant-2*current*p.pseudoscalar_residue))
        if current:
            gmors.append(dict(current_mass_GeV=current,pion_mass_GeV=p.mass,
                              GMOR_ratio=p.mass_squared*p.decay_constant**2/(-2*current*p.condensate_per_flavor)))
        for s in [-.37, -.08, 0., p.mass_squared]:
            residual = 2*current*a.pseudoscalar_mixing(m,s)-s*a.axial_mixing(m,s)-(m-current)*a.pion_inverse(m,s)
            maxima['offshell_source_GeV3'] = max(maxima['offshell_source_GeV3'],abs(residual))
            for eta in ([.5] if s>0 else [0., .23, .5, 1.]):
                i1,i2 = a.regulator.direct_euclidean_integrals(m,-s,routing=eta)
                maxima['direct_tadpole_GeV2'] = max(maxima['direct_tadpole_GeV2'],abs(i1-a.regulator.i1(m*m)))
                maxima['direct_bubble'] = max(maxima['direct_bubble'],abs(i2-a.regulator.bubble(m,s)))
        ds = 1e-5
        r = a.regulator
        i2 = r.direct_euclidean_integrals(m,-p.mass_squared)[1]
        derivative = (r.direct_euclidean_integrals(m,-p.mass_squared-ds)[1]
                      -r.direct_euclidean_integrals(m,-p.mass_squared+ds)[1])/(2*ds)
        independent = 12*(i2+p.mass_squared*derivative)
        maxima['independent_norm_relative'] = max(maxima['independent_norm_relative'],abs(independent/p.norm-1))
    rng = np.random.default_rng(101926)
    for _ in range(24):
        k0 = rng.normal(0,.14,4)
        q = rng.normal(0,.17,4)
        q[0] = .2*np.linalg.norm(q[1:])
        k1 = k0+q
        axial, weighted = action.quark_source_vertices(mass,k1,k0)
        rhs = (slash(k1)-mass*np.eye(4))@G5+G5@(slash(k0)-mass*np.eye(4))+weighted
        maxima['quark_Ward_GeV'] = max(maxima['quark_Ward_GeV'],float(np.linalg.norm(np.einsum("m,mij->ij",q@ETA,axial)-rhs)))
        boost = rng.normal(0,.25,3)
        gen = np.zeros((4,4))
        gen[0,1:] = gen[1:,0] = boost
        lorentz = expm(gen)
        spinor = expm(sum(boost[i]*GAMMA[0]@GAMMA[i+1]/2 for i in range(3)))
        inverse = np.linalg.inv(spinor)
        moved,moved_p = action.quark_source_vertices(mass,lorentz@k1,lorentz@k0)
        maxima['boosted_vertex'] = max(maxima['boosted_vertex'],float(np.linalg.norm(moved-np.einsum("mn,nij->mij",lorentz,spinor@axial@inverse))))
        maxima['boosted_weighted_pseudoscalar_GeV'] = max(maxima['boosted_weighted_pseudoscalar_GeV'],float(np.linalg.norm(moved_p-spinor@weighted@inverse)))
    deformations = []
    for lam in [0., .4, 2.]:
        b = replace(action,interaction=RadialInteraction(.1,lam,(mass-.004)**2/2))
        p = b.solve_pion(b.solve_mass(2.))
        deformations.append(dict(quartic=lam,solution=asdict(p),
                                 scalar_susceptibility_GeV_minus2=1/p.scalar_curvature))
    convergence = []
    for n in [32,64,128]:
        a = replace(action,regulator=CovariantPV(.8,n))
        p = a.solve_pion(a.solve_mass(2.))
        direct = a.regulator.direct_euclidean_integrals(mass,-pion.mass_squared,nodes=n)
        convergence.append(dict(nodes=n,pion_mass_GeV=p.mass,decay_constant_GeV=p.decay_constant,
                                direct_bubble_error=abs(direct[1]-a.regulator.bubble(mass,pion.mass_squared))))
    negatives = dict(
        changed_kernel_chiral_K0_GeV2=replace(chiral,interaction=RadialInteraction(.101)).pion_inverse(m0,0),
        changed_cutoff_chiral_K0_GeV2=replace(chiral,regulator=CovariantPV(.84)).pion_inverse(m0,0),
        omitted_generated_quark_channel_Ward_max_entry_GeV=2*(mass-.004),
    )
    subcritical = replace(chiral,interaction=RadialInteraction(.15))
    subcritical_finite = replace(subcritical,current_mass=.004)
    small_m = subcritical_finite.solve_mass(2.)
    try:
        subcritical_finite.solve_pion(small_m)
    except ValueError as error:
        threshold_status = str(error)
    else:
        raise AssertionError("Negative control unexpectedly supplied a bound pion.")
    return dict(
        scope="Leading-color two-flavor NJL-type reference with a radial invariant extension; NOT production dynamics, QCD matching or a nucleon prediction.",
        units="GeV; inverse pion/scalar kernels GeV^2; norm dimensionless; condensate per flavor GeV^3; pseudoscalar residue GeV^2.",
        inputs=dict(cutoff_GeV=.8,a_GeV2=.1,G_GeV_minus2=5.,current_mass_GeV=.004,colors=3,flavors=2),
        chiral_solution=asdict(p0), finite_mass_solution=asdict(pion),
        critical_a_GeV2=24*action.regulator.i1(0), maxima=maxima,
        GMOR_chiral_approach=gmors, radial_deformations=deformations,
        conditional_pion_calibrations=[conditional_calibration(l) for l in [.7,.85,1.]],
        quadrature_convergence=convergence, negatives=negatives,
        subcritical=dict(chiral_mass_GeV=subcritical.solve_mass(2.),
                         finite_quark_mass_GeV=small_m,bound_pion_status=threshold_status),
        pole_subtracted_longitudinal_exchange=[
            dict(s_GeV2=s,remainder=action.pole_subtracted_axial_exchange(pion,s)) for s in [-.01,-.1,-.25]],
        decision="Retain the common-action derivative and normalized-pole constraints; do not promote contact dynamics or infer p/n from pion agreement. Next qualify a momentum-resolved, source-gauged chiral interaction against canonical amplitudes with retained subtraction; broader matching must constrain radial/transverse freedom.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = evaluate()
    rendered = json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
