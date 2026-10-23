"""Replay tree-level soft-pion matching and its physical scope diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from deuteron_wigner.quantum.chiral_soft_matching import (
    ETA, GAMMA, G5, PionPole, PolynomialQuarkDressing, square,
    pion_charge_generator, static_one_pion_exchange,
)


def evaluate():
    pion = PionPole(.14, .0924)
    rng = np.random.default_rng(1047)
    maxima = dict(explicit_nonlocal=0., PCAC=0., vector_Ward=0.,
                  action_source_difference=0., Lorentz_response=0.,
                  Lorentz_source=0., LF_relative=0., quark_axial_Ward=0.)
    negatives = dict(omit_axial_contact=[], omit_vector_contacts=[], bare_dressed_axial=[])
    dressing = PolynomialQuarkDressing((1., .12), (.28, .025), .8)
    for _ in range(24):
        def momentum():
            spatial = rng.normal(size=3)
            spatial *= rng.uniform(.025, .18)/np.linalg.norm(spatial)
            return np.r_[rng.uniform(-.25, .25)*np.linalg.norm(spatial), spatial]
        k0, k1 = momentum(), momentum()
        response = pion.axial_response(k0)
        source = pion.axial_vector_source(k1, k0)
        pcac = pion.decay_constant**2*pion.mass**2*k0*pion.propagator(square(k0))
        ward = pion.axial_response(k1)-response
        maxima['explicit_nonlocal'] = max(maxima['explicit_nonlocal'], np.linalg.norm(response-pion.explicit_response(k0)))
        maxima['PCAC'] = max(maxima['PCAC'], np.linalg.norm(k0@ETA@response-pcac))
        maxima['vector_Ward'] = max(maxima['vector_Ward'], np.linalg.norm(np.einsum('m,mab->ab', (k1-k0)@ETA, source)-ward))
        negatives['omit_axial_contact'].append(np.linalg.norm(k0@ETA@(response-pion.decay_constant**2*ETA)-pcac))
        without = pion.axial_vector_source(k1, k0, include_contacts=False)
        negatives['omit_vector_contacts'].append(np.linalg.norm(np.einsum('m,mab->ab', (k1-k0)@ETA, without)-ward))
        for mu in range(4):
            epsilon = 1e-6
            diff = (pion.explicit_response_in_vector_source(k1, k0, mu, epsilon)-
                    pion.explicit_response_in_vector_source(k1, k0, mu, -epsilon))/(2*epsilon)
            maxima['action_source_difference'] = max(maxima['action_source_difference'], np.linalg.norm(diff-source[mu]))
        boost = rng.normal(size=3)*.3
        generator = np.zeros((4, 4)); generator[0, 1:] = generator[1:, 0] = boost
        transform = expm(generator)
        maxima['Lorentz_response'] = max(maxima['Lorentz_response'], np.linalg.norm(pion.axial_response(transform@k0)-transform@response@transform.T))
        expected = np.einsum('mi,aj,bk,ijk->mab', transform, transform, transform, source)
        maxima['Lorentz_source'] = max(maxima['Lorentz_source'], np.linalg.norm(pion.axial_vector_source(transform@k1, transform@k0)-expected))
        maxima['LF_relative'] = max(maxima['LF_relative'], abs(pion.light_front_propagator(k0)/pion.propagator(square(k0))-1))
        axial, weighted = dressing.axial_vertices(k1, k0, .0035, pion)
        rhs = dressing.inverse(k1, .0035)@G5+G5@dressing.inverse(k0, .0035)+weighted
        maxima['quark_axial_Ward'] = max(maxima['quark_axial_Ward'], np.linalg.norm(np.einsum('m,mij->ij', (k1-k0)@ETA, axial)-rhs))
        bare = np.array([g@G5 for g in GAMMA])
        rhs_bare = dressing.inverse(k1, .0035)@G5+G5@dressing.inverse(k0, .0035)+.007*G5
        negatives['bare_dressed_axial'].append(np.linalg.norm(np.einsum('m,mij->ij', (k1-k0)@ETA, bare)-rhs_bare))
    expansions = []
    for q in (.035, .07, .10, .14, .21, .28):
        exact = pion.propagator(-q*q)
        expansions.append({'spatial_momentum_GeV':q, 'ratio_to_pion_mass':q/pion.mass,
                           'relative_propagator_errors':{str(n):abs(pion.propagator(-q*q, n)/exact-1) for n in (0, 1, 2, 4)}})
    k0, k1 = np.array([.04, .08, -.03, .02]), np.array([-.02, .02, .1, -.04])
    truncation = []
    for n in (0, 1, 2, 4):
        good = pion.axial_vector_source(k1, k0, n)
        bad = good + pion.decay_constant**2*np.einsum('m,a,b->mab', k1+k0, k1, k0)*(
            pion.propagator(square(k1), n)*pion.propagator(square(k0), n)-
            pion.divided_difference(square(k1), square(k0), n))
        rhs = pion.axial_response(k1, n)-pion.axial_response(k0, n)
        truncation.append({'order':n,
                           'correct_vector_Ward_error':float(np.linalg.norm(np.einsum('m,mab->ab', (k1-k0)@ETA, good)-rhs)),
                           'two_truncated_propagators_Ward_error':float(np.linalg.norm(np.einsum('m,mab->ab', (k1-k0)@ETA, bad)-rhs)),
                           'current_relative_error_to_pole':float(np.linalg.norm(good-pion.axial_vector_source(k1, k0))/np.linalg.norm(pion.axial_vector_source(k1, k0)))})
    iso0 = np.array([0, 1, -1, 0])/np.sqrt(2)
    triplet = [np.array([1, 0, 0, 0]), np.array([0, 1, 1, 0])/np.sqrt(2), np.array([0, 0, 0, 1])]
    subspace = np.column_stack([np.kron(iso0, spin) for spin in triplet])
    ope = subspace.conj().T@static_one_pion_exchange([0., 0., .12], pion, 1.27)@subspace
    return {
        'scope':'LO quadratic-pion source matching and dressed-quark longitudinal constraint; not a physical nucleon or nuclear fit',
        'representative_inputs':{'m_pi_GeV':pion.mass, 'F_GeV':pion.decay_constant,
                                 'm_current_GeV':.0035, 'g_A_N':1.27,
                                 'status':'declared illustrative low-energy scales, not a precision data determination'},
        'random_spacelike_cases':24,
        'maximum_errors':{key:float(value) for key, value in maxima.items()},
        'error_units':'response GeV^2; source derivative GeV; PCAC and vector Ward GeV^3/GeV^2 respectively; quark Ward GeV; LF_relative dimensionless',
        'negative_control_minimum_errors':{key:float(min(value)) for key, value in negatives.items()},
        'derivative_expansion':expansions,
        'consistent_gauging_comparison':truncation,
        'pion_isospin_generator_eigenvalues':np.linalg.eigvalsh(pion_charge_generator()).tolist(),
        'I0_spin_triplet_static_OPE_diagonal_GeV_inverse2':np.diag(ope).real.tolist(),
        'architecture_conclusion':'protect the pion pole, chiral contact and source partners explicitly or nonlocally; fit matched residual content, not a duplicate pole',
        'exact_at_all_chiral_orders':False, 'quark_dressing_generated_by_dynamics':False,
        'transverse_quark_vertex_determined':False, 'bound_nucleon_covariance_solved':False,
        'production_Hamiltonian_selected':False, 'public_push':False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
