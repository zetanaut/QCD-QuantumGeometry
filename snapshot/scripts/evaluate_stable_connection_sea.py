"""Replay constructive stability, chiral and shared connection/sea controls."""
import json
from dataclasses import asdict, replace

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize, minimize_scalar

from deuteron_wigner.quantum.stable_connection_sea import (
    StableConnectionSea, chiral_matrix, connection_profile, dirac_jet,
    full_logdet, negative_logdet_jet, sea_kernel,
    retained_dirac_jet, dirac_determinant_jet,
    texture_hamiltonian,
)


def fit_slice(action):
    attempts = []
    for theta in (-5.,0.,5.):
        for sigma in (.1,.7,1.5):
            fit = minimize(lambda x:action.jet(*x)[:2],[sigma,theta],jac=True,
                           method='L-BFGS-B',bounds=[(0,3),(-2*np.pi,2*np.pi)],
                           options={'ftol':1e-14,'gtol':1e-9,'maxiter':200})
            attempts.append(dict(start=[sigma,theta],success=bool(fit.success),
                                 point=fit.x.tolist(),action=float(fit.fun)))
    best = min(attempts,key=lambda r:r['action'])
    value, gradient, hessian = action.jet(*best['point'])
    return dict(attempts=attempts,best_point=best['point'],action=value,
                gradient_norm=float(np.linalg.norm(gradient)),
                slice_hessian_eigenvalues=np.linalg.eigvalsh(hessian).tolist(),
                all_attempts_success=all(r['success'] for r in attempts),
                maximum_action_spread=max(r['action'] for r in attempts)-value)


def run():
    action = StableConnectionSea()
    unquenched = fit_slice(action)
    quenched = fit_slice(replace(action,sea_weight=0.))
    sigma, theta = unquenched['best_point']
    d, dt, dtt = dirac_jet(connection_profile(theta))
    phi = [sigma,.13,-.29,.07]
    kernel = sea_kernel(d,phi,action.mass)
    full = chiral_matrix(d,phi,action.mass)
    logabs, phase = full_logdet(full)
    determinant_error = abs(logabs+negative_logdet_jet(kernel)[0])
    h = 2e-4
    x = np.array([.63,.43])
    v, gradient, hessian = action.jet(*x)
    gradient_error = []
    hessian_error = []
    for delta in h*np.eye(2):
        vp,gp,_ = action.jet(*(x+delta))
        vm,gm,_ = action.jet(*(x-delta))
        gradient_error.append((vp-vm)/(2*h))
        hessian_error.append((gp-gm)/(2*h))
    chiral = replace(action,mass=0.)
    radius = minimize_scalar(lambda r:chiral.homogeneous_value([r,0,0,0],.43),
                             bounds=(.05,2.),method='bounded',options={'xatol':1e-12}).x
    rng = np.random.default_rng(92)
    orbit = rng.normal(size=(12,4)); orbit *= radius/np.linalg.norm(orbit,axis=1)[:,None]
    base = chiral.homogeneous_value([radius,0,0,0],.43)
    curvature = 2*(chiral.homogeneous_value([radius,h,0,0],.43)-base)/h**2
    k = sea_kernel(d,[sigma,0,0,0],action.mass)
    kt = dt.conj().T@d+d.conj().T@dt
    ktt = dtt.conj().T@d+2*dt.conj().T@dt+d.conj().T@dtt
    _, link_response, link_hessian = negative_logdet_jet(k,[kt],[[ktt]],action.sea_weight)
    _, _, omitted_contact = negative_logdet_jet(k,[kt],weight=action.sea_weight)
    k2 = sea_kernel(dirac_jet(connection_profile(-.6,41))[0],[sigma,0,0,0],action.mass)
    joint = block_diag(k,k2); zero = np.zeros_like(k)
    jvalue,_,jh = negative_logdet_jet(joint,[block_diag(np.eye(96),zero),block_diag(zero,np.eye(96))])
    retained = np.concatenate([np.arange(24*n,24*(n+1)) for n in (0,2,4,6)])
    jets = (full,np.kron(dt,np.eye(2))+.3*np.eye(192),np.kron(dtt,np.eye(2)))
    effective,residual = retained_dirac_jet(jets,retained)
    direct,eff,rest = [dirac_determinant_jet(j) for j in (jets,effective,residual)]
    fields = .7*np.random.default_rng(151).normal(size=(8,4))
    htexture = texture_hamiltonian(connection_profile(.43),fields,mass=.06)
    return dict(
        preedit_commit='5254a42815fa5379647bdbd46fcebd0caa42fca6',
        scope='Sufficient bounded kinetic/coercive potential family; actual finite connection-dependent Nf2 determinant and common responses. Not a physical vacuum/baryon regulator.',
        units='dimensionless reference-cell units; no parameters fitted to physical data',
        parameters=asdict(action),
        carrier=dict(nodes=8,links=12,spin=4,color=3,flavor=2,full_D_dimension=192,
                     role='algebraic graph, not continuum/anomaly-correct lattice or LF dynamics'),
        kinetic_gate=dict(bounds=action.gate.bounds(),reference_value=action.gate.jet(.5)[0],
                          canonical_momentum='Pi=kappa E; Gauss uses D_i Pi_i, not D_i E_i',
                          rejected_linear_gate_at_rho_2=-1.,
                          zero_floor_confinement_mechanism_not_claimed=True),
        full_two_flavor_determinant=dict(logabs=logabs,kernel_identity_norm=float(np.linalg.norm(full.conj().T@full-np.kron(kernel,np.eye(2)))),
                                        independent_logdet_error=determinant_error,phase_error=float(abs(phase-1)),
                                        extra_flavor_factor=False,homogeneous_chiral_scope=True),
        unquenched_two_coordinate_slice=unquenched,
        quenched_two_coordinate_slice=quenched,
        sea_shift=(np.array(unquenched['best_point'])-quenched['best_point']).tolist(),
        source_checks=dict(gradient_fd_max_error=float(np.max(np.abs(np.array(gradient_error)-gradient))),
                           hessian_fd_max_error=float(np.max(np.abs(np.array(hessian_error).T-hessian))),
                           mixed_response=float(hessian[0,1]),
                           mixed_response_without_sea=float(replace(action,sea_weight=0.).jet(*x)[2][0,1]),
                           sea_link_gradient=float(link_response[0]),
                           sea_link_hessian=float(link_hessian[0,0]),
                           sea_link_hessian_with_contact_omitted=float(omitted_contact[0,0])),
        non_singlet_chiral=dict(mass=0.,fixed_theta=.43,stationary_radius=float(radius),
                               gap_residual=float(abs(chiral.jet(radius,.43)[1][0])),
                               orbit_action_spread=float(max(abs(chiral.homogeneous_value(p,.43)-base) for p in orbit)),
                               pion_curvature_fd=float(curvature),
                               finite_orientation_average_norm=float(np.linalg.norm(np.mean(np.r_[orbit,-orbit],axis=0))),
                               physical_spontaneous_condensate_claim=False,U1_A_anomaly_claim=False),
        radial_coercivity_scan=[dict(radius=r,action=action.jet(r,.4)[0]) for r in (2.,4.,8.,16.)],
        disconnected_modules=dict(logdet_additivity_error=abs(jvalue-negative_logdet_jet(k)[0]-negative_logdet_jet(k2)[0]),
                                  mixed_source_response=float(jh[0,1]),physical_nuclear_cluster_claim=False),
        exact_fermionic_retained_matching=dict(retained_full_spin_color_flavor_dimension=len(retained),
            logdet_split_error=abs(direct[0]-eff[0]-rest[0]),
            first_source_split_error=float(abs(direct[2]-eff[2]-rest[2])),
            second_source_split_error=float(abs(direct[3]-eff[3]-rest[3])),
            incorrect_full_plus_retained_first_source_excess=float(abs(eff[2])),
            full_gluon_or_LF_Fock_matching_claim=False),
        arbitrary_pion_texture_operator=dict(hermiticity_error=float(np.linalg.norm(htexture-htexture.conj().T)),
            single_particle_norm=float(np.max(np.abs(np.linalg.eigvalsh(htexture)))),
            single_particle_bound=float(3*.4+.06+np.linalg.norm(fields,axis=1).max()),
            finite_Fock_plus_quartic_coercive_bound_derived=True,
            finite_gauge_field_quantization_performed=False,
            euclidean_to_real_time_equivalence_claim=False),
        limitations=['Not full connection-space minimization or a quantum ensemble',
                     'No positivity claim for arbitrary pion textures or chemical potential',
                     'No continuum renormalization, anomaly, real-time/LF reconstruction or baryon binding',
                     'Color-link, mass and algebraic bilocal sources are not a complete matched electromagnetic/metric current',
                     'Exact finite fermionic Schur matching is not full gluonic/renormalized LF matching'],
        decision='Retain the stable connection/chiral/sea shared-action class as a working exploratory candidate; next qualify full field-space textures, physical regulator/anomaly and canonical LF matching before a physical p/n solve',
        physical_model_complete=False,public_push=False)


if __name__ == '__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))
