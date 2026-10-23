"""Replay finite charged-collective triangle and conditional slope subtraction."""
import json
import numpy as np
from scipy.linalg import expm
from deuteron_wigner.quantum.quantum_collective_current import (
    normalized_loop_shape,lf_overlap_shape,lf_parameter_shape,loop_slope,
    bubble_vertex_norm,triangle_unit,photon_attachments,elastic_momenta,dot,
)
from deuteron_wigner.quantum.quantum_pair_threshold import feynman_nonlocal
from deuteron_wigner.quantum.nonzero_momentum_current import benchmark_reference
from deuteron_wigner.quantum.transverse_source_matching import radius_coefficient
from deuteron_wigner.quantum.common_action_pilot import HBARC


def evaluate():
    reference=benchmark_reference();m=reference.mass
    masses,modes=np.linalg.eigh(reference.pion.hessian);s=float(masses[0]);e=modes[:,0]
    g_squared=float((e@reference.pion.couplings)**2)
    slope=loop_slope(m,s);target=radius_coefficient(.659);local=target-g_squared*slope
    rows=[]
    for q2 in [0.,.02,.05,.1,.5]:
        covariant=normalized_loop_shape(m,s,q2)
        lf=lf_overlap_shape(m,s,q2);reduced=lf_parameter_shape(m,s,q2)
        refined=lf_overlap_shape(m,s,q2,order=96)
        rows.append(dict(Q2_GeV2=q2,unit_triangle_deltaF=covariant,
            unit_LF_overlap_deltaF=lf,unit_LF_integrated_deltaF=reduced,
            overlap_difference=float(abs(covariant-lf)),resolution_difference=float(abs(refined-lf)),
            loop_deltaF=g_squared*covariant,old_tree_linear_F=1-target*q2,
            old_tree_plus_loop_DOUBLE_COUNTS=1-target*q2+g_squared*covariant,
            conditionally_subtracted_one_loop_F=1-local*q2+g_squared*covariant,
            large_transfer_control_only=bool(q2>.1)))
    p1=np.array([.2,.05,.03,.02]);p0=np.array([.13,-.07,.02,-.01])
    vertex=photon_attachments(m,p1,p0,order=16)
    trace=triangle_unit(m,p1,p0,order=16,method='trace')
    inverse_difference=feynman_nonlocal(m,dot(p0,p0),'pion')-feynman_nonlocal(m,dot(p1,p1),'pion')
    generator=np.zeros((4,4));generator[0,1:]=generator[1:,0]=[.3,-.2,.25];boost=expm(generator)
    boosted=triangle_unit(m,boost@p1,boost@p0,order=16,method='trace')
    complete=lf_overlap_shape(m,s,.1);erased=lf_overlap_shape(m,s,.1,orbital=False)
    norm=bubble_vertex_norm(m,s);p=np.array([np.sqrt(s+.08),.2,0,.2])
    return dict(inputs=dict(quark_mass_GeV=m,external_pion_mass_squared_GeV2=s,light_mode_yukawa_squared=g_squared,
                colors=3,flavor_charges=[2/3,-1/3],reference='old tree kinematics/couplings only; no quantum mass/gap or physical pion solve',
                order='one free-quark loop, no gluon or collective loops, electric e stripped',
                finite_convention='same zero-invariant wave subtraction as twice-subtracted pion inverse; external charge-normalized shape at one-loop order',
                prior_radius_input_fm=.659,prior_radius_input_source='https://pdglive.lbl.gov/DataBlock.action?node=S008CR',
                prior_radius_is_reused_calibration_not_new_measurement=True,
                local_coefficient='conditional renormalized on-shell slope, NOT a replacement of bare C matrix'),
            transfer_scan=rows,
            conditional_matching=dict(unit_loop_slope_GeV_minus2=slope,
                loop_slope_GeV_minus2=g_squared*slope,loop_radius_squared_fm2=6*HBARC**2*g_squared*slope,
                prior_target_slope_GeV_minus2=target,residual_local_slope_GeV_minus2=local,
                fraction_of_prior_radius_squared=g_squared*slope/target,
                unit_wave_norm_correction=norm,projected_wave_norm_correction=g_squared*norm,
                local_C_was_modified=False,local_D_was_modified=False,
                qualification='omitted other loops/local source terms and quantum state matching prevent physical refit; two-photon matching must also be updated before using old joint C/D calibration'),
            offshell=dict(outgoing_GeV=p1.tolist(),incoming_GeV=p0.tolist(),
                loop_vertex_GeV=vertex['total'].tolist(),inverse_difference_GeV2=float(inverse_difference),
                Ward_difference_GeV2=float(abs(dot(p1-p0,vertex['total'])-inverse_difference)),
                quark_only_Ward_defect_GeV2=float(abs(dot(p1-p0,vertex['quark'])-inverse_difference)),
                trace_difference_GeV=float(np.linalg.norm(trace-vertex['total'])),
                boosted_trace_difference_GeV=float(np.linalg.norm(boosted-boost@trace))),
            negative_controls=dict(orbital_erasure_relative=abs(erased-complete)/abs(complete),
                orbital_erasure_still_charge_subtracted=True,
                unnormalized_zero_transfer_shift=g_squared*norm),
            checks=dict(maximum_overlap_difference=max(r['overlap_difference'] for r in rows),
                maximum_resolution_difference=max(r['resolution_difference'] for r in rows),
                pole_norm_vertex_difference_GeV=float(np.linalg.norm(triangle_unit(m,p,p)-2*p*norm))),
            claims=dict(independent_transverse_loop_shape=True,independent_quantum_LF_plus_overlap=True,
                independent_quantum_LF_four_current=False,local_C_actually_refitted=False,
                full_quantum_two_photon_matching=False,physical_hadron_form_factor=False,
                quantum_coupled_state_solved=False,production_model_selected=False,public_push=False))


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
