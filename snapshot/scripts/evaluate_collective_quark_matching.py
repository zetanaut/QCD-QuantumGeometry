"""Replay exact Gaussian/source transport and bounded nonlinear TREE matching."""
import json
from dataclasses import replace
import numpy as np
import torch

from deuteron_wigner.quantum.common_action_pilot import calibrated_action
from deuteron_wigner.quantum.transverse_source_matching import SourceCurvature, match_projected, dot
from deuteron_wigner.quantum.collective_quark_matching import (
    routed_jet, response_derivatives, torch_response, one_source_matrix,
    matched_response, potential_derivatives, yukawa_map, exchange_weights,
    induced_many_quark_tensors, tree_series, nonlinear_bilinear_response,
)


def evaluate():
    base=calibrated_action(.13957039,.0922,.515,.004,.6,1.4,.12,[3.,1.])
    s=base.solve(.004);v=s['vacuum'];e=s['pion_modes'][:,0];heavy=s['pion_modes'][:,1]
    c,d=match_projected(.659,2e-4,.13957039)
    model=SourceCurvature(c*np.outer(e,e),d*np.outer(e,e))
    p0=np.array([.07,.3,.12,.11]);q1=np.array([.01,.04,-.01,.02]);q2=np.array([.02,-.03,.02,-.04])
    ea=np.array([.1,.3,-.2,.4]);eb=np.array([.2,-.1,.4,.3])
    hp,hs=base.matrices(v);p,jet=routed_jet(hp,model,p0,q1,q2,ea,eb)
    y=np.kron(np.eye(4),base.yukawa[:,None]);w=response_derivatives(jet,y)
    z=torch.zeros(2,dtype=torch.float64)
    mixed=torch.autograd.functional.hessian(lambda x:torch_response(jet,y,x)[3,0],z).numpy()[0,1]
    _,longjet=routed_jet(hp,model,p0,q1,q2,q1,eb)
    r=[np.linalg.inv(hp-dot(k,k)*np.eye(2)) for k in p];g=base.yukawa
    contracted=response_derivatives(longjet,y)
    ward=g@(r[3]@one_source_matrix(model,p[3],p[1],eb)@r[1]-r[2]@one_source_matrix(model,p[2],p[0],eb)@r[0])@g
    wrong=response_derivatives((*longjet[:3],np.zeros_like(longjet[3])),y)[3][3,0]
    kept=(jet[0]+.07*np.eye(8),.6*jet[1],.7*jet[2],.4*jet[3]);yk=.4*y
    residual=matched_response(jet,y,kept,yk);ret=response_derivatives(kept,yk)
    alternate=SourceCurvature(model.c+.6*(np.outer(e,heavy)+np.outer(heavy,e)),model.d+.8*np.outer(heavy,heavy))
    _,altjet=routed_jet(hp,alternate,p0,q1,q2,ea,eb)
    alt=response_derivatives(altjet,y)
    changed=replace(base,yukawa=base.yukawa+1.2*np.array([-v[1],v[0]])/np.linalg.norm(v))
    other=changed.solve(.004)
    h,t,u=potential_derivatives(base,v);ym=yukawa_map(g);tensors=induced_many_quark_tensors(h,t,u,ym)
    tree=[]
    for scale in [1e-5,3e-5,1e-4,3e-4,1e-3]:
        b=scale*np.array([-1.,.2,-.1,.3]);exact=nonlinear_bilinear_response(h,t,u,ym,b)
        terms=tree_series(tensors,b);value=exact['action_density']
        tree.append(dict(bilinear_probe_GeV3=b.tolist(),exact_tree_action_density_GeV4=value,
            separate_4q_6q_8q_terms_GeV4=terms.tolist(),
            relative_errors_through_4q_6q_8q=(abs(np.cumsum(terms)-value)/abs(value)).tolist(),
            min_local_curvature_GeV2=exact['min_curvature'],stationarity_error_GeV3=exact['stationarity_error'],
            log_det_local_curvature_ratio=float(np.linalg.slogdet(exact['collective_curvature'])[1]-np.linalg.slogdet(h)[1])))
    return dict(
        scope='Resolved S/P collective-to-quark algebra, one/two-source Gaussian response, homogeneous nonlinear tree elimination. Not QCD coefficient matching, a fermion expectation value, a loop calculation or a nucleon Hamiltonian.',
        inputs=dict(prior_calibration='transverse_source_matching_results.json, middle charged-mass variant; same approximate isospin scalar benchmark',
            current_mass_parameter_GeV=.004,yukawa=g.tolist(),vacuum_GeV=v.tolist(),quartics=base.quartic.tolist(),
            source_C_GeV_minus2=model.c.tolist(),source_D_GeV_minus2=model.d.tolist(),
            source_coefficient_scope='Chosen light-projection representative only; null directions explicitly varied, not fixed by data.'),
        resolved_channels=dict(order=['scalar','isovector_pseudoscalar_1','isovector_pseudoscalar_2','isovector_pseudoscalar_3'],
            endpoint_indices='4 Dirac x 2 flavor x 3 color; canonical color-vector exchange remains separate',
            Q2_GeV2=.1,exchange_weights_GeV_minus2=exchange_weights(base,v,.1).tolist()),
        gauged_exchange=dict(momenta_GeV=p.tolist(),q1_GeV=q1.tolist(),q2_GeV=q2.tolist(),polarizations=[ea.tolist(),eb.tolist()],
            response_norms_orders_0_a_b_ab=[float(np.linalg.norm(z)) for z in w],
            response_units=['GeV^-2','GeV^-3','GeV^-3','GeV^-4'],
            direct_mixed_derivative_error=float(abs(mixed-w[3][3,0])),
            one_source_Ward_error=float(abs(contracted[1][1,0]-g@(r[1]-r[0])@g)),
            two_source_Ward_error=float(abs(contracted[3][3,0]-ward)),
            missing_contact_Ward_error=float(abs(wrong-ward)),
            retained_reconstruction_max_error=float(max(np.max(abs(a-b-c)) for a,b,c in zip(w,ret,residual))),
            retained_scope='Explicit synthetic same-coordinate retained functional, NOT a matched QCD content assignment.'),
        nonidentifiability=dict(tree_quark_mass_GeV=s['quark_mass'],alternative_yukawa=changed.yukawa.tolist(),
            tree_mass_change_GeV=float(other['quark_mass']-s['quark_mass']),
            light_pion_yukawa_change=float((changed.yukawa-g)@e),
            alternative_exchange_weights_GeV_minus2=exchange_weights(changed,v,.1).tolist(),
            same_light_C_D_projection_error=float(np.max(abs(np.array(model.projected(e))-alternate.projected(e)))),
            null_direction_change_one_source_norm_GeV_minus3=float(np.linalg.norm(w[1]-alt[1])),
            null_direction_change_two_source_norm_GeV_minus4=float(np.linalg.norm(w[3]-alt[3]))),
        nonlinear_tree=dict(sign_convention='Euclidean mass m+g phi: +B^T Y^T x, stationary J=-Y B; vacuum and linear mass terms separate.',
            W3_SSS_GeV_minus5=float(tensors[1][0,0,0]),W3_S_P1_P1_GeV_minus5=float(tensors[1][0,1,1]),
            W4_SSSS_GeV_minus8=float(tensors[2][0,0,0,0]),
            probes=tree,
            determinant_scope='Finite 8-field local Hessian ratio only: demonstrates background dependence. NOT a regulated spacetime trace, loop energy, relative loop error or QCD fluctuation prediction.'),
        decision='Collective/source transport advances; pairwise-only closure and mean-bilinear replacement are not justified. Keep the implicit nonlinear functional or explicit collective variables. Before a physical quantum state solve, define whether the learned interaction is a scale-dependent action with retained modes quantized once or an already quantum-effective functional. These imply different matching and no-double-counting rules; the tree pilot cannot select between them. Unknown coefficients alone do not bar exploratory fits.')


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
