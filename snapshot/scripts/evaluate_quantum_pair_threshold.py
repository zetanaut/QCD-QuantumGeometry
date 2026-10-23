"""One-loop pair-cut and matching-direction replay; prints JSON only."""
import json
import numpy as np
from deuteron_wigner.quantum.quantum_pair_threshold import (
    analytic_spectral,phase_space_spectral,feynman_nonlocal,dispersive_nonlocal,
    feynman_upper_boundary,collective_cut,tree_width_diagnostics,
)
from deuteron_wigner.quantum.nonzero_momentum_current import (
    benchmark_reference,CollectiveChannel,QuarkCollectiveReference,
)
from deuteron_wigner.quantum.common_action_pilot import calibrated_action


def norm(a): return float(np.linalg.norm(a))


def evaluate():
    model=benchmark_reference();m=model.mass;rows=[]
    for q in [[.7,0,0,0],[1.,.1,-.2,.3],[1.4,-.3,.4,.6]]:
        expected=analytic_spectral(q,m)
        rest=phase_space_spectral(q,m,geometry='rest',order=4)
        lf=phase_space_spectral(q,m,geometry='light_front',order=4)
        refined=phase_space_spectral(q,m,geometry='light_front',order=6)
        qc=np.array(q)*[1,-1,-1,-1]
        removed=expected.copy();removed[1:4,4:16]=0;removed[4:16,1:4]=0
        rows.append(dict(q_GeV=q,spectral_norm_GeV2=norm(expected),
            rest_relative=norm(rest-expected)/norm(expected),LF_relative=norm(lf-expected)/norm(expected),
            quadrature_relative=norm(refined-lf)/norm(expected),
            eigenvalues_GeV2=np.linalg.eigvalsh(expected).tolist(),
            vector_Ward_norm_GeV3=norm(qc@lf[16:20]),
            quark_axial_Ward_norm_GeV3=max(norm(qc@lf[4+4*a:8+4*a]+1j*m*lf[1+a]) for a in range(3)),
            omit_PA_interference_Ward_norm_GeV3=norm(qc@removed[4:8]+1j*m*removed[1])))
    dispersion=[]
    for z in [-1.,-.3,-.1,.1]:
        for channel in ['scalar','pion']:
            a=feynman_nonlocal(m,z,channel);b=dispersive_nonlocal(m,z,channel)
            dispersion.append(dict(z_GeV2=z,channel=channel,Feynman_GeV2=a,dispersive_GeV2=b,relative=abs(a-b)/abs(a)))
    discontinuity=[]
    for s in [.5,1.]:
        for channel,index in [('scalar',0),('pion',1)]:
            cut=analytic_spectral([np.sqrt(s),0,0,0],m)[index,index].real
            samples=[dict(epsilon_GeV2=eps,negative_Im_GeV2=float(-feynman_upper_boundary(m,s,channel,imaginary_part=eps).imag))
                     for eps in [1e-6,1e-8]]
            discontinuity.append(dict(s_GeV2=s,channel=channel,phase_space_cut_GeV2=float(cut),boundary=samples))
    base=calibrated_action(.13957039,.0922,.515,.004,.6,1.4,.12,[3.,1.]);v=base.solve(.004)['vacuum']
    normal=np.array([-v[1],v[0]])/norm(v);g=model.scalar.couplings
    _,scalar_modes=np.linalg.eigh(model.scalar.hessian)
    eta=-(scalar_modes[:,1]@g)/(scalar_modes[:,1]@normal)
    sensitivity=[]
    for t in [0.,eta/2,eta]:
        changed=g+t*normal
        alternative=QuarkCollectiveReference(CollectiveChannel(model.scalar.hessian,changed,model.scalar.c),
            CollectiveChannel(model.pion.hessian,changed,model.pion.c),m)
        sizes=[]
        for q2 in [.1,.3,1.]:
            item=dict(Q2_GeV2=q2)
            for name in ['scalar','pion']:
                c=getattr(alternative,name);h,e=np.linalg.eigh(c.hessian+q2*np.eye(2));inverse_root=(e/np.sqrt(h))@e.T
                delta=feynman_nonlocal(m,-q2,name)*np.outer(changed,changed)
                item[name+'_subtracted_piece_relative_to_tree']=float(np.linalg.norm(inverse_root@delta@inverse_root,2))
            sizes.append(item)
        sensitivity.append(dict(eta=float(t),yukawa=changed.tolist(),fixed_tree_mass_GeV=float(.004+changed@v),
            mode_width_diagnostics=tree_width_diagnostics(alternative),nonlocal_piece_sizes=sizes))
    a=collective_cut(model,1.,'scalar');dark=np.array([-g[1],g[0]])/norm(g)
    return dict(inputs=dict(quark_mass_GeV=m,quark_pair_threshold_GeV=2*m,colors=3,flavors=2,
                    vertices='S,P1..3,A1[0..3],A2,A3,V_EM[0..3]; 20 x 24 x 24 open vertices',
                    cut_convention='A=1/2 integral dPhi2 sum M M*; Im correlator=-Im induced inverse term',
                    quantum_order='one free-quark loop; no gluon or collective loops',
                    subtraction='nonlocal two-point remainder only, DeltaK(z)-DeltaK(0)-z DeltaKprime(0)',
                    local_coefficients='unselected; zero subtraction values do not set physical counterterms',
                    domain='future timelike pair cuts; real nonlocal remainder below 4M^2; listed spacelike probes'),
                spectral_checks=rows,dispersive_checks=dispersion,complex_boundary_checks=discontinuity,
                shared_continuum=dict(s_GeV2=1.,scalar_cut_matrix_GeV2=a.tolist(),
                    rank=int(np.linalg.matrix_rank(a,tol=1e-13)),dark_source_response=norm(a@dark),
                    diagonal_only_dark_response=float(dark@np.diag(np.diag(a))@dark)),
                coupling_sensitivity=sensitivity,
                checks=dict(max_LF_spectral_relative=max(r['LF_relative'] for r in rows),
                    max_rest_spectral_relative=max(r['rest_relative'] for r in rows),
                    max_dispersion_relative=max(r['relative'] for r in dispersion)),
                reassessment='old heavy scalar is not a justified narrow mode at its test coupling; shared coupling freedom can remove that particular cut projection but does not remove nonlocal quantum corrections; preserve common continuum and source interference, do not select a retuned production model',
                claims=dict(one_loop_nonlocal_information=True,physical_hadron_widths=False,finite_local_matching_selected=False,
                    quantum_coupled_state_solved=False,full_quantum_EM_triangle=False,interacting_PCAC_completed=False,
                    confinement_or_QCD_matching=False,production_model_selected=False,public_push=False))


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2,allow_nan=False))
