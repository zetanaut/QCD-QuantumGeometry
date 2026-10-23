"""Replay source accessibility and coupled transport, with no physical fit."""
import json
import numpy as np
import torch
from scipy.linalg import expm
from deuteron_wigner.quantum.source_network_transport import (
    insertion_design,symmetric_basis,diagonal_cycle_null,accessibility,
    source_jet,source_derivatives,two_E_return,canonical_coordinates,
)
from deuteron_wigner.quantum.joint_source_matching import stress_contact
from deuteron_wigner.quantum.transverse_source_matching import dot


def evaluate():
    x=np.geomspace(.002,8.,40);ranks=[]
    for n in [1,2,3,4]:
        h=np.diag(np.geomspace(.1,1.5,n));g=np.linspace(1.,.7,n)
        diagonal=insertion_design(h,g,x,diagonal=True);full=insertion_design(h,g,x[::5])
        ranks.append(dict(nodes=n,symmetric_source_entries=n*(n+1)//2,
            equal_virtuality_rank=int(np.linalg.matrix_rank(diagonal)),independent_virtuality_rank=int(np.linalg.matrix_rank(full)),
            diagonal_singular_values=np.linalg.svd(diagonal,compute_uv=False).tolist(),
            independent_singular_values=np.linalg.svd(full,compute_uv=False).tolist()))
    poles=np.array([.2,.7,1.4]);g=np.array([1.,.8,.6]);h=np.diag(poles)
    cycle=diagonal_cycle_null(poles,g,(0,1,2));probes=np.geomspace(.01,5.,30)
    coefficients=np.einsum('kij,ij->k',symmetric_basis(3),cycle)
    diagonal=insertion_design(h,g,probes,diagonal=True)@coefficients
    independent=insertion_design(h,g,probes[::7])@coefficients
    edge_removed=cycle.copy();edge_removed[0,2]=edge_removed[2,0]=0
    left=g/(poles+.01);right=g/(poles+.8)
    access_h=np.diag([.2,.6,1.2]);access_g=np.array([1.,0,0]);access_e=np.array([[0.,.4,0.],[.4,0.,.3],[0.,.3,0.]])
    conditioning=[]
    for split in [.1,.01,.001,0.]:
        design=insertion_design(np.diag([.2,.2+split]),[1.,.8],np.geomspace(.01,3.,16))
        singular=np.linalg.svd(design,compute_uv=False)
        conditioning.append(dict(pole_splitting_GeV2=split,rank=int(np.linalg.matrix_rank(design)),
            smallest_over_largest_singular=float(singular[-1]/singular[0])))
    rng=np.random.default_rng(24);z=rng.normal(size=(2,2));u=expm(.2*(z-z.T))
    h=u@np.diag([.2,1.1])@u.T;g=u@np.array([1.,.5]);matrices=[]
    for scale in [.4,.2,.7]:
        z=rng.normal(size=(2,2));matrices.append(scale*(z+z.T)/2)
    p0=np.array([.07,.3,.12,.11]);q1=np.array([.01,.04,-.01,.02]);q2=np.array([.02,-.03,.02,-.04])
    eps1=np.array([.1,.3,-.2,.4]);eps2=np.array([.2,-.1,.4,.3]);args=(p0,q1,q2,eps1,eps2)
    ps,jet=source_jet(h,*matrices,*args);y=np.kron(np.eye(4),g[:,None]);values=source_derivatives(jet,y)
    _,without=source_jet(h,*matrices[:2],matrices[2]*0,*args)
    e_piece=float((values[3]-source_derivatives(without,y)[3])[3,0])
    lv=np.linalg.solve(h-dot(ps[3],ps[3])*np.eye(2),g);rv=np.linalg.solve(h-dot(p0,p0)*np.eye(2),g)
    e_independent=stress_contact(ps[3],p0,q1,q2,eps1,eps2,lv@matrices[2]@rv)
    basis=torch.tensor(symmetric_basis(2));zero=torch.zeros(3,dtype=torch.float64)
    def derivative(theta):
        ht=torch.tensor(h)+theta[0]*basis[0];gt=torch.tensor(g)+theta[1]*torch.tensor([.2,-.1],dtype=torch.float64)
        et=torch.tensor(matrices[2])+theta[2]*basis[1]
        _,j=source_jet(ht,matrices[0],matrices[1],et,*args)
        endpoint=torch.kron(torch.eye(4,dtype=torch.float64),gt[:,None])
        return source_derivatives(j,endpoint)[3][3,0]
    gradient=torch.autograd.functional.jacobian(derivative,zero).numpy()
    t=np.array([[1.2,.3],[-.1,.8]]);metric=t.T@t;hm=t.T@h@t;gm=t.T@g;mm=[t.T@a@t for a in matrices]
    _,moved=source_jet(hm,*mm,*args,metric=metric);ym=np.kron(np.eye(4),gm[:,None])
    corrected=source_derivatives(moved,ym);_,wrong=source_jet(hm,*mm,*args)
    hc,gc,mc,_=canonical_coordinates(hm,gm,metric,source_matrices=mm)
    _,canonical=source_jet(hc,*mc,*args);normalized=source_derivatives(canonical,np.kron(np.eye(4),gc))
    h4=np.diag([.2,.7]);e4=np.array([[0.,.6],[.6,0.]]);g4=np.array([1.,0.]);p=np.array([.03,.35,.12,.1])
    q=np.array([[.01,.02,.01,0.],[.02,-.01,.02,.01],[.01,.01,-.02,.01],[.01,-.02,.01,-.01]])
    eps=np.array([[.1,.3,-.2,.4],[.2,-.1,.4,.3],[.1,.2,.3,-.1],[-.1,.4,.1,.2]])
    four,terms=two_E_return(h4,e4,g4,p,q,eps);longitudinal=[]
    for i in range(4):
        moved_eps=eps.copy();moved_eps[i]=q[i]
        longitudinal.append(float(two_E_return(h4,e4,g4,p,q,moved_eps)[0][0,0]))
    return dict(scope='Unfitted off-shell charged-collective matching kernels, not full QCD, measured amplitudes or a quantum bound state.',
        rank_controls=ranks,rank_singular_values_are_not_physical_uncertainties=True,
        exact_cycle=dict(poles_GeV2=poles.tolist(),couplings=[1.,.8,.6],source_direction_dimensionless=cycle.tolist(),
            shape_reference_mass_squared_GeV2=1.,shape_normalization_is_not_a_physical_E_coefficient_or_matching_scale=True,
            max_equal_virtuality_response=float(np.max(abs(diagonal))),max_independent_route_response=float(np.max(abs(independent))),
            selected_independent_response=float(left@cycle@right),erasing_one_edge_diagonal_response=float(left@edge_removed@left),
            interpretation='simple-pole incidence cycle in a response graph; not vacuum topology or confinement'),
        accessibility=dict(source_free_dimension=accessibility(access_h,access_g).shape[1],
            with_all_declared_source_generators_dimension=accessibility(access_h,access_g,source_matrices=[access_e]).shape[1],
            source_free_reachability_is_not_a_global_compression_certificate=True),
        near_degenerate_conditioning=conditioning,
        coupled_transport=dict(response_units='GeV^-4, twice differentiated with respect to electromagnetic sources in GeV',two_source_kernel=float(values[3][3,0]),E_part=e_piece,independent_E_insertion=e_independent,
            E_identity_error=abs(e_piece-e_independent),H_endpoint_E_derivatives=gradient.tolist(),
            parameter_coordinate_units=['GeV^2','dimensionless','GeV^-4'],
            kinetic_metric_transport_error=max(float(torch.max(abs(a-b))) for a,b in zip(values,corrected)),
            recanonicalized_transport_error=max(float(torch.max(abs(a-b))) for a,b in zip(values,normalized)),
            omitted_metric_two_source_error=float(abs(source_derivatives(wrong,ym)[3][3,0]-values[3][3,0]))),
        hidden_mode_return=dict(H_GeV2=h4.tolist(),E_GeV_minus4=e4.tolist(),endpoints=g4.tolist(),
            first_insertion_zero=True,response_units='GeV^-6, four source derivatives',four_photon_E_squared_coefficient=float(four[0,0]),six_ordered_pair_terms=terms[:,0,0].tolist(),
            longitudinal_contractions=longitudinal,after_deleting_hidden_mode=0.,
            scope='complete E^2 coefficient at four-photon order only; NOT complete four-photon kernel. Direct E higher contacts are linear in E.'),
        reassessment=dict(physical_source_directions_kept=True,canonical_color_vector_and_graded_fermion_utilities_unchanged=True,
            source_free_dynamics_or_fitted_coefficients_changed=False,quantum_state_supplied=False,production_model_selected=False,public_push=False,
            next='pre-evaluate the first nonzero-momentum generated quantum state and linked response using the preserved canonical/retained action. Specify actual generator, quantum counting, source partners and independent control before solving; do not relabel this spacelike exchange inverse as a Hamiltonian, choose a production model or launch an inherited bound-state repair agenda.'))


if __name__=='__main__': print(json.dumps(evaluate(),indent=2,allow_nan=False))
