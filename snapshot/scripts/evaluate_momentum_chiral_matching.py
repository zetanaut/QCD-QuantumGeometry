"""Bounded canonical matching and nonlinear functional replay, no data fit."""
import argparse
import json
from pathlib import Path as FilePath
import numpy as np
from scipy.linalg import expm
import torch

from deuteron_wigner.quantum.chiral_soft_matching import G5
from deuteron_wigner.quantum.modular_gauge_network import GaugeGraph,Path,su3,generators
from deuteron_wigner.quantum.momentum_chiral_matching import (
    vector_fierz,color_fierz,two_quark_exchange,SpacelikeKernel,
    SharedInvariantFunctional,FunctionalRemainder,vector_map,vector_map_direct,
)


def evaluate():
    rng=np.random.default_rng(1110926)
    born=[]; max_born=0.; max_frame=0.; max_current=0.;max_combined=0.;max_conditioned=0.
    def lf(v):return np.array([(v[0]+v[3])/np.sqrt(2),v[1],v[2]])
    for _ in range(24):
        masses=rng.uniform(.07,.25,2); k=rng.uniform(.3,.7)
        vec0=rng.normal(size=3);vec0*=k/np.linalg.norm(vec0)
        vec1=rng.normal(size=3);vec1*=k/np.linalg.norm(vec1)
        p=[np.r_[np.sqrt(masses[0]**2+k*k),vec1],np.r_[np.sqrt(masses[1]**2+k*k),-vec1],
           np.r_[np.sqrt(masses[0]**2+k*k),vec0],np.r_[np.sqrt(masses[1]**2+k*k),-vec0]]
        a=two_quark_exchange(*map(lf,p),*masses,1.2)
        gen=np.zeros((4,4));gen[0,1:]=gen[1:,0]=rng.normal(0,.25,3)
        transform=expm(gen)
        b=two_quark_exchange(*[lf(transform@v) for v in p],*masses,1.2)
        error=np.linalg.norm(a["light_front"]-a["covariant"])/np.linalg.norm(a["covariant"])
        error=max(error,np.linalg.norm(b["light_front"]-b["covariant"])/np.linalg.norm(b["covariant"]))
        frame=abs(np.linalg.norm(b["covariant"])/np.linalg.norm(a["covariant"])-1)
        max_born=max(max_born,error);max_frame=max(max_frame,frame)
        for item in [a,b]:
            denominator=np.linalg.norm(item["covariant"])
            raw=np.linalg.norm(item["light_front"]-item["covariant"])/denominator
            condition=(np.linalg.norm(item["propagating"])+np.linalg.norm(item["instantaneous"]))/denominator
            max_conditioned=max(max_conditioned,raw/(np.finfo(float).eps*max(1.,condition)))
            max_combined=max(max_combined,np.linalg.norm(item["light_front_combined"]-item["covariant"])/denominator)
        max_current=max(max_current,a["current_divergence"],b["current_divergence"])
        born.append(dict(Q2_GeV2=a["virtuality"],qplus_GeV=a["plus_transfer"],
                         omitted_instantaneous_relative=np.linalg.norm(a["propagating"]-a["covariant"])/np.linalg.norm(a["covariant"])))
    graph=GaugeGraph({(0,1):su3(rng.normal(0,.3,8)),(1,2):su3(rng.normal(0,.3,8)),(0,2):su3(rng.normal(0,.3,8))})
    paths={(i,i):Path((i,)) for i in range(3)}
    for i,j,vertices in [(0,1,(0,1)),(1,2,(1,2)),(0,2,(0,1,2))]:
        paths[i,j]=Path(vertices,"FINITE_SOURCE_REFERENCE");paths[j,i]=paths[i,j].reversed()
    w=np.array([[graph.transport(paths[i,j]) for j in range(3)] for i in range(3)])
    points=np.array([[0,0,0,0],[.8,.1,-.3,.2],[-.4,.2,.5,.1]])
    qs=np.array([[.3,.1,.2,.4],[.5,-.2,.1,.3],[.2,.6,-.3,.1]]);qs=np.r_[qs,-qs]
    kernel=SpacelikeKernel(1.2,.8,(.3,-.2,.1),(.01,100.))
    pieces=kernel.fourier(points,qs,np.full(6,.03),.4)
    f=SharedInvariantFunctional(pieces["total"],np.array([.6,.8,.7]),w,.8,.8)
    s=rng.normal(0,.13,(3,3,24,24))+1j*rng.normal(0,.13,(3,3,24,24))
    ds=rng.normal(0,.08,s.shape)+1j*rng.normal(0,.08,s.shape)
    sigma=f.self_energy(s);kds=f.kernel_action(s,ds)
    h=1e-3
    fd=(-f.value(s+2*h*ds)+8*f.value(s+h*ds)-8*f.value(s-h*ds)+f.value(s-2*h*ds))/(12*h)
    fd_sigma=-(-f.self_energy(s+2*h*ds)+8*f.self_energy(s+h*ds)-8*f.self_energy(s-h*ds)+f.self_energy(s-2*h*ds))/(12*h)
    epsilon=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    phi=f.torch_value(torch.tensor(s)+epsilon*torch.tensor(ds))
    first=[];second=[]
    for part in [phi.real,phi.imag]:
        one=torch.autograd.grad(part,epsilon,create_graph=True,retain_graph=True)[0]
        two=torch.autograd.grad(one,epsilon,retain_graph=True)[0]
        first.append(one.item());second.append(two.item())
    a=np.kron(np.kron(G5,np.array([[0,1],[1,0]])/2),np.eye(3))
    axial=[c*a for c in [.2,-.6,.9]]
    dax=np.array([[axial[i]@s[i,j]+s[i,j]@axial[j] for j in range(3)] for i in range(3)])
    target=np.array([[axial[i]@sigma[i,j]+sigma[i,j]@axial[j] for j in range(3)] for i in range(3)])
    correct=f.kernel_action(s,dax);frozen=f.kernel_action(s,dax,include_curvature=False)
    radial=f.kernel_action(s,s);radial_frozen=f.kernel_action(s,s,include_curvature=False)
    ret=SharedInvariantFunctional(pieces["retained"],f.weights,w,.8,.8)
    naive=SharedInvariantFunctional(pieces["residual"],f.weights,w,.8,.8)
    residual=FunctionalRemainder(f,ret)
    dw=np.array([[graph.path_tangent(paths[i,j],(0,1),generators()[2]) for j in range(3)] for i in range(3)])
    source,sigma_source=f.path_source(s,dw)
    def moved(e):
        g=GaugeGraph({edge:(expm(1j*e*generators()[2])@u if edge==(0,1) else u) for edge,u in graph.links.items()})
        tw=np.array([[g.transport(paths[i,j]) for j in range(3)] for i in range(3)])
        return SharedInvariantFunctional(f.f,f.weights,tw,.8,.8)
    hh=2e-5
    source_fd=(moved(hh).value(s)-moved(-hh).value(s))/(2*hh)
    source_sigma_fd=(moved(hh).self_energy(s)-moved(-hh).self_energy(s))/(2*hh)
    sf=vector_fierz();ct,cs,co=color_fierz()
    qr=np.geomspace(.02,50.,17);subtractions=[]
    for r in [0.,.4,1.]:
        p=kernel.pieces(qr,r)
        subtractions.append(dict(retained_fraction=r,
            recombination_max_abs_GeV_minus2=float(np.max(abs(p["retained"]+p["residual"]-p["total"]))),
            tree_remainder_max_abs_GeV_minus2=float(np.max(abs(p["tree_remainder"])))))
    return dict(
        scope="Born g2 matching and finite spacelike/Fourier source-functional qualification; no new mass/pion/nucleon solve, loop QCD matching or production causal model.",
        bare_cases=born,
        bare_maxima=dict(LF_covariant_relative=max_born,LF_combined_relative=max_combined,
                         raw_error_over_epsilon_condition=max_conditioned,spin_sum_boost_relative=max_frame,
                         current_divergence_GeV2=max_current),
        recoupling=dict(dirac_reconstruction_norm=float(np.linalg.norm(sf["tensor"]-sf["rebuilt"])),
            SP_only_relative_HS_loss=float(np.linalg.norm(sf["tensor"]-sf["scalar_pseudoscalar"])/np.linalg.norm(sf["tensor"])),
            color_reconstruction_norm=float(np.linalg.norm(ct-cs-co)),
            color_singlet_only_relative_HS_loss=float(np.linalg.norm(ct-cs)/np.linalg.norm(ct)),
            warning="These are fixed-basis open-index tensor norm losses, NOT errors on measured observables."),
        inputs=dict(gs_squared=1.2,scale_GeV=.8,theta=[.3,-.2,.1],spacelike_domain_GeV2=[.01,100.],
                    points_GeV_minus1=points.tolist(),momenta_GeV=qs.tolist(),momentum_weights_GeV4=[.03]*6,
                    coordinate_weights_GeV_minus4=f.weights.tolist(),nonlinear=.8,propagator_seed=1110926,
                    physical_soft_coefficients="unknown; displayed values are diagnostic only"),
        derivative_errors=dict(first_finite_difference=float(abs(fd-f.pairing(sigma,ds))),
            kernel_finite_difference_max_entry=float(np.max(abs(fd_sigma-kds))),
            first_torch=float(abs(complex(*first)-f.pairing(sigma,ds))),
            second_torch=float(abs(complex(*second)+f.pairing(kds,ds))),
            explicit_color_contraction_max_entry=float(np.max(abs(vector_map(s[0,1],w[0,1])-vector_map_direct(s[0,1],w[0,1]))))),
        curvature_negative=dict(full_chiral_max_entry=float(np.max(abs(correct-target))),
            frozen_curvature_chiral_max_entry=float(np.max(abs(frozen-target))),
            omitted_curvature_radial_relative=float(np.linalg.norm(radial-radial_frozen)/np.linalg.norm(radial))),
        tree_subtractions=subtractions,
        nonlinear_subtraction=dict(
            correct_value_recombination=float(abs(residual.value(s)+ret.value(s)-f.value(s))),
            naive_value_relative=float(abs(naive.value(s)-residual.value(s))/abs(residual.value(s))),
            naive_self_energy_relative=float(np.linalg.norm(naive.self_energy(s)-residual.self_energy(s))/np.linalg.norm(residual.self_energy(s))),
            naive_kernel_relative=float(np.linalg.norm(naive.kernel_action(s,ds)-residual.kernel_action(s,ds))/np.linalg.norm(residual.kernel_action(s,ds))),
            naive_path_source_relative=float(abs(naive.path_source(s,dw)[0]-residual.path_source(s,dw)[0])/abs(residual.path_source(s,dw)[0]))),
        source_errors=dict(path_value_difference=float(abs(source_fd-source)),
            path_self_energy_max_entry=float(np.max(abs(source_sigma_fd-sigma_source)))),
        decision="Retain full channel tensors, ordered paths, complete nonlinear derivatives and functional-level subtraction. Next qualify causal/spectral realization and independent scalar/vector/axial matching inputs before selecting a momentum-dependent state/current solve. The spacelike basis alone is not a production model.",
    )


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=FilePath);args=parser.parse_args()
    rendered=json.dumps(evaluate(),indent=2,allow_nan=False)+"\n"
    if args.output:args.output.write_text(rendered)
    print(rendered)
