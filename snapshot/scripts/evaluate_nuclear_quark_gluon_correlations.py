"""Replay the bounded nuclear operator controls; print computed JSON only."""
import json
from dataclasses import asdict

import numpy as np
from scipy.integrate import quad, quad_vec

from deuteron_wigner.quantum.nuclear_quark_gluon_correlations import (
    binding_scalar_match, embed_nucleon_pdfs, evolve_moment,
    kernel_moment_by_quadrature, lo_x_action, moment_generator, moment_null_shape,
)


def run():
    def profile(x):
        return np.array([(j+1)*x**(1+j % 2)*(1-x)**(2+j % 3) for j in range(7)])
    x_moments = []
    for n in (2,3):
        start, _ = quad_vec(lambda x: x**(n-1)*profile(x),0,1,epsabs=1e-11)
        computed, _ = quad_vec(lambda x: x**(n-1)*lo_x_action(profile,x,3),0,1,
                               epsabs=3e-9,epsrel=3e-9)
        x_moments.append(dict(n=n, direct_x_action_moment=computed,
                             analytic_moment_action=moment_generator(n,3)@start,
                             maximum_error=float(np.max(np.abs(computed-moment_generator(n,3)@start)))))
    initial = np.array([.2,.03,.15,.04,.04,.04,.5])
    evolved, alpha = evolve_moment(initial,2,3,.25,2,8)
    direction = np.array([1.,1.,-.3,-.3,.4,.4,-.5])
    null_evolved, _ = evolve_moment(direction/420,3,3,.25,2,6)
    nuclear = []
    def proton(x):
        v = 72*x*(1-x)**7
        return np.array([2*v,0.,v,0.,.8*6*x*(1-x)])
    def neutron(x):
        return proton(x)[[2,3,0,1,4]]
    for a,z in ((1,1),(1,0),(2,1),(3,1),(4,2),(12,6)):
        y = np.arange(1,a+1,dtype=float); y /= y.sum()
        pdfs = [proton]*z + [neutron]*(a-z)
        f = lambda x: embed_nucleon_pdfs(pdfs,y,x,2)
        edges = np.unique(np.r_[0.,y,1.])
        number = sum(quad_vec(f,lo,hi,epsabs=2e-10)[0] for lo,hi in zip(edges[:-1],edges[1:]))
        momentum = sum(quad_vec(lambda x:x*f(x),lo,hi,epsabs=2e-10)[0]
                       for lo,hi in zip(edges[:-1],edges[1:]))
        nuclear.append(dict(A=a,Z=z,nucleon_plus_fractions=y,
                            net_up=float(number[0]-number[1]),net_down=float(number[2]-number[3]),
                            total_momentum=float(momentum.sum())))
    args = (2,1,.94,.95,[.02,.03],[.03,.02])
    scalar1 = binding_scalar_match(*args,.002,[.0004,-.0001],[.2,.2])
    scalar2 = binding_scalar_match(*args,.002,[-.001,.003],[.2,.2])
    # Audit, do not mutate or use the historical matrix to perform evolution.
    from deuteron_wigner.matching.m3.core import splitting_library, collinear_report
    old = splitting_library()["singlet"].moment(2)
    return dict(
        scope="Necessary scalar/trace and LO unpolarized parton matching; reference profiles, no nuclear fit or binding solve",
        whole_target_fraction="X=k+/P_A+; integral net quarks=3A and total momentum=1",
        scalar_reference_inputs="GeV; illustrative masses .94/.95, B=.002, sigma p=(.02,.03), n=(.03,.02); gamma=(.2,.2)",
        scalar_same_binding_different_sources=[asdict(scalar1),asdict(scalar2)],
        scalar_total_trace_connected=[float(s.connected_anomaly+1.2*s.connected_sigma.sum())
                                      for s in (scalar1,scalar2)],
        full_flavor_N2_generator=moment_generator(2,3),
        maximum_mellin_quadrature_error=max(float(np.max(np.abs(moment_generator(n,nf)-kernel_moment_by_quadrature(n,nf))))
                                           for nf,n in ((1,2),(2,3),(3,2),(3,4),(4,5),(6,7))),
        direct_x_checks=x_moments,
        moment_scale_reference=dict(nf=3,mu0_GeV=2,mu_GeV=8,alpha0=.25,alpha=alpha,
                                    initial=initial,evolved=evolved,
                                    quark_total=float(evolved[:-1].sum()),gluon=float(evolved[-1]),
                                    total_momentum=float(evolved.sum()),
                                    finite_x_space_solution=False,threshold_matching=False),
        nuclear_embedding=nuclear,
        moment_null_reference=dict(number=float(quad(moment_null_shape,0,1)[0]),
                                   momentum=float(quad(lambda x:x*moment_null_shape(x),0,1)[0]),
                                   N3=float(quad(lambda x:x*x*moment_null_shape(x),0,1)[0]),
                                   x_action_at_018=lo_x_action(lambda x:direction*moment_null_shape(x),.18,3),
                                   evolved_N3=null_evolved),
        historical_singlet_audit=dict(source="matching/m3/core.py:splitting_library/collinear_report",
                                      actually_integrated_N2=old,column_sums=old.sum(axis=0),
                                      preset_report_momentum_residual=collinear_report()["singlet_momentum_residual"],
                                      historical_files_unchanged=True,used_for_new_evolution=False),
        physical_claim=False,full_x_Q2_prediction=False,new_QCD_vacuum=False)


if __name__ == "__main__":
    print(json.dumps(run(),indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x))
