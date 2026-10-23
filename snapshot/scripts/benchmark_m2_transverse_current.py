"""Free-Q5 canonical transverse-current/covariance diagnostic, NOT a physical fit."""
import argparse,json,time
import numpy as np
from scipy.linalg import eigh,eigvalsh
from scipy.optimize import brentq

from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_shell_fock_reference import build_shell_fock_reference
from deuteron_wigner.quantum.m2_shell_transverse_matching import FreeQ5TransverseMatching
from deuteron_wigner.quantum.m2_shell_current import shell_density_between
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from deuteron_wigner.quantum.m2_current_covariance_diagnostic import current_covariance_diagnostic
from deuteron_wigner.quantum.m2_orientation_current_diagnostic import orientation_current_diagnostic


def run(target,cutoff,coupling):
    start=time.monotonic();delta=(.12,.16);P=1.3
    def build(jz):
        p=build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=coupling)
        return build_shell_fock_reference(p,out_cutoff=cutoff)
    refs=[build(1),build(-1)]
    couplings=[[s.source.full_coupling(coupling) for s in r.sectors] for r in refs]
    def h(e,h=0):
        return refs[h].pilot.matrix+sum(v.conj().T@s.propagation.resolve(e,v)
            for s,v in zip(refs[h].sectors,couplings[h]))
    def objective(e):
        return eigvalsh(h(e),subset_by_index=[0,0])[0]-e
    floor=min(s.propagation.minimum_energy_GeV2 for s in refs[0].sectors)
    upper=min(eigvalsh(refs[0].pilot.matrix,subset_by_index=[0,0])[0],floor-1e-8)
    step=max(1.,sum(np.linalg.norm(v) for v in couplings[0]));lower=upper-step
    for _ in range(32):
        if objective(lower)>0:break
        step*=2;lower=upper-step
    else:raise ArithmeticError("no lower ground-root bracket")
    e=float(brentq(objective,lower,upper,xtol=5e-14,rtol=1e-14))
    _,u=eigh(h(e),subset_by_index=[0,0]);v=u[:,0]
    norm=np.sqrt(np.vdot(v,refs[0].norm_metric(e)@v).real)
    vectors=[v/norm]
    # Explicit positive intrinsic parity, not an independently phased eigensolver.
    parity=transverse_parity_map(refs[1].pilot,refs[0].pilot)
    vectors.append(1j*parity@vectors[0])
    raw=[{s.source.space.sector:s.propagation.resolve(e,c@v[:,None])
          for s,c in zip(r.sectors,cs)} for r,cs,v in zip(refs,couplings,vectors)]
    residuals=[]
    for r,cs,v,q in zip(refs,couplings,vectors,raw):
        rp=r.pilot.matrix@v-e*v;rq2=0.
        for s,c in zip(r.sectors,cs):
            w=q[s.source.space.sector]
            rp+=c.conj().T@w[:,0]
            rq2+=np.linalg.norm(s.propagation.apply(w)-e*w+c@v[:,None])**2
        residuals.append(float(np.sqrt(np.linalg.norm(rp)**2+rq2)))
    plus=np.zeros((2,2),complex);trans=np.zeros((2,2,2),complex);cross=np.zeros_like(trans)
    for hf in range(2):
        for hi in range(2):
            f,i=refs[hf],refs[hi];vf,vi=vectors[hf],vectors[hi]
            m=FreeQ5TransverseMatching(f,i,delta,total_pplus_GeV=P,charge="electromagnetic")
            trans[:,hf,hi]=[np.vdot(vf,j@vi) for j in m.effective_current(e,e)]
            plus[hf,hi]=np.vdot(vf,good_current_density_between(f.pilot,i.pilot,delta,
                                                                charge="electromagnetic")@vi)
            for sf,si,forward,backward in m.sectors:
                name=sf.source.space.sector;qf,qi=raw[hf][name],raw[hi][name]
                plus[hf,hi]+=np.vdot(qf,shell_density_between(sf.source.space,si.source.space,
                    qi,delta,b_GeV=.4,charge="electromagnetic"))
                for j in (0,1):
                    cross[j,hf,hi]+=coupling*(np.vdot(vf[backward.retained_columns],
                        backward.current_per_gs[j].conj().T@qi[:,0])
                        +np.vdot(qf[:,0],forward.current_per_gs[j]@vi[forward.retained_columns]))
    diag=current_covariance_diagnostic(plus,trans,delta,mass_GeV=np.sqrt(e),total_pplus_GeV=P)
    omit=current_covariance_diagnostic(plus,trans-cross,delta,mass_GeV=np.sqrt(e),total_pplus_GeV=P)
    orientation=orientation_current_diagnostic(plus,trans,delta,mass_GeV=np.sqrt(e),total_pplus_GeV=P)
    def pair(z):return [float(np.real(z)),float(np.imag(z))]
    return dict(scope="self-consistent bare free-Q5 reference; no interacting QQ/Q6/current-loop/physical fit",
        target=target,excluded_cutoff=cutoff,coupling=coupling,retained_dimension=37,
        total_pplus_GeV=P,transfer_GeV=delta,M2_GeV2=e,raw_parent_residuals_GeV2=residuals,
        F1_from_plus=pair(diag.F1_from_plus),F2_from_plus=pair(diag.F2_from_plus),
        magnetic_from_plus=pair(diag.magnetic_from_plus),
        magnetic_from_transverse=pair(diag.magnetic_from_transverse),
        magnetic_without_PQ_QP_contacts=pair(omit.magnetic_from_transverse),
        orientation_convention="Carbonell et al. 1998 Eq. (6.25); diagnostic, not renormalization",
        B1=pair(orientation.B1),B2=None,B3=None,
        F1_separated=pair(orientation.F1_separated),F2_separated=pair(orientation.F2_separated),
        orientation_structure_residual_norm=float(np.linalg.norm(orientation.structure_residual)),
        plus_structure_residual_norm=float(np.linalg.norm(diag.plus_structure_residual)),
        transverse_covariance_residual_norm=float(np.linalg.norm(diag.transverse_covariance_residual)),
        transverse_nonmagnetic_residual_norm=float(np.linalg.norm(diag.transverse_nonmagnetic_residual)),
        elastic_continuity_defect_norm=float(np.linalg.norm(diag.elastic_continuity_defect)),
        PQ_QP_contact_current_norm=float(np.linalg.norm(cross)),
        parity_H_error=float(np.linalg.norm(h(e,1)-parity@h(e)@parity.conj().T)),
        physical=False,covariance_repaired=False,seconds=time.monotonic()-start)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",choices=("PROTON","NEUTRON"),required=True)
    parser.add_argument("--cutoff",type=int,required=True)
    parser.add_argument("--coupling",type=float,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.target,args.cutoff,args.coupling),indent=2),flush=True)
