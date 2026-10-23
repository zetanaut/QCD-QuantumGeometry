"""All four canonical free-current components on the unphysical g=0 control."""
import json
import numpy as np
from deuteron_wigner.quantum.m2_coupled_pilot import build_coupled_pilot
from deuteron_wigner.quantum.m2_canonical_good_current import good_current_density_between
from deuteron_wigner.quantum.m2_transverse_current_projection import free_transverse_current_between
from deuteron_wigner.quantum.m2_free_minus_current import free_minus_current_between
from deuteron_wigner.quantum.m2_spin_resolved_current import transverse_parity_map
from deuteron_wigner.quantum.m2_orientation_current_diagnostic import orientation_current_diagnostic


def run(target):
    def build(jz):
        return build_coupled_pilot(total_k2=9,cutoff=0,target=target,jz_twice=jz,b_GeV=.4,
            quark_mass_GeV=.35,gluon_mass_GeV=0.,coupling=0.)
    p=[build(1),build(-1)];e,u=np.linalg.eigh(p[0].matrix)
    v=[u[:,0],1j*transverse_parity_map(p[1],p[0])@u[:,0]]
    delta=(.12,.16);P=1.3;current=np.zeros((4,2,2),complex)
    for hf in range(2):
        for hi in range(2):
            f,i=p[hf],p[hi]
            matrices=[good_current_density_between(f,i,delta,charge="electromagnetic"),
                free_minus_current_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic"),
                *free_transverse_current_between(f,i,delta,total_pplus_GeV=P,charge="electromagnetic")]
            current[:,hf,hi]=[np.vdot(v[hf],j@v[hi]) for j in matrices]
    result=orientation_current_diagnostic(current[0],current[2:],delta,
        mass_GeV=np.sqrt(e[0]),total_pplus_GeV=P,minus=current[1])
    def pair(z):return [float(np.real(z)),float(np.imag(z))]
    return dict(scope="unphysical g=0 free retained equal-share control, not bound nucleon",
        target=target,M2_GeV2=float(e[0]),transfer_GeV=delta,total_pplus_GeV=P,
        coefficients={name:pair(getattr(result,name)) for name in ("F1_separated","F2_separated","B1","B2","B3")},
        full_structure_residual_norm=float(np.linalg.norm(result.structure_residual)),
        physical=False,covariance_repaired=False,interacting_minus_current=False)


if __name__=="__main__":
    print(json.dumps([run(t) for t in ("PROTON","NEUTRON")],indent=2),flush=True)
