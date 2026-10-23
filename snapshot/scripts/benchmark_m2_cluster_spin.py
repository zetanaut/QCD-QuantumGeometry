"""Save one fresh finite-elastic helicity block with all contribution families.

Run blocks independently and resumably; never overwrite old quantum evidence.
This is a classical physics diagnostic, not a fresh quantum optimization.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from deuteron_wigner.quantum.m2_gaussian_cluster_h2 import build_cluster_pilot,GaussianClusterH2
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_cluster_wick import GaussianClusterWick
from deuteron_wigner.quantum.m2_cluster_spin_diagnostic import elastic_spin_pair
from deuteron_wigner.quantum.m2_cluster_quantum_reference import reference_from_current
from deuteron_wigner.quantum.m2_gaussian_cluster_four_current import assemble_cluster_h2_four_current
from deuteron_wigner.quantum.m2_gaussian_cluster_current import PRESCRIPTION
from deuteron_wigner.quantum.m2_smooth_longitudinal_projector import SmoothMinusProjectorContext,SMOOTH_PRESCRIPTION
from deuteron_wigner.quantum.m2_finite_cell_envelope_current import SPECTRAL_PRESCRIPTION
from deuteron_wigner.quantum.m2_total_isospin import H2_ISOSPIN_PRESCRIPTION


def models(target,*,coupling=.3,total_k2=9,retained_cutoff=0,q5=0,q6=0,width=.8):
    reg=GaussianClusterRegulator(width_GeV=width,boson_modes=8,fermion_modes=8)
    wick=GaussianClusterWick(regulator=reg,quark_flavors=2,normal_order_subtraction=True,
        finite_quark_mass2_per_gs2_GeV2=.011,finite_gluon_mass2_per_gs2_GeV2=-.007)
    return [GaussianClusterH2(build_cluster_pilot(total_k2=total_k2,cutoff=retained_cutoff,
        target=target,jz_twice=h,b_GeV=.4,quark_mass_GeV=.35,gluon_mass_GeV=0.,
        coupling=coupling,regulator=reg,total_pplus_GeV=1.3,wick_operators=wick),
        excluded_cutoff=q5,six_cutoff=q6) for h in (1,-1)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('PROTON','NEUTRON'),required=True)
    p.add_argument('--out',type=int,choices=(1,-1),required=True)
    p.add_argument('--incoming',type=int,choices=(1,-1),required=True)
    p.add_argument('--delta',nargs=2,type=float,required=True)
    p.add_argument('--mean',nargs=2,type=float,default=(0.,0.))
    p.add_argument('--q5',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();dest=args.output
    if dest.suffix!='.json' or dest.exists() or dest.with_suffix('.npz').exists():
        p.error('new JSON/NPZ output path required')
    start=time.monotonic()
    pair=elastic_spin_pair(*models(args.target,q5=args.q5),bracket_GeV2=(1.3,1.5),intrinsic_parity=1)
    f,i=[(1,-1).index(h) for h in (args.out,args.incoming)]
    sf,si=[pair['snapshots'][j] for j in (f,i)]
    print('Finite elastic pole',pair['energy_GeV2'],'assembling',args.target,args.out,args.incoming,flush=True)
    current=assemble_cluster_h2_four_current(sf,si,args.delta,charge='electromagnetic',
        loop_flavor_charges=(2/3,-1/3),prescription=PRESCRIPTION,quadrature_order=8,
        minus_context=SmoothMinusProjectorContext(mean_total_transverse_GeV=tuple(args.mean),
                                                  prescription=SMOOTH_PRESCRIPTION),
        field_envelope_prescription=SPECTRAL_PRESCRIPTION,isospin_prescription=H2_ISOSPIN_PRESCRIPTION)
    reference=reference_from_current(current)
    families=tuple(current.contributions)
    family_coeff=np.array([[current.contributions[name][k] for k in range(5)] for name in families])
    uf,ui=[pair['raw_states'][j] for j in (f,i)]
    amplitudes=np.einsum('a,knmab,b->knm',uf.conj(),family_coeff,ui,optimize=True)
    raw=np.sum(family_coeff,axis=0)
    arrays=dict(current_coefficients=raw,family_coefficients=family_coeff,
        family_amplitudes=amplitudes,parity_map=pair['parity_map'])
    for label,j in (('out',f),('in',i)):
        s=pair['snapshots'][j]
        arrays.update({label+'_H':s.effective_hamiltonian,label+'_N':s.metric,
            label+'_W':s.whitener,label+'_L':s.matrix,label+'_raw_state':pair['raw_states'][j],
            label+'_state':pair['states'][j]})
    record=dict(scope='finite elastic H2 g4 spin diagnostic, positive-parity test branch; NOT physical calibration',
        metadata=dict(reference.metadata),families=families,parameters=dict(regulator=asdict(si.owner.pilot.field_regulator),
            wick=asdict(si.owner.pilot.wick_operators),q5=args.q5,q6=0,b_GeV=.4,mq_GeV=.35,mg_GeV=0.,g=.3,Pplus_GeV=1.3),
        energy_GeV2=pair['energy_GeV2'],intrinsic_parity=1,bracket_GeV2=[1.3,1.5],
        independent_spin_poles_GeV2=[v.energy_GeV2 for v in pair['independent_poles']],
        pole_residuals_GeV2=[v.residual_GeV2 for v in pair['independent_poles']],
        parity_errors=pair['parity_errors'],Ward_order_norms=reference.ward_residuals,
        full_matrix_file=dest.with_suffix('.npz').name,physical=False,covariance_repaired=False,
        quantum_execution=False,seconds=time.monotonic()-start)
    with dest.with_suffix('.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    with dest.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(dest),seconds=record['seconds'],pole=record['energy_GeV2'])),flush=True)


if __name__=='__main__':main()
