"""Replay necessary binding/geometry limits; no fit or output-file mutation."""
import json
import numpy as np

from deuteron_wigner.quantum.binding_geometry_preflight import (
    singlet_carrier, longitudinal_threshold, cluster_case,
)


def evaluate():
    _,_,g,_=singlet_carrier()
    return dict(scope='free-threshold and six-static-quark color/geometry preflight, not physical nucleon binding',
        unconfined_three_quark_threshold_GeV=.81,
        illustrative_quark_mass_GeV=.27,
        finite_longitudinal_minima=[dict(k2=k,mass_GeV=longitudinal_threshold([.27]*3,k)[0],
                                       odd_modes=longitudinal_threshold([.27]*3,k)[1])
                                   for k in (3,5,7,9,11,13,15,29,31,61,121)],
        partition_count=10,color_singlet_rank=5,Gram_eigenvalues=np.linalg.eigvalsh(g).tolist(),
        cluster_cases=[cluster_case(r) for r in (2,4,8,16,32,64)],
        physical_bound_states=False,production_confinement_selected=False,
        static_color_basis_includes_independent_flux_states=False,
        full_LF_chiral_current_or_cluster_covariance_proved=False,
        old_BLFQ_H2_rebuilt=False,public_push=False)


if __name__=='__main__':
    print(json.dumps(evaluate(),indent=2))
