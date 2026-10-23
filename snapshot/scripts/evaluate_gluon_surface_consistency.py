"""Print independently computed static/surface and exact SU(3) diagnostics."""
import json

import numpy as np

from deuteron_wigner.quantum.gluon_surface_consistency import (
    Path, baryon_wilson, box_action_exact, closed_box, cube_graph, cube_rewrites,
    factored_loop, interval_integral, quadratic_action, rectangle,
    static_potential, static_slope, surface_pairing,
)
from deuteron_wigner.quantum.modular_gauge_network import generators


def run():
    boxes = []
    for side in (.25,.5,1.,2.):
        exact = box_action_exact([side]*3)
        numerical = quadratic_action(closed_box([side]*3),order=16)
        boxes.append(dict(side_over_ell=side, exact_S2=exact, numerical_S2=numerical,
                          integration_error=abs(exact-numerical),
                          spurious_closed_surface_factor=float(np.exp(-numerical))))
    graph = cube_graph()
    bottom = graph.transport(Path((0,1,2,3,0)))
    steps = cube_rewrites()
    product = factored_loop(graph)
    scalarized = np.prod([graph.loop(s.based_face) for s in steps])*graph.loop(steps[-1].boundary)
    partners = [graph.transport(s.based_face) for s in steps[1:3]]
    response_error = max(float(np.linalg.norm(factored_loop(graph,e,t)[1]
                                             -graph.path_tangent(Path((0,1,2,3,0)),e,t)))
                         for e in graph.links for t in generators())
    reference = [rectangle(.8,1.2)]
    return dict(
        preedit_commit='8ff38e8456aeb81c52d25f9a91e568f5e12fcc4d',
        scope='Necessary test of scalarized two-point surface closure; exact SU(3) connection algebra; no binding solve',
        units='d=1 GeV^4, ell=1 GeV^-1 are illustrative; S2 dimensionless, V GeV and dV/dR GeV^2',
        candidate='C_AB(z)=delta_AB*d*exp(-z^2/ell^2); omit connector dependence and cumulants beyond quadratic surface action',
        string_slope_limit=float(np.pi/2),
        static_rows=[dict(R=r,V=static_potential(r),slope=static_slope(r)) for r in (.25,.5,1.,2.,4.)],
        finite_rectangle_check=dict(numerical_S2=quadratic_action([rectangle(1.3,3.2)],order=22),
                                    analytic_S2=.5*interval_integral(1.3)*interval_integral(3.2)),
        closed_surfaces=boxes,
        separated_surface_cross_terms=[dict(distance=r, pairing=surface_pairing(reference,[reference[0].translated([0.,r,0.,0.])]))
                                       for r in (0.,1.,2.,4.)],
        exact_SU3=dict(seed=17,link_amplitude=.55,edges=len(graph.links),
                       same_boundary_matrix_error=float(np.linalg.norm(product-bottom)),
                       closed_product_identity_error=float(np.linalg.norm(product@bottom.conj().T-np.eye(3))),
                       scalar_face_product_error=float(abs(np.trace(bottom)/3-scalarized)),
                       baryon_color_amplitude=complex(baryon_wilson([bottom]+partners)),
                       baryon_refactorization_error=float(abs(baryon_wilson([product]+partners)-baryon_wilson([bottom]+partners))),
                       all_96_color_connection_tangent_max_error=response_error,
                       frozen_connector_spurious_tangent=float(np.linalg.norm(factored_loop(graph,(0,4),generators()[1],True)[1])),
                       quantum_vacuum_or_em_current=False),
        conclusion='Do not use this independent scalar two-point surface functional as an exact shared binding interaction; retain connection/ordered higher-correlator and source information. Two-point dominance in a qualified approximation is not excluded.',
        physical_nucleon_nuclear_prediction=False,
        physical_confinement_or_LF_covariance_proved=False,
        nonperturbative_action_selected=False,
        full_mass_metric_bilocal_matching_completed=False)


if __name__ == '__main__':
    def encode(value):
        if isinstance(value, complex):
            return dict(real=value.real,imag=value.imag)
        raise TypeError(type(value).__name__)
    print(json.dumps(run(),indent=2,default=encode))
