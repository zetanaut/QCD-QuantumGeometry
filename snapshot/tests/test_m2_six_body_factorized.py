import numpy as np
import pytest

from deuteron_wigner.quantum.m2_excluded_fock_action import IntermediateSpace
from deuteron_wigner.quantum.m2_six_body_cubic import SixBodyCubicAction,five_statistics_project
from deuteron_wigner.quantum.m2_six_body_factorized import FactorizedSixBodyCubicAction


def make(cutoff,outcut=None,total=9):
    spaces={n:IntermediateSpace(n,total,cutoff,"PROTON",1) for n in ("qqqgg","qqqq-qbar")}
    return FactorizedSixBodyCubicAction(spaces,out_cutoff=cutoff if outcut is None else outcut,
                                      b_GeV=.4,mass_GeV=.35)


def random_columns(spaces,seed=789):
    rng=np.random.default_rng(seed)
    return {n:(rng.normal(size=(s.dimension,2))+1j*rng.normal(size=(s.dimension,2)))/np.sqrt(s.dimension)
            for n,s in spaces.items()}


@pytest.mark.parametrize("cutoff,outcut,total",[(0,0,9),(1,1,9),(0,2,9),(0,0,11)])
def test_each_factorized_component_and_reverse_against_full_sparse_reference(cutoff,outcut,total):
    f=make(cutoff,outcut,total)
    old=SixBodyCubicAction(f.incoming,out_cutoff=outcut,b_GeV=.4,mass_GeV=.35)
    x,z=random_columns(f.incoming),random_columns(f.outgoing,791)
    for label,(s,t,w) in old.components.items():
        np.testing.assert_allclose(f.component_action(label,x[s]),w@x[s],atol=3e-16)
        np.testing.assert_allclose(f.component_action(label,z[t],adjoint=True),w.getH()@z[t],atol=3e-16)
    y,back=f.apply(x),f.adjoint(z)
    expected,reverse=old.apply(x),old.adjoint(z)
    for n in y:np.testing.assert_allclose(y[n],expected[n],atol=3e-16)
    for n in back:np.testing.assert_allclose(back[n],reverse[n],atol=3e-16)


def test_complete_second_shell_adjoint_statistics_and_coherent_channels():
    f=make(2);x,z=random_columns(f.incoming),random_columns(f.outgoing,795)
    y,back=f.apply(x),f.adjoint(z)
    np.testing.assert_allclose(sum(np.vdot(z[n],y[n]) for n in y),
                               sum(np.vdot(back[n],x[n]) for n in x),atol=5e-17)
    for n,s in f.outgoing.items():
        assert np.linalg.norm(y[n])>1e-3
        np.testing.assert_allclose(s.statistics_project(y[n]),y[n],atol=5e-16)
    for n,s in f.incoming.items():
        np.testing.assert_allclose(five_statistics_project(s,back[n]),back[n],atol=4e-16)
    separated=[]
    for selected in x:
        source={n:(v if n==selected else np.zeros_like(v)) for n,v in x.items()}
        separated.append(f.apply(source))
    for n in y:
        np.testing.assert_allclose(sum(a[n] for a in separated),y[n],atol=5e-16)
    assert all(np.linalg.norm(a["qqqq-qbar-g"])>1e-3 for a in separated)


def test_factorized_operator_rejects_incomplete_coordinates_and_invalid_values():
    s=IntermediateSpace("qqqgg",9,0,"PROTON",1)
    with pytest.raises(ValueError):
        FactorizedSixBodyCubicAction({"qqqgg":s},out_cutoff=0,b_GeV=.4,mass_GeV=.35)
    f=make(0);x=random_columns(f.incoming)
    with pytest.raises(ValueError):f.apply({"qqqgg":x["qqqgg"]})
    with pytest.raises(ValueError):f.component_action("qqqgg:q_to_qg",x["qqqgg"],adjoint=1)
