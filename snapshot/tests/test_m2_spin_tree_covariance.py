"""On-shell Lorentz/source normalization and explicit regulated failure controls."""
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from deuteron_wigner.bridge.modes.core import GAMMA, polarization, lf_dot
from deuteron_wigner.quantum.m2_canonical_emission import lf_spinor
from deuteron_wigner.quantum.m2_spin_tree_covariance import elastic_distinct_quark_tree
from deuteron_wigner.quantum.m2_gaussian_field_regulator import GaussianClusterRegulator
from deuteron_wigner.quantum.m2_gaussian_reference_shape import GaussianReferenceShape


def scattering(masses, angle=.8):
    incoming = np.array([.4,.2,.5])
    rotation = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    outgoing = rotation@incoming
    vectors = (outgoing,-outgoing,incoming,-incoming)
    return np.array([[np.sqrt(v@v+m*m),*v] for v,m in zip(vectors,(*masses,*masses))])


def lightfront(p):
    return np.column_stack(((p[:,0]+p[:,3])/np.sqrt(2),p[:,1:3]))


def lorentz_pair(name):
    """Independent Cartesian spinor/four-vector transformations."""
    pauli=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
    pairs={}
    for j,a in enumerate("xyz"):
        b=np.zeros((4,4));b[0,j+1]=b[j+1,0]=1
        r=np.zeros((4,4));r[1:,1:]=np.column_stack([np.cross(np.eye(3)[j],e) for e in np.eye(3)])
        pairs["B"+a]=(b,GAMMA[0]@GAMMA[j+1]/2)
        pairs["J"+a]=(r,-.5j*np.kron(np.eye(2),pauli[j]))
    for name0,a,b,s in (("F1","Bx","Jy",-1),("F2","By","Jx",1),
                         ("E1","Bx","Jy",1),("E2","By","Jx",-1)):
        pairs[name0]=tuple((x+s*y)/np.sqrt(2) for x,y in zip(pairs[a],pairs[b]))
    return pairs[name]


def transformed(p,masses,name,t):
    l,s=lorentz_pair(name)
    moved=p@expm(t*l).T
    ds=[]
    for old,new,energy,m in zip(lightfront(p),lightfront(moved),moved[:,0],(*masses,*masses)):
        u=np.column_stack([lf_spinor(*old,m,h) for h in (-1,1)])
        v=np.column_stack([lf_spinor(*new,m,h) for h in (-1,1)])
        spinor=expm(t*s)@u
        d=v.conj().T@spinor/(2*energy)
        assert_allclose(v@d,spinor,atol=2e-14)
        assert_allclose(d.conj().T@d,np.eye(2),atol=2e-14)
        ds.append(d)
    return moved,np.kron(*ds[:2]),np.kron(*ds[2:])


@pytest.mark.parametrize("masses",[(0.,0.),(.3,.3),(.3,.5)])
@pytest.mark.parametrize("angle",[.8,-.6,1.7])
def test_actual_cubic_plus_Gauss_tree_matches_covariant_tensor_in_both_exchange_directions(masses,angle):
    p=scattering(masses,angle)
    result=elastic_distinct_quark_tree(lightfront(p),masses)
    assert result["exchange_GeV2"] < 0
    assert result["external_current_conservation_residual"] < 2e-15
    assert_allclose(result["bare_total"],result["covariant"],atol=5e-13)
    assert np.linalg.norm(result["instantaneous"]) > 1.
    assert np.linalg.norm(result["propagating"]) > 1.
    assert_allclose(result["regulated_total"],result["bare_total"],atol=1e-15)


@pytest.mark.parametrize("name",["Bx","By","Bz","Jx","Jy","Jz","E1","E2","F1","F2"])
@pytest.mark.parametrize("masses",[(0.,0.),(.3,.3),(.3,.5)])
def test_full_bare_tree_external_spin_transport_is_Lorentz_covariant(name,masses):
    p=scattering(masses)
    moved,do,di=transformed(p,masses,name,.35)
    a=elastic_distinct_quark_tree(lightfront(p),masses)
    b=elastic_distinct_quark_tree(lightfront(moved),masses)
    assert_allclose(do.conj().T@b["bare_total"]@di,a["bare_total"],atol=7e-13)
    assert_allclose(b["exchange_GeV2"],a["exchange_GeV2"],atol=1e-15)


@pytest.mark.parametrize("masses",[(0.,0.),(.3,.3),(.3,.5)])
@pytest.mark.parametrize("degree",[None,0,2])
def test_regulator_weights_are_the_actual_physical_leg_Gaussians(masses,degree):
    p=lightfront(scattering(masses))
    shape=None if degree is None else GaussianReferenceShape(base_width_GeV=.8,reference_width_GeV=1.2,degree=degree)
    reg=GaussianClusterRegulator(width_GeV=.8,boson_modes=8,fermion_modes=8,reference_shape=shape)
    d=elastic_distinct_quark_tree(p,masses,regulator=reg)
    k=d["positive_exchange_mode"]
    # Independent evaluation via the exact source field-map transform, not
    # the regulator Lorentz-derivative helper used in the implementation.
    weights=[]
    for legs in (np.array([p[0],k,p[2]]),np.array([p[1],k,p[3]]),p):
        q=np.sum((reg.leg_transform(legs[:,0])@legs[:,1:])**2)
        weights.append(np.exp(-q/(2*.8**2)) if shape is None else shape.value_and_derivative(q)[0])
    assert_allclose([*d["cubic_weights"],d["quartic_weight"]],weights,atol=3e-16)
    assert_allclose(d["regulated_total"],d["covariant_tensor_weight"]*d["covariant"]+d["light_front_direction_term"],atol=1e-14)
    assert np.linalg.norm(d["light_front_direction_term"]) > .1


@pytest.mark.parametrize("masses",[(0.,0.),(.3,.3),(.3,.5)])
@pytest.mark.parametrize("degree",[None,0,2])
def test_current_regulator_preserves_kinematic_but_fails_dynamical_tree_covariance(masses,degree):
    shape=None if degree is None else GaussianReferenceShape(base_width_GeV=.8,reference_width_GeV=1.2,degree=degree)
    reg=GaussianClusterRegulator(width_GeV=.8,boson_modes=8,fermion_modes=8,reference_shape=shape)
    p=scattering(masses)
    a=elastic_distinct_quark_tree(lightfront(p),masses,regulator=reg)
    for name in ("E1","E2","Jz","Bz","F1","F2"):
        moved,do,di=transformed(p,masses,name,.35)
        b=elastic_distinct_quark_tree(lightfront(moved),masses,regulator=reg)
        pulled=do.conj().T@b["regulated_total"]@di
        error=np.linalg.norm(pulled-a["regulated_total"])/np.linalg.norm(a["regulated_total"])
        if name in ("E1","E2","Jz","Bz"):
            assert error < 2e-13
        else:
            assert error > .02
            # Spin-summed probability is independently basis/phase invariant;
            # the failure cannot be repaired by choosing different spin axes.
            rate_a=np.linalg.norm(a["regulated_total"])**2/4
            rate_b=np.linalg.norm(b["regulated_total"])**2/4
            assert abs(rate_b-rate_a)/rate_a > .02


def test_regulated_tree_violation_vanishes_in_removed_smooth_regulator_control():
    masses=(.3,.3);p=scattering(masses)
    moved,do,di=transformed(p,masses,"F1",.35)
    errors=[]
    # Use the asymptotic large-width regime, not Lambda comparable to the
    # external invariant momenta, when checking the leading 1/Lambda² law.
    for width in (8.,16.,32.,64.):
        reg=GaussianClusterRegulator(width_GeV=width,boson_modes=8,fermion_modes=8)
        a=elastic_distinct_quark_tree(lightfront(p),masses,regulator=reg)
        b=elastic_distinct_quark_tree(lightfront(moved),masses,regulator=reg)
        errors.append(np.linalg.norm(do.conj().T@b["regulated_total"]@di-a["regulated_total"]))
    assert np.all(np.diff(errors)<0)
    assert_allclose(np.array(errors[2:])/np.array(errors[1:3]),[.25,.25],rtol=.004)


def test_exchange_and_time_reversal_remain_exact_with_the_same_regulator():
    masses=(.3,.5);p=lightfront(scattering(masses))
    reg=GaussianClusterRegulator(width_GeV=.8,boson_modes=8,fermion_modes=8)
    a=elastic_distinct_quark_tree(p,masses,regulator=reg)
    b=elastic_distinct_quark_tree(p[[2,3,0,1]],masses,regulator=reg)
    c=elastic_distinct_quark_tree(p[[1,0,3,2]],masses[::-1],regulator=reg)
    for key in ("bare_total","regulated_total","covariant"):
        assert_allclose(b[key],a[key].conj().T,atol=3e-14)
        assert_allclose(c[key].reshape(2,2,2,2).transpose(1,0,3,2).reshape(4,4),a[key],atol=3e-14)


def test_energy_nonconservation_and_missing_zero_transfer_domain_are_rejected():
    masses=(.3,.3);p=lightfront(scattering(masses))
    q=p.copy();q[0,1]+=.02
    with pytest.raises(ValueError,match="conservation"):
        elastic_distinct_quark_tree(q,masses)
    with pytest.raises(ValueError,match="nonzero"):
        elastic_distinct_quark_tree(p[[2,3,2,3]],masses)
