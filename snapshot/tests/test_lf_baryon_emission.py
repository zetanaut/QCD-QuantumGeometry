from math import sqrt

import numpy as np
import pytest

from deuteron_wigner.quantum.canonical_lf_fields import GAMMA, lf_spinor, slash
from deuteron_wigner.quantum.canonical_lf_network import four_vector
from deuteron_wigner.quantum.finite_lf_coupling import LFCell, reference_inputs
from deuteron_wigner.quantum.modular_baryon_algebra import wedge
from deuteron_wigner.quantum.lf_baryon_emission import (
    EmissionLayer, canonical_spinors, spin_map, emission_leaf, occupied,
    norm2, difference, total_generator, emission_contact,
)


def layer(p=((.375,.2,0),), shell=1):
    cell,m,g,_=reference_inputs()
    return EmissionLayer(cell,p,.27,g,.43,shell),m


@pytest.mark.parametrize('p', [(.375,0,0),(.625,.2,-.4),(.875,-.6,.2)])
def test_spin_map_is_derived_unitary_change_of_basis(p):
    mass=.27; uc=canonical_spinors(p,mass)
    ul=np.column_stack([lf_spinor(*p,mass,h) for h in (-1,1)])
    r=spin_map(p,mass); cov=ul.conj().T@GAMMA[0]@uc/(2*mass)
    a=mass+sqrt(2)*p[0];pr=p[1]+1j*p[2]
    explicit=np.array([[-pr,a],[a,np.conj(pr)]])/sqrt(a*a+abs(pr)**2)
    assert np.linalg.norm(r-explicit)<3e-15
    assert np.linalg.norm(ul@r-uc)<3e-15
    assert np.linalg.norm(r.conj().T@r-np.eye(2))<3e-15
    assert np.linalg.norm(r-cov)<3e-15
    assert np.linalg.norm((slash(four_vector(p,mass))-mass*np.eye(4))@uc)<3e-15
    if p[1] or p[2]:
        assert np.linalg.norm(r-np.array([[0,1],[1,0]]))>.1


@pytest.mark.parametrize('a,k', [((.125,0,-.2),(.25,.2,.2)),
    ((.375,-.2,.4),(.5,.4,-.2)), ((.625,.2,.2),(.25,-.2,.4))])
def test_emission_crossing_matches_independent_constraint(a,k):
    p=np.array(a)+k
    direct=emission_leaf(a,k,p,.27)
    derived=emission_leaf(a,k,p,.27,constraint=True)
    assert np.linalg.norm(direct)>1
    assert np.linalg.norm(direct-derived)<1e-13
    # All four collective fields and both spin-flip directions remain.
    assert all(np.linalg.norm(direct[:,A,:])>0 for A in range(4))
    assert np.linalg.norm(direct[:6,:,6:])>0
    assert np.linalg.norm(direct[6:,:,:6])>0


@pytest.mark.parametrize('species,q', [('p',1),('n',0)])
@pytest.mark.parametrize('spin', [-1,1])
def test_full_baryon_emission_tensor_color_charge_momentum(species,q,spin):
    obj,_=layer(); p=obj.incoming[0]
    state=obj.interpolator(p,species,spin); out=obj.apply(state)
    assert norm2(state)==pytest.approx(1,abs=3e-15)
    assert norm2(out)>1e-4
    assert difference(out,obj.tensor_reference(state))<1e-15
    assert difference(total_generator(obj,out,charge=True),{k:q*v for k,v in out.items()})<1e-15
    assert max(sqrt(norm2(total_generator(obj,out,color=a))) for a in range(8))<1e-15
    assert {key[0] for key in out}=={'phi','g'}
    assert max(mask.bit_length() for _,_,mask in out)>64
    for (_,k,mask),v in out.items():
        assert len(occupied(mask))==3
        total=np.sum([obj.momenta[i//12] for i in occupied(mask)],axis=0)+k
        assert np.linalg.norm(total-3*np.array(p))<1e-14


def test_all_emitters_are_coherent_not_independent_probabilities():
    obj,_=layer(); state=obj.interpolator(obj.incoming[0],'p',1)
    complete=obj.apply(state)
    single=obj.tensor_reference(state,slots=[0])
    assert difference(complete,{k:3*v for k,v in single.items()})<1e-15
    assert difference(complete,single)>0.01
    assert 3*norm2(single)==pytest.approx(norm2(complete)/3,rel=1e-13)


def test_canonical_spin_is_not_silent_helicity_relabelling():
    obj,_=layer(); p=obj.incoming[0]
    correct=obj.apply(obj.interpolator(p,'p',1))
    relabel=obj.apply(obj.interpolator(p,'p',1,canonical=False))
    assert difference(correct,relabel)>.001
    # Both retain charge/color: those scalar checks cannot qualify spin transport.
    assert difference(total_generator(obj,relabel,charge=True),relabel)<1e-15


def test_unequal_momenta_packet_preserves_correlations_and_fixed_total():
    obj,_=layer(((.375,0,0),(.625,.2,-.2)),shell=1)
    p,q=obj.incoming
    full=obj.wavepacket_interpolator({p:sqrt(.4),q:1j*sqrt(.6)},'n',-1)
    assert norm2(full)==pytest.approx(1,abs=3e-15)
    # Independent occupation projection chooses one p and two q; no dynamics.
    state={mask:v for mask,v in full.items() if sum(i//12==obj.lookup[p] for i in occupied(mask))==1}
    assert norm2(state)==pytest.approx(3*.4*.6**2,abs=3e-15)
    out=obj.apply(state)
    assert difference(out,obj.tensor_reference(state))<2e-15
    assert sqrt(norm2(total_generator(obj,out,charge=True)))<2e-15
    for _,k,mask in out:
        total=np.sum([obj.momenta[i//12] for i in occupied(mask)],axis=0)+k
        assert np.linalg.norm(total-np.array(p)-2*np.array(q))<1e-14


@pytest.mark.parametrize('kind',['phi','g'])
def test_existing_graded_tensor_compiler_agrees_on_whole_channel(kind):
    obj,_=layer(((.375,0,0),),shell=0)
    state=obj.interpolator(obj.incoming[0],'p',1)
    out={key:v for key,v in obj.apply(state).items() if key[0]==kind}
    assert norm2(out)>0
    assert difference(out,obj.graded_reference(state,kind=kind))<1e-15


def test_absorption_is_exact_adjoint_with_all_interference():
    obj,_=layer(); state=obj.interpolator(obj.incoming[0],'n',-1)
    out=obj.apply(state); rng=np.random.default_rng(56)
    probe={k:rng.normal(size=len(v))+1j*rng.normal(size=len(v)) for k,v in out.items()}
    left=sum(np.vdot(probe[k],v) for k,v in out.items())
    back=obj.adjoint(probe)
    right=sum(np.conj(v)*state.get(k,0) for k,v in back.items())
    assert abs(left-right)<3e-15
    assert sum(np.conj(v)*obj.adjoint(out).get(k,0) for k,v in state.items())==pytest.approx(norm2(out),abs=1e-16)


@pytest.mark.parametrize('species',['p','n'])
def test_first_order_wave_matches_independent_time_integration(species):
    obj,m=layer(((.375,0,0),),shell=0)
    state=obj.interpolator(obj.incoming[0],species,1)
    wave=obj.pulse(state,m,12)
    direct=obj.pulse(state,m,12,direct=True)
    assert norm2(wave)>1e-4
    assert difference(wave,direct)<1e-12
    q=int(species=='p')
    assert difference(total_generator(obj,wave,charge=True),{k:q*v for k,v in wave.items()})<1e-15


def test_window_enlargement_keeps_old_amplitudes_and_exposes_missing_wave():
    small,m=layer(shell=1); large,_=layer(shell=2)
    def physical(obj):
        out=obj.pulse(obj.interpolator(obj.incoming[0],'p',1),m,12)
        return {(kind,k,tuple((obj.momenta[i//12],i%12) for i in occupied(mask))):v
                for (kind,k,mask),v in out.items()}
    a,b=physical(small),physical(large)
    assert a.keys()<=b.keys()
    assert max(np.linalg.norm(v-b[k]) for k,v in a.items())<2e-15
    assert norm2({k:v for k,v in b.items() if k not in a})>1e-8


def test_invalid_domains_are_explicit():
    obj,m=layer()
    with pytest.raises(ValueError): spin_map((.375,0,0),0)
    with pytest.raises(ValueError): emission_leaf((.125,0,0),(.25,0,0),(.625,0,0),.27)
    with pytest.raises(ValueError): obj.wavepacket_interpolator({obj.incoming[0]:2},'p',1)
    with pytest.raises(ValueError): obj.apply({1 << (12*len(obj.momenta)):1})
    with pytest.raises(ValueError): obj.pulse({},m,12)


@pytest.mark.parametrize('component',range(4))
def test_crossed_mixed_source_contact_is_derived_not_lost(component):
    obj,_=layer();a=(.125,.2,-.2);k=(.25,-.2,.2);p=(.375,.2,.2)
    e=np.eye(4)[component]
    direct=emission_contact(a,k,p,.27,e,obj.cell.volume)
    cov=emission_contact(a,k,p,.27,e,obj.cell.volume,covariant=True)
    assert np.linalg.norm(direct-cov)<1e-15
    if component==1:
        assert np.linalg.norm(direct)==0  # No mixed J+ contact.
    else:
        assert np.linalg.norm(direct)>.001


def test_variable_six_quark_composition_keeps_exchange_and_emission():
    obj,_=layer(((.375,0,0),),shell=0);p=obj.incoming[0]
    a=obj.interpolator(p,'p',1);b=obj.interpolator(p,'n',-1)
    composite=wedge(a,b)
    assert norm2(composite)>0 and abs(norm2(composite)-1)>.01
    actual=obj.apply(composite);expected={}
    for wave,other,left in ((obj.apply(a),b,True),(obj.apply(b),a,False)):
        for (kind,k,mask),v in wave.items():
            product=wedge({mask:1.},other) if left else wedge(other,{mask:1.})
            for out,value in product.items():
                key=(kind,k,out);expected[key]=expected.get(key,0)+value*v
    assert difference(actual,expected)<1e-15
    assert difference(total_generator(obj,actual,charge=True),actual)<1e-15


def test_longitudinal_boost_of_entire_lf_wave_with_cell_and_state_transport():
    obj,m=layer();p=obj.incoming[0];factor=1.7
    # Cell lengths and LF time transform; compare the SAME physical LF state,
    # not a newly reset canonical-spin interpolator in the boosted frame.
    cell=LFCell((obj.cell.lengths[0]/factor,*obj.cell.lengths[1:]))
    boosted=EmissionLayer(cell,[(p[0]*factor,*p[1:])],obj.mass,obj.g,obj.gs,obj.shell)
    seed=obj.interpolator(p,'p',1)
    assert len(obj.momenta)==len(boosted.momenta)
    wave=obj.pulse(seed,m,12);other=boosted.pulse(seed,m,12*factor)
    other={(kind,(k[0]/factor,*k[1:]),mask):v for (kind,k,mask),v in other.items()}
    assert difference(wave,other)<2e-14
