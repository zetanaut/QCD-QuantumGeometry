from itertools import product
from math import sqrt
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from deuteron_wigner.quantum.canonical_lf_fields import (
    GAMMA, GAMMA_PLUS, GAMMA_MINUS, lf_spinor, antiquark_spinor,
    polarization, polarization_cartesian, slash, three_gluon_numerator,
)
from deuteron_wigner.quantum.canonical_lf_network import (
    cubic_kernel, cubic_word, cubic_vertex_tensor, dot, four_vector, electromagnetic_current,
    fermion_propagator_parts, quark_gluon_tree,
)
from deuteron_wigner.quantum.graded_fock_network import Mode, Word, Ladder, FockRegister
from deuteron_wigner.quantum.modular_gauge_network import generators, su3, adjoint


def scattering(mass, angle=.8):
    v = np.array([.31, .17, .52])
    rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    w = rot@v
    return np.array([[sqrt(x@x+m*m), *x] for x, m in ((w, mass), (-w, 0), (v, mass), (-v, 0))])


def lf(p):
    return np.column_stack(((p[:, 0]+p[:, 3])/sqrt(2), p[:, 1:3]))


def test_new_entry_does_not_import_the_old_model_stack():
    code = '''import sys
import deuteron_wigner.quantum.canonical_lf_network
forbidden=('m2_coupled','m2_gaussian','m2_h2','m2_shell','m2_canonical_emission','m2_excluded_fock','deuteron_wigner.bridge.modes','m2_permutation_complete_h2','m2_total_sign_h2_hamiltonian','m2_k9_exploratory_state','deuteron_wigner.quantum.operator_bundle')
assert not [name for name in sys.modules if any(x in name for x in forbidden)]
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_lazy_legacy_public_exports_preserve_declared_names_and_last_import_wins():
    import importlib
    import deuteron_wigner.quantum as package
    expected = {}
    for module in package._LEGACY_EXPORTS:
        source = importlib.import_module('.'+module, package.__name__)
        for name in source.__all__:
            expected[name] = getattr(source, name)
    assert set(expected) == set(package.__all__)
    assert all(getattr(package, name) is value for name, value in expected.items())
    with pytest.raises(AttributeError):
        getattr(package, 'not_a_physical_model')


def test_legacy_names_are_aliases_to_reused_functions_not_new_formula_copies():
    from deuteron_wigner.bridge.modes import core
    from deuteron_wigner.quantum import m2_canonical_emission, m2_excluded_fock_vertices
    assert core.GAMMA is GAMMA and core.polarization is polarization
    assert m2_canonical_emission.lf_spinor is lf_spinor
    assert m2_excluded_fock_vertices.antiquark_spinor is antiquark_spinor
    assert m2_excluded_fock_vertices.three_gluon_numerator is three_gluon_numerator


@pytest.mark.parametrize('mass', [0., .27, .8])
@pytest.mark.parametrize('species', ['q', 'qbar'])
def test_reused_free_current_all_components_against_independent_Dirac_contraction(mass, species):
    a, b = np.array([.61, .24, -.18]), np.array([1.4, -.31, .13])
    fn = lf_spinor if species == 'q' else antiquark_spinor
    out, inc = [np.column_stack([fn(*p, mass, h) for h in (-1, 1)]) for p in (a, b)]
    gs = (GAMMA_PLUS, GAMMA_MINUS, GAMMA[1], GAMMA[2])
    expected = np.array([out.conj().T@GAMMA[0]@g@inc if species == 'q'
                         else -(inc.conj().T@GAMMA[0]@g@out).T for g in gs])/(2*sqrt(a[0]*b[0]))
    actual = electromagnetic_current(a, b, mass, species=species, quark_charge=2/3)
    np.testing.assert_allclose(actual, expected*2/3, atol=2e-14)
    delta = four_vector(a, mass)-four_vector(b, mass)
    qp, qm = (delta[0]+delta[3])/sqrt(2), (delta[0]-delta[3])/sqrt(2)
    np.testing.assert_allclose(qp*actual[1]+qm*actual[0]-delta[1]*actual[2]-delta[2]*actual[3], 0, atol=2e-14)
    np.testing.assert_allclose(actual[0], np.eye(2)*(2/3 if species == 'q' else -2/3), atol=2e-14)
    reversed_current = electromagnetic_current(b, a, mass, species=species, quark_charge=2/3)
    np.testing.assert_allclose(actual, reversed_current.conj().transpose(0, 2, 1), atol=2e-14)


@pytest.mark.parametrize('channel', ['q_to_qg', 'qbar_to_qbarg', 'g_to_pair', 'g_to_gg'])
def test_cubic_leaves_graded_actions_boson_enhancement_and_source_units(channel):
    a, b = np.array([.7, .17, -.2]), np.array([.9, -.08, .11])
    p = a+b
    choices = list(product((-1, 1), repeat=3))
    values = [cubic_kernel(channel, a, b, p, .3, h) for h in choices]
    h = choices[int(np.argmax(np.abs(values)))]
    c = cubic_kernel(channel, a, b, p, .3, h)
    assert abs(c) > .01
    # Dirac numerator scales as momentum; normalized three-field kernel as E^-1/2.
    np.testing.assert_allclose(cubic_kernel(channel, 4*a, 4*b, 4*p, 1.2, h), c/2, atol=2e-14)
    stats = {'q_to_qg': ('fermion','boson','fermion'),
             'qbar_to_qbarg': ('fermion','boson','fermion'),
             'g_to_pair': ('fermion','fermion','boson'),
             'g_to_gg': ('boson','boson','boson')}[channel]
    register = FockRegister(tuple(Mode(str(i), s, 1 if s == 'fermion' else 3) for i, s in enumerate(stats)))
    word = cubic_word(channel, c, 0, 1, 2)
    for initial in product(*[range(d) for d in register.dimensions]):
        expected = register.occupation_action([word], {initial: 1})
        actual = register.tensor_action([word], {initial: 1})
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-14)
    np.testing.assert_allclose(register.dense([word.adjoint()]), register.dense([word]).conj().T, atol=1e-14)
    with pytest.raises(ValueError, match='conservation'):
        cubic_kernel(channel, a, b, p+[0, .01, 0], .3, h)


def emission_matrix():
    left, gluon = np.array([.7, .17, -.2]), np.array([.9, -.08, .11])
    parent = left+gluon
    modes = tuple(Mode(f'q{i}', 'fermion', 1) for i in range(12))+tuple(Mode(f'g{i}', 'boson', 1) for i in range(16))
    register = FockRegister(modes)
    words, expected = [], np.zeros((96, 6), complex)
    for ho, hi, hg in product(range(2), repeat=3):
        coefficient = cubic_kernel('q_to_qg', left, gluon, parent, .3, (2*ho-1, 2*hg-1, 2*hi-1))
        for co, ci, a in product(range(3), range(3), range(8)):
            value = coefficient*generators()[a, co, ci]
            expected[(ho*3+co)*16+hg*8+a, hi*3+ci] = value
            if value != 0:
                words.append(cubic_word('q_to_qg', value, 6+ho*3+co, 12+hg*8+a, hi*3+ci))
    def occupation(*indices):
        return tuple(int(i in indices) for i in range(28))
    incoming = [occupation(i) for i in range(6)]
    outgoing = [occupation(6+i, 12+j) for i in range(6) for j in range(16)]
    return register, words, incoming, outgoing, expected


def test_color_spin_resolved_canonical_emission_enters_new_tensor_network():
    register, words, incoming, outgoing, expected = emission_matrix()
    actual = register.matrix_elements(words, outgoing, incoming)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    # Check a complex state directly, preserving coherent emitter/spin/color amplitudes.
    rng = np.random.default_rng(44)
    c = rng.normal(size=6)+1j*rng.normal(size=6)
    direct = register.occupation_action(words, dict(zip(incoming, c)))
    np.testing.assert_allclose([direct.get(o, 0) for o in outgoing], actual@c, atol=1e-14)
    g = su3(rng.normal(size=8))
    transformed = np.kron(np.kron(np.eye(2), g), np.kron(np.eye(2), adjoint(g)))
    np.testing.assert_allclose(transformed@actual, actual@np.kron(np.eye(2), g), atol=2e-14)
    # Free additive charge and plus/transverse momentum are the same on each nonzero word.
    assert all(sum(o[:12]) == 1 and sum(o[12:]) == 1 for o in outgoing)


@pytest.mark.parametrize('channel,particles', [
    ('q_to_qg', ('q','g','q')), ('qbar_to_qbarg', ('qbar','g','qbar')),
    ('g_to_pair', ('q','qbar','g')), ('g_to_gg', ('g','g','g')),
])
def test_all_resolved_cubic_tensors_preserve_SU3_and_compile_to_graded_words(channel, particles):
    a,b = np.array([.7,.17,-.2]), np.array([.9,-.08,.11])
    v = cubic_vertex_tensor(channel,a,b,a+b,.3)
    rng = np.random.default_rng(732)
    g = su3(rng.normal(size=8))
    reps = {'q': g, 'qbar': g.conj(), 'g': adjoint(g)}
    maps = [np.kron(np.eye(2),reps[p]) for p in particles]
    matrix = v.reshape(-1,v.shape[-1])
    np.testing.assert_allclose(np.kron(maps[0],maps[1])@matrix,matrix@maps[2],atol=3e-14)
    offsets = np.cumsum([0,*v.shape])
    modes = tuple(Mode(f'{group}:{i}', 'boson' if p=='g' else 'fermion',1)
                  for group,(p,n) in enumerate(zip(particles,v.shape)) for i in range(n))
    register = FockRegister(modes)
    words = [cubic_word(channel,v[i,j,k],i,offsets[1]+j,offsets[2]+k)
             for i,j,k in zip(*np.nonzero(v))]
    def occ(*indices):
        return tuple(int(i in indices) for i in range(len(modes)))
    coefficients = rng.normal(size=v.shape[2])+1j*rng.normal(size=v.shape[2])
    initial = {occ(offsets[2]+k): c for k,c in enumerate(coefficients)}
    tensor = register.tensor_action(words,initial)
    direct = register.occupation_action(words,initial)
    outputs = [occ(i,offsets[1]+j) for i in range(v.shape[0]) for j in range(v.shape[1])]
    np.testing.assert_allclose([tensor.get(o,0) for o in outputs],matrix@coefficients,atol=3e-14)
    np.testing.assert_allclose([direct.get(o,0) for o in outputs],matrix@coefficients,atol=3e-14)
    if channel == 'g_to_gg':
        exchanged = cubic_vertex_tensor(channel,b,a,a+b,.3).transpose(1,0,2)
        np.testing.assert_allclose(v,exchanged,atol=2e-14)


@pytest.mark.parametrize('species', ['q','qbar'])
def test_four_current_uses_the_same_graded_compiler_with_occupied_spectator(species):
    a,b = np.array([.61,.24,-.18]), np.array([1.4,-.31,.13])
    current = electromagnetic_current(a,b,.3,species=species,quark_charge=-1/3)
    register = FockRegister(tuple(Mode(str(i),'fermion',1) for i in range(5)))
    # Spectator lies between incoming and outgoing modes: the transfer sign is negative.
    incoming = [(1,0,1,0,0),(0,1,1,0,0)]
    outgoing = [(0,0,1,1,0),(0,0,1,0,1)]
    for component in current:
        words = [Word(component[i,j],(Ladder(3+i,True),Ladder(j,False)))
                 for i,j in product(range(2),repeat=2)]
        np.testing.assert_allclose(register.matrix_elements(words,outgoing,incoming),-component,atol=2e-14)
        for j,inc in enumerate(incoming):
            direct = register.occupation_action(words,{inc:1})
            np.testing.assert_allclose([direct.get(out,0) for out in outgoing],-component[:,j],atol=2e-14)


@pytest.mark.parametrize('r', [(1.3,.2,-.3,.5), (-.4,.1,.2,-.9), (.2,.7,.1,-.6)])
def test_signed_fermion_constraint_part_completes_the_propagator(r):
    r = np.array(r); mass=.31
    propagating, instantaneous = fermion_propagator_parts(r, mass)
    covariant = (slash(r)+mass*np.eye(4))/(dot(r,r)-mass*mass)
    np.testing.assert_allclose(propagating+instantaneous, covariant, atol=2e-14)
    assert np.linalg.norm(instantaneous) > .1


@pytest.mark.parametrize('mass,angle', product((0., .3, .7), (.35, .8, 1.7, 2.6)))
def test_complete_nonAbelian_tree_LF_covariant_and_both_external_Ward_controls(mass, angle):
    momenta = lf(scattering(mass, angle))
    result = quark_gluon_tree(*momenta, mass)
    scale = max(1., np.linalg.norm(result['covariant']))
    assert np.linalg.norm(result['light_front']-result['covariant'])/scale < 1e-12
    assert result['three_gluon_divergence'] < 1e-12
    assert np.linalg.norm(result['instantaneous'])/scale > .01
    for which in ('replace_incoming', 'replace_outgoing'):
        ward = quark_gluon_tree(*momenta, mass, **{which: True})
        assert np.linalg.norm(ward['covariant'])/scale < 1e-12
        assert np.linalg.norm(ward['light_front'])/scale < 1e-12
        # Non-Abelian three-gluon graph is necessary: two fermion graphs alone fail.
        assert np.linalg.norm(ward['s_covariant']+ward['u_covariant'])/scale > .001


@pytest.mark.parametrize('axis,kind', product(range(3), ('boost','rotation')))
@pytest.mark.parametrize('mass', [0., .3])
def test_complete_tree_under_independent_Lorentz_and_external_spin_polarization_transport(axis, kind, mass):
    original = scattering(mass)
    generator = np.zeros((4,4))
    if kind == 'boost':
        generator[0,axis+1] = generator[axis+1,0] = 1
        spin_generator = GAMMA[0]@GAMMA[axis+1]/2
    else:
        generator[1:,1:] = np.column_stack([np.cross(np.eye(3)[axis], e) for e in np.eye(3)])
        j,k = [(2,3),(3,1),(1,2)][axis]
        spin_generator = GAMMA[j]@GAMMA[k]/2
    transform = expm(.27*generator)
    moved = original@transform.T
    old_lf, new_lf = lf(original), lf(moved)
    quark_maps, gluon_maps = [], []
    for endpoint in (0,2):
        u,v = [np.column_stack([lf_spinor(*p,mass,h) for h in (-1,1)]) for p in (old_lf[endpoint],new_lf[endpoint])]
        d = v.conj().T@expm(.27*spin_generator)@u/(2*moved[endpoint,0])
        np.testing.assert_allclose(d.conj().T@d,np.eye(2),atol=2e-14)
        quark_maps.append(d)
    for endpoint in (1,3):
        old,new = [np.column_stack([polarization_cartesian(polarization(*p,h)) for h in (-1,1)]) for p in (old_lf[endpoint],new_lf[endpoint])]
        metric = np.diag([1,-1,-1,-1])
        d = -new.conj().T@metric@transform@old
        np.testing.assert_allclose(d.conj().T@d,np.eye(2),atol=2e-14)
        gluon_maps.append(d)
    maps = [np.kron(np.kron(np.kron(q,np.eye(3)),g),np.eye(8)) for q,g in zip(quark_maps,gluon_maps)]
    a,b = [quark_gluon_tree(*p,mass)['light_front'] for p in (old_lf,new_lf)]
    assert np.linalg.norm(maps[0].conj().T@b@maps[1]-a)/np.linalg.norm(a) < 2e-12


def test_zero_plus_or_pole_domain_is_not_silently_filled_with_zero():
    with pytest.raises(ValueError):
        fermion_propagator_parts([1,0,0,-1],.3)
    with pytest.raises(ValueError):
        quark_gluon_tree(*lf(scattering(.3,0)),.3)
