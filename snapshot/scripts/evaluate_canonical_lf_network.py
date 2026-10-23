"""Replay portable canonical leaves, graded actions and a complete bare QCD tree.

No bound-state Hamiltonian, model regulator, physical input fit or H2 reduction.
Run with PYTHONPATH=src:. and explicit --output; numerical outputs are generated.
"""
import argparse
import json
from math import sqrt
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.linalg import expm

from deuteron_wigner.quantum.canonical_lf_fields import (
    GAMMA, lf_spinor, polarization, polarization_cartesian,
)
from deuteron_wigner.quantum.canonical_lf_network import (
    quark_gluon_tree, cubic_vertex_tensor, cubic_word,
)
from deuteron_wigner.quantum.graded_fock_network import Mode, FockRegister
from deuteron_wigner.quantum.modular_gauge_network import su3, adjoint


def lf(vectors):
    return np.column_stack(((vectors[:,0]+vectors[:,3])/sqrt(2), vectors[:,1:3]))


def scattering(mass, vector, axis, angle):
    axis = np.asarray(axis)/np.linalg.norm(axis)
    rotation = expm(angle*np.column_stack([np.cross(axis,e) for e in np.eye(3)]))
    outgoing = rotation@vector
    return np.array([[sqrt(v@v+m*m),*v] for v,m in
                     ((outgoing,mass),(-outgoing,0),(vector,mass),(-vector,0))])


def transported_error(original, mass, axis, kind, parameter):
    generator = np.zeros((4,4))
    if kind == 'boost':
        generator[0,axis+1] = generator[axis+1,0] = 1
        spin_generator = GAMMA[0]@GAMMA[axis+1]/2
    else:
        generator[1:,1:] = np.column_stack([np.cross(np.eye(3)[axis],e) for e in np.eye(3)])
        j,k = [(2,3),(3,1),(1,2)][axis]
        spin_generator = GAMMA[j]@GAMMA[k]/2
    transform = expm(parameter*generator)
    moved = original@transform.T
    old_lf, new_lf = lf(original),lf(moved)
    quark,gluon = [],[]
    for endpoint in (0,2):
        u,v = [np.column_stack([lf_spinor(*p,mass,h) for h in (-1,1)])
               for p in (old_lf[endpoint],new_lf[endpoint])]
        quark.append(v.conj().T@expm(parameter*spin_generator)@u/(2*moved[endpoint,0]))
    for endpoint in (1,3):
        u,v = [np.column_stack([polarization_cartesian(polarization(*p,h)) for h in (-1,1)])
               for p in (old_lf[endpoint],new_lf[endpoint])]
        gluon.append(-v.conj().T@np.diag([1,-1,-1,-1])@transform@u)
    maps = [np.kron(np.kron(np.kron(q,np.eye(3)),g),np.eye(8)) for q,g in zip(quark,gluon)]
    a,b = [quark_gluon_tree(*p,mass)['light_front'] for p in (old_lf,new_lf)]
    return {
        'axis': axis, 'kind': kind, 'parameter': parameter,
        'relative_amplitude_error': float(np.linalg.norm(maps[0].conj().T@b@maps[1]-a)/np.linalg.norm(a)),
        'external_map_unitarity_error': float(max(np.linalg.norm(d.conj().T@d-np.eye(len(d))) for d in quark+gluon)),
    }


def tree_case(vectors, mass, label):
    momenta = lf(vectors)
    full = quark_gluon_tree(*momenta,mass)
    scale = np.linalg.norm(full['covariant'])
    energy = max(vectors[:,0])
    wards = {}
    for side in ('incoming','outgoing'):
        w = quark_gluon_tree(*momenta,mass,**{'replace_'+side:True})
        wards[side] = {
            'covariant_relative': float(np.linalg.norm(w['covariant'])/(scale*energy)),
            'LF_relative': float(np.linalg.norm(w['light_front'])/(scale*energy)),
            'omit_three_gluon_graph_relative': float(np.linalg.norm(w['s_covariant']+w['u_covariant'])/(scale*energy)),
        }
    return {
        'label': label, 'mass_GeV': mass, 'external_Cartesian_GeV': vectors.tolist(),
        'matrix_shape': list(full['covariant'].shape),
        'LF_covariant_relative': float(np.linalg.norm(full['light_front']-full['covariant'])/scale),
        'omit_instantaneous_relative': float(np.linalg.norm(full['propagating']-full['covariant'])/scale),
        'crossed_fermion_plus_GeV': full['crossed_fermion_plus'],
        'exchange_GeV2': full['exchange_GeV2'], 'external_gluon_Ward': wards,
    }


def cubic_case(channel, particles):
    a,b = np.array([.7,.17,-.2]),np.array([.9,-.08,.11])
    vertex = cubic_vertex_tensor(channel,a,b,a+b,.3)
    matrix = vertex.reshape(-1,vertex.shape[2])
    rng = np.random.default_rng(20260910)
    g = su3(rng.normal(size=8))
    reps = {'q':g,'qbar':g.conj(),'g':adjoint(g)}
    maps = [np.kron(np.eye(2),reps[p]) for p in particles]
    offsets = np.cumsum([0,*vertex.shape])
    modes = tuple(Mode(f'{group}:{i}','boson' if p=='g' else 'fermion',1)
                  for group,(p,n) in enumerate(zip(particles,vertex.shape)) for i in range(n))
    register = FockRegister(modes)
    words = [cubic_word(channel,vertex[i,j,k],i,offsets[1]+j,offsets[2]+k)
             for i,j,k in zip(*np.nonzero(vertex))]
    def occ(*indices):
        return tuple(int(i in indices) for i in range(len(modes)))
    c = rng.normal(size=vertex.shape[2])+1j*rng.normal(size=vertex.shape[2])
    initial = {occ(offsets[2]+k):value for k,value in enumerate(c)}
    direct = register.occupation_action(words,initial)
    tensor = register.tensor_action(words,initial)
    outputs = [occ(i,offsets[1]+j) for i in range(vertex.shape[0]) for j in range(vertex.shape[1])]
    expected = matrix@c
    return {
        'channel':channel,'tensor_shape':list(vertex.shape),'mode_count':len(modes),
        'formal_full_register_dimension_not_allocated':register.dimension,'nonzero_words':len(words),
        'global_SU3_relative':float(np.linalg.norm(np.kron(maps[0],maps[1])@matrix-matrix@maps[2])/np.linalg.norm(matrix)),
        'tensor_direct_spin_color_relative':float(np.linalg.norm(np.array([tensor.get(o,0) for o in outputs])-expected)/np.linalg.norm(expected)),
        'occupation_direct_spin_color_relative':float(np.linalg.norm(np.array([direct.get(o,0) for o in outputs])-expected)/np.linalg.norm(expected)),
    }


def evaluate():
    grid = [tree_case(scattering(m,np.array([.31,.17,.52]),[0,1,0],angle),m,f'grid_{m}_{angle}')
            for m in (0.,.3,.7) for angle in (.35,.8,1.7,2.6)]
    rng = np.random.default_rng(983041)
    fresh,frames = [],[]
    for n in range(12):
        mass = float(rng.uniform(.05,.8))
        v = rng.normal(size=3)
        v *= rng.uniform(.35,1.1)/np.linalg.norm(v)
        vectors = scattering(mass,v,rng.normal(size=3),rng.uniform(.3,2.8))
        fresh.append(tree_case(vectors,mass,f'fresh_{n}'))
        frames.append(dict(label=f'fresh_{n}', **transported_error(vectors,mass,n%3,'boost' if n<6 else 'rotation',.37)))
    for m in (0.,.3):
        for kind in ('boost','rotation'):
            for axis in range(3):
                frames.append(dict(label=f'grid_mass_{m}',**transported_error(
                    scattering(m,np.array([.31,.17,.52]),[0,1,0],.8),m,axis,kind,.27)))
    cubics = [cubic_case(c,p) for c,p in (
        ('q_to_qg',('q','g','q')),('qbar_to_qbarg',('qbar','g','qbar')),
        ('g_to_pair',('q','qbar','g')),('g_to_gg',('g','g','g')))]
    all_cases = grid+fresh
    summary = {
        'tree_points':len(all_cases),'frame_checks':len(frames),
        'maximum_LF_covariant_relative':max(c['LF_covariant_relative'] for c in all_cases),
        'maximum_Ward_relative':max(w[k] for c in all_cases for w in c['external_gluon_Ward'].values() for k in ('covariant_relative','LF_relative')),
        'maximum_frame_relative':max(c['relative_amplitude_error'] for c in frames),
        'minimum_omit_instantaneous_relative':min(c['omit_instantaneous_relative'] for c in all_cases),
        'minimum_omit_three_gluon_Ward_relative':min(w['omit_three_gluon_graph_relative'] for c in all_cases for w in c['external_gluon_Ward'].values()),
        'negative_crossed_plus_points':sum(c['crossed_fermion_plus_GeV']<0 for c in all_cases),
    }
    assert max(summary[k] for k in ('maximum_LF_covariant_relative','maximum_Ward_relative','maximum_frame_relative')) < 2e-12
    assert summary['minimum_omit_instantaneous_relative'] > .001
    assert summary['minimum_omit_three_gluon_Ward_relative'] > .001
    assert summary['negative_crossed_plus_points'] > 0
    for c in cubics:
        assert max(c[k] for k in ('global_SU3_relative','tensor_direct_spin_color_relative','occupation_direct_spin_color_relative')) < 2e-12
    return {
        'scope':'bare fixed-momentum canonical QCD tree and operator compiler; not a bound nucleon or renormalized model',
        'conventions':'symmetric LF, metric +---, gs squared stripped from scattering; cubic P-minus coefficients strip gs and delta/measure',
        'Ward_normalization':'Frobenius norm divided by physical amplitude norm and maximum external energy; dimensionless',
        'physics_input_role':'explicit test masses/momenta only, not fitted or selected physical parameters',
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'summary':summary,'cubic_tensors':cubics,'tree_grid':grid,'fresh_tree_points':fresh,'Lorentz_frames':frames,
        'BLFQ_H2_selected':False,'physical_nucleon_predictions':False,'renormalization_complete':False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
