"""Replay gauge information, neural response and modular interpolator controls."""

import argparse
import json
import time
from pathlib import Path as FilePath

import numpy as np

from deuteron_wigner.quantum.modular_gauge_network import (
    GaugeGraph, Path, SharedLoopKernel, adjoint, commutator_loop,
    gauge_divergence, su3, positive_character_kernel, dress_spatial_potential,
)
from deuteron_wigner.quantum.modular_baryon_algebra import (
    add, baryon, coupled_cluster, norm2, spin_squared, wedge, ClusterEmbedding,
    quantum_numbers,
)


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2))
TREE = ((0, 1), (1, 2), (2, 3), (1, 4), (4, 5))
ELEMENTARY = (Path((1, 2, 3, 0, 1)), Path((1, 4, 5, 2, 1)))
LOOPS = ELEMENTARY+(commutator_loop(*ELEMENTARY),)


def graph(seed, scale=.8):
    rng = np.random.default_rng(seed)
    return GaugeGraph({e: su3(rng.normal(size=8)*scale) for e in EDGES})


def kernel(coefficients=(.2, -.3, .4, .1)):
    rng = np.random.default_rng(902)
    return SharedLoopKernel([.6, 1.2, -1.8, 2.4], [.1, -.2, .3, -.1],
                            rng.normal(size=(4, 4))*.5, [.2, -.1, .05, .3], coefficients)


def inference():
    """Fixed-feature conditional closure; same invariant functional owns currents.

    Freeze seeds, scales, coefficients and noise before evaluation. No physical
    data, learned hidden layers, state optimization or architecture search.
    """
    net = kernel()
    development = [graph(100+k, scale=.25+.06*k) for k in range(10)]
    reserved = [graph(500+k, scale=.28+.07*k) for k in range(12)]
    values = np.array([net.design(g, LOOPS) for g in development])
    responses = np.concatenate([net.current_design(g, LOOPS).reshape(-1, 4) for g in development])
    held_values = np.array([net.design(g, LOOPS) for g in reserved])
    held_responses = np.concatenate([net.current_design(g, LOOPS).reshape(-1, 4) for g in reserved])
    rng = np.random.default_rng(20260911)
    sigma = 1e-5
    noise = rng.normal(size=(100, len(values)+len(responses)))*sigma
    result = {}
    for name, design in (("functional_only", values), ("joint_functional_and_response", np.vstack((values, responses)))):
        singular = np.linalg.svd(design, compute_uv=False)
        exact = np.linalg.lstsq(design, design@net.coefficients, rcond=None)[0]
        parameters = np.array([np.linalg.lstsq(design, design@net.coefficients+n[:len(design)], rcond=None)[0]
                               for n in noise])
        error = parameters-net.coefficients
        result[name] = {
            "measurements": len(design), "singular_values": singular.tolist(),
            "condition": float(singular[0]/singular[-1]),
            "exact_coefficient_error": float(np.linalg.norm(exact-net.coefficients)),
            "coefficient_RMSE": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "reserved_functional_RMSE": float(np.sqrt(np.mean((error@held_values.T)**2))),
            "reserved_current_RMSE": float(np.sqrt(np.mean((error@held_responses.T)**2))),
        }
    return {"development_seeds": list(range(100, 110)), "reserved_seeds": list(range(500, 512)),
            "sigma": sigma, "realizations": 100, "true_coefficients": net.coefficients.tolist(),
            "hidden_layers_trained": False, "physical_data_used": False,
            "same_functional_noise_in_both_comparisons": True,
            "equal_measurement_budget": False, "results": result}


def evaluate():
    start = time.perf_counter()
    net = kernel()
    gauge_rows = []
    for seed in range(12):
        g = graph(seed)
        rng = np.random.default_rng(1000+seed)
        gauges = {n: su3(rng.normal(size=8)) for n in g.nodes}
        changed = g.transformed(gauges)
        fixed, _ = g.tree_gauge(TREE, 0)
        current = net.current(g, LOOPS)
        current_changed = net.current(changed, LOOPS)
        expected = np.array([adjoint(gauges[a])@j for (a, b), j in zip(g.links, current)])
        erased = GaugeGraph({e: np.eye(3) for e in EDGES})
        missing = current.copy()
        missing[0] = 0
        gauge_rows.append({
            "seed": seed,
            "functional_gauge_error": abs(net.value(g, LOOPS)-net.value(changed, LOOPS)),
            "current_covariance_error": float(np.linalg.norm(current_changed-expected)),
            "gauge_divergence_error": max(float(np.linalg.norm(v)) for v in gauge_divergence(g, current).values()),
            "tree_gauge_functional_error": abs(net.value(g, LOOPS)-net.value(fixed, LOOPS)),
            "tree_gauge_Wilson_error": max(abs(g.loop(p)-fixed.loop(p)) for p in LOOPS),
            "erased_holonomy_functional_error": abs(net.value(g, LOOPS)-net.value(erased, LOOPS)),
            "erased_holonomy_current_error": float(np.linalg.norm(current-net.current(erased, LOOPS))),
            "omitted_current_term_divergence": max(float(np.linalg.norm(v)) for v in gauge_divergence(g, missing).values()),
        })
    g = graph(21)
    exact = net.current(g, LOOPS)
    derivative_rows = []
    for epsilon in (1e-3, 1e-4, 1e-5):
        errors = []
        for i, edge in enumerate(g.links):
            for a in range(8):
                direction = np.eye(8)[a]*epsilon
                plus, minus = g.links.copy(), g.links.copy()
                plus[edge], minus[edge] = su3(direction)@g.links[edge], su3(-direction)@g.links[edge]
                derivative = (net.value(GaugeGraph(plus), LOOPS)-net.value(GaugeGraph(minus), LOOPS))/(2*epsilon)
                errors.append(abs(derivative-exact[i, a]))
        derivative_rows.append({"step": epsilon, "maximum_error": max(errors)})
    identity = {e: np.eye(3) for e in EDGES}
    a = su3([0, 0, 0, 0, 0, 0, .8, .3])
    b = su3([0, 0, 0, 0, 0, 0, -.4, .9])
    r = su3([1.3, .4, -.7, .8, .2, .1, 0, 0])
    first, second = identity.copy(), identity.copy()
    first[3, 0], first[5, 2] = a, b
    second[3, 0], second[5, 2] = a, r@b@r.conj().T
    left, right = GaugeGraph(first), GaugeGraph(second)
    scalar_blindness = {
        "elementary_loop_trace_difference": max(abs(left.loop(p)-right.loop(p)) for p in ELEMENTARY),
        "commutator_trace_difference": abs(left.loop(LOOPS[2])-right.loop(LOOPS[2])),
        "shared_functional_difference": abs(net.value(left, LOOPS)-net.value(right, LOOPS)),
        "not_a_topological_order_invariant": True,
    }
    clusters = []
    for species, couplings, magnetic in ((('p',), (), 1), (('n',), (), -1),
                                         (('p', 'n'), (2,), 0), (('p', 'n'), (0,), 0),
                                         (('p', 'n', 'n'), (0, 1), 1),
                                         (('p', 'p', 'n'), (2, 3), 1),
                                         (('p', 'p', 'n', 'n'), (0, 1, 0), 0)):
        sites = tuple(range(len(species)))
        state = coupled_cluster(species, sites, couplings, magnetic)
        j = (couplings[-1] if couplings else 1)/2
        residual = add(((1, spin_squared(state, sites)), (-j*(j+1), state)))
        clusters.append({"species": species, "A": len(species), "Z": species.count('p'),
                         "spin": j, "magnetic": magnetic/2, "sparse_coefficients": len(state),
                         "norm": norm2(state), "spin_Casimir_residual_norm": np.sqrt(norm2(residual))})
    pa = baryon(0, 'p', 1)
    pn_same = wedge(pa, baryon(0, 'n', 1))
    pn_separate = wedge(pa, baryon(1, 'n', 1))
    inference_result = inference()
    return {
        "scope": "reusable ordered SU3, shared neural-kernel response and fermionic nuclear-interpolator architecture",
        "preedit_checkpoint": "6c409c266342d1ae5f557c3224f7b8556ea66ae1",
        "gauge_cases": gauge_rows, "all_56_link_generator_derivatives": derivative_rows,
        "relative_holonomy_information": scalar_blindness,
        "modular_spin_resolved_interpolators": clusters,
        "overlap_exchange": {"separated_pn_norm": norm2(pn_separate), "overlapping_pn_norm": norm2(pn_same),
                             "repeated_identical_creator_norm": norm2(wedge(pa, pa)),
                             "nuclear_binding_or_medium_effect_prediction": False},
        "shared_neural_inference": inference_result,
        "generated_physical_nucleon_predictions": False, "nuclear_binding_solved": False,
        "LF_current_or_Poincare_covariance_qualified": False, "quantum_gauge_field_state_supplied": False,
        "physical_topological_order_demonstrated": False, "quantum_advantage": False,
        "runtime_seconds": time.perf_counter()-start,
    }


def continuation():
    """Post-result Gram-map and temporal/spatial-role tests, recorded separately."""
    gram_rows = []
    for spin in (0, 2):
        columns = [coupled_cluster(order, (0, 0), (spin,), 0) for order in (("p", "n"), ("n", "p"))]
        embedding = ClusterEmbedding(columns, relative_tolerance=1e-12)
        def charge(state):
            return {mask: quantum_numbers(mask)['charge']*value for mask, value in state.items()}
        gram_rows.append({"spin": spin/2, "columns": 2, "physical_rank": embedding.rank,
                          "Gram_eigenvalues": embedding.gram_eigenvalues.tolist(),
                          "null_image_norm": float(np.linalg.norm(embedding.matrix@embedding.null_vectors)),
                          "isometry_error": float(np.linalg.norm(embedding.isometry.conj().T@embedding.isometry-np.eye(embedding.rank))),
                          "unqualified_Euclidean_charge": float(embedding.kernel(charge)[0, 0].real),
                          "common_map_charge": embedding.operator(charge).real.tolist()})
    a = coupled_cluster(("p", "n"), (0, 1), (2,), 0)
    b = coupled_cluster(("p", "n"), (1, 0), (2,), 0)
    embedding = ClusterEmbedding((a, b, add(((.6, a), (.8j, b)))), relative_tolerance=1e-12)
    def charge_at_zero(state):
        return {mask: quantum_numbers(mask & ((1 << 12)-1))['charge']*value for mask, value in state.items()}
    c = np.array([.3, -.2j, .7])
    state = embedding.full_state(c)
    diagonal = np.array([quantum_numbers(mask & ((1 << 12)-1))['charge'] for mask in embedding.occupations])
    direct = np.vdot(state, diagonal*state)
    transported = (c.conj()@embedding.kernel(charge_at_zero)@c)/(c.conj()@embedding.gram@c)
    overlap_control = {"cluster_columns": 3, "physical_rank": embedding.rank,
                       "full_carrier_expectation": float(direct.real),
                       "common_Gram_expectation": float(transported.real),
                       "difference": float(abs(direct-transported)),
                       "not_a_nuclear_form_factor": True}
    net = kernel()
    positivity_rows = []
    # Seed 81 was preflight; the other three are fresh independent group sets.
    for seed, size in ((81, 40), (192, 32), (293, 64), (394, 96)):
        rng = np.random.default_rng(seed)
        u = np.array([su3(rng.normal(size=8)) for _ in range(size)])
        relative = u[:, None]@u[None, :].conj().swapaxes(-1, -2)
        z = 1-np.trace(relative, axis1=-2, axis2=-1).real/3
        bad = np.exp(-net.features(z)@net.coefficients)
        good = positive_character_kernel(u, [.5, .2, .1])
        potential = net.features(1-np.trace(u, axis1=-2, axis2=-1).real/3)@net.coefficients
        dressed, offset = dress_spatial_potential(good, potential, .2)
        bad_eigen = np.linalg.eigvalsh(bad)
        positivity_rows.append({"seed": seed, "configurations": size,
                                "unrestricted_neural_temporal_min_eigenvalue": float(bad_eigen[0]),
                                "unrestricted_negative_count": int(sum(bad_eigen < -1e-10)),
                                "positive_character_min_eigenvalue": float(np.linalg.eigvalsh(good)[0]),
                                "with_same_neural_spatial_potential_min_eigenvalue": float(np.linalg.eigvalsh(dressed)[0]),
                                "spatial_potential_offset": offset})
    return {"scope": "common modular Gram map and role-sensitive positive finite transfer-kernel controls",
            "preedit_checkpoint": "6c409c266342d1ae5f557c3224f7b8556ea66ae1",
            "overlapping_cluster_Gram_controls": gram_rows,
            "independent_full_carrier_operator_control": overlap_control,
            "temporal_versus_spatial_neural_placement": positivity_rows,
            "negative_temporal_control_does_not_reject_spatial_potential": True,
            "complete_LF_or_continuum_transfer_theory": False,
            "physical_nucleon_or_nuclear_prediction": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=FilePath, required=True)
    parser.add_argument('--continuation', action='store_true')
    args = parser.parse_args()
    report = continuation() if args.continuation else evaluate()
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'gauge_cases'}, indent=2))


if __name__ == '__main__':
    main()
