# Using and extending the tools

[Return to QCD-QuantumGeometry](../README.md)

The snapshot is an inspectable record of implementations, tests, and replay scripts from DeuteronWigner. Its directory layout preserves the original source paths. The runtime has dependencies elsewhere in the upstream project, including canonical physics helpers, bridge modules, and selected data. This companion is not an independently installable Python distribution.

## Choose the correct source revision

The early tree geometry, natural-gradient solver, and selected graph tests are present at the public upstream commit:

```text
b6fbd04239f8cf9573b54aeec75be0b527881012
```

The later scalar/modular/source/path/sea work in this companion is taken from the full local science checkpoint:

```text
dd0d75491aec06d9312d5361eaf0fd56b06e84c7
```

At the October 23, 2026 audit, that later revision was not on upstream public `main`. Fetching or checking out that hash from the public remote is therefore not a guaranteed reproduction path. Use a complete checkout containing the later checkpoint for its integrated tests and replay scripts. The snapshot itself allows inspection of the relevant files and recorded numerical reports.

## Dependencies

The original package uses NumPy and SciPy. Its optional `quantum` extra declares PennyLane 0.38.0, PennyLane-Lightning 0.38.0, autoray 0.6.12, SymPy, and mpmath. The `joint-learning` extra additionally declares PyTorch 2.6.0. These are the project's recorded compatibility choices, not a recommendation to upgrade to current releases without checking compatibility.

From a complete upstream checkout, use an isolated environment and the original package configuration:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[quantum,analysis]'
```

For the later differentiable joint-learning controls, install the corresponding extra from the complete later checkpoint:

```bash
python -m pip install -e '.[joint-learning,analysis]'
```

The original [package configuration](../snapshot/pyproject.toml) and [quantum environment recipe](../snapshot/environment_quantum.yml) are retained. No new environment or physics benchmark is certified by this documentation publication.

## Selected entry points

| Task | Start with | Direct controls |
| --- | --- | --- |
| Exact full-tree metric action | `m2_tree_projective_geometry.py` | `test_m2_tree_projective_geometry.py`, `test_m2_tree_state_retraction.py` |
| Quantum natural-gradient optimization | `m2_graph_natural_gradient.py`, `m2_symmetry_graph_solver.py` | `test_m2_graph_natural_gradient.py`, `test_m2_symmetry_graph_solver.py` |
| Low-rank coefficient geometry and current sensitivity | `scalar_representation_geometry.py` | `test_scalar_representation_geometry.py` |
| Ordered SU(3) paths and common functional derivatives | `modular_gauge_network.py` | `test_modular_gauge_network.py` |
| Fermion composition and common overlaps | `modular_baryon_algebra.py`, `graded_fock_network.py` | Matching `test_*.py` files |
| Canonical LF interaction tensors | `canonical_lf_fields.py`, `canonical_lf_network.py` | `test_canonical_lf_network.py` |
| Joint state/dynamics derivatives | `joint_state_dynamics.py` | `test_joint_state_dynamics.py` |
| Actual path-created Fock states | `lf_path_states.py`, `path_dirac_matching.py` | Matching `test_*.py` files |
| Constructive connection/chiral/sea controls | `stable_connection_sea.py` | `test_stable_connection_sea.py` |

All source paths and available test/replay snapshots are linked from the [catalog](TOOL_CATALOG.md).

## Focused reproduction examples

Run these from the appropriate **complete upstream checkout**, with `src` and the repository root on the Python import path. The commands reproduce existing controls; they do not train a physical nucleon model.

Early projective geometry:

```bash
PYTHONPATH=src:. python -m pytest \
  tests/test_m2_tree_projective_geometry.py \
  tests/test_m2_tree_state_retraction.py \
  tests/test_m2_graph_natural_gradient.py -q
```

Recorded graph optimization configuration:

```bash
PYTHONPATH=src:. python scripts/benchmark_m2_graph_convergence.py \
  --natural --iterations 600 --metric-rcond 1e-18 --spin-current
```

Later scalar and modular controls:

```bash
PYTHONPATH=src:. python -m pytest \
  tests/test_scalar_representation_geometry.py \
  tests/test_modular_gauge_network.py \
  tests/test_modular_baryon_algebra.py -q
```

For a new scalar audit report, choose an unused output filename so the historical result is preserved:

```bash
PYTHONPATH=src:. python scripts/evaluate_scalar_geometry.py \
  --output /tmp/qcd-scalar-geometry-new-result.json
```

For later path and sea controls:

```bash
PYTHONPATH=src:. python -m pytest \
  tests/test_lf_path_states.py \
  tests/test_path_dirac_matching.py \
  tests/test_stable_connection_sea.py -q
```

Saved result notes give the original parameter choices, quadrature orders, finite mode support, and additional reproduction requirements. Some evaluator scripts use fixed report destinations; inspect their entry points before executing them against a preserved checkout.

## Reusing the mathematics in a new model

1. **Declare the carrier and conventions.** Identify field content, spin/flavor/color labels, normalization, modes, regulator, and source conventions before reusing a map.
2. **Derive geometry for the new representation.** The complete-tree inverse does not automatically apply to a truncated tensor, learned path dictionary, or different neural ansatz.
3. **Separate redundancy from sensitivity.** Remove proven coordinate null directions. Test weak directions at refined resolution and against the operators the model must predict.
4. **Transport the same operators and sources.** Carry both endpoints, induced norm, contacts, and required many-body terms when eliminating or composing sectors.
5. **Preserve canonical node information.** Use full qualified Dirac/color tensors at interaction nodes; scalar path weights are not substitutes for missing spin dynamics.
6. **Count retained physics once.** An explicit mode and the residual effective contribution representing that mode must share a consistent subtraction prescription.
7. **Validate beyond the optimization objective.** Check eigenpair residuals, independent currents, new kinematics, enlarged modes, cluster limits, and withheld data as appropriate.

The demonstrated geometry is useful even when a physical candidate fails. Its reuse should preserve the positive and negative controls without adopting that candidate's unqualified dynamics.
