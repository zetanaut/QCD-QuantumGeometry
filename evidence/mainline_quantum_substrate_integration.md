# Main-line quantum substrate integration

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Date: 2026-09-03

## Scope

This is an integration record, not a physical-claim record. The frozen Q0,
Q1, and Q2 worktrees remain unchanged. Their public implementation cores are
now available on the main line at:

- `src/deuteron_wigner/bridge/plhqcd0/`
- `src/deuteron_wigner/bridge/plhqcdstate/`
- `src/deuteron_wigner/quantum/plhqcdobs/`

The copied core files are byte-identical to their frozen sources:

| Main-line file | SHA-256 |
| --- | --- |
| `bridge/plhqcd0/core.py` | `277bf7a38ea7abb86b5145284103f7f803459c75fc6c38fbe0df45a605167390` |
| `bridge/plhqcdstate/core.py` | `4b5fae6e8721892b7c73f73cdfad3d604d5550b11e376714f934496619d291ec` |
| `quantum/plhqcdobs/core.py` | `8e6eb7cdcfefb0a81fbee5695e0eab574fd3667fa169a1bed84579eb9e9e37de` |

The corresponding frozen source worktrees are `deuteron_wigner_q0_plhqcd0`,
`deuteron_wigner_q1_plhqcdstate`, and `deuteron_wigner_q2_plhqcdobs`.

## Main-line seam

`src/deuteron_wigner/quantum/operator_bundle.py` is the new integration seam
for current science. It requires, explicitly:

1. a caller-supplied (H_{0,K}), sparse or matrix-free;
2. both C401 mass-direction coefficients;
3. the Lane-A C411 first-C117 coefficient and residual normalization/mixing
   parameters;
4. an explicit nonphysical claim tier.

It assembles

\[
H_K = H_{0,K}+
 c_{q,K}D_{q,K}+
 c_{g,K}D_{g,K}+
 c_{117,K}D_{117,K},
\]

and provides exact sparse/Krylov state output, sector-weight accounting,
Hellmann--Feynman checks, finite-difference observable responses, and a
diagnostic singular spectrum. It does not provide a default (H_0), select a
deuteron sector, infer a current, choose a physical C117 normalization, fit
parameters, or activate a Hamiltonian.

## Validation

The following focused tests pass:

```text
47 passed
```

This consists of the recovered Q0--Q2 tests and the C117/main-line tests. The
existing Q0 worktree virtual environment supplies PennyLane 0.38 and
PennyLane-Lightning 0.38. Its SymPy dependency was supplied from the existing
base Python installation for this validation run; the clean reproducible
declaration is now `pyproject.toml` extra `quantum` plus
`environment_quantum.yml` and should be used for a fresh environment.

The test result validates source integration, sparse/matrix-free parity,
padding and encoded routes, ordinary-gate decomposition, and diagnostic
observable compilation. It does not validate physical parameters or a
production quantum calculation.

## Next work

1. Validate the declared clean quantum environment.
2. Bind an explicit source-qualified or exploratory (H_{0,K}) through the
   tested K-local basis-map contract to the main-line bundle; do not use C144
   diagnostic components as C396 proxies. The current C7/C8 validation branch
   is not dimension-matched and is not a default H0.
3. Reproduce the bundle's K9 state and derivative observables through Q0--Q2.
4. Connect the state to the existing light-front and LPS current routes and
   carry their discrepancy until a production choice is justified.
5. Only then begin topology/symmetry-aware ansatz design and physical-sector
   selection.
