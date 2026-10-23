# M2h active-quark current and first non-null microscopic response

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

## Result

M2h replaces the internal identity used by M2g with the smallest
source-derived nonzero-transfer current supported by the represented M2
basis: the canonical active-quark plus current at zero skewness and Wilson
order zero. The operator is implemented in
`src/deuteron_wigner/quantum/m2_active_quark_current.py`; its four-direction
response is implemented in
`src/deuteron_wigner/quantum/m2_internal_current_response.py`.

The internal operator is source-qualified within this finite-basis active-
subsystem scope. Its lift to a proton/neutron and then NN current is still
exploratory because the added two-quark spectator has spin, flavor, and color
but no resolved momentum wave function or Hamiltonian. The lift therefore
uses exact constituent charges and the current identity-spectator extension:
the unresolved spectator charge is pointlike. This closes charge at zero
transfer but is not a complete microscopic-to-nucleon matching.

## Recoil derivation

For a retained `qg` partition, C47 uses the intrinsic relative momentum

\[
 q_{\rm rel}
 =\sqrt{x_g}\,q_q-\sqrt{x_q}\,q_g
 =\frac{k_q}{\sqrt{x_qx_g}},
 \qquad k_g=-k_q.
\]

The Volume-XI symmetric zero-skewness recoil map gives

\[
 \kappa_{qT}^{\rm out/in}
 =k_{qT}\pm\frac{x_g}{2}\Delta_T.
\]

Consequently the relative coordinate entering the outgoing and incoming
wave functions is shifted by

\[
 q_{{\rm rel},T}^{\rm out/in}
 =q_{{\rm rel},T}\pm
 \frac12\sqrt{\frac{x_g}{x_q}}\Delta_T,
 \qquad
 \delta q_{{\rm rel},T}
 =\sqrt{\frac{x_g}{x_q}}\Delta_T.
\]

For C45 oscillator modes the finite matrix is therefore

\[
 J^{(p)}_{n'm',nm}(\Delta_T)
 =\int\frac{d^2q}{(2\pi)^2}
 \phi_{n'm'}^*\!\left(q+
 \frac12\sqrt{\frac{x_g^{(p)}}{x_q^{(p)}}}\Delta_T\right)
 \phi_{nm}\!\left(q-
 \frac12\sqrt{\frac{x_g^{(p)}}{x_q^{(p)}}\Delta_T\right).
\]

It is diagonal in longitudinal partition, active-quark helicity, and gluon
helicity. The one-quark block is the exact intrinsic identity because that
sector has no represented relative coordinate. Canonical number-changing
`q <-> qg`, constrained-field, zero-mode, induced, and renormalization/
matching currents require separate named operators and remain unavailable,
not physically zero.

At `K9` the four relative-shift factors are

\[
 \sqrt{2/7},\quad\sqrt{4/5},\quad\sqrt2,\quad\sqrt8.
\]

The matrix is evaluated directly from the C45 momentum-space modes by
Gauss--Hermite quadrature with no threshold pruning. The finite shell is a
projection, so the nonzero-transfer matrix is contractive rather than
unitary; this leakage is retained and reported.

## Exact and numerical closure

At the diagnostic transfer `Delta_T=(0.2,0) GeV`:

- `J(0)=1` exactly in the retained 450-dimensional singlet multiplicity
  space;
- `J(-Delta)=J(Delta)^dagger` closes to `2.3e-16`;
- the smallest retained partition singular value is `0.05232321968`, with no
  contraction excess above `2e-14`;
- the projected K9 helicity-pair current is

  ```text
  [[ 0.99653203761, -0.000706960268],
   [ 0.000706960268, 0.99653203761 ]]
  ```

  up to imaginary parts below `5e-19`;
- the pointlike-spectator proton and neutron lift closes forward charge, and
  the resulting NN current closes deuteron charge to `1.1e-14`, transfer
  reversal to `1.3e-20`, parity below `1.3e-17`, and coherent SS/SD/DS/DD
  reconstruction exactly.

## Four-direction response

The same parameter order and diagnostic scales as M2g are retained:

```text
(c396:D_mu_q_sq, c396:D_delta_mu_g_sq,
 c117:I2_density_projector, c53:g_s_canonical_q_qg)

(0.05 GeV^2, 0.05 GeV^2, 0.03, 0.01)
```

All eight endpoint states are recomputed. The unscaled internal pair-current
Jacobian is

```text
[[ 0.0062869474,  0.0111934901, -0.0030202311, -0.1056393931],
 [-0.0012961253,  0.0012371281,  0.0022510580, -0.0273735595],
 [ 0.0012961253, -0.0012371281, -0.0022510580,  0.0273735595],
 [ 0.0062869474,  0.0111934900, -0.0030202311, -0.1056393930]]
```

Its scale-qualified singular values are

```text
0.001785981969, 0.000240331244, 1.42e-12, 1.21e-16
```

and its diagnostic rank is two. The direct endpoint derivative and the
state-tangent formula

\[
 \partial_r(B^\dagger JB)
 =(\partial_rB)^\dagger JB+B^\dagger J(\partial_rB)
\]

agree within `2.9e-10`.

After the explicitly provisional spectator lift and AV18 NN convolution, the
scale-qualified spin-one response singular values are

```text
0.000350939433, 0.000022626705, 9.11e-11, 4.48e-13.
```

The last two values are below the declared `1e-6` relative numerical-rank
threshold set by endpoint cancellation through the 5376-node convolution, so
the spin-one diagnostic rank is also two. This is not a physical response
rank.

The baseline angular residual changes from the M2g identity-control value
`1.4975815809e-4` to `1.7602917333e-4`, and its derivative is

```text
(4.87978e-5, -4.57125e-5, -8.43522e-5, 1.01864e-3).
```

Thus the current now sees the microscopic state, but it does not restore
rotational covariance or satisfy the angular condition by construction.

## Scientific interpretation and next step

M2h establishes three things.

1. The M2g null current Jacobian was caused by the identity-only current
   interface, not by immobility or irrelevance of the four microscopic
   directions.
2. One source-derived active-current channel resolves two combinations of
   those directions at this transfer and resolution.
3. A physical response rank still cannot be assigned: the spectator momentum
   dynamics, full one-body current, interaction/induced currents, operator
   matching, physical state parameters, and resolution trajectory are not
   complete.

The next current step is to give the two-quark spectator a resolved momentum
and Hamiltonian embedding, then match active and spectator one-body currents
in the same regulator convention. The canonical number-changing and induced
currents must be added as separately sourced operators. In parallel, the
first C117 finite-cell normalization ownership and the M3 topology-aware
quantum encoding remain open work streams.
