# Quartic spin density and necessary tree covariance control

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

Checkpoint: 9 September 2026. **Scientific work stopped for user evaluation.**
This note is the review addendum to the living theory note at `0ff93b19`.
The main TeX note still represents the preceding all-cubic checkpoint; this
addendum has not yet been incorporated there. No regulator repair, physical
fit, public push or review-mirror refresh is part of this checkpoint.
The complete pre-edit source and notes are recoverable at `0ff93b19`.

## Outcome and scope

The bare quartic local Noether spin and coordinate-density improvements are
now derived and tested for the ordered instantaneous-fermion (IF),
instantaneous-gluon (IG) and four-gluon (A4) sources. This is a continuum
local-density result before Wick contraction and finite-domain projection,
not the completed many-body Lorentz algebra.

An independent, on-shell distinguishable-quark tree control reproduces the
covariant amplitude from the actual cubic and Gauss source kernels. The
current physical-leg Gaussian spoils its dynamical Lorentz covariance even
for massless external quarks, before finite-cell/Fock projection. This is a
necessary colored partonic diagnostic, not a nucleon/deuteron observable or
a quantitative attribution of the preceding nucleon-current discrepancy.

## Bare quartic density result

Use the existing symmetric light-front conventions and signed Fourier
creation/annihilation labels. The constrained fermion field is linear in
the transverse gauge field. For the canonical fermion spin matrix
\(H_i=\gamma^0\{\gamma^+,\sigma^{-i}\}/4\),
\(\Lambda_-H_i\Lambda_-=0\): the candidate quartic local fermion spin
vanishes algebraically. The gluon constraint is linear in the coupling, so
the canonical gluon local spin has no quartic term either. Canonical
\(T^{+i}\) is kinematical. **The quartic generator is not zero:** its
coordinate moment includes \(-x^i\mathcal H_4\).

For one ordered IF word let
\(\Delta=p_f+k_L+k_R-p_i\), \(l=p_f+k_L\), \(r=p_i-k_R\),
\(n=\sqrt{4|k_L^+k_R^+|}\), and
\(B=G_o^\dagger(\alpha_T\cdot e_L)(\alpha_T\cdot e_R)G_i\).
The explicitly supplied good-field axes are quark or antiquark axes;
physical CAR reordering and transposition are not guessed. Independent
kinetic and interaction contributions to the local Legendre density give

\[
C_{\rm can}^{\rm IF}=\frac{B}{4n}\left(\frac1{r^+}+\frac1{l^+}\right),
\qquad C_{\rm src}^{\rm IF}=\frac{B}{2nr^+},
\qquad D_{\rm IF}=-\frac{\Delta^+B}{4nl^+r^+}.
\]

Both inverse momenta are signed and nonzero. For one ordered Gauss-current
product, with phases \(e^{iQ^+x^- -iQ_T\cdot x_T}\) and
\(A^-_1=-J/(Q^+)^2\), the independent field terms
\(-\tfrac12(\partial^+A^-_1)^2-JA^-_1\) give

\[
C_{\rm can}^{\rm IG}=J_aJ_b\left(\frac1{2Q_a^+Q_b^+}
 +\frac1{(Q_b^+)^2}\right),\qquad
C_{\rm src}^{\rm IG}=\frac{J_aJ_b}{2(Q_b^+)^2},\qquad
D_{\rm IG}=\frac{(Q_a^++Q_b^+)J_aJ_b}{2Q_a^+(Q_b^+)^2}.
\]

The supplied ordered current product includes its spin, color and physical
field normalization. Both current orders belong in the Hamiltonian. This
top-degree identity does not determine lower-degree Wick reordering terms.
Neither longitudinal improvement may be discarded on a finite domain by
assumption. Both are transverse-momentum independent, so their bare normal
transverse contacts vanish.

Each ordered A4 assignment is
\((e_{0T}\cdot e_{2T})(e_{1T}\cdot e_{3T})
f^{xab}f^{xcd}/[4\sqrt{\prod_j2p_j^+}]\).
The source quarter and four physical field normalizations occur once;
sum all 24 assignments with separate Fock factors. Its local spin and
bare normal transverse contact are zero. Existing retained/excluded IF,
IG and A4 source comparisons retain their cell factor
\(C_{M^2}=2P_N^+ C/(2L)\), \(2L=\pi K_2/P_N^+\).
This mass conversion is not an extra factor in a Lorentz generator.

## Necessary tree test and regulator failure

For distinguishable quarks with conserved on-shell external four-momenta,
let \(q=p_{1i}-p_{1f}\), \(q^2<0\), \(q^+\ne0\), and
\(J_j^\mu=\bar u_{jf}\gamma^\mu u_{ji}\). Per coupling squared and
the color product, the source cubic exchange and both ordered Gauss words
give, after removing the four external flat-field normalizations once,

\[
\mathcal M_{\rm prop}+\mathcal M_{\rm inst}
=\mathcal M_{\rm cov}=-\frac{J_1\cdot J_2}{q^2},
\qquad \mathcal M_{\rm inst}=\frac{J_1^+J_2^+}{(q^+)^2}.
\]

The propagating intermediate mode has positive
\(k^+=|q^+|\), \(k_T=\operatorname{sgn}(q^+)q_T\) and its on-shell
massless minus momentum, with denominator \(q^2/(2k^+)\). It is not
an off-shell covariant momentum substituted into the regulator.
For the actual physical-leg Gaussian, let \(R_{3a},R_{3b}\) be its
two cubic weights and \(R_4\) its four-external-leg contact weight.
Then, exactly,

\[
\mathcal M_R=R_{3a}R_{3b}\mathcal M_{\rm cov}
 +(R_4-R_{3a}R_{3b})\mathcal M_{\rm inst}.
\]

The second term exposes the failed cancellation of light-front-direction
dependence; the first weight is also generally frame dependent. The
implementation returns both terms separately. Independent source-transform
evaluation verifies the weights, including reference-shape degrees 0 and 2.
Bare covariance passes all ten tested Cartesian/light-front transformations
with external spinor transport. Regulated kinematic transformations pass;
dynamical F1/F2 transformations fail in the tested cases. The spin-averaged
squared amplitude also changes, excluding a mere unitary spin-axis/phase fix.

For orientation, the smooth width-0.8 GeV example with center-of-mass incoming
three-momentum (0.4, 0.2, 0.5) GeV, outgoing momentum rotated 0.8 rad about y,
and Cartesian x-boost rapidity 0.35 has relative transported amplitude-matrix
differences of about 18.0% for masses (0,0), 22.0% for (0.3,0.3) GeV and
24.9% for (0.3,0.5) GeV. These are Frobenius amplitude differences in this
partonic control, not cross-section or nucleon uncertainty estimates.
The bare source/covariant matrix discrepancy is below 1e-13 here.
An independent large-width control at 8,16,32,64 GeV confirms leading
inverse-width-squared decay. It does not justify removing the regulator in
the interacting model or claim loop renormalization.

## Verification and replay

945 focused tests passed in 29.76 s, including 101 new quartic-density and
60 new tree-covariance controls. This was not the entire repository suite.
The failure diagnostics pass by detecting the unresolved regulated defect;
passing tests do not mean covariance has been restored.

From the canonical repository, with its scientific Python environment:

```sh
env PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider \
tests/test_m2_quartic_spin_density.py tests/test_m2_spin_tree_covariance.py \
tests/test_m2_cubic_spin_channels.py tests/test_m2_canonical_cubic_spin.py \
tests/test_m2_cubic_lorentz_ward.py tests/test_m2_free_spin_generators.py \
tests/test_m2_constrained_fermion_current.py tests/test_m2_canonical_emission.py \
tests/test_m2_coupled_longitudinal_jacobi.py tests/test_m2_coupled_pilot.py \
tests/test_m2_gaussian_cluster_interactions.py tests/test_m2_excluded_fock_sources.py \
tests/test_m2_six_body_vertices.py
```

## Evaluation point, not an automatic continuation queue

Milestone 1 has advanced, but physical spin/covariance qualification remains
open. Milestones 2 (controlled physical accuracy), 3 (microscopic nuclear
embedding) and 4 (shared physical calibration) also remain open. Earlier
finite-model H/current matching and the small PennyLane benchmark remain
valid within their recorded scopes and were not restarted or modified.

Before resuming many-body spin transport, evaluate the regulator/dynamics
completion required by this necessary tree control. No replacement regulator,
counterterm prescription or physical input has been selected. Higher-order
commutators, regulator/Wick/domain partners and full H2/norm/moving-frame spin
transport are not supplied by this checkpoint. The full theory-note merge
and any public synchronization are pending evaluation. The latest user pause
supersedes older persistent-work instructions; do not automatically resume.
