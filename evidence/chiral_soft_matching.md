# Protected soft-pion matching and the next dynamical constraint

> Historical source note from DeuteronWigner `dd0d7549`. Numerical content and dated scope are preserved; repository links are adapted for this companion. Historical next actions do not supersede the [current synthesis](../docs/RESULTS.md).

10 September 2026. Parent 6076f09f. Local only.

## What this resolves

The first soft correlation now has a concrete physical qualification:
**retain the pion pole and its source partners explicitly or in an exactly
pole-preserving nonlocal term; fit only the matched remainder.** A Taylor
contact expansion about zero momentum is useful only in its stated domain.
There is no reason to choose an entire enlarged Fock space merely to preserve
this tree-level pole, but an instantaneous Hamiltonian cannot simply be
identified with an energy-dependent nonlocal kernel either.

An explicit quadratic pion and its eliminated axial-source functional agree,
including the one-electromagnetic-source response. The contact terms are
essential. This is established chiral physics implemented as a constraint on
our construction, not a new theory of pions and not selection of a nuclear
chiral-EFT, constituent-quark, rainbow-ladder/DSE or BLFQ/H2 model.

The upstream constraint is equally important: **dynamical quark dressing,
the pion amplitude, and axial/pseudoscalar responses cannot be independently
assigned.** A simple Ward-completed vertex demonstrates the relation but
does not solve their shared dynamics. That distinction prevents the next
fit from merely hiding another inconsistent current.

## Scope, sources and what is not being claimed

This is tree-level matching of the leading quadratic pion/axial-source
sector, with one vector-source insertion. It is not all-orders chiral
matching, pion loops, a regulator prescription, a nucleon wavefunction,
confinement, a physical form-factor fit or the completed quark axial vertex.
The static NN one-pion-exchange tensor is a low-energy matching target,
not a production nuclear potential.

The starting physics is the nonsinglet, two-flavor chiral source Lagrangian
and its Ward identities. [Krebs, Appendix A](https://arxiv.org/html/2008.00974v1)
gives the leading pion action and source formulation, and emphasizes the
need for consistent interactions and currents. We do not adopt the paper's
Hamiltonian block-diagonalization or higher-order current program.
The dressed-quark nonsinglet identity and chiral pole/residue relationships
are established independently of a particular solver; see
[Qin, Roberts and Schmidt, Eqs. 1 and 19–22](https://arxiv.org/html/1402.1176v1).
That paper uses Euclidean conventions; our following Minkowski convention
is defined explicitly and tested against the free limit.
The familiar long-range exchange structure and associated pion/contact
currents also appear in [Pastore et al.](https://arxiv.org/abs/0906.1800).
Our contribution is the project's source-consistent explicit/nonlocal
comparison, diagnostic implementation and architectural consequence; no
novelty is claimed for these underlying identities.

## One common physical source functional

Use Cartesian metric \(g={\rm diag}(1,-1,-1,-1)\), momenta in GeV,
\(F\) and \(m_\pi\) in GeV, and
\[
 U=\exp(i\tau^a\pi^a/F),\quad
 {\cal L}_2={F^2\over4}{\rm tr}
   (\nabla_\mu U\nabla^\mu U^\dagger+\chi U^\dagger+U\chi^\dagger).
\]
At quadratic order in pion/axial sources this reduces, with a declared
axial-source sign convention, to
\[
 {\cal L}_{\pi a}={1\over2}(\partial_\mu\pi^a-Fa_\mu^a)^2
                    -{1\over2}m_\pi^2(\pi^a)^2 .
\]
The full chiral completion contains higher pion/source terms. They are not
claimed to be implemented here. EM gauging replaces derivatives by the
adjoint covariant derivative. The Cartesian pion generator is
\((T^3)_{ab}=-i\epsilon_{3ab}\); its eigencharges are \(-1,0,+1\).
The numerical Lorentz vertex below strips the common \(e T^3\).

Define \(s=k^2\) and \(D(s)=1/(m_\pi^2-s-i0)\). Away from the pole,
eliminating the quadratic pion yields
\[
 \Pi_{AA}^{\alpha\beta}(k)
    =F^2[g^{\alpha\beta}+k^\alpha k^\beta D(k^2)],\qquad
 k_\alpha\Pi_{AA}^{\alpha\beta}
    =F^2m_\pi^2 k^\beta D(k^2).
\]
The constant \(F^2g^{\alpha\beta}\) is a required source contact, not a
fittable correction. Omitting it violates this PCAC identity and leaves a
nonzero divergence even in the chiral limit. This is an axial **source
two-point function**, not a nucleon matrix element.

The independent explicit calculation forms the Fourier-space pion/source
action blocks \(K_\pi=s-m_\pi^2,\ L=iFk,\ R=L^\dagger,\ C=F^2g\),
then computes \(C-LK_\pi^{-1}R\). This is ordinary classical field
elimination at the stated source/tree order, not the saved H2 Fock reduction.

For vector transfer \(Q=k'-k\), differentiating that same action gives
\[
 {1\over F^2}\Pi_{VAA}^{\mu;\alpha\beta}(k',k)
 = (k'+k)^\mu k'^\alpha k^\beta D(s')D(s)
   +g^{\mu\alpha}k^\beta D(s)
   +k'^\alpha g^{\mu\beta}D(s').
\]
The first term is the internal pion insertion; the last two are endpoint
derivative contacts. Together they obey
\[
 Q_\mu\Pi_{VAA}^{\mu;\alpha\beta}
   =\Pi_{AA}^{\alpha\beta}(k')-\Pi_{AA}^{\alpha\beta}(k).
\]
The source derivative is separately checked by minimally gauging the
kinetic and endpoint blocks in a two-Fourier-mode explicit calculation.
That finite block is a derivative-at-zero control, not a finite-field or
nonperturbative truncation. No physical full-current claim is inferred from
the identity alone.

## What happens when the pion is represented locally

The unambiguous derivative expansion is
\[
 D_N(s)={1\over m_\pi^2}\sum_{n=0}^{N}(s/m_\pi^2)^n,\qquad
 {D_N(s)-D(s)\over D(s)}=-(s/m_\pi^2)^{N+1}.
\]
It converges for \(|s|<m_\pi^2\), not at the pion pole. For a static spatial
momentum k the absolute relative error is
\((|\boldsymbol k|/m_\pi)^{2(N+1)}\).

| Spatial momentum, mπ=140 MeV | Three-term expansion, N=2 |
|---|---:|
| 35 MeV | 0.0244% |
| 70 MeV | 1.5625% |
| 100 MeV | 13.28% |
| 140 MeV | 100% |
| 210 MeV | 1139% |

These are exact errors for this Taylor representation of this propagator.
They are not the error of the full project, a universal breakdown scale for
every resummed EFT, or proof that no flexible function can approximate a
finite real interval. A polynomial fit on a compact spacelike interval can
behave differently; it still does not possess the pion pole as an analytic
structure. Protecting the pole is the appropriate information-preserving
choice here.

The consistent vector insertion for any \(D_N\) uses
\[
 D(s')D(s)\ \longrightarrow\
 D_N[s',s]\equiv{D_N(s')-D_N(s)\over s'-s},
\]
with the analytic coincident limit and the same \(D_N\) in both contacts.
Multiplying two independently truncated propagators is not differentiation
of the truncated interaction. The implementation checks the latter as a
negative, not as a candidate physical current.

For the replay's representative pair with both virtualities inside the
Taylor domain, N=2 gives a 23.68% current-tensor norm error even though its
own vector Ward identity closes to numerical precision. Conservation and
accuracy are separate tests; this is not a nucleon form-factor error.

The vector identity then holds exactly for the polynomial kernel. The
axial relation to the truncated pion pole has the explicitly known remainder
\[
 k_\alpha\Pi_{AA,N}^{\alpha\beta}
   -F^2m_\pi^2 k^\beta D_N(s)
   =F^2 k^\beta(s/m_\pi^2)^{N+1}.
\]
Thus a finite derivative expansion is not mislabeled an exact full PCAC
completion. Derivative-expansion order N is also not automatically chiral
power-counting order.

## Connection to LF normalization and nuclear tensor information

In the project's symmetric LF convention,
\[
 k^\pm=(k^0\pm k^3)/\sqrt2,\qquad
 D(k^2)={-1\over2k^+[k^--(m_\pi^2+k_\perp^2)/(2k^+)]}.
\]
Both signs of exchanged, off-shell \(k^+\) are tested. The covariant
propagator has a finite two-sided \(k^+\to0\) limit at the tested off-pole
momenta; the singular decomposition is not assigned zero. This does not
resolve interacting loop zero modes or a finite-cell measure. Factors of
i and the common source charge are stripped consistently.

With continuum-normalized NR nucleon states and convolution measure
\(d^3p/(2\pi)^3\), the leading long-range matching target is
\[
 V_{1\pi}(\boldsymbol k)=
 -{g_A^2\over4F^2}\,
 {(\boldsymbol\tau_1\!\cdot\!\boldsymbol\tau_2)
  (\boldsymbol\sigma_1\!\cdot\!\boldsymbol k)
  (\boldsymbol\sigma_2\!\cdot\!\boldsymbol k)
  \over\boldsymbol k^2+m_\pi^2}.
\]
It has units GeV⁻², not the units of a discrete LF mass-squared matrix.
We preserve the full isospin-pair × spin-pair tensor, rather than a scalar
average. In I=0 its isospin factor is −3; in the spin-triplet subspace at
k along z, the relative diagonal entries are \(1,-1,1\). The directional
tensor is not erased. This coefficient uses **nucleon** g_A as an external
low-energy input; it is not assigned as a quark axial coupling. No
iteration, short-range counterterms or deuteron binding prediction is made.

## The upstream constraint on generated quark/nucleon dynamics

Use \(S^{-1}(p)=A(p^2)\not p-[m_q+b(p^2)]\).
For the nonsinglet identity we strip the common flavor generator and define
the Fourier phases so that the free vertices are
\(\Gamma_A^\mu=\gamma^\mu\gamma_5,\ \Gamma_P=\gamma_5\):
\[
 q_\mu\Gamma_A^\mu
 =S^{-1}(p')\gamma_5+\gamma_5 S^{-1}(p)+2m_q\Gamma_P.
\]
The mass here is the renormalized **current** mass at the common scheme/scale,
not an independently chosen constituent mass.

An algebraic witness uses \(\Sigma_A=(A'+A)/2\),
\(\Delta_A=(A'-A)/(p'^2-p^2)\), \(b_\Sigma=b'+b\):
\[
 \Gamma_A^\mu=
 \left[\Sigma_A\gamma^\mu+
 (p'+p)^\mu{\not p'+\not p\over2}\Delta_A
 +q^\mu b_\Sigma D(q^2)\right]\gamma_5,\qquad
 2m_q\Gamma_P=[2m_q+b_\Sigma m_\pi^2D(q^2)]\gamma_5 .
\]
The identity is direct algebra. The massless limit is taken with both
\(m_q,m_\pi\to0\); no \(1/m_q\) is needed in the weighted pseudoscalar
implementation. In the chiral limit the pole residue is tied to the
dynamical scalar dressing, consistent with the quark-level chiral
Goldberger–Treiman constraint.

**This witness is not selected as the physical axial vertex.** Regular
pseudoscalar terms, further pion Dirac amplitudes, renormalization and
transverse axial structures remain unconstrained. For example
\((q^2\gamma^\mu-q^\mu\not q)\gamma_5\) can change the transverse response
without changing this identity. A physical pion amplitude must be generated
and normalized by the same interaction as the quark dressing, not merely
inserted to pass the Ward test. No BSE/DSE solver or its truncation is selected.

The numerical dressing polynomials are explicit algebra fixtures, not a
physical self-energy, UV ansatz or fit. Their purpose is to expose why a
bare current can fail with a dressed propagator and verify the representation
and frame conventions. The original canonical Dirac algebra is reused.

## Numerical result and decision

[Replay data](../snapshot/docs/next_level/chiral_soft_matching_results.json), generated by
[the replay script](../snapshot/scripts/evaluate_chiral_soft_matching.py).
The 24 fresh spacelike pairs use illustrative scales mπ=0.14 GeV,
F=0.0924 GeV, mq=0.0035 GeV and g_A=1.27, not precision input determinations.

- Explicit/nonlocal axial response: maximum difference 2.09e-18 GeV².
- PCAC: 2.49e-19 GeV³; vector-source Ward identity: 2.37e-18 GeV².
- Independent explicit-action source difference: 9.60e-13 GeV.
- Whole response/source Lorentz transport: 1.01e-17 GeV² / 6.35e-17 GeV.
- Signed LF/covariant propagator: relative difference 2.23e-16.
- Dressed-quark axial witness: residual 3.36e-16 GeV.
- Omitting source contacts fails; the bare-current/dressed-propagator
  negative has residual at least 1.116 GeV in the declared matrix norm.
  This is not an explanation or estimate of the old proton discrepancy.

**Advance:** protected pion correlations and their chiral/vector source
partners are now a qualified component/constraint for the next specification.
Represent them with an explicit collective channel or exact pole-preserving
nonlocal kernel, with one source-owned residue and no duplicate contribution
in the fitted remainder. A finite Taylor contact model is rejected for this
component at pion-scale momentum. This conclusion does not force a global
Fock truncation or choose the whole physical Hamiltonian.

**Next bounded dynamical question:** construct a common chirally invariant
interaction functional whose quark dressing and pseudoscalar bound-channel
kernel are related by its variations. Determine which residual spin/flavor/
color structures and source partners are required before solving p/n.
The first solve must generate the dressing and pion mode together and check
the residue/norm against an independent calculation, rather than fitting a
mass and then manually completing its axial vertex. Preserve ordered color
transport and the existing baryon/fermion maps. Any approximation that fails
that joint test remains a bounded candidate, not a new repair agenda.

This still does not establish the full interaction, confinement, physical
nucleon currents, modular nuclear forces, TMD evolution or GTMD predictions.
Those are subsequent shared-state/operator milestones, not implied by a
successful pion or Ward check.

## Replay and preservation

From the canonical checkout with PYTHONPATH=src:.:

    python scripts/evaluate_chiral_soft_matching.py --output docs/next_level/chiral_soft_matching_results.json
    python -m pytest -q tests/test_chiral_soft_matching.py

The existing temporary Python 3.12.14 / NumPy 1.26.4 / SciPy 1.13.1 runtime
was reused; no new dependency was installed. The new module has 19 passing
direct tests, including explicit derivatives, limits, negative controls,
Lorentz spinor/tensor transport and retained spin/isospin. The combined
focused regression and theory-index run passes 143 tests, with 12 existing
code/runtime warnings. This is not a full repository-suite or PDF-build
claim; the previously documented full-suite limitation is not hidden.

All seven original entry, roadmap and main theory/index files are preserved
and verified byte-for-byte against
6076f09f and in
computation_handoff/backups/2026-09-10_pre_chiral_soft_matching.tar.gz.
Earlier physics implementations, AV18/PDF inputs, original volumes,
public/review mirrors and unrelated untracked work remain unchanged.
