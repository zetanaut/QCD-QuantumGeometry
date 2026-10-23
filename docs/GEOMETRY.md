# Geometry and mathematical construction

[Return to QCD-QuantumGeometry](../README.md)

The project uses several geometric structures with different physical meanings. Their role is to preserve information, define valid composition, and improve numerical solution of declared dynamics.

## Distinct structures

| Structure | Mathematical object | Role and boundary |
| --- | --- | --- |
| Projective quantum-state geometry | Rays in the represented Hilbert space and the pullback of their metric | Removes global-phase redundancy and defines state-sensitive optimization |
| Tensor-coordinate geometry | Factorizations related by internal invertible basis changes | Identifies redundant coordinates; it does not identify weak physical directions as gauge |
| Symmetry fusion | Representation spaces, multiplicities, and intertwiners | Enforces admissible couplings and permits exact recoupling on compatible complete domains |
| Gauge transport | Ordered SU(3) matrices along specified paths | Retains endpoint covariance, curvature, noncommuting orientation, and source response |
| Composite-state geometry | A computed Gram matrix of embedded fermionic states | Defines the physical inner product and the common transport of all operators |
| Provenance topology | Typed dependency and replacement/exclusion relations | Records composition and double-counting rules; it supplies no physical amplitude |
| Partonic phase-space geometry | The Fourier-dual impact coordinate and transverse momentum of a GTMD | Organizes Wigner and orbital information; it remains distinct from the TMD impact variable and the nuclear Wigner object |

The original [algebraic/geometric theory](../snapshot/references/algebraic_geometric_next_level_model_note_revised.tex) and [tensor-network volume](../snapshot/references/volume_viii_symmetry_adapted_tensor_networks_prediction_compiler.tex) define the architecture. Their historical solver/model examples are not current production-model selections.

## Projective metric on the actual circuit

For real parameters $\theta^a$ and a normalized complex state, define

$$
Q_{ab}=\langle\partial_a\psi|(I-|\psi\rangle\langle\psi|)|\partial_b\psi\rangle,
\qquad g_{ab}=\operatorname{Re}Q_{ab}.
$$

The implementation differentiates the same complex two-level gates used by the executed circuit. For energy $\mathcal E=\langle H\rangle$, its natural-gradient direction is

$$
\dot\theta=-\tfrac12 g^+\nabla_\theta\mathcal E.
$$

When the full tangent is resolved, the horizontal state direction agrees with $-(H-\mathcal E)\psi$. The dense implementation factors the real/imaginary horizontal Jacobian directly rather than first forming its Gram matrix. A caller-selected cutoff identifies numerically unresolved tangent directions.

Convergence is tested with the executed eigenpair residual. A small energy change or small parameter gradient is insufficient, particularly near tiny amplitudes. Ground-state identification remains a separate check against an independent reference.

Implementation: [dense geometry and optimizer](../snapshot/src/deuteron_wigner/quantum/m2_graph_natural_gradient.py). Evidence: [37-state convergence and currents](../evidence/m2_graph_convergence_result.md).

## Linear-storage geometry of the complete tree

The tree circuit introduces each child once, after its parent is present. Each edge has a rotation angle and a phase. Let $w_e$ be the parent probability immediately before its split and $p_v$ the final vertex probabilities. In the interior of this coordinate chart,

$$
g_{\theta_e\theta_f}=\delta_{ef}\frac{w_e}{4},\qquad
g_{\theta\phi}=0,\qquad
g_{\phi\phi}^{-1}=B\,\operatorname{diag}(1/p_v)B^T,
$$

where $B$ is the oriented edge–vertex incidence matrix. Code applies the incidence operations without constructing the matrices. Every inverse action uses $O(D)$ time and storage.

The API provides four operations:

| Function | Purpose |
| --- | --- |
| `tree_state_weights` | Evaluate the state and pre-split parent probabilities |
| `tree_inverse_metric_action` | Apply the exact interior inverse or an explicitly regularized preconditioner |
| `tree_state_tangent` | Apply the horizontal state Jacobian to a direction without storing that Jacobian |
| `tree_parameters_from_state` | Convert a supplied nonzero state to the same gate coordinates, modulo global phase |

An explicit positive regularization replaces $p,w$ by $p+\epsilon,w+\epsilon$ inside the inverse preconditioner. This does not remove amplitudes from the represented state. With zero regularization a singular chart is rejected. The optional state retraction uses the gradient-generated tangent, not an imported exact Hamiltonian eigenstate.

The complete ansatz has $2(D-1)$ parameters. It is an exact full-state reference, with costs linear in Hilbert-space dimension rather than in qubit count. A compressed tensor or neural ansatz requires a newly derived pullback metric.

Source: [tree projective geometry](../snapshot/src/deuteron_wigner/quantum/m2_tree_projective_geometry.py). The saved direct tests compare chain, star, and binary trees with independently formed dense metrics and test a 32,768-state inverse action without dense linear algebra.

## Factor redundancy and operator-sensitive directions

For the scalar control, a rank-five $57\times57$ coefficient matrix is represented as $C=LR$. Its tangent is

$$
\delta C=\delta L\,R+L\,\delta R,
\qquad (\delta L,\delta R)=(LX,-XR)\Longrightarrow\delta C=0.
$$

There are 570 complex factor coordinates and 25 complex redundancy directions, leaving 545 complex coefficient tangents. The implementation constructs an orthonormal coefficient-tangent chart. It then differentiates the actual quadrature-weighted state $b(C)$:

$$
\psi=\frac{b}{\sqrt{b^\dagger b}},\qquad
A=\frac{(I-\psi\psi^\dagger)\partial b}{\sqrt{b^\dagger b}}.
$$

At the real reference, real and imaginary coefficient directions retain the imaginary cross blocks of the quantum geometric tensor. The external global phase is a null ray direction; it is not an additional free phase of a core with fixed nonzero boundary conditions.

The state and current sample different virtuality ranges. Both current endpoints and the normalization derivative must be retained. The rank-preserving witness in the saved audit changes the state by $2.5885\times10^{-10}$ while changing the pair current by $2.0526\times10^{-4}$. That exceeds the previous $10^{-4}$ absolute-current target while leaving the exact core constraints intact.

Small sampled metric eigenvalues are therefore not proven exact physical null directions. Quadrature refinement, new kinematics, and source response can reveal information they conceal. An augmented response norm can detect such directions, but its weighting does not automatically make it a physical quantum metric.

Source and detailed conditioning controls: [scalar geometry audit](../evidence/quantum_geometric_scalar_geometry.md).

## Symmetry and tensor-network capacity

The original network design places physical labels on edges: sector, species, color representation and multiplicity, spin/helicity, flavor, orbital and momentum coordinates, resolution, and operator scheme. Fixed representation couplers and learnable degeneracy tensors have different responsibilities.

Complete fusion trees can describe equivalent representations when their recoupling maps, conventions, and fermionic swaps are carried correctly. Finite-rank choices define different approximation spaces. The Hamiltonian and operator network need not share the state tree's connectivity; multi-field interactions can require higher-valence tensors or equivalent factorizations.

The early `microscopic/h1/tensor_network.py` is a finite benchmark. Its optimizer diagonalizes nested allowed subspaces and its illustrative recoupling is a discrete Fourier matrix. It should not be presented as a general symmetry-resolved Schmidt optimizer or a complete physical recoupling library. Later canonical emitter recoupling and fermionic composition have separately derived domain-specific maps.

Source: [early benchmark](../snapshot/src/deuteron_wigner/microscopic/h1/tensor_network.py); architecture: [Volume VIII](../snapshot/references/volume_viii_symmetry_adapted_tensor_networks_prediction_compiler.tex).

## Ordered transport and the meaning of a path

For the modular graph convention, $U_{ab}$ transports a fundamental column from $b$ to $a$:

$$
U_{ab}\mapsto G_aU_{ab}G_b^\dagger,\qquad
U_{ba}=U_{ab}^\dagger,\qquad
U_P=U_{v_0v_1}\cdots U_{v_{n-1}v_n}.
$$

A spanning-tree gauge fixing can move loop information into non-tree links; discarding those matrices discards physics carried by the graph. The saved control constructs configurations with equal separate elementary loop traces but a difference of 0.02032736 in a normalized loop-commutator trace.

The old formal metadata method `WilsonPathId.inverted` also reverses future/past orientation. The newer native graph `Path` geometric reverse preserves its process label. These are different operations. Observable process reversal and Hermitian path reversal require an explicit adapter when interfaces are combined.

Curved gauge transport depends on the path geometry, not merely its ordinary homotopy class. Path labels do not establish quantized topological sectors, and a state preparation dressing is distinct from a process-dependent GTMD/TMD staple.

Source: [modular gauge and response controls](../evidence/quantum_geometric_modular_architecture.md).

## Composite overlaps and common operator transport

Let $E$ embed composite coefficients into a common fermionic carrier and let

$$
N=E^\dagger E=V_+\Lambda_+V_+^\dagger,\qquad
W=V_+\Lambda_+^{-1/2},\qquad U=EW.
$$

On the retained non-null range, $U^\dagger U=I$. Every operator uses

$$
O_{\mathrm{cl}}=W^\dagger E^\dagger O E W.
$$

This is a common representation change, with no fitted current rescaling. Exact Pauli-null directions are distinguished from approximate small eigenvalues. In the saved same-orbital proton/neutron example, the unqualified composite norm is $10/9$; the two-ordering Gram matrix has eigenvalues $0$ and $20/9$. Consistent transport restores the physical charge in the qualified coordinate space.

The finite interpolators have declared color, charge, internal spin, and fermionic statistics. Their labels do not identify a physical bound-state wavefunction. Nuclear assembly requires sector-specific variational states, actual overlaps, and common matching of one- and many-body operators.

Source: [modular architecture and Gram controls](../evidence/quantum_geometric_modular_architecture.md).

## Response topology and joint inference

For a declared finite inverse with distinct poles $h_i$, nonzero endpoint components $g_i$, and a symmetric source matrix $E$, define

$$
v_i(x)=\frac{g_i}{x+h_i},\qquad S_E(x,y)=v(x)^TEv(y).
$$

Equal-virtuality data $S_E(x,x)$ have rank $2n-1$, rather than $n(n+1)/2$. Off-diagonal source coordinates contain cycle directions invisible to their vertex divergences. Independent left/right virtualities can recover the full symmetric matrix in the ideal Cauchy-system control. Degeneracy, weak endpoint support, finite sampling, and noise alter accessibility or conditioning.

This is an algebraic response-graph cycle space. It does not establish physical topological order. A source-free hidden mode may also reappear at higher source order, so reachability at one order is not a deletion certificate.

Source: [source/network transport](../evidence/source_network_transport.md).

In joint inference, the state metric and the coefficient information matrix $A^T\Sigma^{-1}A$ describe different objects. Variational parameters solve a supplied state equation; dynamical coefficients require matching and physical constraints. Their full joint geometry need not be block diagonal.

Source: [joint state and dynamics](../evidence/joint_state_dynamics.md).
