"""Quantum infrastructure with opt-in legacy model APIs.

Loading an independent submodule must not instantiate the old BLFQ/H2 stack.
The public exports of the former initializer remain available on demand.
"""
from importlib import import_module as _import_module

# Original import order and explicit __all__ lists; later duplicate names win.
_LEGACY_EXPORTS = {
    'operator_bundle': (
        'CLAIM_TIER PARAMETER_IDS OperatorTerm ExploratoryHamiltonian ExactGroundState '
        'build_exploratory_hamiltonian build_mapped_exploratory_hamiltonian bundle_record '
        'exact_ground_state expectation exploratory_response_map '
    ).split(),
    'm2_k9_exploratory_state': (
        'CLAIM_TIER ExploratoryK9ParameterPoint K9_EXPLORATORY_BASELINE '
        'K9_EXPLORATORY_SENSITIVITY_POINTS K9HamiltonianAssembly ExploratoryK9LowEigenspace '
        'build_parameter_explicit_k9_hamiltonian solve_low_k9_eigenspace stability_report '
        'derivative_report sensitivity_report q0_codec_report q0_stateprep_subspace_report '
        'exploratory_k9_state_record '
    ).split(),
    'm2_state_current_boundary': (
        'M2StateCurrentInterfaceError m2_k9_state_current_interface_audit '
        'require_lawful_m2_k9_current_interface '
    ).split(),
    'm2_color_singlet_bridge': (
        'COLOR_BRIDGE_CLAIM_TIER COLORS DEFAULT_EQUIVARIANCE_TOLERANCE M2ColorBridgeError '
        'M2ColorModules M2SingletOperator m2_color_modules c47_to_m2_target_embedding '
        'map_c47_operator_to_m2_target antitriplet_diquark_isometry color_singlet_embedding '
        'complete_m2_operator_to_singlet color_completion_algebra_record '
        'color_completion_record '
    ).split(),
    'm2_c53_mixed_singlet_state': (
        'K9 C53_MIXED_CLAIM_TIER C53_PARAMETER_ID ExploratoryC53K9Point '
        'C53_MIXED_K9_EXPLORATORY_BASELINE ReducedOperatorTerm C53MixedK9Hamiltonian '
        'C53MixedHelicityState C53MixedK9HelicityPair build_c53_mixed_k9_hamiltonian '
        'solve_c53_mixed_k9_helicity_pair vary_c53_mixed_k9_point helicity_derivative_report '
        'c53_coupling_scan c117_response_report c53_mixed_state_record '
    ).split(),
    'm2_nucleon_spectator_composition': (
        'SCALAR_SPECTATOR AXIAL_SPECTATOR SPECTATOR_CHANNELS TARGETS '
        'SPECTATOR_COMPOSITION_CLAIM_TIER NucleonCompositionComponent NucleonSpectatorState '
        'NucleonSpectatorDoublet spectator_spin_wavefunction spectator_flavor_wavefunction '
        'spectator_exchange_report spectator_coupling_report build_nucleon_spectator_doublet '
        'nucleon_spectator_doublet_record spectator_composition_family_record '
    ).split(),
    'm2_nn_deuteron_state': (
        'HBARC_GEV_FM NN_DEUTERON_CLAIM_TIER OMITTED_NUCLEAR_SECTORS NNNuclearResolution '
        'NN_AV18_NUCLEAR_R1 NNWaveFunctionMember NNLightFrontWaveGrid '
        'ExploratoryNNDeuteronState build_av18_wave_member build_nn_light_front_wave_grid '
        'isospin_singlet_report compose_nn_deuteron_state nn_deuteron_state_record '
        'nn_deuteron_family_record '
    ).split(),
    'm2_nn_transfer_current': (
        'NN_TRANSFER_CURRENT_CLAIM_TIER NN_RECOIL_AUTHORITY_ID REFERENCE_MASS_ROLE '
        'OMITTED_CURRENT_TERMS NNOneBodyCurrentMember NN_POINTLIKE_CHARGE_CURRENT_V1 '
        'NNTransferFibers NNOneBodyDiagnosticCurrent build_nn_transfer_fibers '
        'evaluate_nn_one_body_diagnostic_current nn_one_body_current_pair_record '
    ).split(),
    'm2_nn_current_response': (
        'NN_CURRENT_RESPONSE_CLAIM_TIER NN_CURRENT_RESPONSE_STATUS HELICITY_CURRENT_IDS '
        'EXPECTED_PARAMETER_IDS DEFAULT_PARAMETER_SCALES PARAMETER_UNITS '
        'NNMicroscopicCurrentResponse evaluate_nn_microscopic_current_response '
        'nn_microscopic_current_response_record '
    ).split(),
    'm2_active_quark_current': (
        'M2_ACTIVE_CURRENT_STATUS M2_ACTIVE_CURRENT_CLAIM_TIER NN_ACTIVE_IMPULSE_STATUS '
        'NN_ACTIVE_IMPULSE_CLAIM_TIER CURRENT_COMPONENT_IDS ACTIVE_OPERATOR_SOURCES '
        'UNAVAILABLE_INTERNAL_CURRENT_TERMS UNAVAILABLE_NUCLEON_CURRENT_TERMS '
        'displaced_ho_overlap M2ActiveQuarkImpulseOperator '
        'build_m2_active_quark_impulse_operator active_operator_pair_record '
        'pair_active_current_matrix nucleon_active_impulse_matrix '
        'NNActiveImpulseDiagnosticCurrent evaluate_nn_active_impulse_diagnostic_current '
        'nn_active_impulse_pair_record '
    ).split(),
    'm2_internal_current_response': (
        'M2_INTERNAL_CURRENT_RESPONSE_STATUS M2_INTERNAL_CURRENT_RESPONSE_CLAIM_TIER '
        'PAIR_CURRENT_IDS DIAGNOSTIC_RANK_RELATIVE_TOLERANCE NNInternalCurrentResponse '
        'evaluate_nn_internal_current_response nn_internal_current_response_record '
    ).split(),
    'm2_resolved_spectator_current': (
        'M2_RESOLVED_SPECTATOR_STATUS M2_RESOLVED_CURRENT_STATUS '
        'M2_RESOLVED_SPECTATOR_CLAIM_TIER M2_RESOLVED_CURRENT_CLAIM_TIER '
        'RESOLVED_SPECTATOR_SOURCES UNAVAILABLE_MATCHED_CURRENT_TERMS '
        'ResolvedSpectatorResolution M2_RESOLVED_SPECTATOR_K9_EQUAL_SHARE_S_WAVE_V1 '
        'resolved_sector_longitudinal_records M2ResolvedSpectatorHamiltonian '
        'build_resolved_spectator_hamiltonian resolved_spectator_hamiltonian_record '
        'ResolvedSpectatorCurrentFactors build_resolved_spectator_current_factors '
        'nucleon_resolved_impulse_components NNResolvedSpectatorImpulseCurrent '
        'evaluate_nn_resolved_spectator_impulse_current '
        'nn_resolved_spectator_current_pair_record '
    ).split(),
    'm2_resolved_spectator_response': (
        'M2_RESOLVED_RESPONSE_STATUS M2_RESOLVED_RESPONSE_CLAIM_TIER '
        'NNResolvedSpectatorResponse evaluate_nn_resolved_spectator_response '
        'nn_resolved_spectator_response_record '
    ).split(),
    'm2_permutation_complete_h2': (
        'M2J_BASIS_STATUS M2J_HAMILTONIAN_STATUS M2J_ACTIVE_CURRENT_STATUS M2J_CLAIM_TIER '
        'M2J_TARGET_RESOLUTION_ID UNAVAILABLE_M2J_HAMILTONIAN_TERMS '
        'UNAVAILABLE_M2J_CURRENT_TERMS QuarkPermutationAlgebra build_quark_permutation_algebra '
        'permutation_algebra_record PermutationCompleteH2BasisState PermutationCompleteH2Basis '
        'build_permutation_complete_h2_basis M2ToPermutationCompleteH2Embedding '
        'build_m2_to_permutation_complete_h2_embedding lift_color_scalar_fock_diagonal_operator '
        'lift_active_emitter_operator M2JPermutationCompleteHamiltonian '
        'build_m2j_permutation_complete_hamiltonian M2JPermutationCompleteActiveCurrent '
        'build_m2j_permutation_complete_active_current m2j_embedding_record '
        'm2j_active_current_pair_record '
    ).split(),
    'm2_emitter_tree_recoupling': (
        'M2K_EMITTER_TREE_STATUS M2K_CLAIM_TIER M2K_RESOLUTION_ID EmitterTreeRecouplingAudit '
        'CanonicalEmitterOrbitAudit jacobi_tree_matrix mass_weighted_tree_matrix '
        'transverse_ground_overlap build_emitter_tree_recoupling_audit '
        'build_canonical_emitter_orbit_audit emitter_tree_closure_record '
    ).split(),
    'm2_spin_flavor_s3_completion': (
        'M2L_SPIN_FLAVOR_STATUS M2L_C53_SOURCE_STATUS M2L_CLAIM_TIER M2L_RESOLUTION_ID TARGETS '
        'HELICITIES_TWICE SINGLE_QUARK_STATES SPIN_FLAVOR_BASIS SpinFlavorS3Algebra '
        'GlobalGroundNucleonSpinFlavorState TotalSignSpinFlavorCarrier '
        'C53SpinFlavorSourceChannel C53SpinFlavorOrbitAudit '
        'spin_flavor_permutation_representation build_spin_flavor_s3_algebra '
        'build_global_ground_nucleon_spin_flavor_state build_total_sign_spin_flavor_carrier '
        'build_c53_spin_flavor_source_channels build_c53_spin_flavor_orbit_audit '
        'spin_flavor_s3_completion_record '
    ).split(),
    'm2_total_sign_h2_hamiltonian': (
        'M2M_HAMILTONIAN_STATUS M2M_CLAIM_TIER M2M_RESOLUTION_ID UNAVAILABLE_COMPLETE_H2_TERMS '
        'GlobalGroundMassEmbedding SpinScalarExternalReduction '
        'reduce_spin_scalar_external_operator TotalSignH2BasisState M2TotalSignH2Hamiltonian '
        'TotalSignH2HelicityState TotalSignH2HelicityPair build_total_sign_h2_hamiltonian '
        'solve_total_sign_h2_helicity_pair total_sign_h2_hamiltonian_record zero_c53_point '
    ).split(),
    'm2_hamiltonian_gauged_current': (
        'M2N_DENSITY_STATUS M2N_CURRENT_STATUS M2N_CLAIM_TIER M2N_PROBE_ID '
        'UNAVAILABLE_COMPLETE_CURRENT_TERMS M2BaryonDensityOperator '
        'M2HamiltonianGaugedBaryonCurrent build_m2_baryon_density_operator '
        'build_hamiltonian_gauged_baryon_current hamiltonian_gauged_current_record '
    ).split(),
    'm2_pennylane_contract': (
        'M2O_PENNYLANE_CONTRACT_STATUS M2O_CLAIM_TIER M2O_ENCODING_ID M2CompactQubitBasis '
        'M2PennyLaneContract build_full_qubit_basis build_jz_block_qubit_basis '
        'encode_compact_index decode_bitstring compact_to_padded_state padded_to_compact_state '
        'pad_sparse_operator physical_projector padding_projector build_m2_pennylane_contract '
        'm2_pennylane_contract_record '
    ).split(),
}
_EXPORT_MODULE = {name: module for module, names in _LEGACY_EXPORTS.items() for name in names}
__all__ = list(_EXPORT_MODULE)


def __getattr__(name):
    module = _EXPORT_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
