# Source provenance and publication status

[Return to QCD-QuantumGeometry](../README.md)

## Origin

The implementations and recorded results originate in [uva-spin/DeuteronWigner](https://github.com/uva-spin/DeuteronWigner).

| Reference | Revision | Date and role |
| --- | --- | --- |
| Public upstream checked for this companion | `b6fbd04239f8cf9573b54aeec75be0b527881012` | September 9, 2026; observed on public `main` October 23, 2026 |
| Science and implementation snapshot | `dd0d75491aec06d9312d5361eaf0fd56b06e84c7` | September 11, 2026; includes the staged interpolator and Hamiltonian–Euclidean matching reflection |
| Latest numerical result in that snapshot | `2b257e12` | September 11, 2026; stable connection/chiral/sea controls |

At this audit the local checkpoint was 51 commits ahead of public upstream, with no divergent public-only commits. The quantum subtree contained 135 Python files publicly and 180 locally: 45 added files, six modified existing files, and 129 identical files. These are dated source-publication counts, not scientific milestones.

## Included material

- All 180 quantum Python files at the snapshot revision.
- Original formal-layer sources, the early H1 tensor benchmark, Q0/Q1 backend sources, and the original algebraic/geometric and tensor-network TeX documents.
- 225 selected original test files and 82 replay/benchmark scripts identified by direct imports or backend names.
- 74 historical result notes, with repository-relative links adapted for this companion.
- 103 committed JSON reports no larger than 2 MB each, and the original Python package/environment declarations.
- Newly written overview, mathematical guide, results synthesis, reuse instructions, and the complete quantum catalog.

Source, test, script, TeX, configuration, and JSON snapshot files are copied from Git objects at the stated revision, not from the working tree. The historical notes retain their scientific text and dates; a banner identifies their original status. Links to included material resolve inside this companion. Links to identical public material use the pinned public commit. References to other full-checkout files are shown as source paths instead of broken public URLs.

## Runtime scope

The snapshot preserves tools for inspection and traceability. Dependencies in the broader bridge, physics helper, data, and matching layers remain in the complete DeuteronWigner checkout. The companion does not define a new package API or certify that the copied subset can run independently.

Original large numerical arrays, external physics datasets, vendor trees, private operational records, and historical recovery archives are outside this documentation snapshot. Reproduction that uses them requires the appropriate full checkout and inputs. Historical result notes identify the relevant files and conditions.

## Scientific scope

Original notes may contain a then-current next action or describe a result as local-only. Those statements record the status when that note was written. Publication in this companion makes the included material accessible; it does not change its physical validity or revive the old action queue. [The results synthesis](RESULTS.md#latest-constructive-candidate) gives the latest included scientific boundary.

Recorded tests and numerical replays remain historical evidence. This documentation release checks source fidelity, inventory completeness, and document links; it does not report a new full scientific regression or new physics result.

## Attribution and maintenance

Retain the DeuteronWigner origin, original source docstrings, scientific limitations, and literature citations when reusing the material. No new license grant is introduced by this companion.

A later update should identify its upstream revision, refresh the inventory and affected evidence, and preserve distinctions between demonstrated controls, model assumptions, fitted inputs, and physical predictions. It should not turn unmeasured uncertainties or inaccessible source directions into zeroes.
