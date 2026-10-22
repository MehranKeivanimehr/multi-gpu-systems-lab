# A6000: what can be reproduced, and from what

Author of the repository: Mehran Keivanimehr.

## A. Replayable from the review ZIP alone (any CPU, Python 3 standard library)
| What | How | Expected |
|---|---|---|
| Selected report quantities and the claim/evidence table | From the repository root: `python results/a6000_validation/evidence/70_final_review/recompute_final_tables_v2.py results/a6000_validation/evidence . results/a6000_validation/pilot_evidence` | 49/49 checks hold |
| **The pre-registered R5/C5 control-C endpoints** (G1, U, loss) and verdicts | `replay_control_c_slices.py evidence/70_final_review/calibration_slices evidence/44_c18_comparison/comparison_c18b.json` | all six endpoints pass and agree with the on-node comparator |
| The claim–evidence table | `build_claim_evidence_table_v2.py evidence pilot_evidence/analysis <new csv>` | identical to `CLAIM_EVIDENCE_TABLE.csv` |
| Repository checks of the final source (Candidate 20) | from `evidence/20_candidate_c20/multigpu-bench`: `tools/validate.py`, `unittest discover -s tests`, and the `--check` modes of `scientific_corrections`, `scientific_status`, `build_report` and `prepare_notebook` | 6/6; 269 tests OK. Needs Torch for the Torch-CPU tests (see C) |
| Package integrity | `sha256sum -c SHA256SUMS` | every entry |

**Calibration slices.** `calibration_slices/` holds 4 × 787,968 float32 values (12.6 MB): the pre-registered fixed
slices of R5's and C5's first-update gradient and five-update Δθ. Each slice was cut from a source snapshot whose full
SHA-256 was re-verified at export. `SLICES_MANIFEST.json` gives:
- the source hashes;
- the per-tensor index hashes;
- the offsets;
- the slice hashes.

## B. Replayable only with the on-node tensor evidence (not in the ZIP)
| What | Data needed | Where |
|---|---|---|
| Full-vector R5/C5 comparison; supplementary whole-vector results | R5/C5 `grad_b1.f32` and `dtheta_final.f32` (24.7 GB) | `43_c18_window/w18b_…/cases/*/` |
| ZeRO attribution contrasts (G4 vs G3 bitwise; G3/G1/G0/X1) | 5 first-boundary gradients (30.9 GB) + the sealed S1a reference (12.3 GB) | `47_c20_window/w20fb_…/cases/*/`, `42_s1a_window/…/reference/` |
| S1a full-vector aggregate statistics | Saved per-tensor statistics can be reaggregated | The recipe arrays were not retained. The original elementwise comparison is NOT replayable, even from the listed on-node records; reproducing it would require a new experiment. |

Every snapshot is listed in `SNAPSHOT_MANIFEST.json` with its SHA-256 and size; the ZIP holds the per-tensor index
files with per-tensor hashes. **Keep verified backups of these directories before deleting any working directory on the
node.** Check a backup with `sha256sum -c` against the checkpoint `checkpoints/CHECKPOINT_20261022final.sha256` or its
successor.

## C. Not reproducible without new GPU runs
Every training or inference measurement itself:
- reproducing it needs the same model revisions, software stack and hardware;
- the protocols and environment are in `PROTOCOLS_AND_ENVIRONMENT.md`;
- the frozen candidates and launch scripts are packaged.

**Historical performance numbers:** reproducible only as recorded values (the CSVs and raw logs are packaged), not as
reruns.

**Test tiers:**
| Tier | Needs |
|---|---|
| portable | standard library |
| Torch-CPU | numerics and gloo tests |
| Linux | systemd user scopes: the lifecycle and integrated supervisor self-tests |
| on-node | GPUs: never run by the test suites |

The repository's unit tests need Torch, NumPy and PyYAML. Skips are reported by `unittest`, and all six checks reported
0 skips on-node.
