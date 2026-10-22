# Changelog

## 1.0.0 — 2026-10-22

- Integrated the qualified A6000 closeout into the corrected Titan/1080 base; A100 remains historical.
- Corrected equal-concurrency serving cohorts and repeat-dispersion summaries.
- Added replayable fixed-slice calibration, strict public evidence checks and negative controls.
- Added six reproducible SVG/PNG figures, a labelled walkthrough GIF and engineering/primary-source reading guides.
- Rebuilt the current reports and scope register; kept the detailed limitations at the end of the main README.
- Prepared a metadata-anonymized publication copy with distinct public checksums and preserved private input-archive identities.
- Kept the existing all-rights-reserved license. No remote CI, peer review or new GPU execution is claimed.

Earlier entries below describe their own release states; the current ledgers and reports supersede their completion counts.

## 0.1.0 — 2026-10-14

- Collected the three benchmark campaigns in one repository.
- Retained measured records and added their SHA-256 manifest.
- Added a combined report, full run catalog, protocol matrix, and offline validation.
- Added Linux/Windows CI and contributor documentation.
- Corrected the Titan communication payload unit, precision labels, comparison scope, and reproduction driver.
- Added a cleaned A100 reproduction notebook and retained its historical source separately.
- Assigned authorship and citation metadata to Mehran Keivanimehr.

This is the local repository version; no GitHub release or DOI has been assigned.

## Closing release (2026-10-20)

- The cross-family reference comparison is evaluated and passes. Complete
  final-step logit arrays are now shipped, in a documented non-pickle binary
  format with checksums, so the verdict can be recomputed without a GPU.
- Two new offline commands: `tools/recompute_reference_verdict.py` and
  `tools/recompute_streaming_metrics.py`.
- The streaming Prometheus startup defect is closed; the session itself is
  still not obtained, with its cause identified and a CPU-verified repair not
  yet executed on a GPU.
- Context dispositions after the closing window: 17 completed with
  supported checks, 1 with qualifications, 0 valid numerical
  failure, 1 invalid diagnostic / software failure, of 19.

## Final release (2026-10-20)

- The streaming session is obtained: 100 of 100 requests successful
  under `STREAMING_SUCCESS_RULE.md`, frozen before the session. Raw events,
  server logs, the authorization and the offline recomputation ship with it.
- The cause of the previous empty-bodied 200s is fixed at the request path and
  verified against the installed vLLM (`zd_test_guided_dispatch`).
- `tools/recompute_streaming_metrics.py` rewritten: request accounting,
  per-class counts, lengths supported by the evidence, chunk measurements,
  completion status, and recorded corrections; it refuses damaged evidence.
- Three corrections recorded beside the session evidence rather than edited
  into it (CORR_STREAM_001..003).
- Final dispositions over 19 contexts: 18 supported, 1 qualified,
  0 valid numerical failure, 0 software failure.
