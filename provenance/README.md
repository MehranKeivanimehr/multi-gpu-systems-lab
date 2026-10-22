# Data provenance and the publication copy

Author: Mehran Keivanimehr · Release 2026-10-22

This tree combines the corrected Titan/1080 public base with the reviewed A6000 Candidate 20 source and selected closeout evidence. A100 records remain historical. [Input archive identity](publication.json), [public evidence checksums](measurements.sha256.json).

## Original records versus public bytes

The original private archives are preserved unchanged. The public tree is a derived copy: operator paths, node names and device identifiers are replaced with stable publication labels. All retained FP32 payloads and numerical measurement values keep their original values. The private mapping and original operator/incident records are not part of this repository.

`measurements.sha256.json` authenticates the derived public evidence, not the byte identity of every private metadata file. `publication.json` records the literal input-archive hashes and the transformation scope. Collection-time source hashes identify collection bytes; they are not a claim that edited publication code already ran on hardware.

## Evidence included

- Historical campaign source results and corrected analysis tables.
- A6000 fixed-slice control arrays, raw-sample control records, tensor indices, per-tensor comparison records, current claim/evidence tables and selected window summaries.
- Titan retained inference arrays, numerical context summaries, streaming events and correction sidecars.

The standalone tools verify retained payload integrity and reproduce their stated calculations. A hash of an absent full tensor is a provenance record, not a substitute for the tensor itself.

## Corrections

Current reports supersede earlier interpretations. The new equal-cap serving comparison uses TP2's cap-512 audit rows, and dispersion uses conventional even-sample medians. Original failure states and numerical qualifications are not silently turned into passes. Private superseded/operator records remain outside the public tree; selected historical documents are clearly labelled by the current reports.

## Limitations

See [data availability](../docs/LIMITATIONS.md#what-can-be-replayed) for exactly what can be replayed. Privacy pattern checks cannot prove that no indirect identifier exists. No missing observations have been fabricated and no new GPU runs were performed during publication preparation.
