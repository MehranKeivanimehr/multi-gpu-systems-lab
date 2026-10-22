| context | original outcome | newly established | remaining qualifications | final |
|---|---|---|---|---|
| `D_p1080_2_zero1_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_p1080_4_zero1_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_p1080_6_zero1_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_p1080_2_zero2_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_p1080_4_zero2_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_p1080_6_zero2_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_rtx_2_zero1_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_rtx_2_zero2_fp32` | completed_with_qualifications | working/master consistency established against this rank's own partition view (fp32_exact_equality), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_rtx_2_zero1_amp` | completed_with_qualifications | working/master consistency established against this rank's own partition view (mixed_precision_expected_cast), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `D_rtx_2_zero2_amp` | completed_with_qualifications | working/master consistency established against this rank's own partition view (mixed_precision_expected_cast), bitwise identical, 0 differing elements<br>whole-model coverage reconciled exactly once across the rank set: 124,439,808 of 124,439,808 elements, no missing, duplicate, unexpected or overlapping spans | none | **completed_with_supported_checks** |
| `R_rtx_1_tinyllama_fp32` | invalid_diagnostic_or_software_failure | realized token-id input and attention mask recorded and digested; initial state recorded and digested<br>greedy output token ids compared across families | the final-step logit comparison against the 64u tolerance could NOT be evaluated: only digests of the logits were retained, and a digest mismatch does not say whether the difference is inside the tolerance | **completed_with_qualifications** |
| `R_p1080_1_tinyllama_fp32` | invalid_diagnostic_or_software_failure | realized token-id input and attention mask recorded and digested; initial state recorded and digested<br>greedy output token ids compared across families | the final-step logit comparison against the 64u tolerance could NOT be evaluated: only digests of the logits were retained, and a digest mismatch does not say whether the difference is inside the tolerance | **completed_with_qualifications** |
| `S_rtx_1_stream_256_64` | invalid_diagnostic_or_software_failure | - | no server, request or stream event was produced; there are no streaming observations | **invalid_diagnostic_or_software_failure** |

## Closing window linkage

Three contexts were re-executed under `CLOSED-20261020T125221Z`. For each,
the ledger keeps the original outcome, the follow-up outcome and its
qualifications, and the closing outcome, in separate fields. A follow-up
qualification that the closing window answered is marked `superseded` and
dated; it is not deleted and it is not presented as if it had never applied.
The streaming context's follow-up qualification is marked
`partially_superseded`, because a server and 100
requests were produced this time while the absence of stream events still
holds.

## Streaming-only window linkage

One context was re-executed under `FINAL-20261020T141800Z`. The ledger
keeps its original, follow-up, closing and final outcomes in separate fields,
and `preserved_failure_history` lists every authorization it failed under, with
the reason. A qualification the session answered is marked `superseded` and
dated; nothing is deleted. The limitations the session itself carries are
recorded as its current qualifications.
