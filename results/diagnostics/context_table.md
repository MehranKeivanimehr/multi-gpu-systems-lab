# Context dispositions — original, follow-up, closing, final

> Local correction: streaming is **qualified**, not a fully protocol-conforming timing measurement. True request latency is unavailable; arrival offsets use one session origin. Current census: 17 supported / 2 qualified. See [corrections](../../docs/LOCAL_RELEASE_CORRECTIONS.md).

Append-only: an earlier outcome is never rewritten, and a context that a
later window did not re-execute keeps its previous outcome unchanged.

| context | kind | world size | original | after follow-up | after closing | final |
|---|---|---|---|---|---|---|
| `G0_1080ti_zero1_w1_fp32` | training | 1 | completed with qualifications | completed with qualifications | completed with qualifications | **completed with qualifications** |
| `D_p1080_2_zero1_fp32` | training | 2 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_p1080_4_zero1_fp32` | training | 4 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_p1080_6_zero1_fp32` | training | 6 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_p1080_1_zero2_fp32` | training | 1 | completed with supported checks | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_p1080_2_zero2_fp32` | training | 2 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_p1080_4_zero2_fp32` | training | 4 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_p1080_6_zero2_fp32` | training | 6 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_1_zero1_fp32` | training | 1 | completed with supported checks | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_2_zero1_fp32` | training | 2 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_1_zero2_fp32` | training | 1 | completed with supported checks | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_2_zero2_fp32` | training | 2 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_1_zero1_amp` | training | 1 | completed with supported checks | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_2_zero1_amp` | training | 2 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_1_zero2_amp` | training | 1 | completed with supported checks | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `D_rtx_2_zero2_amp` | training | 2 | completed with qualifications | completed with supported checks | completed with supported checks | **completed with supported checks** |
| `R_rtx_1_tinyllama_fp32` | reference | 1 | invalid diagnostic or software failure | completed with qualifications | completed with supported checks | **completed with supported checks** |
| `R_p1080_1_tinyllama_fp32` | reference | 1 | invalid diagnostic or software failure | completed with qualifications | completed with supported checks | **completed with supported checks** |
| `S_rtx_1_stream_256_64` | streaming | 1 | invalid diagnostic or software failure | invalid diagnostic or software failure | invalid diagnostic or software failure | **completed with qualifications** |

Totals: **17** completed with supported checks, **2** completed with qualifications, **0** valid numerical failure, **0** invalid diagnostic / software failure, of 19 contexts.
