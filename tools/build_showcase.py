"""Build the public evidence summary and current study reports deterministically."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from a6000_evidence import csv_rows, inference, precision, read, replay

ROOT = Path(__file__).resolve().parents[1]


def summary(root=ROOT):
    led = read(root/'results/diagnostics/context_ledger.json')
    ar = root/'results/a6000_validation/evidence/70_final_review'
    claims = csv_rows(ar/'CLAIM_LEDGER_v2.csv')
    scaling = [r for r in csv_rows(root/'studies/titan-1080/results/scaling_efficiency.csv')
               if r['family']=='GTX 1080 Ti' and r['mode']=='ddp' and r['precision']=='fp32'
               and float(r['micro_batch_per_gpu'])==4]
    return {'schema_version':1,'author':'Mehran Keivanimehr','release_date':'2026-10-22',
            'a6000':{'calibration':replay(root),'serving':inference(root),'precision':precision(root),
                     'claim_counts':dict(sorted(Counter(r['status'] for r in claims).items()))},
            'titan':{'contexts':len(led['contexts']),'status_counts':led['by_final_status'],
                     'training_contexts':sum(c['kind']=='training' for c in led['contexts']),
                     'training_world_sizes':sorted({c['world_size'] for c in led['contexts'] if c['kind']=='training'}),
                     'historical_baseline_tokens_s':float(scaling[0]['baseline_1gpu_tokens_per_s']),
                     'historical_weak_scaling':[{'gpus':int(r['n_gpus']),'tokens_s':float(r['tokens_per_s']),
                          'efficiency_pct':float(r['scaling_efficiency_pct'])} for r in scaling]},
            'provenance':'Derived from retained public evidence; no new GPU execution.'}


def compact_numbers(x):
    if isinstance(x,float): return float(format(x,'.12g'))
    if isinstance(x,dict): return {k:compact_numbers(v) for k,v in x.items()}
    if isinstance(x,list): return [compact_numbers(v) for v in x]
    return x


def combined_report(root=ROOT, inv=None):
    s=summary(root); arms=s['a6000']['serving']; c=s['a6000']['calibration']
    out=['# Results and engineering interpretation','','Mehran Keivanimehr · October 2026','',
         'This study keeps historical performance measurements separate from later numerical diagnostics. Each result below identifies its cohort and what was actually checked. No cross-platform speed ranking is calculated. [Project protocols (2026/10), [comparison matrix](../docs/METHODOLOGY.md)]','',
         '## Serving: the concurrency cap changes the comparison','',
         '![Matched-cap A6000 serving throughput](../assets/figures/a6000_serving.svg)','',
         '| Model | TP2 mean output tok/s | DP2 mean output tok/s | DP2 difference | Evidence |',
         '|---|---:|---:|---:|---|']
    for r in arms:
        out.append(f"| {r['model']} | {r['arms']['TP2']['mean_tok_s']:,.1f} | {r['arms']['DP2']['mean_tok_s']:,.1f} | +{r['dp2_advantage_pct']:.2f}% | two repeats/arm; aggregate cap 512 |")
    out += ['', 'Both layouts use two GPUs and a total concurrency cap of 512. TP2 is taken from the matched-cap audit, not the original cap-256 rows. Llama8B is a historical observation, not newly validated 8B training or inference correctness. [A6000 records (2026/10), [comparison CSV](../studies/a6000/results/rerun/vllm_tp_concurrency_comparison.csv), [derived rows and run IDs](../results/showcase.json)]','',
            '**Engineering consequence:** compare complete serving layouts under the same request budget before choosing a deployment configuration. These data favor DP2 for these models and request settings; they do not establish a universal DP-versus-TP rule. [A6000 records (2026/10), [scope and claim ledger](../results/a6000_validation/evidence/70_final_review/CLAIM_LEDGER_v2.csv)]','',
            '## Precision: matching the policy changes the diagnosis','',
            '![A6000 first-boundary precision comparison](../assets/figures/a6000_precision.svg)','',
            'The matched-policy investigation extracts one accumulation-boundary gradient, without executing an optimizer update. Native BF16 and DeepSpeed BF16 have identical recorded hashes over 338 tensors; each differs from the FP32 control by about 0.2175 in relative L2. FP32 parameters with BF16 autocast differ by about 0.0362. [A6000 first-boundary evidence (2026/10), [per-tensor results](../results/a6000_validation/evidence/48_c20_fb_summary/out/fb_summary.json)]','',
            '**Engineering consequence:** a framework label is not a precision contract. The observed deviation can be reproduced under the native BF16 policy without DeepSpeed in this particular case. Weight, accumulation and residual precision inside that policy were not isolated as separate causes. NVIDIA describes precision-dependent rounding behavior; it does not certify the project\'s numerical thresholds. [NVIDIA floating point (accessed 2026/10), [§2.2](https://docs.nvidia.com/cuda/floating-point/index.html#operations-and-accuracy)]; [Project contrast (2026/10), [protocol](../docs/A6000.md)].','',
            '## Calibration: replay the comparison, not just its verdict','',
            '![Replayable calibration](../assets/figures/a6000_calibration.svg)','',
            '| Fixed-slice endpoint | Recomputed value | Frozen limit |', '|---|---:|---:|']
    for k,v in c['endpoints'].items():
        direction,lim=c['limits'][k]
        out.append(f"| {k} | {v:.9g} | {'≤' if direction=='upper' else '≥'} {lim:g} |")
    out += ['', 'The one-GPU FP32 DDP control uses accumulation 4; the two-GPU control uses accumulation 2. Both run five updates over the same 20 global inputs. The four shipped arrays contain the preregistered fixed slices (12,607,488 bytes), not full-model gradients. [Control C (2026/10), [retained data](../results/a6000_validation/evidence/70_final_review/calibration_slices/), [CPU verifier](../tools/verify_a6000.py)]','',
            '## Distributed execution: coverage before a success label','',
            '![Historical weak-scaling observations](../assets/figures/titan_scaling.svg)','',
            'The historical GTX 1080 Ti FP32 DDP arm uses GPT-2, sequence 512 and batch 4 per GPU. Recorded efficiency is 88.23%, 83.73% and 80.83% at two, four and six GPUs against the same-family one-GPU baseline. This is weak scaling: the global batch increases with world size. No uncertainty interval is invented for these selected historical point records. [Titan scaling (2026/10), [source table](../studies/titan-1080/results/scaling_efficiency.csv)]','',
            'Later Titan/1080 diagnostics cover 16 one-boundary training contexts and reconcile rank/partition coverage. They support conditional optimizer-update agreement, not independent gradient correctness or retroactive certification of the historical throughput. The 19-context census is 17 supported and 2 qualified, with Gate-0 and streaming retaining their qualifications. [Titan diagnostics (2026/10), [context ledger](../results/diagnostics/context_ledger.json)]','',
            '## What the infrastructure contributes','',
            '- A frozen input/control design and full-state identity checks rather than seed-only equivalence. [Implementation (2026/10), [A6000 control](../studies/a6000/src/mgbench/control_c.py)]',
            '- Base-optimizer boundary observations, expected/observed inventories and chunked oracle checks. [Implementation (2026/10), [Titan diagnostic](../studies/titan-1080/zerodiag/engine_probe.py), [oracle](../studies/titan-1080/zerodiag/update_oracle.py)]',
            '- Ownership-scoped process cleanup and explicit authorization/resource admission contracts. Source and regression tests are provided; modified publication bytes are not authorized for unattended GPU execution. [Implementation (2026/10), [A6000 scopes](../studies/a6000/src/mgbench/scopes.py), [Titan ownership](../studies/titan-1080/zerodiag/ownership.py)]',
            '- CPU-only replay, data manifests and tests that preserve the difference between a failure, a refusal and a missing observation. [Verification (2026/10), [test profiles](../TESTING.md), [provenance](../provenance/README.md)]','',
            '## Historical A100 reference','',
            'A100 observations remain available as a separate reference campaign. Its model lengths, optimizers and precision paths differ; it is not pooled into A6000 or Titan validation. [A100 records (2026/10), [current interpretation](../studies/a100/CURRENT_REPORT.md), [data](../studies/a100/all_results.csv)]','',
            '## References','',
            'See the [primary-source reading guide](../docs/REFERENCES.md) for NVIDIA documentation and the DDP, FSDP, ZeRO, mixed-precision, AdamW and PagedAttention papers. These references explain mechanisms; the project measurements supply the empirical results.','',
            '## Limitations','',
            'Synthetic inputs and short diagnostics do not establish convergence or model quality. Historical performance and diagnostic cohorts are distinct. Full tensor replay is available for the exported control slices and Titan inference arrays, not every full-model comparison. ResNet agreement, A6000 output/reference agreement and complete all-framework two-GPU validation remain unestablished; four-GPU/8B training and ZeRO-3 work remain deferred. Energy and latency headlines and causal bottleneck claims are not promoted. See [the complete scope statement](../docs/LIMITATIONS.md).','']
    return '\n'.join(out)


def scientific_status(root=ROOT):
    s=summary(root)
    lines=['# Scientific scope and evidence status','','Author: Mehran Keivanimehr · Updated 2026-10-22','',
           'Current scope is a qualified release of two reviewed cluster studies plus a historical A100 reference. A supported check means the stated check has evidence, not that a framework or an entire training recipe is universally correct.','',
           '## Current study ledgers','',
           '- Titan/1080: **17 supported / 2 qualified** over 19 contexts. The streaming observations do not establish request latency; Gate-0 retains its original criterion qualification. [Context ledger](../results/diagnostics/context_ledger.json).',
           '- A6000: replayable five-update fixed-slice control; matched-policy first-boundary contrasts; corrected historical serving and repeat analysis. The 28 claims remain separately classified. [Claim ledger](../results/a6000_validation/evidence/70_final_review/CLAIM_LEDGER_v2.csv).',
           '- A100: historical records only; not certified by either cluster closeout. [Interpretation](../studies/a100/CURRENT_REPORT.md).','',
           '## A6000 claim dispositions','', '| Disposition | Claims |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in s['a6000']['claim_counts'].items()]
    lines += ['', 'The [62-row evidence table](../results/a6000_validation/evidence/70_final_review/CLAIM_EVIDENCE_TABLE.csv) separates calls, gradient coverage, sampled update checks and reference agreement. Old Stage-1 rows are not retrospectively validated. The original finding list remains a [historical correction plan](../validation/findings.json), not the current completion census.','',
              '## Recompute','','```sh','python -I -S tools/verify_a6000.py','python -I -S tools/recompute_reference_verdict.py --a results/diagnostics/closing/reference/TITAN_RTX_1 --b results/diagnostics/closing/reference/GTX_1080Ti_1','python -B tools/release_checks.py --profile portable','```','',
              'The public tree is a metadata-anonymized derivative. Private archive hashes, unchanged tensor payloads and publication checksums have distinct roles; see [provenance](../provenance/README.md). GitHub Actions has not been run as part of this local release.','',
              '## Limitations','','Short diagnostics are not training-quality tests. Missing historical gradients cannot be reconstructed from a verdict or a seed. Full scope exclusions, data-availability boundaries and operational qualifications are in [LIMITATIONS.md](LIMITATIONS.md).','']
    return '\n'.join(lines)


def artifacts(root=ROOT):
    return {'results/showcase.json':json.dumps(compact_numbers(summary(root)),indent=2,sort_keys=True,allow_nan=False)+'\n',
            'docs/SCIENTIFIC_STATUS.md':scientific_status(root)}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--check',action='store_true'); a=ap.parse_args()
    stale=[]
    for name,content in artifacts().items():
        p=ROOT/name; data=content.encode()
        if a.check:
            if not p.exists() or p.read_bytes()!=data: stale.append(name)
        else:
            p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
    if stale: raise SystemExit('Stale showcase outputs: '+', '.join(stale))
    print('Showcase evidence and status verified.' if a.check else 'Built evidence summary and current status.')


if __name__=='__main__': main()
