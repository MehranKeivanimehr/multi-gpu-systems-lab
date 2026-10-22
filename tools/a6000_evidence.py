"""Read retained A6000 evidence without importing any training framework."""
from __future__ import annotations
import array
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXED_NAMES = ['model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.down_proj.weight',
               'model.layers.27.mlp.up_proj.weight', 'model.norm.weight']
LIMITS = {'G1_rel_l2': ('upper', 1e-4), 'G1_cosine': ('lower', .99999),
          'L_k0_rel': ('upper', 1e-6), 'L_max_k0_4_rel': ('upper', 1e-5),
          'U_rel_l2': ('upper', .05), 'U_cosine': ('lower', .999)}


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def stable_mean(values):
    return float(sum((Fraction(x) for x in values), Fraction()) / len(values))


def replay(root=ROOT):
    base = root / 'results/a6000_validation/evidence/70_final_review/calibration_slices'
    man = read(base / 'SLICES_MANIFEST.json')
    require(man['names'] == FIXED_NAMES, 'Unexpected fixed-tensor inventory')
    vectors, records = {}, {}
    expected = {f'{run}_{kind}_slices.f32' for run in ('R5','C5') for kind in ('grad_b1','dtheta_final')}
    expected |= {'R5_control_record.json','C5_control_record.json'}
    require(set(man['files']) == expected, 'Missing/unexpected calibration payload')
    for name, info in man['files'].items():
        require(Path(name).name == name, 'Manifest path is not a basename')
        raw = (base / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == info['sha256'], f'Checksum mismatch: {name}')
        if name.endswith('.json'):
            records[name[:2]] = json.loads(raw)
            continue
        require(len(raw) == info['bytes'] == 787968 * 4, f'Wrong byte count: {name}')
        require(info['elements'] == 787968, 'Wrong element count')
        require([p['name'] for p in info['layout']] == FIXED_NAMES, 'Wrong slice order')
        require([p['elements'] for p in info['layout']] == [262144]*3 + [1536], 'Incomplete slice coverage')
        a = array.array('f'); require(a.itemsize == 4, 'Four-byte float required'); a.frombytes(raw)
        if sys.byteorder != 'little': a.byteswap()
        require(all(math.isfinite(x) for x in a), f'Non-finite payload: {name}')
        vectors[name] = a
    endpoints = {}
    for kind, label in (('grad_b1','G1'),('dtheta_final','U')):
        r, o = vectors[f'R5_{kind}_slices.f32'], vectors[f'C5_{kind}_slices.f32']
        require(len(r) == len(o) == 787968, 'Mismatched vector lengths')
        rr, oo = math.fsum(x*x for x in r), math.fsum(x*x for x in o)
        require(rr > 0 and oo > 0, 'Non-discriminating zero norm')
        endpoints[label+'_rel_l2'] = math.sqrt(math.fsum((x-y)**2 for x,y in zip(o,r)) / rr)
        endpoints[label+'_cosine'] = math.fsum(x*y for x,y in zip(o,r)) / math.sqrt(rr*oo)
    losses, maps = {}, {}
    for run, record in records.items():
        ss = record['global_sample_map']; require(len(ss) == 20, 'Twenty input rows required')
        ls, identity = [], {}
        for step in range(5):
            xs = [s for s in ss if s['step'] == step]
            require(len(xs) == 4 and sorted(s['global_row'] for s in xs) == [0,1,2,3], 'Missing/duplicate global input')
            require(all(math.isfinite(s['raw_loss']) for s in xs), 'Non-finite loss')
            ls.append(math.fsum(s['raw_loss'] for s in xs) / 4)
            identity.update({(step,s['global_row']):s['digest'] for s in xs})
        losses[run], maps[run] = ls, identity
    require(maps['R5'] == maps['C5'], 'R5/C5 inputs differ')
    require(all(x != 0 for x in losses['R5']), 'Zero reference loss')
    rel = [abs(c-r)/abs(r) for c,r in zip(losses['C5'],losses['R5'])]
    endpoints.update(L_k0_rel=rel[0], L_max_k0_4_rel=max(rel))
    verdicts = {k: ('pass' if (v <= LIMITS[k][1] if LIMITS[k][0]=='upper' else v >= LIMITS[k][1]) else 'outside_limit')
                for k,v in endpoints.items()}
    return {'kind':'fixed_slice_control_C', 'slice_elements':787968, 'payload_bytes':12607488,
            'endpoints':endpoints,'limits':LIMITS,'verdicts':verdicts,
            'all_pass':all(v=='pass' for v in verdicts.values()),
            'scope':'Five-update FP32 DDP, Qwen2.5-1.5B, one versus two A6000s; fixed slices, not full-model equality.'}


def inference(root=ROOT):
    rows = csv_rows(root / 'studies/a6000/results/rerun/vllm_tp_concurrency_comparison.csv')
    result = []
    for model in ('qwen2.5-1.5b','llama3.1-8b'):
        arms = {}
        for name,tp,dp,source in [('TP2',2,1,'rerun'),('DP2',1,2,'original')]:
            selected = [r for r in rows if r['model']==model and int(r['tp'])==tp and int(r['dp'])==dp
                        and int(r['total_concurrency_cap'])==512 and r['source']==source]
            require(len(selected)==2 and len({r['run'].rsplit('__r',1)[1] for r in selected})==2,
                    'Exactly two distinct repeats per matched arm required')
            rates = []
            for r in selected:
                cohort = 'vllm_scaling' if source=='original' else 'vllm_tp_concurrency_rerun'
                raw = read(root / 'studies/a6000/results/raw' / cohort / r['run'] / 'result.json')
                tokens, window = float(raw['output_tokens']), float(raw['window_s'])
                require(math.isfinite(tokens) and tokens>0 and math.isfinite(window) and window>0, 'Invalid generation count/window')
                rate = tokens / window
                require(math.isfinite(rate) and rate>0 and math.isclose(rate,float(r['output_tok_s']),rel_tol=1e-10), 'Raw rate differs')
                rates.append(rate)
            arms[name] = {'runs':[r['run'] for r in selected], 'values_tok_s':rates,
                          'mean_tok_s':stable_mean(rates),'n':2,'source':source}
        result.append({'model':model,'aggregate_cap':512,'gpus':2,'arms':arms,
                       'dp2_advantage_pct':100*(arms['DP2']['mean_tok_s']/arms['TP2']['mean_tok_s']-1),
                       'scope':'qualified historical throughput; not output/reference validation'})
    return result


def precision(root=ROOT):
    s = read(root / 'results/a6000_validation/evidence/48_c20_fb_summary/out/fb_summary.json')
    rows = []
    for case,label in [('G1_ddp_autocast','FP32 parameters + BF16 autocast'),
                       ('G3_ddp_bf16_params','Native BF16 policy'),('G4_ds_bf16','DeepSpeed BF16 policy'),
                       ('X1_fp32_order_permuted','FP32, permuted accumulation')]:
        g = s['cases'][case]['vs_sealed_r0']['global']
        rows.append({'case':case,'label':label,'rel_l2':g['rel_l2'],'cosine':g['cosine']})
    return {'rows':rows,'native_vs_deepspeed':s['pairwise']['G4_ds_bf16_vs_G3_ddp_bf16_params']['global'],
            'scope':'One accumulation boundary, one A6000, matched TF32-off/math-SDPA policy; no optimizer update.'}
