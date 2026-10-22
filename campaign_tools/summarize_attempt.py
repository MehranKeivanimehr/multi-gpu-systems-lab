"""Compact one-line-per-rank summary of a diagnostic attempt. Read-only."""
import json, sys, pathlib

def main(paths):
    for p in paths:
        p = pathlib.Path(p)
        if not p.exists():
            print(f"{p.name}: NO RESULT JSON"); continue
        r = json.load(open(p))
        t, c, perf = r.get("telemetry", {}), r.get("config", {}), r.get("perf", {})
        ig = t.get("integration", {}) or {}
        gaps = [v.get("max_support_gap_s") for v in ig.values() if isinstance(v, dict)]
        bad = [k for k, v in ig.items() if isinstance(v, dict) and v.get("status") != "ok"]
        print(f"== {p.stem}: status={r.get('status')} mode={c.get('mode')} "
              f"prec={c.get('precision')} ws={c.get('world_size')} bs={c.get('micro_batch_per_gpu')}")
        print(f"   perf: {perf.get('tokens_per_s')} tok/s  ({perf.get('tokens_per_s_per_gpu')}/gpu)  "
              f"peak {r.get('memory',{}).get('peak_alloc_mib_rank0')} MiB  wall {perf.get('wall_time_s')}s")
        print(f"   tele: interval={t.get('sample_interval_s')} frozen={t.get('sample_interval_matches_frozen')} "
              f"cutoff={t.get('max_permitted_gap_s')} maxgap={max(gaps) if gaps else None} "
              f"energy={t.get('energy_total_j')}J errs={len(t.get('sampling_errors',[]))} bad_integration={bad}")
        print(f"   bind: preserved={c.get('device_binding_preserved_across_memory_cap')} "
              f"identity_certified={c.get('device_identity_established')}")
        for idx, a in enumerate(r.get("optimizer_update_audit_per_rank", [])):
            rid = a.get("rank", a.get("local_rank", idx))  # list order is rank order
            print(f"   rank{rid}: att={a.get('attempted_steps')} logical={a.get('native_successful_logical_updates')} "
                  f"base={a.get('base_optimizer_calls')} skipped={a.get('native_skipped_steps')} "
                  f"cov={a.get('gradient_coverage_complete')} fin={a.get('gradient_observations_finite')} "
                  f"nonfin={a.get('nonfinite_gradient_steps')} miss={a.get('missing_gradient_steps')} "
                  f"unobs={a.get('unobserved_attempt_steps')} path={a.get('observer_path')} "
                  f"verdict={a.get('status_verdict')} cert={a.get('numerical_diagnostic_certified')} "
                  f"why={a.get('certification_reasons')}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
