import json, sys
plan=json.load(open(sys.argv[1])); rows=plan["configurations"]; BLOCKS=6
WALL={"single":60,"ddp":60,"fsdp":60,"zero1":70,"zero2":70,"cnn_forward_only":115,
      "hf_generate":60,"vllm_offline_tp1":90,"vllm_offline_tp2":110,
      "nccl_allreduce_fixed_payload":60,"vllm_loopback_streaming":200}
GATE=15
pb_wall=sum(WALL[r["recipe"]]+GATE for r in rows)
pb_dev=sum((WALL[r["recipe"]]+GATE)*r["world_size"] for r in rows)
energy=sum(1 for r in rows if r["recipe"]=="cnn_forward_only")
train=sum(1 for r in rows if r["recipe"] in ("single","ddp","fsdp","zero1","zero2"))
stream=sum(1 for r in rows if r["recipe"]=="vllm_loopback_streaming")
disk_mb=(energy*260 + train*30 + stream*1200 + len(rows)*25)*BLOCKS/1024.0
est={"basis":"per-case wall times extrapolated from MEASURED pilot timings; NOT measured upper bounds",
 "blocks":BLOCKS,"configurations":len(rows),"planned_job_launches":len(rows)*BLOCKS,
 "unit_note":"job launches, not scalar observations; sweeps contain multiple arms",
 "per_block_job_wall_min":round(pb_wall/60,1),
 "total_job_wall_h":round(pb_wall*BLOCKS/3600,2),
 "total_device_hours":round(pb_dev*BLOCKS/3600,2),
 "allocation_elapsed_h_estimate":round(pb_wall*BLOCKS/3600*1.25,2),
 "allocation_elapsed_basis":"job wall time + ~25% for gating, settle, reservation and inter-case overhead",
 "peak_concurrent_gpus":max(r["world_size"] for r in rows),
 "one_job_at_a_time":True,
 "host_rss_ceiling_gib":32,"host_rss_basis":"planning ceiling per instruction; 6 torchrun ranks x ~2 GiB plus a vLLM parent; actual RSS never measured in the pilot",
 "new_project_data_ceiling_gib":32,"disk_mb_estimate":round(disk_mb),
 "disk_basis":"measured artifact sizes: 60s energy record 260 KB, training record 30 KB, streaming JSONL ~1.2 MB at 100 requests",
 "cpu_analysis_h":[4,7],"cpu_analysis_basis":"independent reintegration of 24 energy windows plus statistics, audit and report generation",
 "uncertainty":{"job_wall":"+/-50%: extrapolated from 5-step diagnostics to 30-step runs; startup dominated the measured cases",
   "device_hours":"+/-50%, same cause","host_rss":"planning ceiling, not a measurement",
   "disk":"+/-30%","not_measured_upper_bounds":"these are estimates, explicitly not measured upper bounds",
   "twelve_hours":"a LIMIT, not a completion guarantee"},
 "supersedes":"39 configurations / 195 runs estimate"}
json.dump(est,open(sys.argv[2],"w"),indent=2)
for k in ("configurations","planned_job_launches","per_block_job_wall_min","total_job_wall_h",
          "total_device_hours","allocation_elapsed_h_estimate","peak_concurrent_gpus","disk_mb_estimate"):
    print(f"  {k:<32} {est[k]}")
print(f"  {'cpu_analysis_h':<32} {est['cpu_analysis_h']}")
