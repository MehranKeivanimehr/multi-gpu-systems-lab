# Correction: resource containment (appended, prior wording preserved)

A previous report said serialising runners makes the per-process and job-wide
limits "coincide". **That does not follow**, because a runner spawns children and
each child receives its own copy of the rlimit. The corrected statement:

## Enforced PER PROCESS (verified)
* `ulimit -v 16777216` (16 GiB address space) and `ulimit -t 1800` (CPU seconds),
  set by `cpu_run.sh` before `exec`, and **inherited by every descendant**.
  Verified this stage: a child allocating 64 MiB succeeded, while a 20 GiB
  attempt in the same subtree raised `MemoryError`
  (`out/receipts/rlimit_child_test.*`).
* `nice -n 19`, inherited.
* `timeout --signal=TERM --kill-after=15s`, bounding wall time of the subtree.
* `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4` — thread caps, which are **not** a
  CPU quota.

## Enforced ACROSS THE PROCESS GROUP / CGROUP
**Nothing.** No aggregate cap is in force. Verified this stage:
`systemd-run --user --scope -p MemoryMax=64M` permitted a 256 MiB allocation and
exited 0 (`out/receipts/cgroup_test.out`), so no delegated cgroup memory
controller is available to this user. Consequently **N concurrent processes could
each approach 16 GiB**; the 16 GiB figure is a per-process ceiling, never a
job-wide one. Strict serialisation reduces concurrency and therefore the
expected aggregate, but it is a scheduling habit, not an enforced cap.

## Observed aggregate usage
* Peak RSS summed over **all processes of this uid** during a deliberate
  three-child 80 MiB test: **10.20 GiB**
  (`out/receipts/aggregate_rss_observed.stdout`). That total includes
  pre-existing unrelated editor and server processes, so it is an upper bound on
  this stage's own footprint, not a measurement of it.
* Earlier campaign sampling recorded a 14.71 GiB peak across the user's
  processes; also an upper bound, also not an enforced cap.

## Not measured and not guaranteed
* Any job-wide memory ceiling.
* "Four CPU-equivalents" as a scheduler guarantee: thread caps plus `nice` shape
  demand, they do not reserve or cap CPU time aggregate across processes.
* Peak aggregate RSS attributable to this stage alone.

**No limit was observed to be exceeded, and no claim is made that one was.** The
remaining acceptance limitation is stated rather than papered over: multiprocess
acceptance in this stage ran under per-process rlimits with observed aggregate
usage, **not** under an enforced job-wide cap.

## Scope deviation preserved
The previous stage's final command invoked
`nvidia-smi --query-compute-apps=pid --format=csv,noheader`, which that stage's
scope prohibited. Contemporaneous evidence: it was the last line of that stage's
closing command and returned a count of `0`; the invocation was read-only and no
CUDA context, allocation, reservation or workload followed. Its individual
receipt was not captured separately, and that missing detail is disclosed rather
than reconstructed. **It has not been rerun to manufacture a receipt**, and the
earlier stage must not be described as zero-query.

This stage installed task-scoped guards instead of relying on discipline:
`guards/nvidia-smi` and four sibling tools refuse with exit 97, and a
`sitecustomize.py` on `PYTHONPATH` replaces `torch.cuda.init`,
`torch.cuda.set_device` and `Tensor.cuda` with refusals when
`TITAN_CPU_ONLY_GUARD=1`. Both verified active
(`out/receipts/guard_selftest.stderr`, `out/receipts/guard_torch.stdout`). No
global shell configuration was changed.
