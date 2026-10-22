# Accounting reconciliation

Two separate batches with separate ceilings. They are never summed.

| quantity | original batch | follow-up batch |
|---|---|---|
| case attempts | 22 | 14 |
| case summaries | 21 | 14 |
| rank process records | 47 | 36 |
| ownership ledgers | 22 | 14 |
| ceiling | 36 | 26 |

## Why the denominators differ

- a case attempt is one launcher invocation; it creates one attempt directory
- a case summary exists only if the launcher reached its post-wait verification; an attempt terminated by the operator has none, which is why summaries can be fewer than attempts
- a rank process record is one per rank per attempt, so a six-rank case contributes six to that denominator and one to the attempt denominator
- an ownership ledger is written once per attempt that reached the spawn stage, so it tracks attempts, not ranks
- the two batches have SEPARATE ceilings and are never summed

## Closing window

| quantity | value |
|---|---|
| authorization | `CLOSED-20261020T125221Z` (nonrenewable) |
| authorized | 7200 s |
| used | 266.1 s |
| unused | 6933.9 s |
| attempts used | 4 of 6 |
| per-case ceiling | 2 |
| safety stops | 0 |
| cleanup verified after every attempt | True |
| devices contacted | 2 (physical indices [0, 1]) |
| concurrent GPUs | 1 |

Elapsed wall time for each case is in `closing/closed_results.json`. The
streaming case used both of its permitted attempts, which is why the repair
found afterwards was not executed on a GPU.

## Streaming-only window

| quantity | value |
|---|---|
| authorization | `FINAL-20261020T141800Z` (nonrenewable, streaming only) |
| authorized / used / unused | 5400 s / 178.1 s / 5221.9 s |
| attempts used | 1 of 3 |
| in-window CPU repair used | 0 of 30 min |
| device | physical index 0 only, one job at a time |
| safety stops | 0 |
| cleanup verified | True |
| monitored client-group peak RSS | 0.418 GiB of 32 GiB |
| aggregate RSS including the server | not measured (CORR_STREAM_003) |
| server process group | confirmed absent after the session, including 2 descendant(s) |
