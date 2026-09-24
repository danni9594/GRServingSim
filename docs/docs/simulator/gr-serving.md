---
title: Generative recommendation serving
sidebar_position: 8
---

# Generative recommendation serving

`python -m serving gr` simulates HSTU and OpenOneRec using **ASTRA-Sim execution
cycles**. Its request stages become operator DAGs in Chakra format, pass through
`serving/core/graph_generator.py`, and execute in the same C++ backend used by
LLMServingSim. The frontend advances on ASTRA completion reports through the
existing controller protocol. It never substitutes a whole-request roofline
latency for the graph's completion time.

The serving semantics follow the supplied manuscript *Enabling High-Bandwidth
Flash for Generative Recommendation Serving with Write-Aware KV Cache Policy*,
Sections II–IV. OpenOneRec uses the requested three semantic-ID decoding steps.
The original LLM CLI, token-prefix cache and continuous batching scheduler remain
available; GR uses a separate request schema and stage scheduler.

## Build and run

From the repository root, with a C++17 compiler, CMake and Protobuf development
libraries plus `protoc` installed:

```bash
python3 -m pip install rich protobuf
bash scripts/compile-gr.sh

python3 -m serving gr \
  --config configs/gr/hstu_hbf.json \
  --dataset workloads/gr_example.jsonl \
  --cache-k 2 \
  --save-trace-text \
  --output outputs/hstu_astra.csv

python3 -m serving gr \
  --config configs/gr/openonerec_hbf.json \
  --dataset workloads/gr_example.jsonl \
  --cache-k 2 \
  --output outputs/openonerec_astra.csv
```

The build script initializes the necessary pinned submodules, applies the
integration patches in `patches/`, generates the pinned Chakra protobuf bindings,
and builds the analytical-network ASTRA executable. Here “analytical network” is
ASTRA's network backend, not the optional Python-only service-time approximation.
`GR_BUILD_JOBS` controls compilation parallelism. Standard `CMAKE_PREFIX_PATH`
and `PATH` can locate a user-installed Protobuf/CMake toolchain. For a static
Protobuf source installation, set `PROTOBUF_FROM_SOURCE=True` so its CMake target
also links transitive dependencies.

`--astra-binary` selects another compatible executable. At startup the adapter
runs a small real memory graph and checks its cycles against the configured
read/write bandwidths. This detects stale binaries that silently ignore the new
memory fields. Initialization and probe cycles are excluded from request time.
An absent or incompatible backend raises an error; there is no silent fallback.

To sweep independent cold-cache policies for both models:

```bash
bash serving/run-gr.sh
python3 -m serving gr --help
```

Other hardware presets are in `configs/gr/`. For a lightweight equation-only
comparison, explicitly choose `--backend analytical`; only that mode uses
`--latency-mode`. The default executable path and other CLI defaults are defined
in `serving/gr/astra.py` and `serving/gr/cli.py`.

## Request schema and cache identity

A request has a long historical sequence and a short increment of new user
interactions. The workload is JSONL, for example:

```json
{"request_id":"a0","user_id":"alice","arrival_time_ns":0,"history_tokens":4096,"incremental_tokens":4,"history_version":"v0","next_history_version":"v1","model_version":"checkpoint-A","candidate_count":500}
{"request_id":"a1","user_id":"alice","arrival_time_ns":1000000000,"history_tokens":4100,"incremental_tokens":3,"history_version":"v1","next_history_version":"v2","model_version":"checkpoint-A","candidate_count":500}
```

The cache lookup key is `(model_version, user_id)`. A hit additionally requires
an exact `history_version`, exact historical token count, and a completed cache
entry. User identity alone is insufficient after a history rewrite or model
change. A hit reuses **the complete historical KV sequence**; it does not search
for a shared token prefix or accept partial-history matches.

At completion the retained entry becomes the `next_history_version` snapshot
covering history plus increment. With an empty increment, the two versions must
match. Candidate KV and semantic-prefix KV are private workspace objects, never
part of the persisted interaction history.

Generate a deterministic, version-consistent workload using:

```bash
python3 -m serving.gr.generate --help
```

## Scheduling and model semantics

The GR scheduler represents one accelerator, with one request in flight, FCFS
queueing and candidate parallelism within the request. Historical and incremental
prefill can be chunked according to `astra.prefill_chunk_tokens`. A stage is
submitted only after its predecessor completes in ASTRA.

| Cache path | Historical prefill | Incremental prefill | Persistent write |
| --- | --- | --- | --- |
| Hit | Reuse complete historical KV | Process new interactions against history | Append increment to a retained matching copy |
| Miss, admitted | Compute full history | Process new interactions | Write history plus increment |
| Miss, rejected | Compute full history | Process new interactions | None |

A retained copy in another tier that did not itself match needs a full write,
even if a faster source supplied a global hit. Admissions in the memory tiers
are independent; retained copies receive write-through updates.

| Model | Decode work | Dependencies |
| --- | --- | --- |
| HSTU | One scoring stage containing all `candidate_count` queries | All candidates attend to the user history; there is no candidate-to-candidate causal chain |
| OpenOneRec | Three semantic decoding stages representing an item | Stage two follows stage one, and stage three follows stage two |

For OpenOneRec, the default beam approximation starts with one query, then uses
up to `candidate_count` beams subject to the previous codebook's number of
scored logits. Thus generating multiple items still uses three sequential
stages, with wider parallel work. `decode_widths` can specify the query width
for each step, for example `[500, 500, 500]` when teacher-forcing a supplied
candidate set. Codebook sizes determine semantic-head work. This is a timing
model, not numerical beam search: it does not produce actual scores or IDs.

## Operator DAG and memory timing

`serving/gr/scheduler.py` emits phases; `serving/gr/trace.py` emits the per-layer
DAG. The current model splits each layer into:

1. Weight load and projection compute.
2. History reads, working-KV append, and attention compute with explicit input dependencies.
3. Output/FFN compute, followed by the next layer.
4. A final candidate-score or semantic-logit head for decoding.

The projection fraction is configurable. Linear and attention coefficients
apply to the whole model and are divided across `model.num_layers`. The history
attention count is causal; a chunk of `q` tokens following `h` tokens contributes
`q*h + q*(q+1)/2` pairs. HSTU scoring contributes `c*(h+i)` pairs, where `c` is the
candidate count and `i` the increment. OpenOneRec also attends to each beam's
already generated semantic prefix.

Compute node durations are `ceil(FLOPs / effective_FLOPs_per_ns)`. Memory sizes,
locations and graph dependencies are separate. ASTRA schedules their completion,
compute-resource serialization, per-tier queuing and cross-tier overlap. This
fork uses nanoseconds in Chakra's field named `duration_micros`, matching its
existing LLM converter; the GR writer follows that convention.

After prefill, cache writeback branches can overlap the first decode stage.
That stage's completion barrier waits for both scoring and writeback, and later
semantic stages start afterward. This is an explicit conservative stage-level
schedule, not an assumption of perfect overlap across all decode steps.

Memory placement is:

- **HBM:** local memory, shared by active workspace and resident user caches.
- **CPU:** remote memory. A CPU hit recalls the historical KV once into private
  HBM workspace, even if persistent HBM admission is denied. Subsequent stages
  use that staged copy.
- **HBF:** storage memory. A hit streams history at each incremental-prefill
  chunk and each decode stage; new interactions and semantic prefixes use HBM.
- **CXL:** an optional direct-streaming endpoint when explicitly configured.

Weights also generate memory traffic from `astra.weight_location`. HBF presets
place their weights in HBF; HBM presets use HBM. There is no implicit weight cache.
All memory nodes using one endpoint share its FIFO, including reads and writes;
separate endpoints can overlap. This is a shared read/write port model, not a
full-duplex flash controller or flash-channel model. CPU link bandwidth caps the
transfer bandwidth using a streaming bottleneck approximation.

The C++ integration adds `is_write` to memory requests, separate read/write
bandwidth and startup-latency fields, and storage endpoint initialization.
Memory-node duration is `ceil(latency_ns + bytes / bandwidth_bytes_per_ns)`.
Configurations without directional fields retain the legacy memory timing path.
The changes are stored as re-applicable patches against pinned submodules,
rather than unpublished submodule commit references.

## LRU-K admission and eviction

`serving/gr/cache.py` owns whole-user, byte-capacity caches:

- Record one reference per arriving request, including rejected misses. The last
  K references survive eviction as ghost metadata.
- Fewer than K references means no admission, including during cold start.
- Otherwise rank an object by its Kth most recent reference: older means colder.
- If capacity is needed, consider entire resident users from coldest upward. For
  K greater than one, admit only if the incoming user's priority is newer than
  every victim required to fit it. A failed admission evicts nobody.
- K equal to one admits every feasible miss and uses LRU victim order. An object
  larger than a tier's capacity cannot be admitted.

Allocation rounds sequence lengths up by `block_tokens`, but the last partial
block is retained and the object is evicted as a whole. Flash writes count the
newly written token bytes rather than allocation padding; physical flash-page
rounding is not modeled. `write_amplification` scales physical persistent writes.

An admission plan is created at service start. Victims and the new version are
committed atomically only after all request stages finish. The single in-flight
scheduler has no concurrent cache writers. An arrival during service sees only
the previously committed snapshot; it can become a service-time hit after it
waits in the queue. Output reports both arrival-time and service-time hit rates.
Completion wins ties with an arrival at the same simulated timestamp.

Cache capacity is explicitly separate from the active KV workspace. Presets
reserve workspace capacity and subtract weights from the relevant memory pool.
The scheduler rejects requests whose full active history plus maximum candidate
or semantic KV scratch cannot fit the reserved workspace. Cache writes represent
a copy from workspace into the retained pool. The model does not implement
workspace spilling or partially resident histories.

## Hit rate, latency and output

For measured requests, let `p_h`, `p_w`, and `p_n` be the hit, miss-write and
miss-no-write fractions. The summary computes:

```text
E[T] = p_h * E[T | hit]
     + p_w * E[T | miss_write]
     + p_n * E[T | miss_no_write]
service_capacity_qps = 1e9 / E[T_ns]
```

All conditional latencies above are measured from ASTRA completion cycles.
They depend on sequence length, candidate count, decode widths, memory placement
and cache admission. A higher hit rate does not guarantee lower latency if a
cache read is slower than recomputation; increasing K can reduce writes while
increasing history recomputation. `achieved_qps` additionally includes arrivals
and idle time. Per-request latency includes FCFS queueing.

Each run writes request CSV, summary JSON, and stage JSONL beside the selected
output path. Useful fields include:

- `service_ns`, `latency_ns`, `queue_ns`, and the three cache paths.
- `historical_flops`, `incremental_flops`, `decode_flops`, and saved history work.
- Persistent logical/physical writes by tier, admission reasons and evicted users.
- `astra_stages`: actual start, finish and duration of each backend iteration.
- Memory-node bytes by location, including working KV, copies and weights.

In ASTRA output, `compute_ns` is the sum of compute-node durations; `read_ns` and
`write_ns` sum memory-node busy times without queue wait. They are diagnostic
resource totals, not a formula for service time. `reads_by_tier` includes all
loads on that tier, including weights; persistent `writes_by_tier` excludes
workspace stores. The stage file separately reports all store traffic.

Flash lifetime uses physical persistent bytes and the configured or trace-derived
arrival rate. It does not assume workload rate equals saturated service capacity.
The endurance calculation excludes weight provisioning and background flash work.

`--save-trace-text` retains `.et` graphs, readable operator JSON, generated
hardware configs and `astra.log`; the summary gives their directory and the
executable SHA-256. Without retention flags, successful runs remove intermediate
inputs. The CLI bounds backend handshakes and reports crashes/timeouts.

## Calibration and scope

The manuscript does not supply complete operator profiles, layer dimensions,
codebooks or kernel launch costs. The checked-in model sizes, FLOP/KV coefficients,
weight footprints and projection split are **illustrative**, with provenance in
each configuration's `calibration_note`. Replace them using the intended HSTU or
OpenOneRec checkpoint before making quantitative claims. Executing real ASTRA
cycles does not make uncalibrated operator costs hardware measurements.

This implementation covers TP=1, PP=1 and one request in flight. It does not
model inter-device collectives, cross-request batching, numerical recommendation
quality, activation allocations/traffic, flash garbage collection, or kernel
launch overhead. Workspace capacity checking covers KV only. These are explicit
scope limits, rather than parameters inferred from the paper's performance plots.

## Validation

Build first, then run:

```bash
python3 -m unittest discover -s serving/gr/tests -t . -v
./serving/validate.sh single local_offloading prefix_cpu_pool cxl
```

The GR tests include real backend read/write timing, independent-tier overlap,
same-tier contention, cache completion visibility, LRU-K admission, CPU recall,
chunked prefill, workspace overflow, and dependent semantic decoding. ASTRA tests
skip when its executable has not been built; cache and equation tests can still
run. The legacy validation command compares recorded clocks for affected LLM
memory paths without updating their baselines.
