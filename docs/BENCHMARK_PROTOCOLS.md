# Benchmark protocols

Each benchmark has an isolated environment, fixed task order and its own
runner. Run commands from the selected benchmark directory. Use a complete
repository checkout so the installer can install the sibling shared core.

| Benchmark | Python | Guide |
|---|---|---|
| LifelongAgentBench | 3.11 | [Install and run](../LifelongAgentBench/README.md) |
| BigCodeBench | 3.10 | [Install and run](../BigCodeBench/README.md) |
| ALFWorld | 3.9 | [Install and run](../ALFWorld/README.md) |

## LifelongAgentBench

Each task is executed once, evaluated externally, then updates memory for
subsequent tasks. There is no separate offline training phase. The processed
inputs, prompts and task limits are fixed by `data/manifest.json`.

| Setting | DBBench | OS Interaction | Knowledge Graph |
|---|---:|---:|---:|
| Tasks | 500 | 500 | 150 |
| Interaction rounds | 3 | 5 | 15 |
| Total output tokens per task | 128 | 512 | 128 |
| Retrieval bank capacity | 64 | 64 | 64 |
| Injected records | 1 | 1 | 1 |
| Complete injection block, characters | 900 | 800 | 4096 |
| Candidate shortlist | 12 | 12 | 12 |
| KRR / AGOP feedback interval | 50 / 200 | 25 / 100 | 25 / 100 |
| Residual influence eta | .15 | .05 | .10 |
| Minimum residual feedback | 4 | 4 | 24 |

DBBench stores valid SQL, observations and final answers. OS Interaction
stores reusable execution idioms from successful Bash actions. Knowledge
Graph stores valid graph tool actions and observations. A correct final
answer or state receives positive feedback even when the interaction ends
at a valid round or output limit.

All methods share the task stream, model, evaluator, action normalization and
per-task output allowance. The allowance includes execution and auxiliary
calls and is reduced by actual provider-reported usage after each request.
Auxiliary calls cannot borrow the next task's allowance. Complete memory
injection blocks include titles and separators within the character cap.

Retrieval banks are capped at 64 records. MemSkill allocates 48 experience
and 16 controller-skill slots, including its four initial instructions.
EvoSC allocates 32 success, 31 failure and one summary record. EvoSkill's cap
includes its pending candidate. Feedback and learning buffers are accounted
for separately from the retrieval bank. Baselines are adaptations to this
resource-constrained interface.

The primary metric is success rate. Knowledge Graph exact match defines
success; F1 is supplementary. OutputTok is total execution and auxiliary
completion tokens divided by the selected task count. Reports use all
planned valid repetitions with mean and sample standard deviation. Incomplete
groups remain diagnostics and are excluded from final comparisons.

## BigCodeBench and ALFWorld

Both adapters use the [TokenWorth algorithm](TOKENWORTH.md). Reference
solutions, unit tests and expert trajectories are excluded from memory.

| Setting | BigCodeBench | ALFWorld |
|---|---|---|
| Dataset | Full v0.1.4 (1,140), Hard subset (148) | 300 sampled train, 140 seen, 134 unseen |
| Order | Full minus Hard, then Hard | Train, seen, unseen |
| Memory updates | After every task | After every episode in all phases |
| Selection seed | Fixed dataset order | 42; stratified train sampling |
| Bank / injection | 50 / top-1 | 128 / top-1 |
| Snippet / complete context cap, characters | 4096 / 4096 | 4096 / 4200 |
| KRR / AGOP feedback interval | 75 / 225 | 75 / 225 |
| Residual influence eta / minimum feedback | .10 / 4 | .10 / 4 |
| External reward | Official evaluator pass | TextWorld `won` flag |
| Experience | Verified generated program; long programs become labeled non-executable outlines | Successful action and observation history |
| Execution budget | Local/no-thinking: 1280 output tokens; thinking: 4096 | 50 steps; 128 output tokens per action; one invalid-action repair |

BigCodeBench memory methods share a 4096-character injection cap. NoMemory
uses the same runner, sanitizer and evaluator. Generated-program failures
and timeouts are evaluated outcomes; worker infrastructure failures abort.

ALFWorld methods share the prompt and action controller. Memory-dependent
loop removal is disabled. The engine's game order is checked against the
manifest before each model call. To restrict memory updates to training,
set `memory.update_phases: [train]`; this defines a separate protocol.

Auxiliary baseline calls are recorded separately from solver calls. These
budgets apply per call or episode as listed above; LifelongAgentBench uses
a pooled per-task allowance. Report each adapter's configuration with its
results and use its own reporting format.

BigCodeBench writes `results_summary.json`. Its top-level `pass_at_1` covers
all processed Full tasks; `phases.simple.pass_at_1` covers Full minus Hard,
and `phases.hard.pass_at_1` covers the Hard subset. For a complete run their
denominators are 1140, 992 and 148 respectively. Report the selected suite
(`local`, `api_no_thinking` or `api_thinking`) with these metrics.

ALFWorld writes `summary.json` with separate `phases.train`,
`phases.valid_seen` and `phases.valid_unseen` statistics. Each phase records
its episode count, successes, success rate and model usage. If reporting a
combined success rate, sum successes and divide by the summed episode count;
state the included phases explicitly. A full online run contains 574 episodes.

## Run identity and recovery

Runs record the resolved model name and endpoint, configuration, ordered task
stream, data checksums, source and dependency identity, and evaluator identity.
Docker image IDs are included where applicable. These must match on resume.
For reproducibility, retain the exact provider model revision and the
Knowledge Graph database snapshot in addition to endpoint URLs.

Every generation attempt has a durable start record and a response or error
entry. SDK retries are disabled. Only explicitly rejected rate-limit requests
have bounded retries. Missing or invalid usage, empty or over-budget replies,
transport errors, service faults and evaluator errors invalidate the run.
Costs with unavailable usage remain unknown in reports.

An atomic checkpoint commits evaluated records and complete learner state
together. Completed task or episode boundaries resume without replay. An
uncommitted active task with a started request requires a new independent
run directory; keep its cost ledger. Checkpoints must match the implementation
and run identity. Use one process per method and output directory.

Partial summaries include planned and completed counts and
`is_complete=false`. Reports distinguish models, task orders, resource limits
and method settings. Full-stream results require complete runs with the
specified model, data and protocol.
