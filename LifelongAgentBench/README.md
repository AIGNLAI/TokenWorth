# TokenWorth on LifelongAgentBench

This adapter contains DBBench, OS Interaction and Knowledge Graph. Its code,
configurations, fixed data, prompts, installation and tests live together in
this directory. Use commands below from `LifelongAgentBench/`.

See the [TokenWorth algorithm](../docs/TOKENWORTH.md) for memory retrieval and
learning details. The [repository overview](../README.md) links BigCodeBench and
ALFWorld adapters, each with its own environment and protocol.

## 1. Install

Requirements:

- Linux and Python 3.11
- Docker Engine for DBBench and OS Interaction
- An OpenAI-compatible model endpoint, provided by local vLLM or a cloud API
- A Freebase-compatible SPARQL endpoint for Knowledge Graph experiments

Clone the repository and run the installer:

```bash
git clone https://github.com/AIGNLAI/TokenWorth.git
cd TokenWorth/LifelongAgentBench
bash scripts/install.sh
source .venv/bin/activate
```

The installer creates `.venv`, installs the benchmark using the verified Linux/Python 3.11 versions in `requirements-lock.txt`, checks the CLI, and
verifies the fixed data checksums. To use another Python 3.11 executable or
environment path:

```bash
PYTHON_BIN=python3.11 \
  bash scripts/install.sh --venv venv
```

Use `--constraints constraints.txt` for another explicitly managed dependency set; its actual versions will be locked into the runtime manifest.

Optional dependency groups are installed only when needed:

```bash
bash scripts/install.sh --extras dev
```

TokenWorth uses the built-in lightweight hashing encoder. vLLM is installed
separately in section 3 so its CUDA dependencies do not affect the benchmark
environment.

## 2. Prepare Data and Task Environments

The exact processed evaluation files used by the release configs are committed
under `data/` and `system_prompts/`. A normal `git clone` downloads them;
no additional benchmark split download is required. Verify that the clone is
complete and unchanged:

```bash
source .venv/bin/activate
python scripts/verify_data.py
```

`data/manifest.json` fixes the protocol version, sample counts, round limits,
and SHA-256 checksums.

Download the DBBench runtime image and build the OS Interaction image:

```bash
docker pull mysql:8.0
docker build \
  --file scripts/dockerfile/os_interaction/default \
  --tag local-os/default \
  scripts/dockerfile/os_interaction
```

Knowledge Graph additionally requires the Freebase Virtuoso database. Clone
[dki-lab/Freebase-Setup](https://github.com/dki-lab/Freebase-Setup) outside this
repository and follow its download and deployment instructions:

```bash
git clone https://github.com/dki-lab/Freebase-Setup.git ../../Freebase-Setup
```

The default SPARQL URL is `http://127.0.0.1:3001/sparql`. Override it at runtime
with `--kg-sparql-url`.

## 3. Configure a Model Endpoint

The benchmark uses an OpenAI-compatible endpoint. Configure either a local
vLLM server or a cloud API before running the smoke example. Both options use
the same experiment YAML files.

### Local vLLM

Create a separate vLLM environment. Install `huggingface_hub` so the model can
be downloaded explicitly with the official `hf` CLI:

```bash
python3.11 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install vllm huggingface_hub

export HF_HOME="$HOME/.cache/huggingface"
.venv-vllm/bin/hf download Qwen/Qwen2.5-14B-Instruct
```

The download is stored in the Hugging Face cache. See the
[Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)
for authentication, cache, and offline options.

Start the OpenAI-compatible server:

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen2.5-14B-Instruct \
  --served-model-name qwen2.5-14b-instruct \
  --port 8000
```

In a second shell, configure the benchmark client:

```bash
source .venv/bin/activate
export LIFELONG_AGENT_MODEL_NAME=qwen2.5-14b-instruct
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export NO_PROXY=127.0.0.1,localhost
```

The paper specifies `Qwen/Qwen2.5-14B-Instruct` for its local deployment. vLLM hardware and CUDA
installation requirements are documented in the
[vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/).

### Cloud API

Set the provider's model name, OpenAI-compatible `/v1` endpoint, and API key:

```bash
source .venv/bin/activate
export LIFELONG_AGENT_MODEL_NAME=provider-model-name
export OPENAI_BASE_URL=https://provider.example/v1
export OPENAI_API_KEY=your-api-key
```

Do not put API keys in YAML. The configs keep the key unset, and the client
reads `OPENAI_API_KEY` when the run starts. The same experiment YAML works with
local and cloud endpoints. A different model does not reproduce the released
Qwen2.5-14B results. `.env.example` is a template only; it is not loaded
automatically.

## 4. Run Smoke Example

The smoke config runs TokenWorth on DBBench samples `0`, `1`, and `2`. It needs
Docker, the `mysql:8.0` image, and one of the model endpoints configured in
section 3.

```bash
source .venv/bin/activate
python run.py \
  --config_path configs/examples/TokenWorth_db_smoke.yaml \
  --output-dir 'outputs/runs/smoke/TokenWorth-db-{TIMESTAMP_MS}'
```

`{TIMESTAMP_MS}` expands to `YYYY-MM-DD-HH-MM-SS-mmm`, so every new run gets a
directory such as `TokenWorth-db-2026-08-24-10-15-30-123`. To resume an
interrupted run, pass its exact existing directory without the placeholder; its
resolved model, endpoint, source, dependencies, dataset, task order and container image IDs must match. Resume is permitted only at a committed task boundary; an uncommitted task with a recorded model request cannot be replayed under the paper protocol.

Knowledge Graph smoke testing is optional because a local Freebase Virtuoso
deployment requires substantially more disk and memory than DBBench. After
starting a compatible endpoint as described in section 2, run its one-sample
config with:

```bash
python run.py \
  --config_path configs/examples/TokenWorth_knowledge_graph_smoke.yaml \
  --output-dir 'outputs/runs/smoke/TokenWorth-knowledge_graph-{TIMESTAMP_MS}' \
  --kg-sparql-url http://127.0.0.1:3001/sparql
```

## 5. Run the Paper Protocol

The release matrix contains 10 methods for each of 3 benchmarks, for 30
experiments in total:

`standard`, `RAG`, `Reflexion`, `Mem0`, `MemP`, `MemRL`, `MemSkill`, `EvoSkill`,
`EvoSC`, and `TokenWorth`.

The online protocol is fixed by the configs and `data/manifest.json`:

| Benchmark | Samples | Max rounds | Total output tokens per task | Primary metric |
|---|---:|---:|---:|---|
| DBBench | 500 | 3 | 128 | Success rate |
| Knowledge Graph | 150 | 15 | 128 | Success rate (exact match); F1 supplementary |
| OS Interaction | 500 | 5 | 512 | Success rate |

The output limit is shared by all execution and auxiliary calls within a task, not refreshed at each interaction. The client requires provider token usage, reduces each request cap to the remaining allowance, and skips auxiliary generation when it is exhausted. Unknown usage, service outages, empty replies, and evaluator faults invalidate the run and return a nonzero exit status. The shared memory bank limit is 64 records, retrieval injects at most one record, and the complete inserted block (including its title and separators) is capped at DB 900 / OS 800 / KG 4096 characters. Learning buffers and fixed method instructions are reported separately from the retrieval bank.

The protocol is online/prequential: begin with an empty learned bank, solve each task once, evaluate it, then update memory and the controller for later tasks. It does not create a held-out training/test split. Baselines use adaptations to this resource-constrained interface. See the [TokenWorth algorithm](../docs/TOKENWORTH.md) and [benchmark protocols](../docs/BENCHMARK_PROTOCOLS.md) for details.

Configure the paper's Qwen2.5-14B endpoint as in section 3, prepare all task
environments from section 2, and launch the matrix:

```bash
source .venv/bin/activate
CONFIG_ROOT=configs/experiments
CONFIGS=(
  "$CONFIG_ROOT"/db/*.yaml
  "$CONFIG_ROOT"/knowledge_graph/*.yaml
  "$CONFIG_ROOT"/os_interaction/*.yaml
)

python scripts/run_experiments.py \
  --jobs 1 \
  --output-root outputs/runs/reproduction \
  "${CONFIGS[@]}"
```

The launcher preflights an actual small model completion, usable Docker images and a KG query, freezes a batch manifest, launches runs and generates its report. The probe is separate from benchmark usage and may incur a small provider charge. `--repeats N` schedules N independent runs of the same task order. Resume an existing batch with:

```bash
python scripts/run_experiments.py --resume outputs/runs/reproduction/BATCH_DIRECTORY
```

Use `--db-image` and `--os-image` with either entry point to select independent image tags; their immutable IDs are locked at runtime. The launcher accepts the same `--llm-*` options and `--kg-sparql-url` overrides as `run.py`.

For a non-default Knowledge Graph endpoint, run its configs with
`--kg-sparql-url`:

```bash
python run.py \
  --config_path "$CONFIG_ROOT/knowledge_graph/TokenWorth.yaml" \
  --output-dir 'outputs/runs/reproduction/TokenWorth-knowledge_graph-{TIMESTAMP_MS}' \
  --kg-sparql-url http://127.0.0.1:3001/sparql
```

Increase `--jobs` only when GPU, API, Docker, memory, and SPARQL capacity allow
independent experiments to run concurrently.

## 6. Interpret Outputs

Each run directory contains:

- `config.yaml`: merged experiment configuration with credentials removed
- `runtime_manifest.json`: resolved runtime identity, protocol hashes and source/dependency fingerprints
- `run_status.json`: completion and validity, expected and committed sample counts
- `usage.jsonl`, `request_attempts.jsonl`: all execution/auxiliary request attempts and provider token counts; these journals survive checkpoint rollback
- `checkpoints/CURRENT`: authoritative atomic generation containing sessions and all callback states
- `runs.json`: per-sample conversations, raw responses, executed actions, injection lengths, evaluator outcomes and status
- `metric.json`: aggregate benchmark metrics
- `singleton_logger.log`, `exception.txt`, and callback-specific files:
  execution diagnostics and resumable memory state

Summarize one output tree:

```bash
python scripts/summarize_outputs.py \
  --outputs-dir outputs/runs/reproduction \
  --report-dir outputs/reports/reproduction
```

Generated artifacts remain under `outputs/`, which is excluded from Git by
default. Use `runtime_manifest.json` to identify the effective model and protocol. A configuration may still contain an `env:` model reference; a changed environment cannot silently alter a resumed run.

Reports use all valid repetitions (mean and sample SD), never the best run. They separate different models, task orders, resource budgets, method settings and implementations. Only a complete planned group enters the final comparison; invalid/incomplete runs remain visible as diagnostics. Smoke subsets are identified by their actual IDs rather than assumed full benchmark counts. OutputTok includes every generated execution/auxiliary token, including uncommitted attempts; missing usage is shown as unknown, not zero. A crashed task with a started request requires a new independent experiment because the protocol forbids replay. Checkpoints must contain the complete runtime identity and learner state to be resumable.

For API providers, model weight revisions and the external Freebase deployment remain external reproducibility inputs. Supply and retain the exact provider revision and KG database snapshot; matching an endpoint URL alone cannot prove their contents are unchanged.

## 7. Add a New Benchmark or Method

Experiments are configuration-driven. Do not edit `run.py` to
select a model, method, dataset, or output directory.

Adapter layout (relative to `LifelongAgentBench/`):

```text
configs/
  tasks/             Task definitions and fixed benchmark protocols
  methods/           Reusable method and callback definitions
  examples/          Small runnable examples
  experiments/       Paper matrix, grouped by benchmark
  utils/             Shared agent, model, logger, and registry configuration
data/
  lifelongagent_bench/  Fixed DB, KG, and OS benchmark inputs
  manifest.json         Protocol metadata and input checksums
system_prompts/  Fixed task prompts and demonstrations
src/
  agents/            Agent loop implementations
  callbacks/         Memory methods, including TokenWorth
  language_models/   OpenAI-compatible local/API client
  tasks/             Benchmark environments and evaluators
run.py                Configuration-driven experiment entry point
scripts/              Install, data validation, batch launch, and result reports
outputs/              Generated trajectories and metrics (ignored by Git)
tests/                Configuration and protocol tests
```

At runtime, an experiment YAML imports reusable components, `ConfigLoader`
resolves them into one configuration, and `run.py` constructs the
task, agent, model client, and callbacks. Local vLLM and cloud API runs share
the same language-model adapter; only environment variables change.

To add a method:

1. Add the callback implementation under `src/callbacks/instance/`.
2. Add its constructor YAML under `configs/methods/` and register it in
   `configs/utils/definition.yaml`.
3. Add `configs/experiments/<benchmark>/<method>.yaml`, importing the matching
   `configs/tasks/<benchmark>.yaml` configuration.

To add a benchmark:

1. Add the task implementation and evaluator under
   `src/tasks/instance/<benchmark>/`.
2. Add `configs/tasks/<benchmark>.yaml` containing both the task definition and
   the fixed agent, model, round, and output-token protocol.
3. Add method experiments under `configs/experiments/<benchmark>/`.
4. Put fixed processed inputs under `data/`, add their checksums and protocol
   metadata to `data/manifest.json`, and add a minimal config under
   `configs/examples/`.

The GitHub workflow in `../.github/workflows/tests.yml` installs the locked Python 3.11 environment, verifies data and runs the workflow regression tests. It exercises fixtures without requiring a paid model or a Freebase deployment; full environment/model evaluation remains a separate check.

Validate additions before publishing:

```bash
bash scripts/install.sh --extras dev
source .venv/bin/activate
python scripts/verify_data.py
python -m pytest -q
```
