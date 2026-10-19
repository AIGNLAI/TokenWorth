# TokenWorth on BigCodeBench

This directory contains the BigCodeBench adapter, experiment configurations
and evaluation runner. See the [TokenWorth algorithm](../docs/TOKENWORTH.md)
and [benchmark protocols](../docs/BENCHMARK_PROTOCOLS.md) for learning,
evaluation and resume settings.

The adapter uses BigCodeBench v0.1.4 and pins the upstream implementation to
commit `09dd993f46c3fbf3a799465bb96d524edcb0b199`.

## Published configurations

The configuration files are fully resolved JSON. They do not inherit from
experiment outputs or depend on machine-specific paths.

- `configs/local`: local `qwen2.5-14b-instruct`, 1,280 completion tokens, and
  the ten methods used by the local experiment. The TokenWorth configuration
  selects the shared `paper` implementation.
- `configs/api_no_thinking`: `deepseek/deepseek-v4-flash`, thinking
  disabled, 1,280 completion tokens, and all ten methods. The baseline memory
  methods use the common cosine retrieval variant. TokenWorth uses its
  shared `paper` implementation and built-in lightweight representation.
- `configs/api_thinking`: `deepseek/deepseek-v4-flash`, thinking
  enabled for the main generation call, 4,096 completion tokens, and all ten
  methods. Auxiliary memory-maintenance calls keep thinking disabled. This
  suite uses the same `paper` TokenWorth implementation. Baselines use the
  common cosine retrieval variant.

## Installation

Python 3.10 is recommended. From this directory:

```bash
bash scripts/install.sh --extras test
source .venv/bin/activate
```

The installer installs the sibling `../tokenworth_core` package from this
complete repository checkout and checks out the pinned BigCodeBench upstream commit inside the
virtual environment and exposes its Python package without registering its
unused model-serving dependency set. It then installs only the host-side
packages needed for data loading, sanitization, API generation, and evaluation.
The full upstream dependency set, including vLLM and multiple provider SDKs,
is not required because model inference is accessed through an external
OpenAI-compatible endpoint. Use `--venv` or `PYTHON_BIN` to override the
environment path or Python executable. `requirements.lock` constrains the
host-side packages to fixed versions.

Download both datasets from immutable revisions of the official Hugging Face
repositories. The script verifies the source Parquet SHA-256 values, checks
the expected 1,140/148 row counts and deterministically exports JSONL:

```bash
python scripts/download_data.py --output-dir ../../tokenworth-data/bigcodebench-v0.1.4
```

Keep the data outside the Git checkout if desired; the output and cache can be
placed on a larger disk with `--output-dir` and `--cache-dir`. The script
prints the exact environment-variable values to use. Then copy the template:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Loopback model endpoints bypass global proxy settings. For local inference,
point `OPENAI_BASE_URL` at the server exposing
`qwen2.5-14b-instruct`. For the API suites, point it at an OpenAI-compatible
endpoint exposing `deepseek/deepseek-v4-flash`.

The model is selected by `model.name` in the suite JSON files, not by `.env`:

- `configs/local/*.json` selects `qwen2.5-14b-instruct`.
- `configs/api_no_thinking/*.json` selects
  `deepseek/deepseek-v4-flash` with thinking disabled.
- `configs/api_thinking/*.json` selects
  `deepseek/deepseek-v4-flash` with thinking enabled.

The launcher reads the selected JSON file and sends its `model.name` value to
the endpoint configured by `OPENAI_BASE_URL`. Changing that value changes the
published protocol.

### API compatibility

The local suite uses standard Chat Completions fields. The two API suites also
send the provider extension `thinking: {type: enabled|disabled}` in
`extra_body` and use `max_completion_tokens`. The endpoint must accept these
fields; thinking responses may expose reasoning as `reasoning_content` or
`reasoning`. This extension is not part of the base OpenAI API contract.

`deepseek/deepseek-v4-flash` is the endpoint alias in the API configurations.
Use `--model` to select another endpoint-specific model name; this changes the
experiment protocol. SDK retries are disabled; only explicitly
rejected rate-limit requests are retried (three bounded retries). Every attempt
is durably recorded. Transport errors, timeouts, 5xx responses, missing usage
and evaluator faults invalidate the run rather than silently changing reward
or cost.

## Isolated evaluator

Generated code is evaluated inside a long-lived Docker worker. Build the image
from this directory:

```bash
docker build \
  --file docker/Dockerfile.evaluator \
  --tag bigcodebench-evaluator-local:latest \
  .
```

At runtime the adapter starts the worker with networking disabled, a read-only
root filesystem, process and memory limits, and only the selected Full dataset
JSONL mounted read-only. The dataset's parent directory and the Hard JSONL are
not exposed to generated code. Docker access is therefore required for the
published configurations.

## Running experiments

Run one method first as a smoke test:

```bash
python scripts/run_suite.py local --method no_memory --limit 1
python scripts/run_suite.py api_no_thinking --method tokenworth --limit 1
python scripts/run_suite.py api_thinking --method tokenworth --limit 1
```

Remove `--limit 1` for a complete method. Omit `--method` to run every method
in a suite sequentially:

```bash
python scripts/run_suite.py local
python scripts/run_suite.py api_no_thinking
python scripts/run_suite.py api_thinking
```

Every method processes BigCodeBench Full minus Hard first and then the Hard
subset. Memory is updated after each task, so task order is part of the
protocol. Runs write beneath ignored `outputs/`. Completed task boundaries are
resumable; an uncommitted active task refuses replay and needs a fresh output
directory. Checkpoints must match the run's implementation and configuration.
Partial runs have explicit planned/completed counts and must not be reported
as complete comparisons.
All memory-method configs share a 4,096-character context cap. TokenWorth
uses verified code or labeled non-executable outlines without auxiliary calls.
NoMemory runs through the same runner and normalization/evaluation path.

Use `--dry-run` to inspect the commands without calling a model or evaluator:

```bash
python scripts/run_suite.py api_no_thinking --dry-run
```

## Tests

Run the offline tests with:

```bash
python -m pytest
```

The tests cover configuration, model and thinking settings, task ordering,
learner persistence, cached residual targets, fault handling,
ordered-prefix recovery, cost accounting and shared retrieval representations.

## Layout

```text
BigCodeBench/
├── configs/                 # Only the three published protocol groups
├── docker/                  # Isolated evaluator image and dependencies
├── experiments/             # Sequential runners and memory methods
├── scripts/                 # Installer, data downloader, and suite launcher
└── tests/                   # Offline tests
```

BigCodeBench is distributed under the Apache License 2.0. See `LICENSE`.
