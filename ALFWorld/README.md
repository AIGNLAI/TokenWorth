# TokenWorth on ALFWorld

This directory contains the continual ALFWorld adapter, experiment
configuration and evaluation runner. See the
[TokenWorth algorithm](../docs/TOKENWORTH.md) and
[benchmark protocols](../docs/BENCHMARK_PROTOCOLS.md) for learning, evaluation
and resume settings.

The adapter pins the upstream ALFWorld implementation to commit
`aaba6870f86c5be6a08a491f32a50b906227bc3e` and uses the ALFWorld 2.1.1 text
environment data layout.

## Published protocol

The single fully resolved configuration is `configs/local.yaml`. It runs
the following ten methods:

`TokenWorth`, `RAG`, `MemRL`, `Reflexion`, `MemP`, `EvoSC`, `EvoSkill`, `Mem0`,
`MemSkill`, and `NoMemory`.

The protocol uses:

- `qwen2.5-14b-instruct` through an OpenAI-compatible local endpoint;
- seed 42 and stratified sampling of 300 training games;
- every game in `valid_seen` and `valid_unseen`;
- online memory updates after every episode in all three phases;
- admissible commands only, without expert trajectories;
- a 50-step episode limit and one invalid-action repair attempt;
- 128 memory records, top-1 retrieval, and a 4,200-character memory context;
- deterministic 512-dimensional hashing, BM25 recall and learned TokenWorth ranking;
- no memory-dependent loop removal, expert access or TokenWorth auxiliary LLM calls.

## Installation

Python 3.9 was used for the reference environment. From this directory:

```bash
bash scripts/install.sh
source .venv/bin/activate
```

The installer installs the sibling `../tokenworth_core` package from this
complete repository checkout and fetches the pinned upstream ALFWorld commit.
It activates the virtual environment for native builds and checks the actual
TextWorld environment import. Dependencies include spaCy 3.7.5 and TextWorld
1.6.2, with versions fixed in `requirements.lock`.
Download the text-environment data to a
directory with sufficient space:

```bash
python scripts/download_data.py --data-dir ../../tokenworth-data/alfworld
```

The downloader fetches the three official text-environment assets, verifies
their SHA-256 values and archive entry counts, resumes truncated range
downloads, extracts them safely, and checks
the expected game counts for every split. It does not download the Mask R-CNN
weights or pretrained agents, which are unnecessary for this text-only
protocol. Use `--archive-dir` to keep the verified archives on a separate disk.

Copy and edit the environment template:

```bash
cp .env.example .env
```

Set `OPENAI_BASE_URL` to the local OpenAI-compatible endpoint serving
`qwen2.5-14b-instruct`, and set `ALFWORLD_DATA_ROOT` to the downloaded data
root. `OPENAI_API_KEY=local` is sufficient for a local server that ignores
authentication. The release configuration reads these values without writing
them into generated configuration files.

## Running the experiment

Inspect the command without starting ALFWorld or calling the model:

```bash
python scripts/run_suite.py --dry-run
```

Run one method first:

```bash
python scripts/run_suite.py --method no_memory --limit 1
python scripts/run_suite.py --method tokenworth
```

`--limit` caps the number of new episodes run for each selected method. It is
intended for smoke tests; removing it resumes the same run from the next
episode.

Repeat `--method` to select several methods, or omit it to run all ten methods
sequentially:

```bash
python scripts/run_suite.py --method rag --method no_memory
python scripts/run_suite.py
```

The complete suite contains 574 episodes per method. The launcher uses one
endpoint sequentially; `--api-base` can override it for an individual
invocation. Each run records its resolved endpoint and model identity, which
must match when resuming.

Runs write beneath ignored `outputs/`. Completed episode boundaries resume
without replay; an uncommitted active episode requires a fresh experiment.
Changing model/config/data/code/dependencies requires a new run directory.
Each response requires valid actual usage; only
explicitly rejected 429 requests are retried, with three attempts total.
Service/evaluator faults do not become negative learning feedback. The
checkpoint atomically stores learner state and evaluated records; partial
summaries have explicit planned/completed counts and `is_complete=false`.
Use one process per method/output directory.

## Tests

Run all offline checks with:

```bash
python -m unittest discover -s tests -v
```

The tests cover all memory methods, the shared retrieval encoder, deterministic
stratified sampling, prompt/state isolation, loop handling, API accounting,
resume helpers, portable environment resolution and experiment protocols.

## Layout

```text
ALFWorld/
├── configs/
│   ├── alfworld_base.yaml   # MIT-licensed upstream base environment config
│   └── local.yaml           # The only published experiment configuration
├── continual_alfworld/      # Continual runner and ten memory methods
├── scripts/                 # Installer and experiment launcher
├── tests/                   # Offline tests
└── run_continual_eval.py    # Direct CLI entry point
```

ALFWorld and this adapter are distributed under the MIT License. See `LICENSE`.
