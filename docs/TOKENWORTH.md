# TokenWorth

TokenWorth retrieves experiences by their learned value for the current task.
The model backbone stays frozen. Each run starts with an empty memory bank,
executes each task once, evaluates the outcome, then updates memory for later
tasks. TokenWorth makes no auxiliary model calls.

## Representation and retrieval

Queries and memories use deterministic 512-dimensional token and character
3–5-gram hashing with L2 normalization. BM25 selects at most 12 candidates;
the learned worth score selects one memory for injection.

The base score uses cosine similarity, raw utility, reliability and recency.
Its weights start at `[1, .10, .10, .02]` and minimize logistic loss with
regularization `.1 * ||w-w0||²`. Deterministic Newton steps with backtracking
solve the objective. The learner uses a FIFO of 256 feedback pairs, refits
every 50 pairs, applies EMA .5 and clips weights between `[0,0,0,0]` and
`[2,1,1,1]`.

Residual features are:

```text
[abs(e_query - e_memory), cosine, raw utility, reliability, recency, frequency]
```

Reliability is `(help+1)/(help+harm+2)`, recency is
`exp(-.02*age_since_access)`, and frequency is
`accesses / max(1, step-created+1)`. Features and the base score are cached at
retrieval time, before access statistics and reward updates.

## Feedback and learning

Only an injected memory receives feedback. Let `r` be the externally evaluated
binary outcome and `y=2r-1`. The base learns `y`; residual kernel ridge
regression learns `y-tanh(s_base/2)`. Residual targets and predictions are
unclipped.

The fast learner uses a Laplacian kernel with a diagonal metric, FIFO 128,
ridge .01 and gamma 1. The residual contribution ramps linearly over the first
three KRR fits after the minimum feedback count is reached.

The slow learner computes diagonal AGOP from the current fast predictor's
gradients at points in a FIFO of 256. It blends the metric with .25, clips to
[.05,8], normalizes its mean to one and immediately refits fast KRR.

Memory utility starts at 1. Successful use adds .25; failed use subtracts .45;
utility is clipped to [-2,5]. Infrastructure and evaluator failures abort the
run without producing negative learning feedback.

## Memory management

Only externally verified successful interactions create experiences. New
experiences are admitted directly while the bank is below capacity.

At capacity, the controller applies these rules in order:

1. Admit the candidate if novelty `1-max_cosine>.35`, or if its probe-average
   worth exceeds the highest existing memory's probe-average worth by more
   than .02 and at least .45 of its probe scores are strictly positive.
2. If neither admission condition holds, replace the nearest memory only when
   that memory's utility is below .5 and the candidate's probe-average worth
   exceeds the nearest memory's by more than .10.
3. Otherwise reject the candidate.

The positive-score ratio counts scores greater than zero. It does not compare
the candidate with another memory on each probe. Nearest-memory comparisons
are used by the fallback replacement rule, while ordinary value-based admission
compares against the highest probe-average worth in the entire bank.

Probes comprise the current query and at most eight prior retrieval queries.
They use the same learned worth score as retrieval. If the bank temporarily
exceeds capacity, the memory with the lowest probe-average worth is evicted,
including the candidate if it has the lowest worth. Direct replacement keeps
the bank at its current size and does not run this overflow-eviction step.

## State and benchmark settings

Checkpoints persist base weights and buffers, residual buffers and fitted
rows, KRR coefficients, AGOP geometry and counters, probe history, memory
statistics and the next memory ID together.

BigCodeBench and ALFWorld use the shared
[`tokenworth_core`](../tokenworth_core/__init__.py) package. LifelongAgentBench
integrates the learner into its callback interface. Bank sizes, injection
limits, update intervals and evaluation budgets are listed in
[benchmark protocols](BENCHMARK_PROTOCOLS.md).
