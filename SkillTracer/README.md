# SkillTracer

SkillTracer is a research implementation for watermarking agent skills in the
**specified-skill** setting. It places a functional-summary payload in a skill
resource script, adds a conditional workflow to `SKILL.md`, and keeps the
verification trigger with the publisher. The Codex runner evaluates the original
skill, the protected skill without the trigger, and the protected skill with the
trigger in separate workspaces.

This repository contains implementation code and tests. Benchmark skills, task
inputs, API credentials, generated skill packages, logs, and experiment results
are intentionally excluded.

## What is implemented

| Component | File | Purpose |
| --- | --- | --- |
| Specified-setting experiment | `scripts/run_skilltracer_specified.py` | Generate protected skills and run the three-phase Codex evaluation. Defaults to three functional dimensions and one trial per skill. |
| Skill-level hypothesis experiment | `scripts/run_skilltracer_hypothesis.py` | Use five dimensions and five distinct verification queries per skill; calculate skill-level theoretical p-values. |
| Hypothesis readback | `scripts/finalize_skilltracer_hypothesis.py` | Rebuild summary tables from saved query evidence without new model calls. |
| Case verification | `scripts/verify_skilltracer_case.py` | Check a saved successful case, including the publisher-held trigger and file hashes. |
| Specified-run rejudgment | `scripts/finalize_skilltracer_results.py` | Rejudge saved three-phase runs and rebuild case-study folders. This step makes model calls. |

The hypothesis experiment uses the complete verification rule: the expected
functional-summary artifact must be present and named in the final response;
it must contain five normal and five abnormal modules; and the semantic labels
must form a positive-module permutation followed by the corresponding reversed
negative modules. A query with an execution or judge error is incomplete, not
a failed detection.

For the idealized null in which the ten distinct semantic modules are uniformly
permuted, the one-query match probability is
\(p_0=5!/10!=1/30240\). For a skill with \(K\) successes in five completed
queries, the reported theoretical p-value is
\(P[\operatorname{Binomial}(5,p_0)\ge K]\), evaluated with
`scipy.stats.binom.logsf`. These values describe that theoretical null; they
are not measured false-positive rates. Repeated queries against the same skill
may be dependent.

## Requirements

- Python 3.10 or newer and `pip`
- Codex CLI available as `codex` (or pass `--codex-bin`)
- An API endpoint that supports both Chat Completions and Responses
- A model available through that endpoint; the default is `openai/gpt-5.6-sol`
- Your own skill dataset and task instructions in the layout below

Install the Python dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the endpoint if needed. The current default is
`https://router.shengsuanyun.com/api/v1`:

```bash
export SKILLTRACER_BASE_URL="https://your-compatible-endpoint.example/v1"
export SKILLTRACER_MODEL="openai/gpt-5.6-sol"
```

Store the API key in a permission-restricted file **outside this repository**.
Pass its path with `--api-key-file`; the runner uses a temporary Codex home for
each execution and removes it afterward. Do not commit the key or generated
`auth.json` files.

## Dataset layout

Place your data under the project root:

```text
data/
├── skills_sample/
│   └── <skill-id>/
│       ├── SKILL.md
│       └── resources/               # optional skill files
└── instruction/
    └── skills_sample/
        └── <skill-id>/
            ├── instruction.md
            └── <task-input-files>     # optional inputs copied into the run
```

The hypothesis runner sorts skill directory names lexicographically and groups
them into consecutive subsets of ten. Each skill is tested independently;
subset rows summarize skill-level results and do not combine queries into a
subset-level p-value.

## Run

From the repository root, first try a focused specified-setting case:

```bash
python scripts/run_skilltracer_specified.py \
  --api-key-file /secure/path/router_api_key \
  --skills <skill-id> \
  --output results/specified \
  --workers 1
```

Run the five-query, five-dimension experiment:

```bash
python scripts/run_skilltracer_hypothesis.py \
  --api-key-file /secure/path/router_api_key \
  --output results/hypothesis \
  --workers 2 \
  --seed 20261030
```

Use `--limit-skills 10` for a smaller run or `--skills <skill-id>` for one
skill. Keep the output directory when resuming: completed query records are
reused. Rebuild the summary files from saved evidence with:

```bash
python scripts/finalize_skilltracer_hypothesis.py results/hypothesis
```

The hypothesis output includes `experiment_config.json`,
`skill_level_results.csv`, `subset_level_results.csv`, and
`query_level_results.jsonl`, plus per-query prompts, raw outputs, logs, and
verification records under `results/hypothesis/skills/`. Only skills with five
completed queries receive a final skill-level p-value.

For a successful specified-setting case, run the independent check with:

```bash
python scripts/verify_skilltracer_case.py \
  results/specified/successful_cases/<skill-id>
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_skilltracer*.py'
```

The tests use local fixtures and mocks. They do not consume API credits.

## Scope and reproducibility

The repository focuses on the paper-aligned specified-skill Codex workflow.
It does not ship the older baseline methods or the private experiment runs
from the source workspace. The recorded theoretical p-values assume the
uniform-permutation null and independent Bernoulli query outcomes; both
assumptions should be stated when reporting results.
