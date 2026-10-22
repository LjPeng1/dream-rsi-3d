# DREAM-RSI

**3D research fork:** this fork adds a runnable procedural 3D layout task,
a shared-start fixed-policy comparison, Blender GLB export/verification, and
current Codex/Windows support. See [3D baseline guide](docs/3D_BASELINE.md).
The original framework and paper description follow below.

This is the code for **DREAM-RSI: Recursive Self-Improvement Through Evolving
Worlds**.

DREAM-RSI improves the *exploration policy*, which decides how a coding agent
spends its discovery budget. The method repeats rounds of two phases:

1. **Online rollout.** The current policy directs a fixed coding agent. The
   agent's attempts form a discovery tree. The tree has at most `BRANCHES`
   branches of `REFINE + 1` attempts each. A fixed task evaluator scores every
   attempt. The finished tree is kept as a *replay world*.
2. **Offline dreaming.** An agent rewrites the policy code. Each version is
   replayed on all replay worlds so far, once per budget preference
   ρ ∈ {0, 0.25, 0.5, 0.75, 1}. The replay objective (Eq. 1 of the paper)
   scores each version. The best version, which may be the current policy,
   explores the next round.

Only the policy changes. The agent, the evaluator and the task stay fixed.

The baseline is **Recursive Fixed Exploration**. It runs the same rounds with
the hand-designed starting policy and never updates it. In both methods, the
agents can read all earlier rounds.

## Repository layout

| Path | Contents |
| --- | --- |
| `dream_rsi/policy/` | Exploration policies. `baseline.py` is the hand-designed starting policy. Offline learning rewrites `discovered/method.template.py` |
| `dream_rsi/exploration/` | Online rollout. Runs the agent on the selected nodes and evaluates its programs |
| `dream_rsi/env/` | `TraceEnv`, the decision interface shared by online rollout and replay |
| `dream_rsi/learning/` | Offline learning: the replay worlds (`trace_pool.py`), policy rewriting (`discover.py`), the replay objective (`heuristic_search/eval.py`), and the direction summaries of the guidance ablation (`directions/`) |
| `dream_rsi/orchestrate.py` | The DREAM-RSI loop: explore, learn, promote |
| `dream_rsi/agents/` | Coding agents: Gemini CLI (used in the paper), Claude Agent SDK, Codex CLI |
| `dream_rsi/prompts/` | Prompts. `explore.txt` and `heuristic_proposer.txt` are Listings 1 and 2 of the paper |
| `dream_rsi/cli/` | The `dream-rsi` command line |
| `scripts/` | Launch scripts for the paper's experiments |
| `tasks/` | Benchmark tasks, see [`tasks/README.md`](tasks/README.md) |
| `tests/` | Tests |

## Installation

You need Python 3.10 or newer. From the repository root:

```bash
pip install -e ".[tasks]"             # numpy, scipy, psutil, scikit-learn for the evaluators
npm install -g @google/gemini-cli     # the coding agent used in the paper
cp .env.example .env                  # then add your Gemini API keys to .env
bash scripts/fetch_tasks.sh           # downloads the math and Lasso task files (see below)
```

The `dream_rsi` package itself has no dependencies. Install it in editable
mode (`-e`), because the launch scripts and the default task paths point into
the repository. The Gemini agent passes `--policy` and `--skip-trust` to the
CLI, so it needs a recent Gemini CLI. The code was written for version 0.50.

Some tasks need more setup:

- **Math and Lasso tasks:** their problem statements, evaluators and initial
  programs come from [SimpleTES](https://github.com/wq-will/SimpleTES), which
  is licensed under the GNU AGPL-3.0 and therefore not included in this
  repository. `bash scripts/fetch_tasks.sh` downloads them from the pinned
  SimpleTES commit into `tasks/<task>/`. The launch scripts tell you if they
  are missing.
- **Lasso path:** run `bash tasks/lasso_path/setup.sh` to download Eigen. The
  evaluator also needs `g++`.
- **KernelBench:** you need a GPU evaluation server, see
  [`tasks/kernelbench/README.md`](tasks/kernelbench/README.md). The client
  side needs `pip install -e ".[kernel]"`. The four KernelBench problems that
  seed the `kb_*` tasks are vendored under
  [`third_party/kernelbench/`](third_party/kernelbench/) (MIT License).

## Quick start

With `DRY_RUN=1`, a launch script runs all of its checks and prints the
commands without running them. It checks the task files, the evaluator's
dependencies, the agent and the API keys:

```bash
DRY_RUN=1 bash scripts/run_dream_rsi.sh circle_packing_26
```

A short real run, with 2 rounds of 3 branches × 2 attempts:

```bash
WORKSPACE=workspace/smoke ROUNDS=2 BRANCHES=3 REFINE=1 MAX_WORKERS=3 \
  bash scripts/run_dream_rsi.sh circle_packing_26
```

Every script lists its settings with `--help`. All settings are environment
variables.

## Reproducing the paper

The launch scripts default to the paper's configuration (Table 3):

- Gemini 3.1 Pro (`gemini-3.1-pro-preview`).
- W = 10 parallel agents.
- 10 branches × 11 attempts per round (B = 110).
- M = 4 policy versions per learning step: the current policy and
  3 rewrites.

The number of rounds T depends on the task:

| Paper name | `TASK` | T |
| --- | --- | --- |
| Sum–Difference | `sums_diffs` | 10 |
| Autocorrelation | `autocorrelation_third` | 10 |
| Circle Packing | `circle_packing_26` | 10 |
| Lasso path | `lasso_path` | 5 |
| VGG16 | `kb_l3_11_vgg16` | 9 |
| LayerNorm | `kb_l1_40_layernorm` | 9 |
| ConvDiv | `kb_l2_8_convdiv` | 9 |
| ConvMax | `kb_l2_43_convmax` | 9 |

In the paper, both methods continue from the same first round E0, which is
explored with the starting policy:

```bash
bash scripts/run_e0.sh sums_diffs
E0_WORKSPACE=workspace/e0/sums_diffs bash scripts/run_dream_rsi.sh sums_diffs
E0_WORKSPACE=workspace/e0/sums_diffs bash scripts/run_baseline.sh sums_diffs
```

Each method copies E0 into its own workspace (`workspace/dream_rsi/<task>` and
`workspace/baseline/<task>`), so the E0 workspace is never changed. Without
`E0_WORKSPACE`, each script explores its own first round.

The two runs are independent and can run at the same time. The Lasso and
kernel scores depend on measured runtimes, though, so concurrent runs should
not share a machine or a GPU. For the `kb_*` tasks, give each run its own evaluation server
with `GPUKERNEL_SERVER_INFO`. A server evaluates one kernel at a time.

**Lasso with Gemini 3.7 Flash** (W = 32, B = 640 = 32 × 20). Use separate
workspaces so these runs do not collide with the Gemini 3.1 Pro runs:

```bash
export GEMINI_MODEL=gemini-3.7-flash BRANCHES=32 REFINE=19 MAX_WORKERS=32
WORKSPACE=workspace/e0/lasso_path_flash bash scripts/run_e0.sh lasso_path
E0_WORKSPACE=workspace/e0/lasso_path_flash WORKSPACE=workspace/dream_rsi/lasso_path_flash \
  bash scripts/run_dream_rsi.sh lasso_path
E0_WORKSPACE=workspace/e0/lasso_path_flash WORKSPACE=workspace/baseline/lasso_path_flash \
  bash scripts/run_baseline.sh lasso_path
```

**Directional guidance (Figure 6).** With `GUIDANCE=1`, an agent summarizes the
explored directions after every round, and the summary goes into the next
round's prompts. These runs write to `workspace/dream_rsi_guidance/<task>` and
`workspace/baseline_guidance/<task>`. E0 has no guidance, so the same E0 can be
used:

```bash
GUIDANCE=1 E0_WORKSPACE=workspace/e0/kb_l2_8_convdiv bash scripts/run_dream_rsi.sh kb_l2_8_convdiv
GUIDANCE=1 E0_WORKSPACE=workspace/e0/kb_l2_8_convdiv bash scripts/run_baseline.sh kb_l2_8_convdiv
```

The paper ran the Lasso and math tasks on an 8-vCPU, 32 GB virtual machine.
It ran the kernel tasks on an NVIDIA L40S. The Lasso and kernel scores are
inverse runtimes, so they depend on the hardware.

These experiments are not included:

- The harness optimization experiment (Section 4.4).
- The "Current History Only" ablation (Table 2). There is no switch for it:
  offline learning always replays all rounds so far.

## Interrupting and resuming

The scripts never overwrite an existing workspace. After an interruption, run
the same command again with `RESUME=1`. It continues from the first round
that was not finished. If that round's exploration was complete, the round is
not explored again: DREAM-RSI goes on with its learning step. To extend a
finished run, use `RESUME=1` with a larger `ROUNDS`.

## Results

At the end, the scripts print the best score of each round. A workspace looks
like this:

```
workspace/dream_rsi/<task>/
├── problem.txt                   problem statement given to the agents
├── .baseline/                    the task's initial program and its score
├── history/
│   ├── iter000/                  round 1 (E0)
│   │   ├── raw_scores.json       every attempt of the round and its score
│   │   └── explore_b0/attempt_00/
│   │       ├── proposal.md       the agent's plan
│   │       ├── artifact/         the candidate program (program.py)
│   │       ├── eval/score.json   the evaluator's result
│   │       └── agent_log/        the agent's full trajectory
│   └── iter001/ ...
├── trace_pool/                   the replay worlds: a frozen copy of each round's scores
├── learning/iter000/ ...         the learning step after each round (DREAM-RSI)
├── policy/method.py              the latest promoted policy (DREAM-RSI)
├── pipeline_summary.json         what each round explored, learned and promoted (DREAM-RSI)
└── directions/                   the direction summaries (GUIDANCE=1)
```

In `raw_scores.json`, the node `b<B>_a<A>` is branch B, attempt A. Its files
are in `explore_b<B>/attempt_<AA>/`, and `raw.combined_score` is its score.
Failed attempts score 0.

In `learning/iter<NNN>/`, `heuristic_history/` keeps every policy version and
its replay results. `discover_summary.json` names the selected version. If a
rewrite beats the current policy, it is copied to `policy/method.py` and
explores the next round.

## Agents

The paper's experiments use Gemini (`AGENT=gemini`, the default). By
default, the Gemini agent can only use file tools in exploration, not the
shell; the evaluator runs the submitted program. With `EXPLORE_SHELL=1` (or
`--explore-shell` on the command line), it may also run shell commands, for
example to test a program before it submits it. Use the same setting for E0
and for both methods. In offline learning, the agent always has the shell.

Two other agents are supported, but the paper did not use them:

- `AGENT=claude`: Claude Agent SDK. Needs `pip install -e ".[claude]"` and
  Claude Code credentials, e.g. `ANTHROPIC_API_KEY`.
- `AGENT=codex`: Codex CLI. Reads `CODEX_BIN` and `CODEX_MODEL`.

Both of them can always run shell commands during exploration;
`EXPLORE_SHELL` does not change them.

## Environment variables

Set these variables in the shell or in `.env`. Variables set in the shell win.
`DRSI_ENV_FILE` selects another `.env` file. The defaults are those of the
launch scripts.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEYS` | – | Comma-separated Gemini API keys. Agent runs take them in turn and skip a key for 60 s after a rate-limit error. Keys only add quota if they belong to different Google Cloud projects |
| `GEMINI_API_KEY` | – | A single key, used if `GEMINI_API_KEYS` is not set. `GOOGLE_API_KEY` also works |
| `GEMINI_BIN` | `gemini` | Gemini CLI executable |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | Model of the agent |
| `GEMINI_TIMEOUT_SEC` | 12000; 3600 for `kb_*` | Time limit of one agent run |
| `GEMINI_RETRIES` | 3; 20 for `kb_*` | Retries after a transient API error |
| `GEMINI_RETRY_BASE_SEC` | 60; 120 for `kb_*` | Base delay of the exponential backoff between retries |
| `GEMINI_EXPLORE_SHELL` | off | `1` lets the Gemini agent run shell commands in exploration. `--explore-shell` and `--no-explore-shell` override it; the launch scripts use `EXPLORE_SHELL` instead |
| `DRSI_EVAL_PYTHON` | the interpreter running DREAM-RSI | Interpreter for the evaluators and the candidate programs |
| `DRSI_HISTORY_SEED_MODE` | `copy` | How each branch gets the earlier rounds: `copy` (as in the paper) or `symlink` (saves disk space) |
| `DRSI_REPLAY_PARALLELISM_PENALTY` | 0.05 | λ in the replay objective |
| `DRSI_REPLAY_BETA_GRID` | `0,0.25,0.5,0.75,1` | The budget preferences ρ |

The task evaluators read their own variables, which are listed in
[`tasks/README.md`](tasks/README.md).

## Command line

The launch scripts call the `dream-rsi` command line. You can also run it as
`python -m dream_rsi`:

| Command | What it does |
| --- | --- |
| `dream-rsi explore init` | Create a workspace |
| `dream-rsi explore run` | Explore one online round with a policy |
| `dream-rsi explore archive` | Move the finished round to `history/iter<NNN>/` |
| `dream-rsi learning export` | Write a round's `raw_scores.json` |
| `dream-rsi learning replay` | Replay the starting policy and another policy on a recorded round |
| `dream-rsi learning allocate` | Create a learning job from a workspace |
| `dream-rsi learning discover` | Run offline policy learning on a learning job |
| `dream-rsi learning directions` | Summarize the explored directions (guidance ablation) |
| `dream-rsi evolve` | Run the full DREAM-RSI loop |

Add `--help` to any command for its options.

## Notes

- Every round starts from the task's initial program, as in the paper. The
  agents still see the best programs of earlier rounds in the history.
  `dream-rsi evolve --update-baseline-champion` makes each round start from
  the previous round's best program instead.
- The policy learned after the last round explores nothing. It is kept so
  that a run extended with `RESUME=1` can use it.

## Tests

```bash
pip install -e ".[tasks,dev]"
pytest                   # no API calls
pytest -m integration    # one live Gemini run on the Lasso task
```

The integration test needs the Gemini CLI, an API key, `g++` and Eigen.

## Acknowledgements

The mathematical and Lasso tasks are those of SimpleTES (Ye et al., 2026);
`scripts/fetch_tasks.sh` downloads their files from the SimpleTES repository.
The kernel tasks are KernelBench problems (Ouyang et al., 2025), scored with
KernelBench's evaluation code.

## License

This repository is licensed under the [Apache License 2.0](LICENSE), except
for the files under [`third_party/`](third_party/), which keep their original
licenses (see the `LICENSE` file in each subdirectory). The task files that
`scripts/fetch_tasks.sh` downloads are licensed by SimpleTES under the GNU
AGPL-3.0 and are not part of this repository.

Eligibility for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).

## Disclaimer

This is not an officially supported Google product. This project is research
code released to accompany a publication and is not intended for use in a
production environment.
