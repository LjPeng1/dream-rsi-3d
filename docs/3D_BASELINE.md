# A runnable 3D baseline for policy self-improvement

This fork adapts [Google Research DREAM-RSI](https://github.com/google-research/dream-rsi)
at `8ec35bd2538220517ce1bfca4282db21b8c518db` (Apache-2.0).
It is a small research baseline, not a reproduction of the paper's headline results.

The fixed coding LLM writes a reusable `solve(scene)` program. A fixed evaluator
scores its 3D placements. DREAM-RSI then rewrites the **exploration policy** using
offline replay, selects a replay-qualified policy, and applies it in the next
round. It does not update LLM weights or change its own evaluator. The upstream
release also excludes the paper's harness-optimization experiment (Section 4.4).

## Task and measurements

`tasks/blender_layout/` contains six deterministic development scenes and eight
heldout scenes with different dimensions and relation configurations. Every
object is an axis-aligned cuboid with immutable identity, size and orientation.
The candidate returns centers only. Constraints include room boundaries,
nonpenetration, ground/parent support, `left_of`, and `near`.

The host independently validates schema and geometry. Missing objects, NaN,
extra identities and out-of-room objects score zero. Schema-valid, in-bounds output receives:

```
feasibility = 0.55 * supported_fraction + 0.45 * noncolliding_pair_fraction
score = feasibility * (0.6 + 0.4 * satisfied_relation_fraction)
```

Strict success additionally requires **all** constraints. Always report strict
success and components alongside the soft score: many noncolliding object pairs
can otherwise hide a small number of important collisions. The initial solver
is a simple grid/stack placement and intentionally has room for improvement.

Online scores use analytic AABB geometry, not a Blender render or a VLM judge.
`export.py` builds real meshes in Blender, exports GLB and PNG, then reimports
GLB and checks every mesh's world-space center and dimensions. This checks
serialization consistency, not whether all scene constraints were satisfied.
This task does not measure arbitrary asset generation, topology, texture quality,
part segmentation or realistic physical stability.

## Setup and offline checks

Python 3.10+; tested locally with Python 3.14 and Blender 4.5.11 LTS. Core task and
framework need only Python's standard library. `pytest` is a development extra.
Blender is needed only for export. Install/login to the official Codex CLI for
live experiments; these runs use the configured account and model.

PowerShell, from this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
New-Item -ItemType Directory -Force workspace | Out-Null
.\.venv\Scripts\python.exe -m pytest tests/test_codex.py tests/test_codex_json.py tests/test_blender_layout.py tests/test_blender_trial.py -q --basetemp workspace/pytest-3d
.\.venv\Scripts\python.exe tasks/blender_layout/evaluator.py --program tasks/blender_layout/init_program.py
codex login status
```

On Linux/macOS use `.venv/bin/python` for the same commands. The task itself is
portable; this fork's actual live pilot was run on Windows. Candidate subprocesses
are timeout-bounded and receive copied input, but are **not a hostile-code sandbox**.
Run untrusted or adversarial candidates in a separate secured environment.

## Bounded live comparison

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_blender_trial.py --agent codex_json --output workspace/blender-json-trial --agent-timeout 240
```

This explicitly caps each online round at two branches with two attempts per
branch, four evaluations, one worker, and 240 seconds per agent run. There are
no automatic policy-rewrite retries and no direction-memory LLM calls.

1. Run a shared E0 with the original policy (up to four agent episodes).
2. Copy that exact E0 to adaptive and fixed workspaces.
3. Adaptive performs one policy rewrite/replay/selection (L0). The incumbent may win.
4. Adaptive and fixed each run E1 from the same initial program, with the same E0
   history and online caps. Only adaptive may use the learned policy.
5. Select each round's champion using dev score, then evaluate those programs on
   heldout scenes. Verify that E1 actually used the selected policy's hash.

The maximum is **13 agent episodes**, not 13 underlying API requests. Each
episode can use many reasoning/tool/model turns. Tokens and elapsed time are
recorded; equal episode caps are not equal compute. Adaptive has the additional
learning cost. No second unused learning round is run.

Two coding backends are available. `--agent codex` uses the usual tool-equipped
CLI. `--agent codex_json` supplies the complete permitted context as text and
requests structured source-code output, then the host validates and writes only
the fixed candidate files. Both experiment arms use the same backend. The latter
is a separate experimental interface, not the paper's tool-equipped coding agent.
It was added after the local Windows sandbox denied the ordinary CLI's directory
access. No ACL, sandbox or network permission was disabled or weakened.

The JSON backend records context hashes, rejects truncated context, duplicate
JSON keys, malformed Python, unexpected fields and tool-use events. The official
`--output-schema` option constrains final output but does **not** disable tools;
the no-tool rule is a prompt/protocol condition, with rejection if violated.
Policy learning receives only the public policy API, method history and filtered
feedback, never raw trace/grid ceilings or trace-pool future outcomes. Directions
summarization is unsupported by this backend and is disabled in this pilot.

Separate `--phase e0`, `--phase learn`, `--phase e1`, `--phase report` commands
allow inspection between stages. Existing live rounds and learning phases are
not silently repeated. If a paid phase fails, inspect its logs before deciding
whether to start a fresh trial. `report` starts no model calls but executes the
selected candidate programs. Use a new output directory when changing task code.

`plan.json` freezes task hashes and caps before calls. `report.json` records dev,
heldout, selected/used policy identity, usage and failures. Raw logs stay in ignored
`workspace/`. The usage calculation counts shared E0 once and excludes copied
history. Missing usage means unknown cost, not zero. The backend keeps the existing
Codex model default unless `CODEX_MODEL` is explicitly set.

The Codex backend now uses `exec --sandbox workspace-write`, sends long prompts
through UTF-8 stdin, retains usage/failure logs, and cleans up its own process tree
on timeout (Windows Job Objects; POSIX process groups). It does not require the
obsolete `--full-auto` option.

## Export a selected layout

Use a `program` path from `report.json` and your local Blender executable:

```powershell
.\.venv\Scripts\python.exe tasks/blender_layout/export.py --program PATH_TO_PROGRAM --output workspace/preview --blender PATH_TO_BLENDER --split heldout --scene-index 0
```

Outputs: `layout.glb`, `layout.blend`, `layout.png`, `layout.json`, and
`reimport_check.json`. Inspect both analytic scores and the rendered scene.

## What this pilot can establish

A passing run establishes that task adaptation, candidate evaluation, policy
rewriting/replay/selection, next-round policy application, and 3D export work.
One trace and one rewrite cannot establish a reliable RSI advantage. Better
geometry after search alone also cannot establish one.

For an effectiveness study, use multiple independent shared E0 seeds, compare
fixed versus adaptive under matched **total** token/time budgets including L0,
evaluate learning curves and uncertainty, and test frozen learned policies on
new task families. Solver generalization to unseen room dimensions is a separate
question from policy transfer to new 3D tasks. Heldout source in this local repo
is readable; it is kept out of search feedback, not cryptographically hidden.
