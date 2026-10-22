# Framework audit — 2026-10-22

The question was whether a real, public 3D + LLM self-improvement framework could
serve as a runnable RSI baseline, or whether a general RSI framework could be
adapted. Repository availability and licenses were checked against the official
repositories, not inferred from paper titles.

| Candidate | Verified release and scope | Decision |
|---|---|---|
| [Google Research DREAM-RSI](https://github.com/google-research/dream-rsi) | Apache-2.0, actual online exploration/offline replay/policy rewriting/promotion code, pluggable task evaluator; pinned upstream `8ec35bd2538220517ce1bfca4282db21b8c518db` | Forked and adapted. Small core dependency footprint and explicit fixed-policy control. |
| [DGM](https://github.com/jennyzzt/dgm) | Public Apache-2.0 agent evolution implementation; tightly coupled to Docker, SWE-bench/Polyglot and coding evaluation | Real alternative, but substantially more migration work for this Windows/3D pilot. Not executed here. |
| [Skill-3D](https://github.com/skill-3d/Skill-3D) | Public Apache-2.0 implementation; skill accumulation for spatial understanding/QA, with substantial model/service dependencies | Relevant to persistent 3D skills, but does not directly implement Blender scene construction/repair. Not executed here. |
| [3D-Generalist](https://github.com/sunfanyunn/3D-Generalist) | Author repository exists, but GitHub API reported `size: 0`; contents returned `This repository is empty.` at audit time | No released executable framework could be validated there. |

Do not confuse [WecoAI/aideml](https://github.com/WecoAI/aideml) (AIDE) with a
verified full AIDE² implementation. The [AIDE² issue](https://github.com/WecoAI/aideml/issues/87)
linked the new paper/report; that alone did not establish a released matching
framework. Likewise, the selected Google Research release is different from
older placeholder repositories named Dream-RSI.

## Reliability conclusion

DREAM-RSI is concrete enough to adapt: a task needs a problem statement, initial
program, and an evaluator returning a score. That interface is easy to migrate.
The research validity does not come for free: 3D rewards, unchanged object
constraints, independent test instances, budget accounting and a fixed control
must be designed and checked.

The upstream system changes the exploration controller, while keeping the
coding agent and task evaluator fixed. Its offline replay objective estimates
the cost/benefit of exploring branches and refinements from previous traces.
This is a meaningful outer self-improvement loop, but not LLM weight training
and not unrestricted recursive modification of the entire system.

## Local integration findings

- The original Codex backend used an obsolete `--full-auto` flag. The fork uses
  the currently supported workspace-write interface and UTF-8 stdin prompts.
- Original POSIX-only process cleanup needed Windows support. Tests now check
  actual child/grandchild process termination on timeout.
- A successful Codex process exit does not imply a successful candidate: the
  first live probe returned four zero-exit responses saying workspace access
  was denied, with no proposal/program files. The trial rejected this trace.
- Windows sandbox probes reproduced the directory-access failure independently
  of the model. No ACL or sandbox permission setting was weakened.
- The task evaluator and actual Blender GLB roundtrip were verified separately.

See [the runnable guide](3D_BASELINE.md) for the bounded experiment and its
limitations. Passing an integration test, generating a better layout, selecting
a new policy, and proving a statistically reliable RSI gain are distinct claims.
