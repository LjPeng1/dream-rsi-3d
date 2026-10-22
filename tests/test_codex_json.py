"""Offline structured-output tests; no Codex/model request is made."""

import json
from pathlib import Path

import pytest

from dream_rsi.agents import AGENTS, load_agent
from dream_rsi.agents.base import AgentResult
from dream_rsi.agents.codex_json import CodexJsonAgent, _reject_tool_use, _write_outputs
from dream_rsi.policy.constants import METHOD_REL


class FakeCodex:
    def __init__(self, response, *, status="ok", tool=False):
        self.extra_args = ["--config", "model_reasoning_effort=low"]
        self.plain_exec = False
        self.response = response
        self.status = status
        self.tool = tool
        self.calls = []

    def run(self, **kwargs):
        self.calls.append({**kwargs, "extra_args": list(self.extra_args)})
        log = kwargs["log_dir"]
        (log / "codex_last_message.txt").write_text(self.response, encoding="utf-8")
        (log / "usage.json").write_text('{"available":true,"total":{"input_tokens":10}}')
        event = {"type": "item.completed", "item": {"type": "command_execution" if self.tool else "agent_message"}}
        (log / "stdout.txt").write_text(json.dumps(event))
        (log / "prompt.txt").write_text(kwargs["prompt"], encoding="utf-8")
        return AgentResult(status=self.status, message="fake")


def put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def setup_explore(root):
    put(root, "problem.txt", "Pack 3D boxes")
    put(root, "baseline/proposal.md", "BASELINE_PROPOSAL")
    put(root, "baseline/artifact/program.py", "BASELINE_CODE = 1")
    put(root, "baseline/eval/score.json", '{"baseline":0.3}')
    put(root, "history/iter000/explore_b0/attempt_00/proposal.md", "HISTORY_PROPOSAL")
    put(root, "history/iter000/explore_b0/attempt_00/artifact/program.py", "HISTORY_CODE = 1")
    put(root, "attempt_00/proposal.md", "SIBLING_PROPOSAL")
    put(root, "attempt_00/error.txt", "SIBLING_ERROR")
    put(root, "attempt_01/proposal.md", "OWN_OLD")
    put(root, "attempt_01/artifact/program.py", "OLD_CODE = 1")
    put(root, "attempt_00/agent_log/proposal.md", "SECRET_LOG")
    put(root, "history/iter000/raw_scores.json", "SECRET_RAW")
    put(root, "trace_pool/iter000/raw_scores.json", "SECRET_POOL")
    return root / "attempt_01"


def run_explore(root, fake, **options):
    node = setup_explore(root)
    agent = CodexJsonAgent(codex=fake, **options)
    result = agent.run(prompt="Solve the task", workspace=root, run_cwd=root, node_dir=node,
                       log_dir=node / "agent_log/codex_json", guide={}, stage="explore")
    return result, node


def test_optional_backend_registered():
    assert "codex" in AGENTS and "codex_json" in AGENTS
    assert isinstance(load_agent("codex_json"), CodexJsonAgent)
    assert load_agent("codex").name == "codex"


def test_explore_full_allowed_context_and_fixed_outputs(tmp_path):
    fake = FakeCodex(json.dumps({"proposal": "New proposal", "program": "answer = 42\n"}))
    result, node = run_explore(tmp_path, fake)
    assert result.status == "ok"
    prompt = fake.calls[0]["prompt"]
    for word in ("BASELINE_PROPOSAL", "BASELINE_CODE", "HISTORY_PROPOSAL", "HISTORY_CODE", "SIBLING_PROPOSAL", "SIBLING_ERROR"):
        assert word in prompt
    for word in ("OWN_OLD", "OLD_CODE", "SECRET_LOG", "SECRET_RAW", "SECRET_POOL"):
        assert word not in prompt
    assert (node / "proposal.md").read_text() == "New proposal"
    assert (node / "artifact/program.py").read_text() == "answer = 42\n"
    assert "--output-schema" in fake.calls[0]["extra_args"]
    assert "--output-schema" not in fake.extra_args  # no shared-instance mutation
    assert fake.calls[0]["run_cwd"] == tmp_path
    manifest = json.loads((node / "agent_log/codex_json/context_manifest.json").read_text())
    assert manifest["truncated"] is False
    assert all(len(row["source_sha256"]) == 64 and len(row["sent_sha256"]) == 64 for row in manifest["files"])
    assert json.loads((node / "agent_log/codex_json/usage.json").read_text())["total"]["input_tokens"] == 10


@pytest.mark.parametrize("response", [
    '```json\n{"proposal":"x","program":"pass"}\n```',
    '{"proposal":"x","proposal":"y","program":"pass"}',
    '{"proposal":"x","program":"pass","path":"../outside.py"}',
    '{"proposal":"x","program":"def broken("}',
    '{"proposal":"x","program":"return 1"}',
    '{"proposal":"x","program":null}',
    '{"proposal":"","program":"pass"}',
])
def test_invalid_response_preserves_both_existing_files(tmp_path, response):
    result, node = run_explore(tmp_path, FakeCodex(response))
    assert result.status == "failed"
    assert (node / "proposal.md").read_text() == "OWN_OLD"
    assert (node / "artifact/program.py").read_text() == "OLD_CODE = 1"


@pytest.mark.parametrize("options", [{"status": "failed"}, {"tool": True}])
def test_failed_or_tool_using_generation_never_commits(tmp_path, options):
    fake = FakeCodex('{"proposal":"changed","program":"pass"}', **options)
    result, node = run_explore(tmp_path, fake)
    assert result.status == "failed"
    assert (node / "proposal.md").read_text() == "OWN_OLD"


def test_oversize_context_fails_without_truncation_or_model_call(tmp_path):
    fake = FakeCodex("unused")
    result, node = run_explore(tmp_path, fake, max_context_bytes=50)
    assert result.status == "failed" and "no truncation" in result.message
    assert fake.calls == []
    manifest = json.loads((node / "agent_log/codex_json/context_manifest.json").read_text())
    assert manifest["prompt_bytes"] > 50
    assert (node / "proposal.md").read_text() == "OWN_OLD"


def setup_learning(root, **manifest_fields):
    manifest = {"method_file": METHOD_REL, "baseline_preset": {"max_width": 2, "max_refine": 1}, **manifest_fields}
    put(root, "manifest.json", json.dumps(manifest))
    put(root, METHOD_REL, "class OptimalPolicy: pass\n")
    put(root, "dream_rsi/policy/api.py", "API_SOURCE = 1\n")
    put(root, "heuristic_history/r0000/dream_rsi/policy/discovered/method.py", "PREVIOUS_POLICY = 1\n")
    put(root, "heuristic_history/r0000/proposal_results/beta_sweep.json", json.dumps({
        "pareto": {"reward": 0.4, "auc": 0.5, "parallel_penalty": 0.1, "target": 987654.0},
        "grid": [{"beta": 0.5, "per_trace": [{"best": 0.3, "run_best": 987655.0, "target": 987656.0}]}],
        "raw_scores": {"all": "SECRET_FEEDBACK"},
    }))
    put(root, "heuristic_results/heuristic_eval.json", '{"status":"ok","run_best":987657,"heuristic":{"best":0.3,"total_budget":2,"reached_target":true}}')
    put(root, "trace/raw_scores.json", "SECRET_FROZEN_TRACE")
    put(root, "trace_pool/iter000/raw_scores.json", "SECRET_TRACE_POOL")
    put(root, "heuristic_history/r0000/proposal_results/policy_execution_traces.jsonl", "SECRET_EXECUTION_DUMP")
    put(root, "logs/elsewhere/beta_sweep.json", "SECRET_LOG")


def test_learning_projects_feedback_without_hidden_future_scores(tmp_path):
    setup_learning(tmp_path)
    fake = FakeCodex(json.dumps({"method": "class OptimalPolicy:\n    NAME = 'new'\n"}))
    result = CodexJsonAgent(codex=fake).run(prompt="Improve prefix-only policy", workspace=tmp_path,
        node_dir=tmp_path / "logs/r0001", log_dir=tmp_path / "logs/r0001", guide={}, stage="learning")
    assert result.status == "ok"
    prompt = fake.calls[0]["prompt"]
    for word in ("API_SOURCE", "PREVIOUS_POLICY", '"reward"', '"max_width"'):
        # Context is a JSON string array, so quotes are escaped within content.
        assert word.replace('"', '\\"') in prompt
    for word in ("SECRET_", "987654", "987655", "987656", "987657"):
        assert word not in prompt
    assert (tmp_path / METHOD_REL).read_text() == "class OptimalPolicy:\n    NAME = 'new'\n"
    assert not (tmp_path / "logs/r0001/proposal.md").exists()
    inventory = json.loads((tmp_path / "logs/r0001/context_manifest.json").read_text())["files"]
    assert not any(row["path"].startswith(("trace/", "trace_pool/", "logs/")) for row in inventory)


@pytest.mark.parametrize("manifest", [
    {"method_file": "../outside.py"}, {"history_dir": "trace_pool"}, {"results_dir": "../outside"},
])
def test_learning_rejects_manifest_path_redirection(tmp_path, manifest):
    setup_learning(tmp_path, **manifest)
    fake = FakeCodex("unused")
    result = CodexJsonAgent(codex=fake).run(prompt="x", workspace=tmp_path,
        node_dir=tmp_path / "logs/r1", log_dir=tmp_path / "logs/r1", guide={}, stage="learning")
    assert result.status == "failed"
    assert not fake.calls


def test_output_transaction_rolls_back_first_file_if_second_replace_fails(tmp_path, monkeypatch):
    first = put(tmp_path, "proposal.md", "old proposal")
    second = put(tmp_path, "artifact/program.py", "old code")
    import dream_rsi.agents.codex_json as module
    original = module.os.replace

    def replace(src, dst):
        if Path(dst) == second:
            raise OSError("simulated write failure")
        return original(src, dst)

    monkeypatch.setattr(module.os, "replace", replace)
    with pytest.raises(OSError, match="simulated"):
        _write_outputs({first: "new proposal", second: "new code"}, tmp_path)
    assert first.read_text() == "old proposal"
    assert second.read_text() == "old code"
    assert not list(tmp_path.rglob(".codex-json-*"))


_WARNING_MESSAGE = (
    "Under-development features enabled: respect_system_proxy. "
    "Under-development features are incomplete and may behave unpredictably. "
    "To suppress this warning, set `suppress_unstable_features_warning = true` "
    "in C:\\Users\\test\\.codex\\config.toml."
)


def test_exact_pre_turn_configuration_warning_is_not_a_tool_call(tmp_path):
    events = [
        {"type": "thread.started", "thread_id": "test"},
        {"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": _WARNING_MESSAGE}},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ]
    put(tmp_path, "stdout.txt", "\n".join(map(json.dumps, events)))
    _reject_tool_use(tmp_path)


@pytest.mark.parametrize("after_turn,item_type,message", [
    (True, "error", _WARNING_MESSAGE),
    (False, "command_execution", _WARNING_MESSAGE),
    (False, "error", "shell failed: Access denied"),
    (False, "error", _WARNING_MESSAGE + "\nTool failed"),
])
def test_startup_warning_exception_never_allows_tool_errors(tmp_path, after_turn, item_type, message):
    events = [{"type": "thread.started"}]
    if after_turn:
        events.append({"type": "turn.started"})
    events.append({"type": "item.completed", "item": {"type": item_type, "message": message}})
    put(tmp_path, "stdout.txt", "\n".join(map(json.dumps, events)))
    with pytest.raises(ValueError, match="zero tool use"):
        _reject_tool_use(tmp_path)
