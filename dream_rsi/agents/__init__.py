# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Coding agent backends for DREAM-RSI exploration."""

from dream_rsi.agents.base import Agent, AgentResult
from dream_rsi.agents.gemini import GeminiAgent
from dream_rsi.agents.codex import CodexAgent
from dream_rsi.agents.codex_json import CodexJsonAgent
from dream_rsi.agents.claude import ClaudeAgent

AGENTS = ("gemini", "codex", "codex_json", "claude")


def load_agent(name: str, *, explore_shell: bool | None = None) -> Agent:
    """Build an agent from the environment.

    ``explore_shell`` overrides GEMINI_EXPLORE_SHELL for the Gemini agent.  The
    Claude and Codex agents always have a shell, so they ignore it.
    """
    key = name.strip().lower()
    if key == "gemini":
        from dream_rsi.agents.gemini import from_env

        agent = from_env()
        if explore_shell is not None:
            agent.explore_shell = explore_shell
        return agent
    if key == "codex":
        from dream_rsi.agents.codex import from_env

        return from_env()
    if key == "codex_json":
        from dream_rsi.agents.codex_json import from_env

        return from_env()
    if key == "claude":
        from dream_rsi.agents.claude import from_env

        return from_env()
    raise ValueError(f"Unknown agent {name!r}. Choose: {', '.join(AGENTS)}")


__all__ = [
    "Agent",
    "AgentResult",
    "GeminiAgent",
    "CodexAgent",
    "CodexJsonAgent",
    "ClaudeAgent",
    "load_agent",
    "AGENTS",
]
