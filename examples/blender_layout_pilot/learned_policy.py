"""Prefix-only trajectory portfolio controller.

Successful anchors, consecutive-success gains, regressions, repair outcomes and
current failure episodes drive ranking. A fixed beta controls width, patience,
reserve admission and waiting. Batches protect successful refinements, represent
exploration and admit at most one actual repair; remaining useful slots are filled
by priority. Shallow branches and first implementation failures receive patience.
Default beta is 0.6: supplied history has no paired live manifests and informative
beta sweep supporting a causal default change. Grid planning explicitly retains
bounded fallback dimensions when history cannot establish width-versus-depth
returns. No hidden outcomes, score targets, filesystem access or ceiling tests.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import median

from dream_rsi.policy.api import (
    GridPlan, GridPlanningContext, LLMDesignedMethod, SimResult,
    _record_curve, finalize_result,
)


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def __init__(self, config=None):
        super().__init__(config)
        beta = float(self.config.get("beta", 0.6))
        self.beta = beta
        self.sch = self._schedule(beta)

    def _schedule(self, beta):
        b = min(1.0, max(0.0, beta)) if math.isfinite(beta) else 0.6
        return {
            "width_fraction": 0.25 + 0.75 * b,
            "minimum_successes": 2 + int(3 * b),
            "stagnation": 2 + int(6 * b),
            "hard_patience": 2 + int(2 * b),
            "repair_patience": 2 + int(5 * b),
            "repair_bonus_cap": 1 + int(2 * b),
            "lag_tolerance": 0.35 + 1.65 * b,
            "continue": 0.9 - 0.9 * b,
            "reserve": 0.55 - 1.15 * b,
            "wait_rounds": 3 + int(4 * b),
            "wait_weight": 0.15 + 0.25 * b,
            "anchor_weight": 1.3,
            "trend_weight": 0.65,
            "gain_weight": 0.55,
            "regression_weight": 0.35,
            "stale_weight": 0.16 - 0.1 * b,
            "failure_weight": 0.35 - 0.2 * b,
            "repair_weight": 0.3,
            "depth_weight": 0.25 + 0.25 * b,
            "uncertainty": 0.7 + 0.8 * b,
            "root_priority": 0.8 + 0.8 * b,
            "novelty_weight": 0.5,
            "root_success_weight": 0.5,
            "hard_priority": 0.15 + 0.3 * b,
            "recovery_margin": 0.4 - 0.25 * b,
            "history_min": 2,
        }

    @staticmethod
    def _number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _success(obs):
        return bool(obs.evaluated) and obs.error is None and obs.fail_class == "ok"

    @classmethod
    def _failure(cls, obs):
        if cls._success(obs):
            return "success"
        text = (str(obs.fail_class) + " " + str(obs.error or "")).lower()
        repair = (
            "mismatch", "incorrect", "correctness", "output", "shared memory",
            "shared_memory", "resource", "out of memory", "outofmemory",
            "variable", "nameerror", "syntax", "typeerror", "indexerror",
            "mask", "layout", "shape", "dimension", "stride", "assert",
            "illegal memory", "launch", "register", "code", "compile_other",
        )
        hard = (
            "modulenotfounderror", "no module named", "dependency",
            "driver unavailable", "no cuda device", "device unavailable",
            "permission denied", "compiler not found", "toolchain missing",
            "unsupported platform", "compile_env",
        )
        # Concrete implementation evidence takes precedence over coarse labels.
        if any(token in text for token in repair):
            return "repairable"
        if any(token in text for token in hard):
            return "hard"
        # Zero-valid and generic compilation/runtime failures are not proofs
        # that the direction is intrinsically unworkable.
        return "repairable"

    @staticmethod
    def _tags(meta):
        tags = getattr(meta, "tags", ()) or ()
        if isinstance(tags, dict):
            return frozenset(str(k) + "=" + str(v) for k, v in tags.items())
        if isinstance(tags, str):
            return frozenset((tags,))
        return frozenset(str(x) for x in tags)

    def _tie(self, cell, meta):
        # Identity only breaks ties; numerical branch order is not a prior.
        payload = repr((sorted(self._tags(meta)), str(cell)))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        width_cap = int(context.hard_max_branch_count)
        depth_cap = int(context.hard_max_refine_count)
        if context.trace_branch_count is not None:
            width_cap = min(width_cap, int(context.trace_branch_count))
        if context.trace_refine_count is not None:
            depth_cap = min(depth_cap, int(context.trace_refine_count))
        width = max(1, min(width_cap, int(context.fallback_branch_count)))
        depth = max(0, min(depth_cap, int(context.fallback_refine_count)))
        # Aggregate scores, work and grid sizes alone do not identify whether
        # gains came from distinct roots or late refinements. No manifest field
        # schema for that attribution is supplied, so do not invent one.
        if len(context.history) < self.sch["history_min"]:
            reason = "Insufficient completed-cycle history; retain fallback dimensions within hard caps and replay support."
        else:
            reason = "Completed history is available, but aggregate manifest statistics do not establish direction diversity or late-gain attribution; retain the bounded fallback grid."
        return GridPlan(branch_count=width, refine_count=depth, reason=reason)

    def _trajectory(self, entries, baseline, scale):
        s = self.sch
        observations = [obs for _, obs in entries]
        successes = []
        gains = []
        regressions = []
        anchor = None
        stale = 0
        repairs = 0
        episode = []
        previous = None
        for obs in observations:
            kind = self._failure(obs)
            score = self._number(obs.score) if kind == "success" else None
            if kind == "success":
                if episode and any(x == "repairable" for x in episode):
                    repairs += 1
                episode = []
                if score is not None:
                    if previous is not None and self._success(previous):
                        parent = self._number(previous.score)
                        delta = self._number(getattr(obs, "delta_vs_parent", None))
                        if delta is None and parent is not None:
                            delta = score - parent
                        if delta is not None:
                            gains.append(delta / scale)
                            regressions.append(max(0.0, -delta / scale))
                    if anchor is None or score > anchor:
                        anchor = score
                        stale = 0
                    else:
                        stale += 1
                    successes.append(score)
            else:
                episode.append(kind)
            previous = obs
        n = len(successes)
        trend = ((successes[-1] - successes[0]) / (scale * max(1, n - 1))) if n else 0.0
        last_gain = gains[-1] if gains and self._success(observations[-1]) else 0.0
        kind = self._failure(observations[-1])
        hard_count = 0
        for item in reversed(episode):
            if item != "hard":
                break
            hard_count += 1
        return {
            "anchor": anchor, "n": n, "trend": trend,
            "gain": last_gain, "regression": sum(regressions) / max(1, len(regressions)),
            "stale": stale, "repairs": repairs, "episode": len(episode),
            "hard_count": hard_count, "kind": kind,
            "depth": max(obs.attempt for obs in observations),
            "count": len(observations),
            "under": n < s["minimum_successes"],
        }

    def solve(self, question, budget=None):
        question.reset()
        result = SimResult()
        s = self.sch
        waiting = {}
        round_index = 0
        while True:
            prefix = question.observed()
            remaining_budget = None if budget is None else max(0, int(budget) - len(prefix))
            if remaining_budget == 0:
                break
            legal = list(dict.fromkeys(question.legal_actions()))
            if not legal:
                break
            workers = max(1, int(question.max_parallelism or 1))
            capacity = workers if remaining_budget is None else min(workers, remaining_budget)
            roots = set(question.legal_roots()).intersection(legal)
            metas = {cell: question.meta(cell) for cell in legal}
            grouped = defaultdict(list)
            used_tags = set()
            baseline = self._number(question.baseline_score)
            baseline = 0.0 if baseline is None else baseline
            scores = [baseline]
            for cell, obs in prefix.items():
                meta = question.meta(cell)
                grouped[meta.branch].append((meta, obs))
                used_tags.update(self._tags(meta))
                score = self._number(obs.score)
                if self._success(obs) and score is not None:
                    scores.append(score)
            # Numerical fallback only: all behavioral comparisons are normalized
            # by observed score dispersion and baseline-relative differences.
            differences = [abs(x - baseline) for x in scores if x != baseline]
            scale = max(max(scores) - min(scores), median(differences) if differences else 0.0)
            if scale == 0.0:
                scale = max(math.ulp(abs(baseline)), math.ulp(1.0))
            best = max(scores)
            trajectories = {}
            for branch, entries in grouped.items():
                entries.sort(key=lambda pair: (pair[0].attempt, pair[0].seq))
                trajectories[branch] = self._trajectory(entries, baseline, scale)
            successful_roots = sum(
                self._success(entries[0][1]) and entries[0][1].score > baseline
                for entries in grouped.values()
            )
            root_yield = successful_roots / max(1, len(grouped))
            available_width = len(grouped) + len(roots)
            width_fraction = min(1.0, s["width_fraction"] + (1.0 - s["width_fraction"]) * root_yield)
            width_limit = min(available_width, max(workers, math.ceil(available_width * width_fraction)))
            candidates = []
            decisions = {}
            # Recompute closures every round; no earlier failure can lock a
            # subsequently successful trajectory out of consideration.
            closed = set()
            for cell in legal:
                meta = metas[cell]
                waiting.setdefault(cell, round_index)
                age = round_index - waiting[cell]
                age_bonus = s["wait_weight"] * min(age, s["wait_rounds"]) / s["wait_rounds"]
                tags = self._tags(meta)
                novelty = len(tags - used_tags) / max(1, len(tags))
                if cell in roots:
                    priority = s["root_priority"] + s["novelty_weight"] * novelty + s["root_success_weight"] * root_yield
                    eligible = len(grouped) < width_limit
                    role = "explore"
                    mandatory = eligible
                    branch = meta.branch
                else:
                    branch = meta.branch
                    t = trajectories.get(branch)
                    if t is None:
                        decisions[cell] = "reserve: missing observed parent"
                        continue
                    anchor = baseline if t["anchor"] is None else t["anchor"]
                    lag = (best - anchor) / scale
                    # Only the legal next cell is known to remain. Do not
                    # fabricate a hidden maximum depth for an irregular branch.
                    remaining_known = int(meta.attempt > t["depth"])
                    depth_option = remaining_known / (t["count"] + remaining_known)
                    priority = (
                        s["anchor_weight"] * (anchor - baseline) / scale
                        + s["trend_weight"] * t["trend"]
                        + s["gain_weight"] * t["gain"]
                        - s["regression_weight"] * t["regression"]
                        - s["stale_weight"] * t["stale"]
                        + s["depth_weight"] * depth_option
                        + s["uncertainty"] / math.sqrt(t["n"] + 1)
                    )
                    if t["kind"] == "hard":
                        closed_now = t["hard_count"] >= s["hard_patience"]
                        role = "explore"
                        mandatory = not closed_now
                        eligible = not closed_now
                        priority += s["hard_priority"] - s["failure_weight"] * t["episode"]
                        classification = "hard-unrecoverable" if closed_now else "weak-but-underexplored environment retry"
                    elif t["kind"] == "repairable":
                        role = "recover"
                        allowance = s["repair_patience"] + min(t["repairs"], s["repair_bonus_cap"])
                        priority += s["repair_weight"] * min(t["repairs"], s["repair_bonus_cap"]) - s["failure_weight"] * t["episode"]
                        mandatory = t["episode"] == 1
                        # Repeated failures lower relative value, rather than
                        # deleting the historical anchor or closing permanently.
                        eligible = mandatory or (t["episode"] <= allowance and priority + age_bonus >= s["reserve"])
                        closed_now = False
                        classification = "repairable implementation failure"
                    else:
                        closed_now = (not t["under"] and t["stale"] >= s["stagnation"] and t["gain"] <= 0 and lag > s["lag_tolerance"])
                        role = "explore" if t["under"] else "normal"
                        mandatory = t["under"]
                        eligible = not closed_now and (mandatory or priority + age_bonus >= s["continue"] or (t["stale"] < s["stagnation"] and priority + age_bonus >= s["reserve"]))
                        classification = "repeatedly unpromising" if closed_now else ("weak-but-underexplored" if t["under"] else "successful refinement")
                    if closed_now:
                        closed.add(branch)
                        decisions[cell] = "close: " + classification
                        continue
                priority += age_bonus
                if not eligible:
                    decisions[cell] = "reserve: insufficient marginal prefix evidence"
                    continue
                decisions[cell] = "continue"
                candidates.append({"cell": cell, "branch": branch, "role": role, "priority": priority, "mandatory": mandatory, "age": age, "tie": self._tie(cell, meta)})
            candidates.sort(key=lambda c: (-c["priority"], c["tie"]))
            if not candidates:
                # Every legal action was explicitly classified above. Reserved
                # actions cannot acquire new evidence without an eligible probe.
                break
            batch = []
            branches = set()
            recovery_used = False

            def take(candidate):
                nonlocal recovery_used
                if candidate is None or len(batch) >= capacity or candidate["branch"] in branches:
                    return False
                if candidate["role"] == "recover" and recovery_used:
                    return False
                batch.append(candidate["cell"])
                branches.add(candidate["branch"])
                recovery_used |= candidate["role"] == "recover"
                return True

            normal = [c for c in candidates if c["role"] == "normal"]
            exploration = [c for c in candidates if c["role"] == "explore"]
            recovery = [c for c in candidates if c["role"] == "recover"]
            if capacity == 1:
                # Serial hardware still rotates overdue eligible exploration
                # and repairs; one promising branch cannot starve the portfolio.
                overdue = [c for c in candidates if c["age"] >= s["wait_rounds"]]
                overdue.sort(key=lambda c: (-c["age"], -c["priority"], c["tie"]))
                take(overdue[0] if overdue else candidates[0])
            else:
                if normal:
                    take(normal[0])
                if exploration:
                    take(exploration[0])
                # Recovery never evicts a normal successful refinement. It
                # competes for capacity left after those refinements and the
                # exploration representative have been accounted for.
                unselected_normals = [c for c in normal if c["branch"] not in branches]
                if recovery and capacity - len(batch) > len(unselected_normals):
                    r = recovery[0]
                    alternatives = [c for c in candidates if c["role"] != "recover" and c["branch"] not in branches]
                    justified = r["mandatory"] or r["age"] >= s["wait_rounds"] or not alternatives or r["priority"] + s["recovery_margin"] >= alternatives[0]["priority"]
                    if justified:
                        take(r)
                for candidate in normal:
                    take(candidate)
                for candidate in candidates:
                    take(candidate)
            if not batch:
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(result, question))
            for cell in batch:
                waiting.pop(cell, None)
            round_index += 1
        return finalize_result(question, result)
