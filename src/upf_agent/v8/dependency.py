"""Tracked program reads, reachable invalidation, local waves and safe fallback."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from time import perf_counter

import networkx as nx

from upf_agent.v8.actions import (
    _gate,
    action_templates,
    full_actions,
    impact_scope,
    propagation_graph,
)
from upf_agent.v8.compiler import slot
from upf_agent.v8.evaluation_context import certify, resolve_context
from upf_agent.v8.evaluator import (
    EvidenceSnapshot,
    TrackedStates,
    evaluate_requirement,
    full_evaluate,
)
from upf_agent.v8.models import EvidenceBinding, EvidenceContract, OrchestrationTask, digest


class DependencyIndex:
    def __init__(self, task: OrchestrationTask, contract: EvidenceContract, actions=None):
        self.graph = nx.DiGraph()
        self.requirements = {r.requirement_id: r for r in contract.requirements}
        self.requirement_count = len(contract.requirements)
        self.actions = {
            a.action_id: a
            for a in (actions if actions is not None else action_templates(task, contract))
        }
        self.context = digest({"task": task, "contract": contract})
        for req in contract.requirements:
            target = f"state:{req.requirement_id}"
            self.graph.add_node(target)
            for dep in req.upstream:
                self.graph.add_edge(f"state:{dep}", target, relation="state_state")
            if req.kind in {"K", "R", "P"}:
                for obj in req.objects:
                    for field in [f"__record__:{req.requirement_id}", *req.fields]:
                        self.graph.add_edge(
                            slot(obj, req.kind, field), target, relation="field_state"
                        )
        propagation = propagation_graph(task)
        self.propagation = propagation
        for action in self.actions.values():
            objects, _ = impact_scope(action, task, propagation)
            for req in contract.requirements:
                if set(req.objects) & set(objects) or req.requirement_id in action.required_states:
                    self.graph.add_edge(
                        f"state:{req.requirement_id}", action.action_id, relation="state_action"
                    )
                    if req.kind in {"K", "R", "P"}:
                        for obj in req.objects:
                            for field in [f"__record__:{req.requirement_id}", *req.fields]:
                                self.graph.add_edge(
                                    slot(obj, req.kind, field),
                                    action.action_id,
                                    relation="field_action",
                                )
            for pred in action.predicates:
                for field in [pred.field]:
                    self.graph.add_edge(
                        slot(pred.object_id, pred.category, field),
                        action.action_id,
                        relation="field_action",
                    )
                for req in contract.requirements:
                    if req.kind == pred.category and pred.object_id in req.objects:
                        for field in [f"__record__:{req.requirement_id}", *req.fields]:
                            self.graph.add_edge(
                                slot(pred.object_id, pred.category, field),
                                action.action_id,
                                relation="field_action",
                            )
        self.expected_edges = frozenset(self.graph.edges)
        self.actual_edges = frozenset()

    def certify_reads(self, read_sets):
        """Certify independently observed Full reads, including unscheduled readers."""
        self.actual_edges |= frozenset(
            (source, target) for target, sources in read_sets.items() for source in sources
        )
        missing = sorted(self.actual_edges - set(self.graph.edges))
        return {
            "reader_count": len(read_sets),
            "observed_edges": len(self.actual_edges),
            "missing_edges": missing,
            "trusted": self.audit(),
        }

    def audit(self) -> bool:
        return (
            self.expected_edges <= set(self.graph.edges)
            and self.actual_edges <= set(self.graph.edges)
            and len(self.requirements) == self.requirement_count
            and all(
                parent in self.requirements
                for req in self.requirements.values()
                for parent in req.upstream
            )
        )

    def check_reads(self, target: str, reads: set[str]) -> bool:
        return all(self.graph.has_edge(source, target) for source in reads)


class LocalEvaluator:
    def __init__(
        self,
        task: OrchestrationTask,
        contract: EvidenceContract,
        bindings: list[EvidenceBinding],
        batch: str = "initial",
        actions=None,
        *,
        cutoff: bool = True,
        audit_reads: bool = True,
    ):
        started = perf_counter()
        self.cutoff, self.audit_reads = cutoff, audit_reads
        self.task, self.contract = task, contract
        self.index = DependencyIndex(task, contract, actions)
        built = perf_counter()
        self.snapshot = EvidenceSnapshot(bindings)
        fingerprint_started = perf_counter()
        # Capture the previous batch before a caller can mutate its input list or records.
        _ = self.snapshot.fingerprints
        fingerprint_seconds = perf_counter() - fingerprint_started
        rule_started = perf_counter()
        self.context = resolve_context(task, contract, batch)
        reads = {}
        self.states = full_evaluate(
            task, contract, bindings, batch, context=self.context, read_sets=reads
        )
        self.actions = full_actions(
            task,
            contract,
            bindings,
            self.states,
            batch,
            list(self.index.actions.values()),
            context=self.context,
            read_sets=reads,
        )
        rule_seconds = perf_counter() - rule_started
        audit_started = perf_counter()
        self.initial_audit = self.index.certify_reads(reads)
        audit_seconds = perf_counter() - audit_started
        self.batch = batch
        self.now = task.as_of
        self._interrupted = False
        self.initial_cost = {
            "index_seconds": built - started,
            "fingerprint_seconds": fingerprint_seconds,
            "initial_evaluation_seconds": rule_seconds,
            "initial_read_audit_seconds": audit_seconds,
            "total_seconds": perf_counter() - started,
        }

    def _check_reads(self, node, reads):
        if not self.audit_reads:
            return True
        started = perf_counter()
        trusted = self.index.check_reads(node, reads)
        self._read_audit_seconds += perf_counter() - started
        self._read_audit_count += 1
        return trusted

    def _full(self, bindings, batch, now, reason):
        context = resolve_context(self.task, self.contract, batch, now)
        reads = {}
        self.states = full_evaluate(
            self.task, self.contract, bindings, batch, now, context=context, read_sets=reads
        )
        self.actions = full_actions(
            self.task,
            self.contract,
            bindings,
            self.states,
            batch,
            list(self.index.actions.values()),
            context=context,
            read_sets=reads,
        )
        return {
            "mode": "full_fallback",
            "reason": reason,
            "recomputed_states": len(self.states),
            "recomputed_actions": len(self.actions),
            "read_audit": self.index.certify_reads(reads),
        }

    def update(
        self,
        bindings: list[EvidenceBinding],
        batch: str,
        now: datetime | None = None,
        max_recomputations: int | None = None,
    ) -> dict:
        started = perf_counter()
        self._read_audit_seconds = 0.0
        self._read_audit_count = 0
        self.context = resolve_context(self.task, self.contract, batch, now)
        now = self.context.as_of
        fresh = EvidenceSnapshot(bindings)
        old = self.snapshot
        fingerprint_started = perf_counter()
        changed = {
            s
            for s in old.fingerprints.keys() | fresh.fingerprints.keys()
            if old.fingerprints.get(s) != fresh.fingerprints.get(s)
        }
        fingerprint_seconds = perf_counter() - fingerprint_started
        audit_started = perf_counter()
        index_trusted = self.index.audit()
        index_audit_seconds = perf_counter() - audit_started
        recompute_started = perf_counter()
        context = digest({"task": self.task, "contract": self.contract})
        if context != self.index.context:
            self.index = DependencyIndex(
                self.task, self.contract, list(self.index.actions.values())
            )
            result = self._full(bindings, batch, now, "context_changed")
        elif not index_trusted:
            result = self._full(bindings, batch, now, "incomplete_read_index")
        elif not nx.is_directed_acyclic_graph(self.index.graph):
            # Conservative bounded-cycle result: no reusable satisfaction remains.
            result = self._full(bindings, batch, now, "dependency_cycle")
            prior = semantic_results(self.states, self.actions)
            converged = False
            for iteration in range(1, 4):
                result["cycle_iterations"] = iteration
                self._full(bindings, batch, now, "dependency_cycle")
                current = semantic_results(self.states, self.actions)
                if current == prior:
                    converged = True
                    break
                prior = current
            result.update(cycle_iterations=iteration, cycle_converged=converged)
            for s in self.states.values():
                s.pending = True
            self.actions = full_actions(
                self.task,
                self.contract,
                bindings,
                self.states,
                batch,
                list(self.index.actions.values()),
                context=self.context,
            )
        elif self._interrupted and max_recomputations is not None:
            # Repeated bounded retries cannot certify unfinished actions just because no
            # new fingerprints changed. An unbounded recovery will refresh the whole batch.
            for state in self.states.values():
                state.pending = True
            result = {
                "mode": "local",
                "reason": "recompute_budget_exhausted",
                "recomputed_states": 0,
                "recomputed_actions": 0,
            }
        elif (
            self._interrupted or any(s.pending for s in self.states.values())
        ) and max_recomputations is None:
            # A previous interrupted batch already advanced the snapshot. Replaying the same
            # bindings must still settle its pending readers before suggestions can resume.
            result = self._full(bindings, batch, now, "pending_recovery")
        else:
            if now != self.now:
                # Time is an input even without a new evidence revision.
                changed.update(
                    slot(o, r.kind, f"__record__:{r.requirement_id}")
                    for r in self.contract.requirements
                    if r.kind in {"K", "R", "P"}
                    for o in r.objects
                )
            affected = set()
            for source in changed:
                if source in self.index.graph:
                    affected.update(nx.descendants(self.index.graph, source))
            prior = {rid: state.state_version for rid, state in self.states.items()}
            for rid, state in self.states.items():
                certify(state, self.context)
                if f"state:{rid}" in affected:
                    state.pending = True
            # Topological waves process direct readers only after all upstream readers settle.
            queue = deque(n for n in nx.topological_sort(self.index.graph) if n in affected)
            computed, reused, action_count = 0, 0, 0
            changed_outputs = set(changed)
            result = {"mode": "local", "reason": None}
            while queue:
                node = queue.popleft()
                if max_recomputations is not None and computed + action_count >= max_recomputations:
                    result["reason"] = "recompute_budget_exhausted"
                    break
                readers = set(self.index.graph.predecessors(node))
                needs = not self.cutoff or bool(readers & changed_outputs)
                if node.startswith("state:"):
                    rid = node.removeprefix("state:")
                    if not needs:
                        # Every declared input is unchanged and settled: reuse is safe.
                        if any(
                            self.states[n.removeprefix("state:")].pending
                            for n in readers
                            if n.startswith("state:")
                        ):
                            continue
                        self.states[rid].pending = False
                        reused += 1
                        continue
                    fresh.reads.clear()
                    state = evaluate_requirement(
                        self.index.requirements[rid],
                        self.task,
                        self.contract,
                        fresh,
                        TrackedStates(self.states, fresh.reads),
                        batch,
                        now,
                        self.context,
                    )
                    if not self._check_reads(node, fresh.reads):
                        result = self._full(bindings, batch, now, "undeclared_actual_read")
                        break
                    self.states[rid] = state
                    computed += 1
                    if prior[rid] != state.state_version:
                        changed_outputs.add(node)
                elif node in self.index.actions:
                    if needs:
                        fresh.reads.clear()
                        # The public update entry validated this batch's context once.
                        # Full uses the same internal rule; rehashing the complete contract
                        # for every action adds work without changing its input context.
                        self.actions[node] = _gate(
                            self.index.actions[node],
                            self.task,
                            self.contract,
                            TrackedStates(self.states, fresh.reads),
                            fresh,
                            batch,
                            self.index.propagation,
                            self.context,
                        )
                        if not self._check_reads(node, fresh.reads):
                            result = self._full(bindings, batch, now, "undeclared_action_read")
                            break
                        action_count += 1
                    else:
                        self.actions[node].batch_id = batch
            if result["mode"] == "local":
                result.update(
                    recomputed_states=computed,
                    recomputed_actions=action_count,
                    reused_states=reused,
                    invalidated_readers=len(affected),
                )
                if any(s.pending for s in self.states.values()):
                    self.actions = full_actions(
                        self.task,
                        self.contract,
                        bindings,
                        self.states,
                        batch,
                        list(self.index.actions.values()),
                        context=self.context,
                    )
        recompute_seconds = perf_counter() - recompute_started - self._read_audit_seconds
        if result.get("reason") == "recompute_budget_exhausted":
            for decision in self.actions.values():
                decision.outcome = "human_review"
                decision.reasons.append("recompute_budget_exhausted")
        # Readers outside the affected subgraph still need current-batch certification.
        # Their computation origin remains unchanged and their inputs were verified stable.
        for value in [*self.states.values(), *self.actions.values()]:
            certify(value, self.context)
        self.snapshot, self.batch, self.now = fresh, batch, now
        self._interrupted = result.get("reason") == "recompute_budget_exhausted"
        result.update(
            changed_slots=sorted(changed),
            batch_id=batch,
            duration_seconds=perf_counter() - started,
            pending_states=[r for r, s in self.states.items() if s.pending],
            evaluation_context=self.context.model_dump(mode="json"),
            cutoff_enabled=self.cutoff,
            observed_read_audit_enabled=self.audit_reads,
            cost={
                "fingerprint_seconds": fingerprint_seconds,
                "index_integrity_audit_seconds": index_audit_seconds,
                "observed_read_audit_seconds": self._read_audit_seconds,
                "observed_read_audits": self._read_audit_count,
                "propagation_recompute_seconds": recompute_seconds,
                "fallback_seconds": recompute_seconds if result["mode"] == "full_fallback" else 0,
            },
        )
        return result


def semantic_results(states, actions) -> dict:
    return {
        "states": {k: v.semantic() for k, v in sorted(states.items())},
        "actions": {
            k: v.model_dump(
                mode="json",
                exclude={
                    "batch_id",
                    "computed_batch_id",
                    "evaluation_as_of",
                    "context_fingerprint",
                },
            )
            for k, v in sorted(actions.items())
        },
    }


def complete_results(states, actions):
    """Compare published certification as well as semantics; computation origin may differ."""
    return {
        "states": {
            k: v.model_dump(mode="json", exclude={"computed_batch_id"})
            for k, v in sorted(states.items())
        },
        "actions": {
            k: v.model_dump(mode="json", exclude={"computed_batch_id"})
            for k, v in sorted(actions.items())
        },
    }
