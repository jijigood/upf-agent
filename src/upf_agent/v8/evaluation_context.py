"""Construct and validate one evaluation context at each public entry point."""

from upf_agent.v8.models import EvaluationContext


def resolve_context(task, contract, batch, now=None, context=None):
    effective_time = now if now is not None else context.as_of if context else task.as_of
    expected = EvaluationContext.from_task(task, contract, batch, effective_time)
    if context is not None and context != expected:
        raise ValueError("evaluation context does not match task, contract, batch or time")
    return expected


def certify(value, context, *, computed=False):
    """Attach current certification while preserving the computation's origin."""
    value.batch_id = context.batch_id
    value.evaluation_as_of = context.as_of
    value.context_fingerprint = context.fingerprint
    if computed:
        value.computed_batch_id = context.batch_id
    return value


def bind_snapshot(snapshot, context):
    """Eligibility caches are reusable within one certified snapshot context only."""
    prior = getattr(snapshot, "evaluation_fingerprint", None)
    if prior is not None and prior != context.fingerprint:
        raise ValueError("EvidenceSnapshot cannot be reused across evaluation contexts")
    snapshot.evaluation_fingerprint = context.fingerprint
