"""v8 runners use the same source APIs as P5, with explicit no-graph dispatch skipping."""

from upf_agent.agent_contracts import AgentTask, ToolCall
from upf_agent.contracts import TaskType
from upf_agent.v8.engine import OrchestrationEngine
from upf_agent.v8.models import OrchestrationData, QueryPlan, QueryStep, digest
from upf_agent.v8.store import BudgetStop


class CommonToolV8Engine(OrchestrationEngine):
    def __init__(self, root, tools):
        super().__init__(root)
        self.tools = tools

    def _compile(self, state):
        compiled = super()._compile(state)
        data = OrchestrationData.model_validate(compiled["payload"])
        self.tools.task, self.tools.contract = data.task, data.contract
        steps = []
        generation = digest([data.task.observation_files, "public-synthetic-generation"])[:20]
        for obj in data.task.objects:
            for kind, name, args in [
                ("K", "search_manual", {"query": data.task.question}),
                ("R", "analyze_log_window", {"log_id": generation, "device_id": obj}),
                ("P", "lookup_topology", {"topology_id": generation, "device_id": obj}),
            ]:
                ids = [
                    r.requirement_id
                    for r in data.contract.requirements
                    if r.kind == kind and obj in r.objects
                ]
                if ids:
                    steps.append(
                        QueryStep(
                            step_id=f"public:{kind}:{obj}",
                            requirement_ids=ids,
                            tool=name,
                            object_id=obj,
                            arguments=args,
                            capability="common-public-synthetic",
                        )
                    )
        data.query_plan = QueryPlan(
            contract_id=data.contract.contract_id, steps=steps, gaps=data.contract.gaps
        )
        return {"payload": data.model_dump(mode="json")}

    def _collect(self, state):
        data = OrchestrationData.model_validate(state["payload"])
        if data.stop_reason:
            return state
        generation = digest([data.task.observation_files, "public-synthetic-generation"])[:20]
        scoped = AgentTask(
            run_id=data.task.run_id,
            question=data.task.question,
            task_type=TaskType.DIAGNOSE,
            device_ids=[],
            product=data.task.product,
            version=data.task.version,
            log_generation=generation,
            topology_generation=generation,
            manual_generation=generation,
        )
        try:
            for step in data.query_plan.steps:
                request = step.model_dump(mode="json")
                key = digest(request)
                result = self.ledger.charge(data.task, key, "tool", request)
                if result is None:
                    packet = self.tools.tool(
                        # The legacy validator only authorizes device-scoped reads when
                        # device_ids or earlier native evidence establish that scope.
                        # Each compiled query already targets one authorized object;
                        # use its single-device task rather than an empty legacy scope.
                        scoped.model_copy(update={"device_ids": [step.object_id]}),
                        ToolCall(name=step.tool, arguments=step.arguments),
                        [],
                        self.tools.budget.remaining(),
                    )
                    selected = [
                        b
                        for b in self.tools.bindings.values()
                        if b.requirement_id in step.requirement_ids
                    ]
                    result = {
                        "bindings": [b.model_dump(mode="json") for b in selected],
                        "gaps": packet.gaps,
                    }
                    self.ledger.commit_receipt(data.task, key, result, selected, 0)
                data.query_plan.gaps.extend(result.get("gaps", []))
            data.bindings = self.ledger.evidence(data.task.run_id)
        except (TimeoutError, BudgetStop) as error:
            data.stop_reason = str(error)
        data.status = "collected"
        return {"payload": data.model_dump(mode="json")}

    def _graph(self, state):
        data = OrchestrationData.model_validate(state["payload"])
        if not data.task.graph_enabled:
            return state
        request = {"graph": "contract_bfs", "round": data.rounds}
        self.tools.budget.check("tool", request)
        with self.tools.budget.collector.measure("tool", "contract_bfs", request):
            return super()._graph(state)

    def _route(self, state):
        # Sources are immutable in this pilot. Dynamic round-wise evidence belongs to E3.
        return "report"


def run_v8(root, task, tools):
    engine = CommonToolV8Engine(root, tools)
    try:
        return engine.run(task)
    finally:
        engine.close()
