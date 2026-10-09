"""Reference branch matching and explicit denominators. Unreviewed references are not VPR."""

import math

import networkx as nx


def score_case(plan, validation, gold):
    matches = []
    forbidden = bool(plan and any(s.action_type in gold.forbidden_actions for s in plan.steps))
    if plan:
        graph = nx.DiGraph()
        graph.add_nodes_from(s.step_id for s in plan.steps)
        graph.add_edges_from((d, s.step_id) for s in plan.steps for d in s.depends_on)
        for branch in gold.any_of:
            options = {}
            for pattern in branch.required:
                options[pattern.pattern_id] = [
                    step.step_id
                    for step in plan.steps
                    if (
                        step.action_type == pattern.action_type
                        and step.target == pattern.target
                        and all(step.parameters.get(k) == v for k, v in pattern.parameters.items())
                    )
                ]
            # A valid branch requires an injective match and the prescribed reachability.
            best = {
                "branch_id": branch.branch_id,
                "required": len(branch.required),
                "matched": 0,
                "edges": sum(len(p.depends_on) for p in branch.required),
                "matched_edges": 0,
                "complete": False,
            }
            visited = 0

            def search(index, assignment, branch=branch, options=options):
                nonlocal visited, best
                visited += 1
                if visited > 10000:
                    return
                if index == len(branch.required):
                    edges = sum(
                        1
                        for p in branch.required
                        for d in p.depends_on
                        if p.pattern_id in assignment
                        and d in assignment
                        and assignment[d] != assignment[p.pattern_id]
                        and nx.has_path(graph, assignment[d], assignment[p.pattern_id])
                    )
                    complete = len(assignment) == best["required"] and edges == best["edges"]
                    if (complete, len(assignment), edges) > (
                        best["complete"],
                        best["matched"],
                        best["matched_edges"],
                    ):
                        best = {
                            **best,
                            "matched": len(assignment),
                            "matched_edges": edges,
                            "complete": complete,
                        }
                    return
                pattern = branch.required[index]
                for sid in options[pattern.pattern_id]:
                    if sid not in assignment.values():
                        search(index + 1, {**assignment, pattern.pattern_id: sid})
                search(index + 1, assignment)

            search(0, {})
            best["matching_exhausted"] = visited > 10000
            if best["matching_exhausted"]:
                best["complete"] = False
            matches.append(best)
    best = (
        max(matches, key=lambda r: (r["complete"], r["matched"], r["matched_edges"]))
        if matches
        else None
    )
    reference_pass = bool(
        best and best["complete"] and validation.raw_plan_validity and not forbidden
    )
    return {
        "case_id": gold.case_id,
        "reviewed": gold.reviewed,
        "solvability": gold.solvability,
        "parsed": plan is not None,
        "unscorable": plan is None,
        "validator_pass": validation.raw_plan_validity,
        "published_validator_pass": validation.published_plan_validity,
        "reference_pass": reference_pass,
        "vpr": reference_pass if gold.reviewed and gold.solvability == "solvable" else None,
        "forbidden_action": forbidden,
        "hard_constraint_violation": any(i.severity == "hard" for i in validation.issues),
        "branch": best,
        "issues": [i.code for i in validation.issues],
        "retained_steps": validation.retained_steps,
        "rejected_steps": validation.rejected_steps,
        "rewritten_steps": validation.rewritten_steps,
    }


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return [max(0, center - half), min(1, center + half)]


def summarize_cases(rows):
    reviewed = [r for r in rows if r["vpr"] is not None]
    valid = sum(r["validator_pass"] for r in rows)
    successes = sum(r["vpr"] for r in reviewed)
    return {
        "runs": len(rows),
        "parsed": sum(r["parsed"] for r in rows),
        "validator_pass_count": valid,
        "validator_pass_rate": valid / len(rows) if rows else None,
        "reference_pass_count": sum(r["reference_pass"] for r in rows),
        "vpr_denominator": len(reviewed),
        "vpr": successes / len(reviewed) if reviewed else None,
        "vpr_wilson95": wilson(successes, len(reviewed)),
        "hard_violation_count": sum(r["hard_constraint_violation"] for r in rows),
        "parse_failure_count": sum(not r["parsed"] for r in rows),
        "unscorable_count": sum(r["unscorable"] for r in rows),
    }
