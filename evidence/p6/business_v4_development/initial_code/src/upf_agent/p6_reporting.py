"""Aggregate reviewed retrieval without discarding failed cases from the denominator."""

ROUTES = ("exact", "bm25", "dense", "rrf", "reranked")


def _routes(rows: list[dict]) -> dict:
    routes = {}
    for route in ROUTES:
        values = [r.get("retrieval_metrics", {}).get(route, {}) for r in rows]
        routes[route] = {
            "n": len(values),
            "failed_or_unscored": sum(
                value.get("recall_at_20") is None or value.get("ndcg_at_8") is None
                for value in values
            ),
            "recall_at_20": (
                sum(value.get("recall_at_20") or 0 for value in values) / len(values)
                if values
                else None
            ),
            "ndcg_at_8": (
                sum(value.get("ndcg_at_8") or 0 for value in values) / len(values)
                if values
                else None
            ),
        }
    return routes


def retrieval_summary(rows: list[dict]) -> dict:
    manual = [r for r in rows if r["category"] in {"command", "knowledge", "troubleshooting"}]
    routes = _routes(manual)
    families = {}
    for row in manual:
        if row["category"] == "command":
            group = next(
                (g for g in row.get("source_groups", []) if g.startswith("command-family:")),
                "unassigned",
            )
            families.setdefault(group, []).append(row)
    final, fused = routes["reranked"], routes["rrf"]
    return {
        "scope": "macro mean on human-reviewed relevant chunk sets; failures count as zero",
        "routes": routes,
        "by_command_family": {family: _routes(selected) for family, selected in families.items()},
        "retrieval_gate_pass": bool(
            manual
            and final["failed_or_unscored"] == 0
            and fused["failed_or_unscored"] == 0
            and final["recall_at_20"] >= 0.9
            and final["ndcg_at_8"] >= fused["ndcg_at_8"]
        ),
    }
