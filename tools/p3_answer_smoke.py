"""Real local-model answer checks on saved P3 retrieval evidence."""

import json
from pathlib import Path

from upf_agent.config import load_model_config
from upf_agent.models import LocalChatClient
from upf_agent.p3_answer import answer_from_retrieval, render_answer

root = Path(__file__).resolve().parents[1]
model = load_model_config(root / "config/models.yaml").generation["baseline"]
results = []
with LocalChatClient(model) as client:
    for name in ["show_cp_node", "netman_v13", "n4_pfcp", "unrelated"]:
        retrieval = json.loads(
            (
                root / "evidence/p3" / f"{'show_cp_node' if name == 'unrelated' else name}.json"
            ).read_text()
        )
        question = "苹果蛋糕需要多少克面粉？" if name == "unrelated" else retrieval["query"]
        result = answer_from_retrieval(root, question, retrieval, client)
        (root / "evidence/p3" / f"answer_{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        (root / "evidence/p3" / f"answer_{name}.md").write_text(render_answer(result))
        expected = (
            "missing_source"
            if name == "unrelated"
            else "completed"
            if name == "netman_v13"
            else "partial"
        )
        results.append(
            {
                "name": name,
                "status": result["status"],
                "claims": len(result["claims"]),
                "citations": len(result["citations"]),
                "expected_status": expected,
                "pass": result["status"] == expected,
                "error": result.get("error"),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
(root / "evidence/p3/answer_smoke.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2) + "\n"
)
raise SystemExit(0 if all(r["pass"] for r in results) else 1)
