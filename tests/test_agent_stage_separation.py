"""Offline wire/stage regressions, never live Agent quality evidence."""
import json

import pytest

from aml_qc import llm, workflow
from aml_qc.contracts import contract_schemas
from aml_qc.depgraph import canonical
from test_model_safety import case, json_message, response_message, ScriptedModel, tool_message


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Stage tests must not read credentials or use transport")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)
    monkeypatch.setattr(llm.httpx, "stream", forbidden)


def test_actual_planning_wire_has_no_final_support_contract():
    value = case()
    model = ScriptedModel([json_message({"claims": [], "unresolved": []}), response_message(value),
        tool_message("query_transactions", {"direction": "out"}, "chosen-read"),
        {"role": "assistant", "content": "No further obtainable evidence."},
        json_message({"gaps": [], "leads": []})])
    workflow.run_review(value, mode="agent", provider="frozen", model=model)
    planners = [row["request"] for row in model.calls if row["request"].get("tools")]
    assert len(planners) == 2
    support_prompt = (workflow.PROMPT_ROOT / "semantic.txt").read_text().strip()
    for request in planners:
        system = request["messages"][0]["content"]
        assert support_prompt not in system
        assert '"basis_kind"' not in system
        assert canonical(contract_schemas()["support"]) not in system
    final = model.calls[-1]["request"]
    assert not final.get("tools")
    assert support_prompt in final["messages"][0]["content"]
    assert canonical(contract_schemas()["support"]) in final["messages"][0]["content"]
    assert json.loads(final["messages"][-2]["content"])["read_scope_progress"]["out"]["whole_visible_interval_read"]
    assert len(model.calls) == 5 and not model.responses


def test_planner_completion_is_not_promoted_or_queried_by_engine():
    value = case()
    premature = '{"gaps": [{"basis_kind": "planner-only-sentinel"}], "leads": []}'
    model = ScriptedModel([json_message({"claims": [], "unresolved": []}), response_message(value),
        {"role": "assistant", "content": premature}, json_message({"gaps": [], "leads": []})])
    result = workflow.run_review(value, mode="agent", provider="frozen", model=model)
    final = model.calls[-1]["request"]
    assert "planner-only-sentinel" not in json.dumps(final, ensure_ascii=False)
    assert result["stats"]["adaptive_tool_calls"] == 0
    assert not result["lead_candidates"]
    assert len(model.calls) == 4 and not model.responses
