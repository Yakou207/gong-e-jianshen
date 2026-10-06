"""Execution telemetry wire checks; scripted support is not model quality proof."""
from copy import deepcopy
from datetime import timedelta
import json

import pytest

from aml_qc import core, llm, workflow
from test_model_safety import case, json_message, response_message, ScriptedModel, tool_message


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Scope mechanism tests must not read credentials or call transport")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)
    monkeypatch.setattr(llm.httpx, "stream", forbidden)


@pytest.mark.parametrize("explained_direction", ["in", "out"])
def test_actual_partial_query_progress_measures_reads_without_assigning_narrative_scope(explained_direction):
    value = case()
    incoming = [row["transaction_id"] for row in value["transactions"] if row["direction"] == "in"][:2]
    outgoing = [row["transaction_id"] for row in value["transactions"] if row["direction"] == "out"]
    value["materials"] = [{"material_id": "visible-pair", "revision": "1", "material_type": "synthetic_relation",
        "source_transaction_ids": incoming, "transfer_transaction_id": outgoing[0],
        "text": "Synthetic relation references only these explicit endpoints."}]
    value["material_links"] = []
    workflow.narrative(value)["text"] = "本期全部" + ("转入用途" if explained_direction == "in" else "转出去向") + "按已给材料所列记录对应。"
    value["alert"]["focuses"][0]["text"] = workflow.narrative(value)["text"]
    gap_start = core._time(value["coverage_start"]) + timedelta(days=3)
    for row in value["coverage"]:
        row.update(status="partial", missing_ranges=[{"start": gap_start.isoformat(),
            "end": (gap_start + timedelta(days=1)).isoformat()}])
    saved = deepcopy(value)
    model = ScriptedModel([json_message({"claims": [], "unresolved": []}), response_message(value),
        tool_message("query_transactions", {"transaction_id": incoming[0]}, "chosen-endpoint-1"),
        tool_message("query_transactions", {"transaction_id": incoming[1]}, "chosen-endpoint-2"),
        tool_message("query_transactions", {"direction": "out"}, "chosen-out-enumeration"),
        json_message({"gaps": [], "leads": []})])
    result = workflow.run_review(value, mode="agent", provider="frozen", model=model)
    wire = model.calls[-1]["request"]
    final = json.loads(wire["messages"][-2]["content"])
    progress = final["read_scope_progress"]
    assert progress["out"]["whole_visible_interval_read"] is True
    assert progress["in"] == {"whole_visible_interval_read": False, "successful_result_refs": []}
    assert "不分配" in progress["meaning"] and "不能反推" in progress["meaning"]
    assert all("required" not in entry for entry in (progress["in"], progress["out"]))
    endpoints = final["material_transaction_read_progress"]
    assert endpoints["pending_transaction_ids"] == []
    assert all(row["status"] == "read_returned" for row in endpoints["entries"])
    observations = final["transaction_observations"]
    assert {row["transaction_id"] for observation in observations for row in observation["rows"]} == set(incoming + outgoing)
    assert all(observation["coverage"]["status"] == "partial" for observation in observations)
    assert any(issue["type"] == "insufficient_coverage" for issue in result["issues"])
    assert "先确定被解释的记录集合" in wire["messages"][0]["content"]
    assert wire["messages"][0]["content"] == workflow.SEMANTIC_SYSTEM
    for call in [row["request"] for row in model.calls if row["request"].get("tools")][1:]:
        feedback = json.loads(call["messages"][-1]["content"])
        assert feedback["read_scope_progress"]["meaning"] == progress["meaning"]
    assert result["stats"]["adaptive_tool_calls"] == 3 and len(model.calls) == 6
    assert not model.responses and value == saved
