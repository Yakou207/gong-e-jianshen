"""Offline input/dispatch boundaries, not evidence of live Agent tool quality."""
from copy import deepcopy
import json

import pytest

from aml_qc import llm, workflow
from aml_qc.depgraph import Evaluator, sources_for
from test_model_safety import case, json_message, response_message, ScriptedModel, support_response, tool_message


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Mechanism tests must not read credentials or call a model")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)


def evaluator(value, previous=None):
    value.setdefault("schema", workflow.default_schema())
    return Evaluator(sources_for(value, value["schema"], {"provider": "frozen"}),
                     previous, "incremental" if previous else "full")


@pytest.mark.parametrize("material_id", ["invented-mapping-table", ""])
def test_invented_material_id_does_not_create_a_read_node(material_id):
    value = case()
    graph = evaluator(value)
    with pytest.raises(ValueError, match="材料ID"):
        workflow.execute_tool(value, value["schema"], graph, "read_material", {"material_id": material_id})
    assert graph.nodes == {} and graph.trace == []


def test_explicitly_linked_missing_material_remains_a_known_empty_read():
    value = case()
    value["materials"] = []
    material_id = value["material_links"][0]["material_id"]
    graph = evaluator(value)
    result = workflow.execute_tool(value, value["schema"], graph, "read_material", {"material_id": material_id})
    assert result["material"] is None
    assert result["scope"]["material_ids"] == []
    assert result["scope"]["referenced_material_ids"] == [material_id]
    assert graph.trace[0]["status"] == "computed" and len(graph.nodes) == 1


def test_link_directory_change_invalidates_cached_missing_material_scope():
    value = case()
    value["materials"] = []
    material_id = value["material_links"][0]["material_id"]
    first = evaluator(value)
    args = {"material_id": material_id}
    workflow.execute_tool(value, value["schema"], first, "read_material", args)
    changed = deepcopy(value)
    changed["material_links"].append({**deepcopy(changed["material_links"][0]),
                                     "link_id": "additional-link", "material_id": "known-missing-2"})
    incremental = evaluator(changed, first.snapshot())
    full = evaluator(changed)
    actual = workflow.execute_tool(changed, changed["schema"], incremental, "read_material", args)
    expected = workflow.execute_tool(changed, changed["schema"], full, "read_material", args)
    assert actual == expected and actual["material"] is None
    assert actual["scope"]["referenced_material_ids"] == sorted([material_id, "known-missing-2"])
    assert incremental.trace[0]["status"] == "computed"
    removed = deepcopy(changed)
    removed["material_links"] = []
    removed_graph = evaluator(removed, incremental.snapshot())
    with pytest.raises(ValueError, match="材料ID"):
        workflow.execute_tool(removed, removed["schema"], removed_graph, "read_material", args)
    assert not removed_graph.nodes


def test_agent_initial_context_contains_real_coverage_and_full_versions_without_unread_rows():
    value = case()
    value["material_links"][0]["material_id"] = "known-missing-material"
    value["coverage"][0]["status"] = "partial"
    value["coverage"][0]["missing_ranges"] = [{"start": value["coverage_start"], "end": value["coverage_end"]}]
    model = ScriptedModel([json_message({"claims": [], "unresolved": []}), response_message(value),
                           {"role": "assistant", "content": "No extra tool information requested."},
                           json_message(support_response(value))])
    result = workflow.run_review(value, mode="agent", provider="frozen", model=model)
    first_input = json.loads(model.calls[2]["request"]["messages"][1]["content"])
    assert first_input["case"]["coverage"] == value["coverage"]
    assert first_input["case"]["documents"] == sorted(value["documents"], key=lambda row: row["document_id"])
    assert first_input["case"]["materials"] == value["materials"]
    info = first_input["initial_information"]
    for field, identifier in [("documents", "document_id"), ("materials", "material_id")]:
        assert info[field]["provided"] == "full_current_versions"
        assert info[field]["versions"] == [{identifier: row[identifier], "revision": row["revision"]}
                                            for row in first_input["case"][field]]
    assert info["coverage"]["provided"] == "source_declarations"
    assert info["checks"]["provided"] == "deterministic_summaries"
    assert info["known_material_ids"] == sorted({row["material_id"] for row in value["materials"]}
                                               | {link["material_id"] for link in value["material_links"]})
    assert info["transaction_rows"] == {"provided": False, "read_with": "query_transactions"}
    assert "known-missing-material" in info["known_material_ids"]
    assert all(row["material_id"] != "known-missing-material" for row in first_input["case"]["materials"])
    assert "transactions" not in first_input["case"]
    assert first_input["case"].get("transaction_observations", []) == []
    assert not [row for row in result["trace"] if row.get("tool_call_id")]
    assert result["stats"]["adaptive_tool_calls"] == 0
    assert len(model.calls) == 4 and not model.responses


def test_bad_material_request_returns_real_failure_and_planner_can_choose_a_valid_query():
    value = case()
    model = ScriptedModel([json_message({"claims": [], "unresolved": []}), response_message(value),
        tool_message("read_material", {"material_id": "invented-mapping-table"}, "bad-read"),
        tool_message("query_transactions", {"direction": "out"}, "actual-read"),
        {"role": "assistant", "content": "Read completed."}, json_message(support_response(value))])
    result = workflow.run_review(value, mode="agent", provider="frozen", model=model)
    bad, good = [row for row in result["trace"] if row.get("tool_call_id")]
    assert bad["status"] == "failed" and bad["result_ref"] is None
    assert bad["result"]["execution_status"] == "failed"
    assert good["status"] == "completed" and good["result_ref"]
    assert not any(node["kind"] == "read_material" for node in result["snapshot"]["nodes"].values())
    next_request = model.calls[3]["request"]
    rejection = next(message for message in next_request["messages"] if message.get("tool_call_id") == "bad-read")
    assert json.loads(rejection["content"])["lead_basis_ref"] is None
    assert json.loads(rejection["content"])["execution_status"] == "failed"
    assert result["stats"]["adaptive_tools_failed"] == 1
    assert result["stats"]["adaptive_tools_completed"] == 1
    assert len(model.calls) == 6 and not model.responses
