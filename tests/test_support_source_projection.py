"""Declared tool fields and visible source directories; no live model claims."""
from copy import deepcopy
import json

import pytest

from aml_qc import core, llm, workflow
from aml_qc.depgraph import Evaluator, sources_for
from test_model_safety import case, json_message, response_message, ScriptedModel, support_response, tool_message


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Mechanism tests must not read credentials or call a model")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)
    monkeypatch.setattr(llm.httpx, "stream", forbidden)


def run(value, mode):
    replies = [json_message({"claims": [], "unresolved": []}), response_message(value)]
    if mode == "agent":
        replies += [tool_message("query_transactions", {}, "actual-read"),
                    {"role": "assistant", "content": "Read completed."}]
    replies.append(json_message(support_response(value)))
    model = ScriptedModel(replies)
    result = workflow.run_review(value, mode=mode, provider="frozen", model=model)
    assert not model.responses
    return result, model


@pytest.mark.parametrize("mode", ["fixed", "agent"])
def test_unrequested_transaction_metadata_cannot_enter_either_model_request(mode):
    value = case()
    value["transactions"][0].update(memo="UNREQUESTED-MEMO", channel="UNREQUESTED-CHANNEL",
                                     masked_name="UNREQUESTED-NAME")
    original = deepcopy(value)
    direct = core.query_transactions(value)
    assert direct["rows"][0]["memo"] == "UNREQUESTED-MEMO"
    result, model = run(value, mode)
    query = next(row for row in (result.get("fixed_policy_reads", []) if mode == "fixed" else result["trace"])
                 if row.get("tool") == "query_transactions" and row.get("result_ref"))
    returned = query["result"]
    expected = {**direct, "rows": [{k: row[k] for k in direct["scope"]["fields"] if k in row} for row in direct["rows"]]}
    assert returned == expected
    assert set(returned["rows"][0]) == set(core.TRANSACTION_FIELDS)
    for request in model.calls:
        encoded = json.dumps(request["request"])
        assert all(marker not in encoded for marker in ["UNREQUESTED-MEMO", "UNREQUESTED-CHANNEL", "UNREQUESTED-NAME"])
    data = json.loads(model.calls[-1]["request"]["messages"][1]["content"])
    supplied = data if mode == "fixed" else data["case"]
    assert supplied["materials"] == value["materials"]
    assert supplied["relationship_support_source_candidates"]["materials"] == [
        {"material_id": row["material_id"], "revision": row["revision"]} for row in value["materials"]]
    assert value == original and core.query_transactions(value) == direct


def test_visible_support_candidates_are_revision_bound_directories_without_transaction_or_missing_source_inference():
    value = case()
    value["material_links"].append({**deepcopy(value["material_links"][0]),
        "link_id": "missing-source-link", "material_id": "not-currently-supplied"})
    value["transactions"] = object()
    value["hidden_support_pool"] = [{"material_id": "hidden-candidate", "revision": "hidden"}]
    data = workflow.semantic_input(value, {})
    candidates = data["relationship_support_source_candidates"]
    assert candidates["materials"] == [{"material_id": row["material_id"], "revision": row["revision"]} for row in value["materials"]]
    assert candidates["other_documents"] == [{"document_id": row["document_id"], "revision": row["revision"]}
                                               for row in value["documents"] if row["document_id"] != "narrative"]
    assert all(row["document_id"] != "narrative" for row in candidates["other_documents"])
    assert "not-currently-supplied" not in json.dumps(candidates) and "hidden-candidate" not in json.dumps(candidates)
    value["materials"][0]["revision"] = "updated-visible-version"
    assert workflow.semantic_input(value, {})["relationship_support_source_candidates"]["materials"][0]["revision"] == "updated-visible-version"


@pytest.mark.parametrize("mode", ["fixed", "agent"])
def test_empty_candidate_directory_does_not_create_a_gap_for_a_pure_action_statement(mode):
    value = case()
    value["materials"] = []
    value["material_links"] = []
    value["documents"] = [{"document_id": "narrative", "revision": "2", "text": "已逐项核对，将继续复核。"}]
    result, model = run(value, mode)
    data = json.loads(model.calls[-1]["request"]["messages"][1]["content"])
    supplied = data if mode == "fixed" else data["case"]
    candidates = supplied["relationship_support_source_candidates"]
    assert candidates["materials"] == candidates["other_documents"] == []
    assert not any(row["type"] == "unsupported_explanation" for row in result["issues"])
    assert result["semantic_results"][0]["status"] == "addressed"
    assert len(model.calls) == (3 if mode == "fixed" else 5)


def test_tool_projection_is_cached_without_mutating_source_or_direct_deterministic_query():
    value = case()
    value["transactions"][0]["memo"] = "not-a-declared-field"
    schema = workflow.default_schema()
    value["schema"] = schema
    graph = Evaluator(sources_for(value, schema, {"provider": "frozen"}), None, "full")
    direct = core.query_transactions(value)
    result = workflow.execute_tool(value, schema, graph, "query_transactions", {})
    assert all(set(row) <= set(result["scope"]["fields"]) for row in result["rows"])
    assert core.query_transactions(value) == direct and direct["rows"][0]["memo"] == "not-a-declared-field"
    incremental = Evaluator(sources_for(value, schema, {"provider": "frozen"}), graph.snapshot(), "incremental")
    assert workflow.execute_tool(value, schema, incremental, "query_transactions", {}) == result
    assert incremental.trace[-1]["status"] == "reused"
