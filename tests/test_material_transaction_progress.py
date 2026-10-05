"""Actual reads of explicitly declared IDs; offline mechanism fixtures only."""
from copy import deepcopy
from datetime import timedelta
import json

import pytest

from aml_qc import core, llm, workflow
from aml_qc.depgraph import business_result
from test_model_safety import case, json_message, response_message, ScriptedModel, support_response, tool_message


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Mechanism tests must not read credentials or call a model")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)
    monkeypatch.setattr(llm.httpx, "stream", forbidden)


def query_trace(value, args, ref="actual-query"):
    return {"tool": "query_transactions", "status": "completed", "dispatch_status": "sent",
            "result_ref": ref, "result": core.query_transactions(value, args)}


def with_reference():
    value = case()
    value["materials"][0]["refund_transaction_id"] = value["transactions"][0]["transaction_id"]
    return value


def agent(value, queries=(), *, previous=None):
    replies = [] if previous else [json_message({"claims": [], "unresolved": []}), response_message(value)]
    replies += [tool_message("query_transactions", args, f"read-{i}") for i, args in enumerate(queries)]
    if len(queries) < 3:
        replies.append({"role": "assistant", "content": "No further read chosen."})
    replies.append(json_message(support_response(value)))
    model = ScriptedModel(replies)
    result = workflow.run_review(value, mode="agent", provider="frozen", model=model,
                                 strategy="incremental" if previous else "full", previous=previous)
    assert not model.responses
    return result, model


def test_only_explicit_material_id_fields_are_recursively_declared():
    value = {"materials": [{"transaction_id": "literal", "refund_transaction_id": "refund-in2",
        "transaction_ids": ["literal", "second", False],
        "nested": [{"related_transaction_ids": ["third"], "source_transaction_id": " fourth "}],
        "text": "A payment called unmentioned-id", "order_id": "not-a-transaction",
        "transaction_id_hint": "not-declared", "invalid_transaction_id": 23,
        "empty_transaction_id": "   "}],
        "transactions": object(), "hidden_transaction_id": "not-material"}
    assert workflow.material_transaction_ids(value) == [" fourth ", "literal", "refund-in2", "second", "third"]
    progress = workflow.material_transaction_read_progress(value, [])
    assert progress["pending_transaction_ids"] == workflow.material_transaction_ids(value)
    assert all(row["status"] == "pending" and not row["successful_result_refs"] for row in progress["entries"])
    assert not any(field in json.dumps(progress) for field in ("counterparty_token", "amount", "timestamp", "not-material"))


def test_successful_return_reads_declared_id_even_with_filters_and_preserves_real_reference():
    value = with_reference()
    row = value["transactions"][0]
    trace = query_trace(value, {"transaction_id": row["transaction_id"], "direction": row["direction"]})
    saved = deepcopy(trace)
    projected = {"materials": value["materials"], "coverage_start": value["coverage_start"],
                 "coverage_end": value["coverage_end"], "transactions": object()}
    progress = workflow.material_transaction_read_progress(projected, [trace, trace])
    assert progress["entries"] == [{"transaction_id": row["transaction_id"], "status": "read_returned",
                                     "successful_result_refs": ["actual-query"]}]
    assert progress["pending_transaction_ids"] == [] and trace == saved
    replayed = deepcopy(trace)
    replayed.update(status="reused", original_status="completed")
    assert workflow.material_transaction_read_progress(projected, [replayed]) == progress


@pytest.mark.parametrize("restriction", ["direction", "counterparty", "subscope", "failed", "rejected", "execution", "no_ref", "other_tool"])
def test_unread_id_remains_pending_after_filtered_empty_or_failed_read(restriction):
    value = with_reference()
    value["materials"][0]["refund_transaction_id"] = "absent-declared-id"
    args = {"transaction_id": "absent-declared-id"}
    if restriction == "direction":
        args["direction"] = "out"
    elif restriction == "counterparty":
        args["counterparty_token"] = value["transactions"][0]["counterparty_token"]
    elif restriction == "subscope":
        args["start"] = (core._time(value["coverage_start"]) + timedelta(hours=1)).isoformat()
    trace = query_trace(value, args)
    if restriction == "failed": trace["status"] = "failed"
    elif restriction == "rejected": trace["dispatch_status"] = "not_sent"
    elif restriction == "execution": trace["result"]["execution_status"] = "identity_unresolved"
    elif restriction == "no_ref": trace["result_ref"] = None
    elif restriction == "other_tool": trace["tool"] = "compute_features"
    progress = workflow.material_transaction_read_progress(value, [trace])
    assert progress["entries"] == [{"transaction_id": "absent-declared-id", "status": "pending", "successful_result_refs": []}]


@pytest.mark.parametrize("query", [{}, {"transaction_id": "absent-declared-id"}])
def test_unfiltered_full_visible_interval_can_establish_no_matching_visible_row_without_full_coverage(query):
    value = with_reference()
    value["materials"][0]["refund_transaction_id"] = "absent-declared-id"
    value["coverage"][0]["status"] = "partial"
    trace = query_trace(value, query)
    progress = workflow.material_transaction_read_progress(value, [trace])
    assert progress["entries"] == [{"transaction_id": "absent-declared-id", "status": "queried_no_visible_row",
                                     "successful_result_refs": ["actual-query"]}]
    assert progress["pending_transaction_ids"] == []
    assert trace["result"]["coverage"]["status"] == "partial"
    assert "来源覆盖完整性" in progress["meaning"] and "现实不存在" in progress["meaning"]


def test_agent_unread_explicit_reference_is_partial_without_retry_or_loss_of_validated_response():
    value = with_reference()
    result, model = agent(value)
    progress = result["material_transaction_read_progress"]
    assert progress["pending_transaction_ids"] == [value["materials"][0]["refund_transaction_id"]]
    required = next(row for row in result["required_checks"] if row["check_id"] == "material_transaction_refs")
    assert required["status"] == "pending" and result["run_status"] == "partial"
    assert result["semantic_results"][0]["status"] == "addressed"
    assert any(row["type"] == "unread_material_transaction" for row in result["issues"])
    assert result["stats"]["adaptive_tool_calls"] == 0 and len(model.calls) == 4
    initial = json.loads(model.calls[2]["request"]["messages"][1]["content"])
    assert initial["case"]["material_transaction_read_progress"] == progress


def test_next_planner_sees_pending_declared_id_and_autonomously_reads_novel_actual_fields():
    value = with_reference()
    tid = value["materials"][0]["refund_transaction_id"]
    value["transactions"][0]["counterparty_token"] = "actual-unseen-party"
    class DependingModel(ScriptedModel):
        def complete(self, messages, tools=None, stage=None):
            if len(self.calls) == 2:
                data = json.loads(messages[1]["content"])["case"]
                assert data["material_transaction_read_progress"]["pending_transaction_ids"] == [tid]
                assert "actual-unseen-party" not in json.dumps(data["material_transaction_read_progress"])
            if len(self.calls) == 3:
                feedback = json.loads(messages[-1]["content"])
                assert feedback["material_transaction_read_progress"]["pending_transaction_ids"] == [tid]
            if len(self.calls) == 4:
                feedback = json.loads(messages[-1]["content"])
                assert feedback["material_transaction_read_progress"]["pending_transaction_ids"] == []
            return super().complete(messages, tools, stage)
    model = DependingModel([json_message({"claims": [], "unresolved": []}), response_message(value),
        tool_message("read_document", {"document_id": workflow.narrative(value)["document_id"]}, "old-full-text"),
        tool_message("query_transactions", {"transaction_id": tid}, "chosen-id-read"),
        {"role": "assistant", "content": "Read completed."}, json_message(support_response(value))])
    result = workflow.run_review(value, mode="agent", provider="frozen", model=model)
    final = json.loads(model.calls[-1]["request"]["messages"][-2]["content"])
    assert final["material_transaction_read_progress"]["entries"][0]["status"] == "read_returned"
    assert final["transaction_observations"][0]["rows"][0]["counterparty_token"] == "actual-unseen-party"
    assert final["material_transaction_read_progress"]["entries"][0]["successful_result_refs"] == [final["transaction_observations"][0]["result_ref"]]
    assert result["stats"]["adaptive_tool_calls"] == 2 and len(model.calls) == 6 and not model.responses


@pytest.mark.parametrize("present", [True, False])
def test_fixed_actual_full_read_uses_same_explicit_reference_rule(present):
    value = with_reference()
    if not present: value["materials"][0]["refund_transaction_id"] = "absent-declared-id"
    model = ScriptedModel([json_message({"claims": [], "unresolved": []}), response_message(value), json_message(support_response(value))])
    result = workflow.run_review(value, mode="fixed", provider="frozen", model=model)
    progress = result["material_transaction_read_progress"]
    assert progress["pending_transaction_ids"] == []
    assert progress["entries"][0]["status"] == ("read_returned" if present else "queried_no_visible_row")
    assert progress["entries"][0]["successful_result_refs"] == [result["fixed_policy_reads"][1]["result_ref"]]
    data = json.loads(model.calls[-1]["request"]["messages"][1]["content"])
    assert data["material_transaction_read_progress"] == progress and len(model.calls) == 3


def test_material_reference_changes_and_deletion_recompute_independent_business_results():
    value = with_reference()
    original_id = value["materials"][0]["refund_transaction_id"]
    first, _ = agent(value, [{"transaction_id": original_id}])
    changed = deepcopy(value)
    new_id = next(row["transaction_id"] for row in value["transactions"] if row["transaction_id"] != original_id)
    changed["materials"][0].update(refund_transaction_id=new_id, revision="2")
    incremental, inc_model = agent(changed, [{"transaction_id": new_id}], previous=first["snapshot"])
    full, _ = agent(changed, [{"transaction_id": new_id}])
    assert business_result(incremental) == business_result(full)
    assert incremental["material_transaction_read_progress"] == full["material_transaction_read_progress"]
    assert [row["transaction_id"] for row in incremental["material_transaction_read_progress"]["entries"]] == [new_id]
    assert len(inc_model.calls) == 3
    assert "source:materials" in incremental["snapshot"]["nodes"]["agent_stage"]["dependencies"]
    removed = deepcopy(changed)
    del removed["materials"][0]["refund_transaction_id"]
    removed["materials"][0]["revision"] = "3"
    inc_removed, _ = agent(removed, previous=incremental["snapshot"])
    full_removed, _ = agent(removed)
    assert business_result(inc_removed) == business_result(full_removed)
    assert inc_removed["material_transaction_read_progress"]["entries"] == []
    assert not any(row["check_id"] == "material_transaction_refs" for row in inc_removed["required_checks"])
