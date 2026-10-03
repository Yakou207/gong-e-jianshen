"""Offline B0 contract checks; no live model quality or human truth asserted."""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from aml_qc import baseline, core, llm
from aml_qc.ingest import load_case
from test_model_safety import ScriptedModel, json_message


@pytest.fixture(autouse=True)
def no_credentials_or_live_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("baseline mechanism tests cannot read credentials or call a live API")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)


def case():
    return load_case(Path(__file__).resolve().parents[1] / "data/synthetic/seed-01.json")


def answer(package):
    """A mechanical response fixture, deliberately not a correct reference."""
    doc = next(d for d in package["documents"] if d["document_id"] == "narrative")
    scope = {"account_id": package["subject_account_id"], "direction": None, "counterparty_tokens": [],
             "start": package["coverage_start"], "end": package["coverage_end"]}
    values = {"F1": "met", "F2": "not_met", "count": "supported", "amount_sum": "supported", "counterparty": "supported",
              "time_range": "supported", "material_relation": "corresponds", "alert_response": "addressed"}
    propositions = {"count": 1, "amount_sum": "1050.00", "counterparty": ["supplier-b"],
                    "time_range": {"start": package["coverage_start"], "end": package["coverage_end"]}}
    result = {"checks": [], "coverage": [], "issues": [], "unfinished": []}
    for label, value in values.items():
        anchor = {}
        if label in propositions:
            anchor = {"document_id": doc["document_id"], "revision": doc["revision"], "span": [0, len(doc["text"])],
                      "quote": doc["text"], "claim": {"operator": "only", "value": propositions[label], "unit": None}}
        elif label == "material_relation":
            anchor = {"material_link_id": package["material_links"][0]["link_id"]}
        elif label == "alert_response":
            anchor = {"focus_id": package["alert"]["focuses"][0]["focus_id"]}
        result["checks"].append({"check_id": "local-" + label, "label": label, "status": "completed", "value": value,
                                 "object_scope": deepcopy(scope), "anchor": anchor, "reason": "Offline mechanism fixture only.", "evidence": []})
        result["coverage"].append({"label": label, "status": "completed", "reason": "Fixture declares coverage; no completeness proof."})
    return result


def test_once_raw_reading_has_no_tools_or_business_answer_repair(monkeypatch):
    package = case()
    package["transactions"] = []
    raw = answer(package)
    raw["checks"][0]["value"] = "met"  # Clearly wrong with no transactions; must remain a candidate.
    for name in ("query_transactions", "compute_features", "verify_claim", "check_materials", "resolve_entity", "check_coverage"):
        monkeypatch.setattr(core, name, lambda *args, **kwargs: pytest.fail("B0 invoked a business checker"))
    model = ScriptedModel([json_message(raw)])
    before = deepcopy(package)
    run = baseline.run_baseline(package, model=model)
    assert package == before
    assert len(model.calls) == run["stats"]["model_calls"] == 1
    assert run["stats"]["tool_calls"] == 0 and model.calls[0]["request"]["tools"] is None
    assert run["mode"] == "b0" and run["method"] == "B0" and run["run_status"] == "completed"
    assert run["parsed_output"]["checks"][0]["value"] == "met"
    assert run["parsed_output"]["checks"][2]["anchor"]["claim"]["operator"] == "only"
    assert run["raw_response"] == json_message(raw)
    request = json.loads(model.calls[0]["request"]["messages"][1]["content"])
    assert request == {"raw_case": baseline.raw_case_input(package)}
    assert run["input_projection"] == request["raw_case"]
    assert run["execution"]["input_hash"] == baseline.digest(run["input_projection"])
    assert run["execution"]["max_model_calls"] == 1 and run["execution"]["thinking"] == "disabled"
    assert run["execution"]["model"] == model.model


def test_recursive_projection_keeps_raw_sources_but_excludes_private_and_precomputed_data():
    package = case()
    marker = "PRIVATE_ANSWER_MUST_NOT_REACH_B0"
    package.update(case_family=marker, split=marker, claims=[{"value": marker}], features=[marker], deliveries=[marker])
    package["profile"]["reference_answers"] = {"F1": marker}
    package["transactions"][0]["ground_truth"] = marker
    package["documents"][0]["expected_results"] = marker
    package["materials"][0]["subject"]["intended_mismatch"] = marker
    package["materials"][0]["period"]["condition_id"] = marker
    package["coverage"][0]["hidden_truth"] = marker
    package["entity_mappings"][0]["review_record"] = marker
    package["alert"]["focuses"][0]["expected_label"] = marker
    package["review_scope"] = {"lead_dispositions": [{"verdict": marker}]}
    package["schema"] = llm_schema = baseline.load_schema()
    llm_schema["labels"]["F1"]["reference_answers"] = marker
    llm_schema["material_templates"]["single_purchase_payment"]["generator_truth"] = marker
    visible = baseline.raw_case_input(package)
    assert marker not in baseline.canonical(visible)
    assert visible["documents"][0]["text"] == package["documents"][0]["text"]
    assert visible["transactions"][0]["amount"] == package["transactions"][0]["amount"]
    assert visible["materials"][0]["subject"]["account_id"] == package["materials"][0]["subject"]["account_id"]
    assert visible["schema"]["features"] == package["schema"]["features"]
    model = ScriptedModel([json_message(answer(package))])
    run = baseline.run_baseline(package, model=model)
    assert marker not in baseline.canonical(model.calls)
    assert marker not in baseline.canonical(run["input_projection"])


@pytest.mark.parametrize("location", ["document", "transaction", "schema"])
def test_private_dictionary_cannot_hide_in_an_allowed_scalar_leaf(location):
    package = case()
    if location == "document": package["documents"][0]["text"] = {"hidden_truth": "bad"}
    elif location == "transaction": package["transactions"][0]["memo"] = {"hidden_truth": "bad"}
    else:
        package["schema"] = baseline.load_schema()
        package["schema"]["labels"]["F1"]["definition"] = {"hidden_truth": "bad"}
    model = ScriptedModel([])
    run = baseline.run_baseline(package, model=model)
    assert run["run_status"] == "failed" and run["stats"]["model_calls"] == 0 and model.calls == []
    assert run["raw_response"] is None


def test_human_claim_amendments_are_rejected_not_silently_used():
    package = case(); package["claim_amendments"] = [{"proposed_claim": {"value": 99}}]
    model = ScriptedModel([])
    result = baseline.run_baseline(package, model=model)
    assert result["run_status"] == "failed" and not model.calls
    assert any("human claim amendments" in str(e) for e in result["validation_errors"])


def test_source_citations_are_not_checked_or_repaired_into_true_evidence():
    package = case(); raw = answer(package)
    raw["checks"][0]["evidence"] = [{"type": "document_span", "document_id": "absent-source", "revision": "999",
                                    "span": [10000, 10005], "quote": "invented quote"}]
    raw["issues"] = [{"issue_id": "local-issue", "type": "claim_error", "check_ids": ["local-count"],
                      "reason": "Fixture, not a reference ID.", "evidence": deepcopy(raw["checks"][0]["evidence"])}]
    result = baseline.run_baseline(package, model=ScriptedModel([json_message(raw)]))
    assert result["run_status"] == "completed"
    assert result["parsed_output"]["checks"][0]["evidence"] == raw["checks"][0]["evidence"]
    assert result["parsed_output"]["issues"][0]["check_ids"] == ["local-count"]


@pytest.mark.parametrize("label,value", [("F1", "undeterminable"), ("count", "insufficient_evidence"),
                                          ("material_relation", "insufficient"), ("alert_response", "pending_judgement")])
def test_business_abstention_values_stay_completed_business_answers(label, value):
    package = case(); raw = answer(package)
    next(c for c in raw["checks"] if c["label"] == label)["value"] = value
    result = baseline.run_baseline(package, model=ScriptedModel([json_message(raw)]))
    assert result["run_status"] == "completed" and result["validation_errors"] == []
    assert next(c for c in result["parsed_output"]["checks"] if c["label"] == label)["value"] == value


@pytest.mark.parametrize("mutation,target", [("missing_label", "label:F2"), ("missing_focus", "focus:focus-1"),
                                            ("missing_link", "material_link:purchase-link"), ("unfinished", "check:local-count")])
def test_missing_or_unfinished_obligations_are_partial_without_synthesizing_answers(mutation, target):
    package = case(); raw = answer(package)
    if mutation == "missing_label": raw["coverage"] = [r for r in raw["coverage"] if r["label"] != "F2"]
    elif mutation == "missing_focus": raw["checks"] = [c for c in raw["checks"] if c["label"] != "alert_response"]
    elif mutation == "missing_link": raw["checks"] = [c for c in raw["checks"] if c["label"] != "material_relation"]
    else: next(c for c in raw["checks"] if c["label"] == "count").update(status="unfinished", value=None)
    result = baseline.run_baseline(package, model=ScriptedModel([json_message(raw)]))
    assert result["run_status"] == "partial" and target in result["unfinished_targets"]
    assert result["raw_response"] == json_message(raw)
    assert len(result["parsed_output"]["checks"]) == len(raw["checks"])


def test_no_candidate_claim_declaration_is_not_proof_of_no_missed_claims():
    package = case(); raw = answer(package)
    raw["checks"] = [c for c in raw["checks"] if c["label"] != "count"]
    next(c for c in raw["coverage"] if c["label"] == "count")["status"] = "no_candidate"
    result = baseline.run_baseline(package, model=ScriptedModel([json_message(raw)]))
    assert result["run_status"] == "completed"  # Declared processing only; an independent scorer can count the omission.
    assert not any(c["label"] == "count" for c in result["parsed_output"]["checks"])


def test_upgraded_focus_completeness_check_does_not_mutate_the_frozen_input_projection():
    package = case(); package["review_scope"] = {"upgraded_leads": [{"focus_id": "extra", "text": "Raw human-scoped question", "origin_issue_id": "PRIVATE"}]}
    expected = baseline.raw_case_input(package)
    raw = answer(package)
    result = baseline.run_baseline(package, model=ScriptedModel([json_message(raw)]))
    assert result["run_status"] == "partial" and "focus:extra" in result["unfinished_targets"]
    assert result["input_projection"] == expected
    assert result["execution"]["input_hash"] == baseline.digest(expected)
    assert len(result["input_projection"]["alert"]["focuses"]) == 1


@pytest.mark.parametrize("mutation", ["extra_key", "bad_value", "duplicate_id", "dangling_issue", "missing_proposition", "old_abstained"])
def test_invalid_output_is_failed_and_raw_is_preserved_without_retry(mutation):
    package = case(); raw = answer(package)
    if mutation == "extra_key": raw["verified_truth"] = True
    elif mutation == "bad_value": raw["checks"][0]["value"] = "supported"
    elif mutation == "duplicate_id": raw["checks"][1]["check_id"] = raw["checks"][0]["check_id"]
    elif mutation == "dangling_issue": raw["issues"] = [{"issue_id": "x", "type": "other", "check_ids": ["missing-local-id"], "reason": "fixture", "evidence": []}]
    elif mutation == "missing_proposition": raw["checks"][2]["anchor"].pop("claim")
    else: raw["checks"][0]["status"] = "abstained"
    message = json_message(raw)
    model = ScriptedModel([message])
    result = baseline.run_baseline(package, model=model)
    assert result["run_status"] == "failed" and result["parsed_output"] is None
    assert result["raw_response"] == message and result["validation_errors"]
    assert len(model.calls) == 1


@pytest.mark.parametrize("content", ["not JSON", '{"checks":[],"checks":[],"coverage":[],"issues":[],"unfinished":[]}',
                                     '{"checks":NaN,"coverage":[],"issues":[],"unfinished":[]}'])
def test_invalid_json_is_an_execution_failure_not_business_abstention(content):
    message = {"role": "assistant", "content": content}
    model = ScriptedModel([message])
    result = baseline.run_baseline(case(), model=model)
    assert result["run_status"] == "failed" and result["parsed_output"] is None
    assert result["raw_response"] == message and len(result["call_records"]) == 1
    assert result["usage"] == {"complete": False, "input_tokens": None, "output_tokens": None, "total_tokens": None}


def test_unexpected_tool_request_is_preserved_but_never_executed():
    message = {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "query_transactions", "arguments": "{}"}}]}
    result = baseline.run_baseline(case(), model=ScriptedModel([message]))
    assert result["run_status"] == "failed" and result["stats"]["tool_calls"] == 0
    assert result["raw_response"] == message


def test_provider_exception_before_its_own_call_record_still_has_failed_attempt():
    class Failure:
        model = "offline-failure"
        calls = []
        def complete(self, messages):
            raise RuntimeError("fixture failure, not a live API")
    result = baseline.run_baseline(case(), model=Failure())
    assert result["run_status"] == "failed" and result["stats"]["model_calls"] == 1
    assert result["raw_response"] is None and result["parsed_output"] is None
    assert result["call_records"][0]["status"] == "failed" and result["call_records"][0]["usage"] is None


def test_empty_provider_usage_stays_unknown_instead_of_zero():
    package = case()
    class WithoutUsage(ScriptedModel):
        def complete(self, messages, tools=None):
            response = super().complete(messages, tools)
            self.calls[-1]["usage"] = {}
            return response
    result = baseline.run_baseline(package, model=WithoutUsage([json_message(answer(package))]))
    assert result["call_records"][0]["usage"] is None
    assert result["usage"] == {"complete": False, "input_tokens": None, "output_tokens": None, "total_tokens": None}


def test_repeated_identical_requests_remain_two_attempts_without_copying_prior_call_history():
    package = case(); response = json_message(answer(package))
    model = ScriptedModel([response, response])
    runs = [baseline.run_baseline(package, model=model) for _ in range(2)]
    assert len(model.calls) == 2
    assert sum(len(run["call_records"]) for run in runs) == 2
    assert runs[0]["call_records"][0]["request_hash"] == runs[1]["call_records"][0]["request_hash"]
    assert all(len(run["call_records"][0]["provider_records"]) == 1 for run in runs)


@pytest.mark.parametrize("outcome", ["success", "http_error", "length"])
def test_deepseek_budget_usage_and_safe_truncated_raw_response(monkeypatch, outcome):
    package = case(); raw = answer(package); captured = []
    monkeypatch.setattr(llm, "settings", lambda: {"DEEPSEEK_API_KEY": "TEST_CREDENTIAL_DO_NOT_PERSIST", "DEEPSEEK_MODEL": "mock-direct",
                                                "DEEPSEEK_BASE_URL": "https://example.invalid"})
    text = '{"checks":[' if outcome == "length" else json.dumps(raw)
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "prompt_cache_hit_tokens": 10}
    def post(url, **kwargs):
        captured.append(kwargs["json"])
        if outcome == "http_error": return httpx.Response(503)
        return httpx.Response(200, json={"usage": usage, "model": "mock-returned", "choices": [{"finish_reason": "length" if outcome == "length" else "stop",
            "message": {"role": "assistant", "content": text, "reasoning_content": "HIDDEN_REASONING_DO_NOT_PERSIST"}}]})
    monkeypatch.setattr(llm.httpx, "post", post)
    original = baseline.DeepSeek
    def one_call_factory(*, max_calls):
        assert max_calls == 1
        return original(max_calls=max_calls)
    monkeypatch.setattr(baseline, "DeepSeek", one_call_factory)
    result = baseline.run_baseline(package, provider="deepseek")
    assert len(captured) == 1 and "tools" not in captured[0]
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["max_tokens"] == result["execution"]["max_output_tokens"] == 4096
    assert len(result["call_records"]) == 1
    assert "TEST_CREDENTIAL_DO_NOT_PERSIST" not in baseline.canonical(result)
    assert "HIDDEN_REASONING_DO_NOT_PERSIST" not in baseline.canonical(result)
    if outcome == "http_error":
        assert result["run_status"] == "failed" and result["raw_response"] is None
        assert result["usage"]["complete"] is False and result["call_records"][0]["usage"] is None
    else:
        assert result["raw_response"] == {"role": "assistant", "content": text}
        assert result["call_records"][0]["usage"] == usage
        assert result["usage"] == {"complete": True, "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        if outcome == "length":
            assert result["run_status"] == "failed" and result["parsed_output"] is None
            assert result["call_records"][0]["finish_reason"] == "length"
            assert result["call_records"][0]["response"] == result["raw_response"]
        else:
            assert result["run_status"] == "completed"
