"""Closed synthetic acquisition mechanisms; no real provider or quality claims."""
from copy import deepcopy
import json

import pytest

from aml_qc import llm, workflow
from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import BudgetLedger, BudgetedModel, TOKEN_RESERVATION
from aml_qc.llm import GENERATION, ModelError, generation_request
from aml_qc.material_acquisition import MaterialSession, run_acquisition_review
from test_model_safety import case, json_message, response_message, ScriptedModel, tool_message


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Acquisition fixtures must not read credentials or call transport")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)
    monkeypatch.setattr(llm.httpx, "stream", forbidden)


def inputs():
    value = case()
    pool = [{"case_id": value["case_id"], "material": deepcopy(value["materials"][0]),
             "material_links": deepcopy(value["material_links"]), "credit_cost": 3, "available": True}]
    pool[0]["material"]["text"] = "PRIVATE-MATERIAL-BODY"
    pool[0]["material_links"][0]["provenance"] = "PRIVATE-OWN-LINK"
    value.update(materials=[], material_links=[])
    return value, pool


def review_replies(value):
    return [json_message({"claims": [], "unresolved": []}), response_message(value),
            json_message({"gaps": [], "leads": []})]


def acquisition_call(material_id, call_id="acquire-1"):
    return tool_message("acquire_material", {"material_id": material_id}, call_id)


def test_initial_request_is_independent_of_private_content_and_links():
    value, pool = inputs()
    changed = deepcopy(pool)
    changed[0]["material"].update(text="OTHER-PRIVATE-BODY", amount="9999.99")
    changed[0]["material_links"][0]["provenance"] = "OTHER-PRIVATE-LINK"
    models = []
    for hidden in [pool, changed]:
        model = ScriptedModel([{"role": "assistant", "content": "No acquisition chosen."}] + review_replies(value))
        run_acquisition_review(value, hidden, credit_budget=3, experiment_method="agent", provider="frozen", model=model)
        models.append(model)
    assert models[0].calls[0]["request"] == models[1].calls[0]["request"]
    initial = json.loads(models[0].calls[0]["request"]["messages"][1]["content"])
    assert set(initial) == {"response_input", "visible_materials", "public_catalog", "credits"}
    assert initial["public_catalog"] == [{"material_id": "purchase-contract", "material_type": "purchase_contract", "credit_cost": 3}]
    assert set(initial["credits"]) == {"budget", "spent", "remaining", "unit"}
    serialized = json.dumps(initial)
    assert all(secret not in serialized for secret in ["PRIVATE-MATERIAL-BODY", "PRIVATE-OWN-LINK", "OTHER-PRIVATE-BODY", "seed-01-out-00"])
    assert "transactions" not in initial and "material_links" not in initial and "private_pool" not in initial


def test_real_acquisition_reveals_only_returned_material_and_charges_once():
    value, pool = inputs(); original = deepcopy((value, pool))
    session = MaterialSession(value, pool, credit_budget=3)
    assert session.revealed_case() == value
    assert session.acquire_material("purchase-contract")["charged_credits"] == 3
    returned = session.acquire_material("purchase-contract")
    assert returned["status"] == "acquired" and returned["charged_credits"] == 0
    returned["material"]["text"] = "caller edit"
    exposed = session.revealed_case()
    assert exposed["materials"][0]["text"] == "PRIVATE-MATERIAL-BODY"
    assert exposed["material_links"][0]["provenance"] == "PRIVATE-OWN-LINK"
    exposed["materials"].clear()
    assert len(session.revealed_case()["materials"]) == 1
    assert session.credits() == {"budget": 3, "spent": 3, "remaining": 0, "unit": "hypothetical_integer_credits"}
    assert (value, pool) == original
    fresh = MaterialSession(value, pool, credit_budget=3)
    assert fresh.acquire_material("purchase-contract")["charged_credits"] == 3


def test_unavailable_missing_and_foreign_requests_have_identical_zero_cost_shape():
    value, pool = inputs(); pool[0]["available"] = False
    foreign = deepcopy(pool[0]); foreign["case_id"] = "other-case"; foreign["material"]["material_id"] = "foreign"
    session = MaterialSession(value, pool + [foreign], credit_budget=3)
    responses = [session.acquire_material(mid) for mid in ["purchase-contract", "missing", "foreign"]]
    assert [{k: v for k, v in row.items() if k != "material_id"} for row in responses] == [
        {"status": "unavailable", "charged_credits": 0}] * 3
    assert session.credits()["spent"] == 0 and session.revealed_case() == value
    assert all(row["material_id"] != "foreign" for row in session.catalog())


def test_insufficient_credits_return_no_content_no_charge_and_no_reveal():
    value, pool = inputs(); session = MaterialSession(value, pool, credit_budget=2)
    assert session.acquire_material("purchase-contract") == {"material_id": "purchase-contract", "status": "insufficient_credits", "charged_credits": 0}
    assert session.credits()["spent"] == 0 and session.revealed_case() == value


@pytest.mark.parametrize("failure", ["unavailable", "insufficient_credits"])
def test_failed_acquisition_does_not_leak_private_body_or_own_links(failure):
    value, pool = inputs(); changed = deepcopy(pool)
    changed[0]["material"]["text"] = "DIFFERENT-PRIVATE-BODY"
    changed[0]["material_links"][0]["provenance"] = "DIFFERENT-PRIVATE-LINK"
    responses = []
    for hidden in [pool, changed]:
        if failure == "unavailable": hidden[0]["available"] = False
        session = MaterialSession(value, hidden, credit_budget=2 if failure == "insufficient_credits" else 3)
        responses.append(session.acquire_material("purchase-contract"))
        assert session.credits()["spent"] == 0 and session.revealed_case() == value
    assert responses[0] == responses[1]
    assert responses[0]["status"] == failure
    assert not {"material", "material_links"} & set(responses[0])


@pytest.mark.parametrize("bad", [True, False, -1, 1.0, "3", None])
@pytest.mark.parametrize("field", ["budget", "cost"])
def test_credits_require_nonnegative_integers(bad, field):
    value, pool = inputs()
    if field == "cost": pool[0]["credit_cost"] = bad
    with pytest.raises(ValueError):
        MaterialSession(value, pool, credit_budget=bad if field == "budget" else 3)


def test_duplicate_pool_or_visible_identifiers_are_rejected_before_dispatch():
    value, pool = inputs()
    with pytest.raises(ValueError): MaterialSession(value, pool + deepcopy(pool), credit_budget=6)
    value["materials"] = [deepcopy(pool[0]["material"])]
    with pytest.raises(ValueError): MaterialSession(value, pool, credit_budget=3)


def test_acquired_links_reach_actual_fixed_full_checks_without_changing_author_response():
    value, pool = inputs(); model = ScriptedModel(review_replies(value)); original = deepcopy(value)
    before = workflow.run_review(value, mode="fixed", provider="frozen", strategy="full", previous=None,
                                 model=ScriptedModel(review_replies(value)))
    assert before["material_results"] == []
    result = run_acquisition_review(value, pool, credit_budget=3, experiment_method="fixed", provider="frozen", model=model)
    assert result["material_results"][0]["result"] == "corresponds"
    assert result["fixed_policy_reads"][1]["result"]["rows"]
    assert result["mode"] == "fixed" and result["strategy"] == "full" and result["stats"]["reused"] == 0
    assert result["material_acquisition"]["experiment_method"] == "fixed"
    assert result["material_acquisition"]["review_engine_mode"] == "fixed"
    assert result["material_acquisition"]["quality_or_agent_advantage_claimed"] is False
    response = next(row["request"] for row in model.calls if row["request"]["messages"][0]["content"] == workflow.RESPONSE_SYSTEM)
    assert json.loads(response["messages"][1]["content"]) == workflow.response_input(value)
    assert value == original


@pytest.mark.parametrize("method", ["fixed", "agent"])
def test_previously_used_model_is_rejected_before_dispatch_for_both_methods(method):
    value, pool = inputs(); model = ScriptedModel(review_replies(value))
    model.calls.append({"request": "old-run"})
    before = deepcopy((value, pool, model.calls))
    with pytest.raises(ValueError, match="尚未调用"):
        run_acquisition_review(value, pool, credit_budget=3, experiment_method=method, provider="frozen", model=model)
    assert (value, pool, model.calls) == before


def test_frozen_label_cannot_dispatch_an_unmetered_deepseek():
    value, pool = inputs(); model = object.__new__(llm.DeepSeek); model.calls = []
    with pytest.raises(ValueError, match="离线机制"):
        run_acquisition_review(value, pool, credit_budget=3, experiment_method="agent", provider="frozen", model=model)
    assert not model.calls


def test_second_agent_selector_sees_actual_acquired_content_and_links():
    value, pool = inputs()
    class DependingModel(ScriptedModel):
        def complete(self, messages, tools=None, stage=None):
            if len(self.calls) == 1:
                returned = json.loads(messages[-2]["content"])
                assert returned["material"]["text"] == "PRIVATE-MATERIAL-BODY"
                assert returned["material_links"][0]["provenance"] == "PRIVATE-OWN-LINK"
                assert json.loads(messages[-1]["content"])["credits"]["spent"] == 3
            return super().complete(messages, tools, stage)
    model = DependingModel([acquisition_call("purchase-contract"), {"role": "assistant", "content": "Done."}] + review_replies(value))
    result = run_acquisition_review(value, pool, credit_budget=3, experiment_method="agent", provider="frozen", model=model)
    assert result["material_results"][0]["result"] == "corresponds" and len(model.calls) == 5
    assert result["material_acquisition"]["credits"]["spent"] == 3


@pytest.mark.parametrize("bad_calls", [None, {}, ["not-object"], [
    {"id": "bad", "function": {"name": "acquire_material", "arguments": "[]"}}], [
    {"id": "bad", "function": {"name": "read_hidden_pool", "arguments": "{}"}}], [
    {"id": "bad", "function": {"name": "acquire_material", "arguments": '{"material_id":1}'}}], [
    {"id": "bad", "function": {"name": "acquire_material", "arguments": '{"material_id":"x","other":1}'}}]])
def test_malformed_tool_proposals_do_not_acquire_or_start_review(bad_calls):
    value, pool = inputs(); model = ScriptedModel([{"role": "assistant", "tool_calls": bad_calls}])
    with pytest.raises(ModelError):
        run_acquisition_review(value, pool, credit_budget=3, experiment_method="agent", provider="frozen", model=model)
    assert len(model.calls) == 1


@pytest.mark.parametrize("kind", ["over_limit", "duplicate_id"])
def test_batch_rejection_precedes_any_material_charge(kind, monkeypatch):
    value, pool = inputs(); message = acquisition_call("purchase-contract")
    message["tool_calls"] *= 3 if kind == "over_limit" else 2
    model = ScriptedModel([message])
    monkeypatch.setattr(MaterialSession, "acquire_material", lambda *a, **k: pytest.fail("Malformed batch must not execute"))
    with pytest.raises(ModelError):
        run_acquisition_review(value, pool, credit_budget=3, experiment_method="agent", provider="frozen", model=model)


@pytest.mark.parametrize("method", ["fixed", "agent"])
@pytest.mark.parametrize("kind", ["insufficient", "catalog_too_large", "missing_model", "ungoverned_deepseek"])
def test_same_experiment_preconditions_fail_before_model_dispatch(method, kind):
    value, pool = inputs(); budget = 3; model = ScriptedModel([]); provider = "frozen"
    if kind == "insufficient": budget = 2
    elif kind == "catalog_too_large":
        for i in range(4):
            row = deepcopy(pool[0]); row["material"]["material_id"] = f"extra-{i}"
            row["material_links"] = []; pool.append(row)
        budget = 15
    elif kind == "missing_model": model = None
    else: provider = "deepseek"
    with pytest.raises(ValueError):
        run_acquisition_review(value, pool, credit_budget=budget, experiment_method=method, provider=provider, model=model)
    assert model is None or not model.calls


class ReceiptModel(ScriptedModel):
    def complete(self, messages, tools=None, stage=None):
        answer = super().complete(messages, tools, stage)
        request = generation_request(self.model, messages, tools, stage=stage)
        self.calls[-1].update(request=request, request_hash=digest(request), response=deepcopy(answer),
            generation=deepcopy(GENERATION), stage=stage or ("tool_review" if tools else "final"),
            status="completed", finish_reason="tool_calls" if answer.get("tool_calls") else "stop",
            usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                   "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 100})
        return answer


@pytest.mark.parametrize("method", ["fixed", "agent"])
def test_one_injected_budgeted_model_covers_selection_and_common_review(method):
    value, pool = inputs(); replies = review_replies(value)
    if method == "agent": replies = [acquisition_call("purchase-contract"), {"role": "assistant", "content": "Done."}] + replies
    ledger = BudgetLedger("20", {"currency": "CNY", "unit_tokens": 1000000,
        "rates": {"input_cache_hit": "0.02", "input_cache_miss": "1", "output": "4"}})
    events = []; budget = {"max_calls": 6, "max_output_tokens": 16384,
        "total_token_budget": TOKEN_RESERVATION * 6, "currency_limit": "20"}
    model = BudgetedModel(ReceiptModel(replies), ledger, "offline-p1", budget, events.append)
    result = run_acquisition_review(value, pool, credit_budget=3, experiment_method=method, provider="frozen", model=model)
    assert len(model.calls) == (3 if method == "fixed" else 5) <= 6
    assert result["model_requests"] == model.calls
    assert all(event["run_id"] == "offline-p1" for event in events)
    assert len([e for e in events if e["event"] == "call_reserved"]) == len(model.calls)
    assert ledger.snapshot()["held"] == "0" and ledger.snapshot()["total"] == "20"
    assert result["material_acquisition"]["credits"]["unit"] == "hypothetical_integer_credits"
    assert result["material_acquisition"]["credits"]["spent"] == 3 and ledger.spent > 0


def test_four_item_catalog_agent_two_rounds_and_fixed_reveal_same_complete_set():
    value, pool = inputs()
    for i in range(3):
        row = deepcopy(pool[0]); row["material"]["material_id"] = f"other-material-{i}"
        row["material_links"] = []; row["credit_cost"] = i + 1; pool.append(row)
    ids = sorted(row["material"]["material_id"] for row in pool)
    batches = []
    for i in range(2):
        batch = acquisition_call(ids[i * 2], f"chosen-{i}-a")
        batch["tool_calls"] += acquisition_call(ids[i * 2 + 1], f"chosen-{i}-b")["tool_calls"]
        batches.append(batch)
    fixed_model = ScriptedModel(review_replies(value))
    agent_model = ScriptedModel(batches + review_replies(value))
    results = [run_acquisition_review(value, pool, credit_budget=9, experiment_method=method,
                                     provider="frozen", model=model)
               for method, model in [("fixed", fixed_model), ("agent", agent_model)]]
    assert all(result["material_acquisition"]["revealed_material_ids"] == ids for result in results)
    assert all(result["material_acquisition"]["credits"]["spent"] == 9 for result in results)
    assert all(len(result["material_acquisition"]["trace"]) == 4 for result in results)
    material_inputs = [json.loads(model.calls[-1]["request"]["messages"][1]["content"])["materials"]
                       for model in [fixed_model, agent_model]]
    assert material_inputs[0] == material_inputs[1]
    assert len(agent_model.calls) == 5 and len(fixed_model.calls) == 3
