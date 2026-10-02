from copy import deepcopy
from pathlib import Path

import pytest

from aml_qc import core
from aml_qc.depgraph import Evaluator, sources_for
from aml_qc.ingest import load_case
from aml_qc.llm import ModelError
from aml_qc.workflow import default_schema, execute_tool, normalize_claims, normalize_semantic


def case(number=4):
    return load_case(Path(__file__).resolve().parents[1] / f"data/synthetic/seed-{number:02d}.json")


def claim(value):
    return normalize_claims(value, {"claims": [{"kind": "count", "value": 1,
        "direction": "out", "counterparty_ref": "乙公司", "quote": "仅向乙公司支付一次货款"}], "unresolved": []})["claims"][0]


def response(value, gap):
    return {"focuses": [{"focus_id": f["focus_id"], "status": "pending_judgement", "quote": "",
                        "reason": "开发结构测试，不是模型质量证据"} for f in value["alert"]["focuses"]], "gaps": [gap]}


def test_material_mismatch_cannot_override_corresponds():
    value = case(); facts = core.check_materials(value)
    assert facts[0]["result"] == "corresponds"
    gap = {"basis_kind": "material_mismatch", "basis_ref": facts[0]["link_id"],
           "quote": "仅向乙公司支付一次货款", "reason": "合同字段不对应", "requested_material": "合同"}
    with pytest.raises(ModelError, match="确定性结果冲突"):
        normalize_semantic(value, response(value, gap), {"material_results": facts})


def test_identity_gap_keeps_corresponding_material_separate():
    value = case(); extracted = claim(value)
    checked = core.verify_claim(value, extracted)
    assert checked["execution_status"] == "identity_unresolved"
    gap = {"basis_kind": "identity_unresolved", "basis_ref": extracted["claim_id"],
           "quote": extracted["text"], "reason": "这个自由文本甚至错误声称合同字段不能核验", "requested_material": "原文名称对应资料"}
    result = normalize_semantic(value, response(value, gap), {"claim_results": [checked]})["gaps"][0]
    assert "材料字段对应不等于" in result["reason"]
    assert "合同字段不能核验" not in result["reason"]
    assert "合同字段不能核验" in result["model_draft"]["reason"]
    assert result["model_draft"]["status"] == "unverified_model_suggestion"


def test_nonexistent_claim_reference_is_rejected():
    value = case()
    gap = {"basis_kind": "identity_unresolved", "basis_ref": "invented-claim", "quote": "仅向乙公司支付一次货款",
           "reason": "虚构对象ID", "requested_material": "映射"}
    with pytest.raises(ModelError, match="没有绑定"):
        normalize_semantic(value, response(value, gap), {})


def test_existing_material_cannot_be_claimed_missing():
    value = case(); facts = core.check_materials(value)
    gap = {"basis_kind": "missing_linked_material", "basis_ref": facts[0]["link_id"],
           "quote": "仅向乙公司支付一次货款", "reason": "不存在合同", "requested_material": "合同"}
    with pytest.raises(ModelError, match="实际存在"):
        normalize_semantic(value, response(value, gap), {"material_results": facts})


def test_explanation_suggestion_cannot_change_material_or_fact_assertions():
    value = case()
    gap = {"basis_kind": "explanation_support", "basis_ref": "narrative", "quote": "仅向乙公司支付一次货款",
           "reason": "模型可能误写合同字段不匹配", "requested_material": "业务解释资料"}
    checked = normalize_semantic(value, response(value, gap))["gaps"][0]
    assert checked["title"] == "业务解释支持程度需人工核实"
    assert "字段不匹配" not in checked["reason"]
    assert checked["model_draft"]["reason"] == gap["reason"]


@pytest.mark.parametrize("kind,value", [("count", True), ("count", 1.5), ("amount_sum", 10.05),
    ("amount_sum", "10.001"), ("counterparty", 3), ("time_range", {"start": "2026-01-01", "end": "2026-02-01"})])
def test_bad_claim_shapes_are_unresolved_not_silently_coerced(kind, value):
    result = normalize_claims(case(), {"claims": [{"kind": kind, "value": value, "quote": "仅向乙公司支付一次货款"}], "unresolved": []})
    assert result["claims"] == [] and result["unresolved"]


def test_ambiguous_name_can_be_extracted_without_inventing_a_token():
    result = claim(case())
    assert result["counterparty_ref"] == "乙公司"
    assert core.verify_claim(case(), result)["execution_status"] == "identity_unresolved"


def test_numeric_only_is_normalized_to_exact_without_changing_raw_response():
    response = {"claims": [{"kind": "count", "operator": "only", "value": 1,
                           "quote": "仅向乙公司支付一次货款"}], "unresolved": []}
    original = deepcopy(response)
    extracted = normalize_claims(case(), response)
    assert extracted["claims"][0]["operator"] == "exact"
    assert response == original


def test_resolver_loses_token_after_sole_transaction_deleted_incremental_equals_full():
    old = case(); old["counterparties"] = []; old["entity_mappings"] = []
    old["transactions"] = [t for t in old["transactions"] if t["direction"] == "out"]
    schema = default_schema(); execution = {"test": "resolver-dependency"}
    first = Evaluator(sources_for(old, schema, execution))
    actual = execute_tool(old, schema, first, "resolve_entity", {"entity": "supplier-b"})
    assert actual["execution_status"] == "completed"
    new = deepcopy(old); new["transactions"] = []
    incremental = Evaluator(sources_for(new, schema, execution), first.snapshot(), "incremental")
    full = Evaluator(sources_for(new, schema, execution))
    inc_result = execute_tool(new, schema, incremental, "resolve_entity", {"entity": "supplier-b"})
    full_result = execute_tool(new, schema, full, "resolve_entity", {"entity": "supplier-b"})
    assert inc_result == full_result
    assert inc_result["execution_status"] == "identity_unresolved"
    assert incremental.recomputed == 1 and incremental.reused == 0
