"""Validate authored task requirements without inventing response obligations."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aml_qc.ingest import load_case, validate_case


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "seed-01.json"


@pytest.fixture(params=["alert", "upgraded_leads"])
def focus_case(request):
    case = load_case(DATA)
    focus = {"focus_id": "authored-focus", "text": "说明该交易的业务用途，并列明核对结果及未解决事项。"}
    if request.param == "alert":
        case["alert"]["focuses"] = [focus]
    else:
        case.setdefault("review_scope", {})["upgraded_leads"] = [focus]
    return case, focus


def test_optional_requirements_preserve_old_focuses(focus_case):
    case, _ = focus_case
    assert validate_case(case)["valid"]


def test_all_authored_requirement_kinds_are_valid_and_input_is_not_rewritten(focus_case):
    case, focus = focus_case
    focus["response_requirements"] = [
        {"kind": "explanation", "text": "  说明款项用途。  "},
        {"kind": "verification_result", "text": "列明交易时间核对结果。"},
        {"kind": "unresolved_item", "text": "列明尚未解决的对象定位问题。"},
    ]
    before = deepcopy(case)
    assert validate_case(case)["valid"]
    assert case == before


@pytest.mark.parametrize("requirements", [None, [], {}, "说明用途", False])
def test_present_requirements_must_be_nonempty_array(focus_case, requirements):
    case, focus = focus_case
    focus["response_requirements"] = requirements
    result = validate_case(case)
    assert not result["valid"]
    assert any("response_requirements" in error for error in result["errors"])


@pytest.mark.parametrize("requirement", [
    None,
    "说明用途",
    {},
    {"kind": "explanation"},
    {"text": "说明用途"},
    {"kind": "explanation", "text": "说明用途", "model_inferred": True},
    {"kind": "other", "text": "说明用途"},
    {"kind": ["explanation"], "text": "说明用途"},
    {"kind": True, "text": "说明用途"},
    {"kind": "explanation", "text": " \n\t "},
    {"kind": "explanation", "text": None},
    {"kind": "explanation", "text": 1},
])
def test_requirement_items_have_only_valid_kind_and_nonblank_text(focus_case, requirement):
    case, focus = focus_case
    focus["response_requirements"] = [requirement]
    result = validate_case(case)
    assert not result["valid"]
    assert any("response_requirements" in error for error in result["errors"])


@pytest.mark.parametrize("second_text", ["说明用途。", "  说明用途。\n"])
def test_duplicate_requirement_kind_and_trimmed_text_are_rejected(focus_case, second_text):
    case, focus = focus_case
    focus["response_requirements"] = [
        {"kind": "explanation", "text": "说明用途。"},
        {"kind": "explanation", "text": second_text},
    ]
    result = validate_case(case)
    assert not result["valid"]
    assert any("response_requirements" in error and "重复" in error for error in result["errors"])


def test_same_text_under_distinct_authored_kinds_is_not_semantically_deduplicated(focus_case):
    case, focus = focus_case
    focus["response_requirements"] = [
        {"kind": "verification_result", "text": "对象定位结果。"},
        {"kind": "unresolved_item", "text": "对象定位结果。"},
    ]
    assert validate_case(case)["valid"]


@pytest.mark.parametrize("focuses", [None, {}, [None], [{"focus_id": "f", "text": " "}],
                                    [{"focus_id": [], "text": "说明用途。"}],
                                    [{"focus_id": "f", "text": "事项一"}, {"focus_id": "f", "text": "事项二"}]])
def test_both_focus_collections_validate_shape_and_identity(focus_case, focuses):
    case, focus = focus_case
    if case["alert"].get("focuses") == [focus]:
        case["alert"]["focuses"] = focuses
    else:
        case["review_scope"]["upgraded_leads"] = focuses
    assert not validate_case(case)["valid"]


def test_original_focus_fallback_and_missing_alert_remain_compatible():
    case = load_case(DATA)
    case["alert"].pop("focuses", None)
    case["alert"]["original_focus"] = "说明款项用途。"
    assert validate_case(case)["valid"]
    case["alert"]["focuses"] = []
    assert validate_case(case)["valid"]
    case["alert"] = None
    assert validate_case(case)["valid"]
    del case["alert"]
    assert validate_case(case)["valid"]


def test_case_loader_rejects_invalid_authored_requirements(focus_case, tmp_path):
    case, focus = focus_case
    focus["response_requirements"] = [{"kind": "verification_result", "text": ""}]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="response_requirements"):
        load_case(path)
