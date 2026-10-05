"""One raw-case model reading, without tools, reference answers or answer repair.

Local check/issue IDs only link this response. They are never reference IDs.
Structural validity and declared coverage do not prove semantic correctness,
citation support, or that every claim in the source has been found.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .ingest import validate_case
from .llm import GENERATION, DeepSeek, generation_request
from .schema import LABEL_VALUES, load_schema


VERSION = "b0-direct-1.0"
ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "config/prompts/b0-v1/direct.txt"
Label = Literal["F1", "F2", "count", "amount_sum", "counterparty", "time_range", "material_relation", "alert_response"]
CLAIM_LABELS = {"count", "amount_sum", "counterparty", "time_range"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


# None means a JSON scalar, [spec] an array, and {field: spec} a closed
# projection. An allowed leaf never copies an arbitrary nested dictionary.
_PARTY = dict.fromkeys(("counterparty_token", "display_name_masked", "display_name", "credit_code", "account_no_masked", "type", "description"))
_TRANSACTION = dict.fromkeys(("transaction_id", "account_id", "direction", "amount", "amount_cents", "currency", "timestamp", "counterparty_token", "channel", "counterparty_name_masked", "memo"))
_DOCUMENT = dict.fromkeys(("document_id", "revision", "source", "text"))
_FOCUS = dict.fromkeys(("focus_id", "text"))
_RANGE = dict.fromkeys(("start", "end"))
_LABEL_RULE = {**dict.fromkeys(("definition", "target", "positive_example", "negative_example", "abstain_when")),
               "allowed_values": [None], "required_data": [None]}
_MATERIAL_TEMPLATE = dict.fromkeys(("subject_role", "counterparty_role", "direction", "transaction_count", "amount_relation", "period_relation", "amount_tolerance_cents"))
_SCHEMA = {**dict.fromkeys(("schema_version", "calculator_version", "purpose", "material_boundary")),
           "features": {"F1": dict.fromkeys(("minimum_days", "ratio_numerator", "ratio_denominator", "window")),
                        "F2": dict.fromkeys(("minimum_in_counterparties", "minimum_out_counterparties", "maximum_out_counterparties", "ratio_numerator", "ratio_denominator", "window_days", "anchor"))},
           "claims": {**dict.fromkeys(("exact_requires_full_coverage", "observed_counterexample_can_contradict_partial", "name_similarity_resolves_identity", "amount_unit")), "kinds": [None], "results": [None]},
           "material_templates": {name: _MATERIAL_TEMPLATE for name in ("single_purchase_payment", "contract_installments")},
           "labels": {name: _LABEL_RULE for name in LABEL_VALUES}}
RAW_FIELDS = {
    **dict.fromkeys(("case_id", "task_mode", "subject_account_id", "data_version", "coverage_start", "coverage_end", "currency", "timezone", "schema_version")),
    "profile": dict.fromkeys(("business_type", "data_origin")),
    "transactions": [_TRANSACTION], "counterparties": [_PARTY], "documents": [_DOCUMENT],
    "coverage": [{**dict.fromkeys(("coverage_id", "source", "account_id", "start", "end", "status", "reliable", "revision", "meaning")),
                  "fields": [None], "missing_ranges": [{**_RANGE, "fields": [None]}]}],
    "entity_mappings": [dict.fromkeys(("mapping_id", "source_ref", "target_token", "confirmed", "revision", "basis"))],
    "materials": [{**dict.fromkeys(("material_id", "revision", "material_type", "source", "amount", "amount_cents", "currency", "text",
                                    "order_id", "original_payment_transaction_id", "refund_transaction_id")),
                   "subject": dict.fromkeys(("account_id", "role")),
                   "counterparty": dict.fromkeys(("counterparty_token", "credit_code", "account_no_masked", "role")), "period": _RANGE}],
    "material_links": [{**dict.fromkeys(("link_id", "material_id", "revision", "material_revision", "claim_or_issue_id", "relation_template", "schema_version", "provenance")),
                        "transaction_ids": [None], "field_paths": [None]}],
    "alert": {**dict.fromkeys(("alert_id", "revision", "original_focus", "subject_account_id", "start", "end", "rule_source", "rule_version")),
              "focuses": [_FOCUS], "trigger_features": [None]},
    "review_scope": {"target_labels": [None], "upgraded_leads": [_FOCUS]}, "schema": _SCHEMA,
}


def project(value, fields, path="case"):
    if fields is None:
        if isinstance(value, (dict, list)) or not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(f"non-scalar raw field: {path}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite raw field: {path}")
        return value
    if isinstance(fields, list):
        if not isinstance(value, list):
            raise ValueError(f"raw array required: {path}")
        return [project(row, fields[0], path + "[]") for row in value]
    if not isinstance(value, dict):
        raise ValueError(f"raw object required: {path}")
    return {key: project(value[key], spec, path + "." + key) for key, spec in fields.items() if key in value}


def raw_case_input(case):
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    if "claim_amendments" in case and case["claim_amendments"] != []:
        raise ValueError("B0 cannot read human claim amendments")
    source = deepcopy(case)
    source["schema"] = case.get("schema") or load_schema()
    if source.get("alert") is None:
        source.pop("alert", None)
    visible = project(source, RAW_FIELDS)
    # The loader needs family metadata, but family/split identity is never sent
    # to the model. This is input syntax validation, not business computation.
    validation = validate_case({**visible, "case_family": "b0-input-validation"})
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    return visible


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Interval(Strict):
    start: str = Field(min_length=1)
    end: str = Field(min_length=1)


class Proposition(Strict):
    operator: Literal["exact", "only", "at_least", "at_most", "exists", "none"]
    value: int | str | list[str] | Interval | None
    unit: str | None = None


class Scope(Strict):
    account_id: str | None = None
    direction: Literal["in", "out"] | None = None
    counterparty_tokens: list[str] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None


class Anchor(Strict):
    document_id: str | None = None
    revision: str | int | None = None
    span: list[int] | None = Field(default=None, min_length=2, max_length=2)
    quote: str | None = None
    focus_id: str | None = None
    material_link_id: str | None = None
    claim: Proposition | None = None


class DocumentReference(Strict):
    type: Literal["document_span"]
    document_id: str = Field(min_length=1)
    revision: str | int
    span: list[int] = Field(min_length=2, max_length=2)
    quote: str


class TransactionReference(Strict):
    type: Literal["transactions"]
    transaction_ids: list[str]
    fields: list[str]


class MaterialReference(Strict):
    type: Literal["material"]
    material_id: str = Field(min_length=1)
    revision: str | int
    field_paths: list[str]


class FocusReference(Strict):
    type: Literal["alert_focus", "upgraded_focus"]
    focus_id: str = Field(min_length=1)
    revision: str | int | None = None


class CoverageReference(Strict):
    type: Literal["coverage"]
    coverage_id: str = Field(min_length=1)
    revision: str | int


class MappingReference(Strict):
    type: Literal["entity_mapping"]
    mapping_id: str = Field(min_length=1)
    revision: str | int


Evidence = Annotated[DocumentReference | TransactionReference | MaterialReference | FocusReference | CoverageReference | MappingReference,
                     Field(discriminator="type")]


class Check(Strict):
    check_id: str = Field(min_length=1)
    label: Label
    status: Literal["completed", "unfinished"]
    value: str | None
    object_scope: Scope
    anchor: Anchor
    reason: str = Field(min_length=1)
    evidence: list[Evidence]

    @model_validator(mode="after")
    def coherent_candidate(self):
        if self.status == "completed" and self.value not in LABEL_VALUES[self.label]:
            raise ValueError("completed check needs a value from its label vocabulary")
        if self.status == "unfinished" and self.value is not None:
            raise ValueError("unfinished check must keep value null")
        if self.label in CLAIM_LABELS:
            if not self.anchor.claim or not self.anchor.document_id or self.anchor.revision is None or self.anchor.span is None or not self.anchor.quote:
                raise ValueError("claim check needs its proposition and document anchor")
        return self


class Coverage(Strict):
    label: Label
    status: Literal["completed", "no_candidate", "unfinished"]
    reason: str = Field(min_length=1)


class Issue(Strict):
    issue_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    check_ids: list[str]
    reason: str = Field(min_length=1)
    evidence: list[Evidence]


class Unfinished(Strict):
    target: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DirectOutput(Strict):
    checks: list[Check]
    coverage: list[Coverage]
    issues: list[Issue]
    unfinished: list[Unfinished]

    @model_validator(mode="after")
    def local_identifiers(self):
        check_ids = [c.check_id for c in self.checks]
        issue_ids = [i.issue_id for i in self.issues]
        labels = [c.label for c in self.coverage]
        if len(set(check_ids)) != len(check_ids) or len(set(issue_ids)) != len(issue_ids) or len(set(labels)) != len(labels):
            raise ValueError("duplicate local check/issue/coverage identifier")
        if any(set(i.check_ids) - set(check_ids) for i in self.issues):
            raise ValueError("issue references an absent local check")
        return self


def parse_output(message):
    if not isinstance(message, dict) or message.get("tool_calls"):
        raise ValueError("B0 must return JSON without requesting tools")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON output key")
            value[key] = item
        return value
    def invalid_constant(_value):
        raise ValueError("non-finite JSON output")
    decoded = json.loads(message.get("content") or "", object_pairs_hook=unique, parse_constant=invalid_constant)
    return DirectOutput.model_validate(decoded).model_dump()


def incomplete_targets(visible, output):
    """Only declared/source-ID coverage, never an inferred list of true claims."""
    targets = visible.get("review_scope", {}).get("target_labels", list(LABEL_VALUES))
    coverage = {r["label"]: r for r in output["coverage"]}
    checks = output["checks"]
    missing = []
    for label in targets:
        row = coverage.get(label)
        matching = [c for c in checks if c["label"] == label]
        if row is None or row["status"] == "unfinished" or (row["status"] == "completed" and not matching):
            missing.append("label:" + label)
        if row and row["status"] == "no_candidate" and (label not in CLAIM_LABELS or matching):
            missing.append("inconsistent_coverage:" + label)
    alert = visible.get("alert") or {}
    focuses = list(alert.get("focuses", []))
    if not focuses and alert.get("original_focus"):
        focuses = [{"focus_id": alert.get("alert_id", "original-focus")}]
    focuses += visible.get("review_scope", {}).get("upgraded_leads", [])
    if visible["task_mode"] == "alert_review" or "alert_response" in targets or focuses:
        if not focuses:
            missing.append("alert:missing_focus")
        for focus in focuses:
            if not any(c["label"] == "alert_response" and c["anchor"]["focus_id"] == focus["focus_id"] for c in checks):
                missing.append("focus:" + focus["focus_id"])
    if "material_relation" in targets:
        for link in visible.get("material_links", []):
            if not any(c["label"] == "material_relation" and c["anchor"]["material_link_id"] == link["link_id"] for c in checks):
                missing.append("material_link:" + link["link_id"])
    missing.extend("check:" + c["check_id"] for c in checks if c["status"] == "unfinished")
    missing.extend("reported:" + u["target"] for u in output["unfinished"])
    return sorted(set(missing))


def run_baseline(case, *, provider="frozen", model=None):
    started = perf_counter()
    result = {"method": "B0", "mode": "b0", "baseline_version": VERSION, "case_id": case.get("case_id") if isinstance(case, dict) else None,
              "provider": provider, "run_status": "failed", "raw_response": None, "parsed_output": None,
              "validation_errors": [], "unfinished_targets": [], "call_records": [], "execution": {}, "input_projection": None,
              "usage": {"complete": False, "input_tokens": None, "output_tokens": None, "total_tokens": None},
              "stats": {"model_calls": 0, "tool_calls": 0, "duration_ms": None}}
    try:
        if provider not in {"frozen", "deepseek"}:
            raise ValueError("unknown B0 provider")
        visible = raw_case_input(case)
        result["input_projection"] = deepcopy(visible)
        prompt = PROMPT_PATH.read_text()
        output_schema = DirectOutput.model_json_schema()
        messages = [{"role": "system", "content": prompt + "\n输出JSON结构：\n" + canonical(output_schema)},
                    {"role": "user", "content": canonical({"raw_case": visible})}]
        result["execution"] = {"source_hash": digest(case), "input_hash": digest(visible),
                               "prompt_hash": digest(messages[0]["content"]), "schema_hash": digest(visible["schema"]),
                               "prompt_file_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
                               "output_contract_hash": digest(output_schema), "implementation_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "max_model_calls": 1, "tools_available": False, **deepcopy(GENERATION)}
        if model is None:
            if provider != "deepseek":
                raise ValueError("frozen B0 requires an injected model")
            model = DeepSeek(max_calls=1)
        result["execution"]["model"] = getattr(model, "model", None)
        if getattr(model, "execution_budget_spec", None) is not None:
            result["execution"]["evaluation_budget"] = deepcopy(model.execution_budget_spec)
        prior_calls = len(getattr(model, "calls", []))
        request = generation_request(getattr(model, "model", None), messages)
        record = {"request": request, "request_hash": digest(request), "status": "started", "usage": None,
                  "provider_records": [], "stage": "final", "generation": deepcopy(GENERATION)}
        result["call_records"].append(record)
        result["stats"]["model_calls"] = 1
        try:
            call_started = perf_counter()
            message = model.complete(messages)
            result["raw_response"] = {k: deepcopy(v) for k, v in message.items() if k in {"role", "content", "tool_calls"}} if isinstance(message, dict) else deepcopy(message)
            record["status"] = "completed"
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__)
            result["validation_errors"].append("model_call_failed:" + type(error).__name__)
            return result
        finally:
            record["duration_ms"] = round((perf_counter() - call_started) * 1000, 3)
            records = getattr(model, "calls", [])[prior_calls:]
            record["provider_records"] = [{k: deepcopy(r[k]) for k in ("request", "request_hash", "status", "usage", "duration_ms", "model_returned", "system_fingerprint", "finish_reason", "dispatch_status", "budget_event_id", "stage", "generation") if k in r} for r in records]
            if len(records) == 1:
                for key in ("dispatch_status", "budget_event_id"):
                    if key in records[0]:
                        record[key] = deepcopy(records[0][key])
                usage = records[0].get("usage") or {}
                if result["raw_response"] is None and isinstance(records[0].get("response"), dict):
                    result["raw_response"] = {k: deepcopy(v) for k, v in records[0]["response"].items() if k in {"role", "content", "tool_calls"}}
                record["response"] = deepcopy(result["raw_response"])
                record["usage"] = deepcopy(records[0].get("usage")) or None
                if "request_hash" in records[0]:
                    record["provider_request_hash"] = records[0]["request_hash"]
                record["model_returned"] = records[0].get("model_returned")
                record["finish_reason"] = records[0].get("finish_reason")
                values = [usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
                if all(type(n) is int and n >= 0 for n in values) and values[2] == values[0] + values[1]:
                    result["usage"] = dict(zip(("input_tokens", "output_tokens", "total_tokens"), values)) | {"complete": True}
        if len(records) > 1:
            raise ValueError("injected model recorded more than one call")
        result["parsed_output"] = parse_output(message)
        result["unfinished_targets"] = incomplete_targets(visible, result["parsed_output"])
        result["run_status"] = "partial" if result["unfinished_targets"] else "completed"
    except ValidationError as error:
        result["validation_errors"] = error.errors(include_input=False, include_context=False)
    except (ValueError, TypeError, KeyError, OSError) as error:
        result["validation_errors"].append(str(error))
    finally:
        result["stats"]["duration_ms"] = round((perf_counter() - started) * 1000, 3)
    return result
