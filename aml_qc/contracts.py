"""Strict model-output contracts; schema validity does not prove semantic truth."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ingest import amount_cents, parse_time


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClaimOutput(StrictOutput):
    kind: Literal["count", "amount_sum", "counterparty", "time_range"]
    operator: Literal["exact", "only", "at_least", "at_most", "exists", "none"] = "exact"
    value: Any = None
    quote: str = Field(min_length=1)
    document_id: Literal["narrative"] = "narrative"
    unit: str | None = None
    direction: Literal["in", "out"] | None = None
    counterparty_ref: str | None = None
    start: str | None = None
    end: str | None = None

    @model_validator(mode="after")
    def kind_and_scope(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("查询起止时间须成对提供")
        if self.start is not None and parse_time(self.start) >= parse_time(self.end):
            raise ValueError("查询范围必须非空")
        if self.kind in {"count", "amount_sum"}:
            if self.operator == "only":
                # For a numeric quantity, "only N" is the equality relation.
                # Raw model output remains immutable in the request record.
                self.operator = "exact"
            if self.operator not in {"exists", "none"}:
                if self.kind == "count" and (type(self.value) is not int or self.value < 0):
                    raise ValueError("次数必须是非负整数")
                if self.kind == "amount_sum":
                    if not isinstance(self.value, str):
                        raise ValueError("金额必须是十进制字符串，单位元")
                    amount_cents(self.value)
        elif self.operator not in {"exact", "only", "exists", "none"}:
            raise ValueError("对手或时间不接受数值上下界限定词")
        if self.kind == "counterparty":
            values = self.value if isinstance(self.value, list) else [self.value]
            if not values or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError("对手值必须是非空名称/明确标识或其数组")
        if self.kind == "time_range":
            if not isinstance(self.value, dict) or set(self.value) != {"start", "end"}:
                raise ValueError("时间事实必须有start/end")
            if not all(isinstance(v, str) for v in self.value.values()):
                raise ValueError("时间值必须为文本")
            if parse_time(self.value["start"]) >= parse_time(self.value["end"]):
                raise ValueError("时间事实区间必须非空")
        return self


class ExtractionOutput(StrictOutput):
    # Individual claims are validated separately so one rejected item does not
    # discard valid items; rejected items always keep extraction unfinished.
    claims: list[dict]
    unresolved: list[str] = Field(default_factory=list)


class FocusOutput(StrictOutput):
    focus_id: str = Field(min_length=1)
    status: Literal["addressed", "not_addressed", "pending_judgement"]
    quote: str
    reason: str = Field(min_length=1)


class GapOutput(StrictOutput):
    basis_kind: Literal["identity_unresolved", "missing_linked_material", "material_mismatch", "explanation_support"]
    basis_ref: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    requested_material: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class LeadOutput(StrictOutput):
    observation: str = Field(min_length=1, max_length=1200)
    question: str = Field(min_length=1, max_length=600)
    basis_refs: list[str] = Field(min_length=1, max_length=5)


class SemanticOutput(StrictOutput):
    focuses: list[FocusOutput]
    gaps: list[GapOutput]
    leads: list[LeadOutput] = Field(default_factory=list, max_length=3)


def contract_schemas():
    return {"claim": ClaimOutput.model_json_schema(), "extraction": ExtractionOutput.model_json_schema(),
            "semantic": SemanticOutput.model_json_schema()}
