"""Closed synthetic material acquisition, separate from the provider CNY ledger."""
from copy import deepcopy
import json
from threading import RLock

from .depgraph import canonical
from .evaluation_budget import BudgetedModel
from .ingest import validate_case
from .llm import DeepSeek, ModelError
from .workflow import response_input, run_review


VERSION = "material-acquisition-1"
ACQUISITION_SYSTEM = """你在封闭合成实验中自主选择额外材料，之后由独立Fixed全量流程核验。
初始只有原理由/作者任务、已可见材料、公开目录和假设整数credits，不含隐藏材料正文或流水。
仅从公开目录选择能缩小原必需事项的acquire_material；收到实际返回后才能依其内容继续选择。
最多2轮，每轮最多2次调用。内容未知不能编造，失败/额度不足不视为已取得。
目录不是材料正文或关系证明；已有可见资料足够或工具不能增益时可简短结束。
这是取证阶段，不输出最终业务JSON，不改写作者原理由或回应义务。credits不是CNY或银行实际费用。"""
ACQUISITION_TOOLS = [{"type": "function", "function": {"name": "acquire_material",
    "description": "按本案公开目录ID获取额外合成材料；成功首次扣假设credits，重复成功扣0。",
    "parameters": {"type": "object", "properties": {"material_id": {"type": "string"}},
                   "required": ["material_id"], "additionalProperties": False}}}]


def _credits(value):
    if type(value) is not int or value < 0:
        raise ValueError("credits必须为非负整数")
    return value


class MaterialSession:
    """One run owns one private pool and credit balance; never a cross-run cache."""

    def __init__(self, case, private_pool, *, credit_budget):
        validation = validate_case(case)
        if not validation["valid"] or case.get("profile", {}).get("data_origin") != "synthetic":
            raise ValueError("材料实验仅接受有效合成案件")
        if not isinstance(private_pool, list):
            raise ValueError("private_pool必须为对象数组")
        self._case, self._pool, self._acquired, self._events = deepcopy(case), {}, {}, []
        self._budget, self._spent, self._lock = _credits(credit_budget), 0, RLock()
        seen = set()
        for item in private_pool:
            if (not isinstance(item, dict) or not {"case_id", "material", "credit_cost"} <= set(item)
                    or set(item) - {"case_id", "material", "credit_cost", "material_links", "available"}):
                raise ValueError("私有材料记录格式无效")
            material, links = item["material"], item.get("material_links", [])
            if (not isinstance(item["case_id"], str) or not item["case_id"].strip()
                    or not isinstance(material, dict) or not isinstance(links, list)
                    or any(not isinstance(link, dict) for link in links)
                    or type(item.get("available", True)) is not bool
                    or any(not isinstance(material.get(key), str) or not material[key].strip()
                           for key in ("material_id", "material_type", "revision", "text"))):
                raise ValueError("私有材料记录格式无效")
            _credits(item["credit_cost"])
            identity = item["case_id"], material["material_id"]
            if identity in seen:
                raise ValueError("私有材料标识重复")
            seen.add(identity)
            if item["case_id"] != case["case_id"]:
                continue
            if (any(row["material_id"] == material["material_id"] for row in case.get("materials", []))
                    or any(link.get("material_id") != material["material_id"] for link in links)):
                raise ValueError("私有材料与可见材料或自有关联冲突")
            self._pool[material["material_id"]] = deepcopy(item)
        candidate = self.revealed_case()
        candidate["materials"] += [deepcopy(row["material"]) for row in self._pool.values()]
        candidate["material_links"] += [deepcopy(link) for row in self._pool.values() for link in row.get("material_links", [])]
        if not validate_case(candidate)["valid"]:
            raise ValueError("私有材料或关联无法构成有效合成案件")

    def catalog(self):
        return [{"material_id": mid, "material_type": row["material"]["material_type"],
                 "credit_cost": row["credit_cost"]} for mid, row in sorted(self._pool.items())]

    def credits(self):
        return {"budget": self._budget, "spent": self._spent, "remaining": self._budget - self._spent,
                "unit": "hypothetical_integer_credits"}

    def revealed_case(self):
        value = deepcopy(self._case)
        value.setdefault("materials", []).extend(deepcopy(row["material"]) for row in self._acquired.values())
        value.setdefault("material_links", []).extend(deepcopy(link) for row in self._acquired.values()
                                                     for link in row.get("material_links", []))
        return value

    def initial_input(self):
        return {"response_input": response_input(self._case),
                "visible_materials": deepcopy(self._case.get("materials", [])),
                "public_catalog": self.catalog(), "credits": self.credits()}

    def events(self):
        return deepcopy(self._events)

    def acquire_material(self, material_id):
        if not isinstance(material_id, str) or not material_id.strip():
            raise ValueError("material_id必须为非空文本")
        with self._lock:
            row = self._pool.get(material_id)
            result = {"material_id": material_id, "status": "unavailable", "charged_credits": 0}
            if row is not None and row.get("available", True):
                charge = 0 if material_id in self._acquired else row["credit_cost"]
                if charge > self._budget - self._spent:
                    result["status"] = "insufficient_credits"
                else:
                    self._spent += charge
                    self._acquired[material_id] = row
                    result.update(status="acquired", charged_credits=charge,
                                  material=deepcopy(row["material"]), material_links=deepcopy(row.get("material_links", [])))
            self._events.append({"tool": "acquire_material", "arguments": {"material_id": material_id},
                                 "result": deepcopy(result), "credits_after": self.credits()})
            return result


def _tool_calls(message, seen):
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ModelError("获取阶段模型消息格式无效")
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list) or len(calls) > 2:
        raise ModelError("获取阶段每轮最多2次工具调用")
    parsed, batch_ids = [], set()
    for call in calls:
        try:
            cid, function = call["id"], call["function"]
            args = json.loads(function["arguments"])
            if (not isinstance(cid, str) or not cid.strip() or cid in seen or cid in batch_ids
                    or function["name"] != "acquire_material" or not isinstance(args, dict)
                    or set(args) != {"material_id"} or not isinstance(args["material_id"], str)
                    or not args["material_id"].strip()):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ModelError("获取阶段工具调用格式无效") from None
        parsed.append((cid, args["material_id"]))
        batch_ids.add(cid)
    seen.update(batch_ids)
    return parsed


def run_acquisition_review(case, private_pool, *, credit_budget, experiment_method, provider, model):
    """Same pool/credit rules, then the same P0 Fixed full engine; no default LLM."""
    if experiment_method not in {"fixed", "agent"} or provider not in {"frozen", "deepseek"}:
        raise ValueError("未知材料实验方式")
    if model is None or not callable(getattr(model, "complete", None)) or getattr(model, "calls", None) != []:
        raise ValueError("须显式注入本次运行尚未调用的模型")
    if provider == "deepseek" and not isinstance(model, BudgetedModel):
        raise ValueError("DeepSeek材料实验须注入累计预算BudgetedModel")
    if provider == "frozen" and isinstance(model.base if isinstance(model, BudgetedModel) else model, DeepSeek):
        raise ValueError("frozen仅接受离线机制模型")
    if isinstance(model, BudgetedModel) and (model.max_calls > 6 or model.ledger.currency != "CNY" or model.ledger.total != 20):
        raise ValueError("材料实验沿用20CNY累计账本及最多6次模型调用")
    session = MaterialSession(case, private_pool, credit_budget=credit_budget)
    catalog = session.catalog()
    if len(catalog) > 4 or credit_budget < sum(row["credit_cost"] for row in catalog):
        raise ValueError("两种方式使用最多4项完整目录且相同额度须覆盖目录全部成本")
    if experiment_method == "fixed":
        for row in catalog:
            session.acquire_material(row["material_id"])
    else:
        messages = [{"role": "system", "content": ACQUISITION_SYSTEM},
                    {"role": "user", "content": canonical(session.initial_input())}]
        seen = set()
        for _ in range(2):
            message = model.complete(messages, tools=deepcopy(ACQUISITION_TOOLS), stage="tool_review")
            calls = _tool_calls(message, seen)
            messages.append(deepcopy(message))
            if not calls:
                break
            for cid, mid in calls:
                result = session.acquire_material(mid)
                messages.append({"role": "tool", "tool_call_id": cid, "content": canonical(result)})
            messages.append({"role": "user", "content": canonical({"credits": session.credits()})})
    revealed = session.revealed_case()
    assert response_input(revealed) == response_input(case)
    result = run_review(revealed, mode="fixed", provider=provider, strategy="full", previous=None, model=model)
    result["material_acquisition"] = {"version": VERSION, "experiment_method": experiment_method,
        "review_engine_mode": "fixed", "review_strategy": "full", "max_model_calls": 6,
        "max_acquisition_rounds": 2, "tools_per_round": 2, "public_catalog": catalog,
        "fixed_policy": "acquire_complete_catalog_in_material_id_order_then_fixed_full_review",
        "credits": session.credits(), "trace": session.events(),
        "revealed_material_ids": sorted({row["arguments"]["material_id"] for row in session.events() if row["result"]["status"] == "acquired"}),
        "model_evidence": "frozen_mechanism" if provider == "frozen" else "governed_model_requests",
        "quality_or_agent_advantage_claimed": False}
    return result
