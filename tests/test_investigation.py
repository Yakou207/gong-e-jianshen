"""AI 研判 Agent mechanics with a scripted stand-in model (tests only; never shipped as an Agent)."""
import json
from copy import deepcopy
from pathlib import Path

from aml_qc import investigation as inv
from aml_qc.flows import flow_profile

ROOT = Path(__file__).resolve().parents[1]


def seed(name="seed-02"):
    return json.loads((ROOT / f"data/synthetic/{name}.json").read_text())


class Scripted:
    model = "scripted-test"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, tools=None, stage=None):
        self.calls.append({"stage": stage, "tools": bool(tools), "messages": deepcopy(messages),
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_cache_hit_tokens": 0}})
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


def call(cid, name, **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def verdict(**overrides):
    body = {"recommendation": "insufficient_evidence", "summary": "收付集中，收款来源缺少订单明细。",
            "risk_findings": [{"finding": "检查期内向乙公司转出 1050.00 元", "severity": "medium",
                               "evidence": [{"ref": "T2", "transaction_ids": ["seed-02-out-00"]}]}],
            "mitigating_findings": [],
            "focus_answers": [{"focus_id": "focus-1", "status": "needs_information", "answer": "付款对象与合同一致，收款背景待补订单。",
                               "evidence": [{"ref": "T1", "transaction_ids": []}]}],
            "information_requests": ["零售订单明细"], "draft_opinion": "建议补充零售订单明细后再作结论。"}
    body.update(overrides)
    return {"role": "assistant", "content": json.dumps(body, ensure_ascii=False)}


def happy_script(final=None):
    return [{"role": "assistant", "content": "先看资金画像，再核对转出明细。",
             "tool_calls": [call("c1", "flow_profile"), call("c2", "query_transactions", direction="out")]},
            {"role": "assistant", "content": "调查结束。"},
            final or verdict()]


def test_investigation_streams_real_tool_steps_and_validates_the_draft():
    events = []
    session = inv.investigate(seed(), Scripted(happy_script()), emit=events.append)
    assert session["status"] == "completed" and session["errors"] == []
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["stage", "context"] and "tool_call" in kinds and kinds[-1] == "verdict"
    results = [e for e in events if e["type"] == "tool_result"]
    assert [r["tool"] for r in results] == ["flow_profile", "query_transactions"]
    assert results[1]["summary"].startswith("返回 3 笔") and "转出 1050.00 元" in results[1]["summary"] and results[1]["status"] == "completed"
    assert session["stats"]["model_calls"] == 3 and session["stats"]["tool_calls"] == 2
    assert [r["ref"] for r in results] == ["T1", "T2"]


def test_final_step_lists_only_successful_refs_as_citable():
    model = Scripted(happy_script())
    inv.investigate(seed(), model)
    final = json.loads(model.calls[-1]["messages"][-1]["content"])
    assert [r["ref"] for r in final["citable_refs"]] == ["T1", "T2"] and model.calls[-1]["stage"] == "final"


def test_agent_context_withholds_the_analyst_narrative():
    model = Scripted(happy_script())
    inv.investigate(seed(), model)
    first_user = model.calls[0]["messages"][1]["content"]
    narrative = next(d["text"] for d in seed()["documents"] if d["document_id"] == "narrative")
    assert narrative not in first_user and "rule_results" in first_user
    refused = inv.investigate(seed(), Scripted([
        {"role": "assistant", "content": "读理由", "tool_calls": [call("c1", "read_document", document_id="narrative")]},
        {"role": "assistant", "content": ""}, verdict(risk_findings=[], focus_answers=[
            {"focus_id": "focus-1", "status": "needs_information", "answer": "待补", "evidence": [{"ref": "T1"}]}])]))
    assert refused["trace"][0]["status"] == "failed" and "经办理由" in refused["trace"][0]["result"]["error"]


def test_evidence_must_point_at_real_successful_calls_and_their_rows():
    fake = inv.investigate(seed(), Scripted(happy_script(verdict(risk_findings=[
        {"finding": "x", "severity": "low", "evidence": [{"ref": "made-up", "transaction_ids": []}]}]))))
    assert fake["status"] == "needs_attention" and any("made-up" in e for e in fake["errors"])
    wrong_row = inv.investigate(seed(), Scripted(happy_script(verdict(risk_findings=[
        {"finding": "x", "severity": "low", "evidence": [{"ref": "T2", "transaction_ids": ["seed-02-in-00"]}]}]))))
    assert any("未返回的交易" in e for e in wrong_row["errors"])


def test_numbers_not_in_tool_returns_are_flagged_for_humans():
    session = inv.investigate(seed(), Scripted(happy_script(verdict(summary="共向乙公司转出 9999.00 元，合计 3 笔。"))))
    assert session["status"] == "completed"
    assert any("9999.00元" in w for w in session["warnings"]) and not any("3笔" in w for w in session["warnings"])


def test_exclude_is_blocked_when_coverage_is_partial_and_every_focus_needs_an_answer():
    case = seed()
    case["coverage"][0]["status"] = "partial"
    session = inv.investigate(case, Scripted(happy_script(verdict(recommendation="exclude"))))
    assert any("覆盖不完整" in f for f in session["policy_flags"])
    missing = inv.investigate(seed(), Scripted(happy_script(verdict(focus_answers=[]))))
    assert any("关注点" in e for e in missing["errors"])


def test_per_round_tool_budget_and_malformed_final_output():
    many = [call(f"c{i}", "check_coverage") for i in range(1, 5)]
    session = inv.investigate(seed(), Scripted([{"role": "assistant", "content": "多查", "tool_calls": many},
                                               {"role": "assistant", "content": ""}, verdict()]))
    assert [t["status"] for t in session["trace"]] == ["completed"] * 3 + ["failed"]
    broken = inv.investigate(seed(), Scripted(happy_script({"role": "assistant", "content": "{\"recommendation\": \"maybe\"}"})))
    assert broken["status"] == "failed" and broken["verdict"] is None and broken["errors"]


def test_chat_answers_with_tools_and_checks_citations():
    first = inv.investigate(seed(), Scripted(happy_script()))
    turn = inv.chat(seed(), [first], "转给乙公司的是哪几笔？", Scripted([
        {"role": "assistant", "content": "按对手查询", "tool_calls": [call("q1", "query_transactions", direction="out", counterparty_token="supplier-b")]},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": json.dumps({"answer": "共 3 笔，合计 1050.00 元 [Q1]", "citations": ["Q1", "T2", "nope"]}, ensure_ascii=False)}]))
    assert turn["status"] == "completed" and turn["answer"]["answer"].startswith("共 3 笔")
    assert turn["errors"] == ["引用了不存在的工具返回 nope"]


def test_flow_profile_is_exact_and_coverage_aware():
    profile = flow_profile(seed())
    assert profile["inflow"]["amount"] == "1200.00" and profile["outflow"]["amount"] == "1050.00"
    assert profile["outflow"]["top_counterparties"][0]["share_percent"] == 100.0
    assert profile["coverage"]["status"] == "full" and profile["out_to_in_percent"] == 87.5


def test_material_read_carries_its_declared_links_so_citations_can_be_checked():
    session = inv.investigate(seed(), Scripted([
        {"role": "assistant", "content": "读材料", "tool_calls": [call("m1", "read_material", material_id="purchase-contract")]},
        {"role": "assistant", "content": ""},
        verdict(risk_findings=[], mitigating_findings=[{"finding": "合同对应三笔付款", "severity": "low",
                "evidence": [{"ref": "T1", "transaction_ids": ["seed-02-out-00"]}]}],
                focus_answers=[{"focus_id": "focus-1", "status": "explained", "answer": "合同对应", "evidence": [{"ref": "T1"}]}])]))
    assert session["errors"] == [] and session["trace"][0]["result"]["material_links"][0]["transaction_ids"]
