import json
from pathlib import Path

from fastapi.testclient import TestClient

from aml_qc import agent_api, api, assistant
from tests.test_investigation import Scripted, call

ROOT = Path(__file__).resolve().parents[1]
SEED = json.loads((ROOT / "data/synthetic/seed-02.json").read_text())


def bench():
    rows = [{"case_id": "seed-02", "ai_recommendation": None, "run_status": "not_run"}]
    return assistant.Workbench(lambda: rows, lambda case_id: {"case_id": case_id, "qc_issues": []}, lambda case_id: SEED)


def answer(text, citations=()):
    return {"role": "assistant", "content": json.dumps({"answer": text, "citations": list(citations)}, ensure_ascii=False)}


def test_help_question_reads_the_builtin_help_and_cites_it():
    turn = assistant.converse("怎么上传流水？", [], {"view": "tasks"}, bench(), Scripted([
        {"role": "assistant", "content": "查说明", "tool_calls": [call("h", "read_help", topic="上传")]},
        {"role": "assistant", "content": ""}, answer("在案件列表点“上传资料新建案件” [A1]", ["A1"])]))
    assert turn["status"] == "completed" and turn["errors"] == []
    assert "transaction_id" in turn["trace"][0]["result"]["text"]


def test_navigation_is_returned_as_ui_actions_and_unknown_cases_are_refused():
    events = []
    turn = assistant.converse("打开 seed-02 的研判", [], {"view": "tasks"}, bench(), Scripted([
        {"role": "assistant", "content": "打开", "tool_calls": [call("o", "open_case", case_id="seed-02", stage="ai"),
                                                               call("x", "open_case", case_id="nope")]},
        {"role": "assistant", "content": ""}, answer("已打开 [A1]", ["A1"])]), emit=events.append)
    assert turn["ui_actions"] == [{"action": "open_case", "case_id": "seed-02", "stage": "ai"}]
    assert [e["type"] for e in events].count("ui_action") == 1 and turn["trace"][1]["status"] == "failed"


def test_case_tools_need_an_open_case_and_proposals_are_never_executed():
    model = Scripted([
        {"role": "assistant", "content": "查流水", "tool_calls": [call("q", "query_transactions", direction="out")]},
        {"role": "assistant", "content": ""}, answer("请先打开案件")])
    closed = assistant.converse("转出几笔？", [], {"view": "tasks"}, bench(), model)
    assert closed["trace"][0]["status"] == "failed"
    assert "query_transactions" not in {d["function"]["name"] for d in assistant.tools(False)}
    assert "query_transactions" in {d["function"]["name"] for d in assistant.tools(True)}
    opened = assistant.converse("转出几笔？要不要研判？", [], {"view": "workspace", "case_id": "seed-02", "stage": "ai"}, bench(), Scripted([
        {"role": "assistant", "content": "查", "tool_calls": [call("q", "query_transactions", direction="out"),
                                                             call("p", "propose_action", action="start_investigation", reason="尚未研判")]},
        {"role": "assistant", "content": ""}, answer("转出 3 笔，合计 1050.00 元 [A1]", ["A1"])]))
    assert opened["trace"][0]["status"] == "completed" and opened["warnings"] == []
    assert opened["ui_actions"] == [{"action": "propose", "proposal": "start_investigation", "label": assistant.ACTIONS["start_investigation"],
                                     "reason": "尚未研判", "case_id": "seed-02"}]


def test_assistant_endpoint_streams_actions_and_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_api, "_model", lambda purpose: Scripted([
        {"role": "assistant", "content": "去规则检验", "tool_calls": [call("g", "go_to_stage", stage="rules")]},
        {"role": "assistant", "content": ""}, answer("已切换到规则检验 [A1]", ["A1"])]))
    c = TestClient(api.create_app(tmp_path / "a.sqlite3", seed=False))
    c.post("/api/cases", json={"package": SEED})
    lines = [json.loads(l[6:]) for l in c.post("/api/assistant", json={"message": "看看规则检验",
             "context": {"view": "workspace", "case_id": "seed-02", "stage": "ai"}}).text.splitlines() if l.startswith("data: ")]
    assert {"type": "ui_action", "action": "go_to_stage", "stage": "rules"}.items() <= next(e for e in lines if e["type"] == "ui_action").items()
    assert lines[-1]["type"] == "done" and lines[-1]["answer"].startswith("已切换")


def test_figures_quoted_in_returned_text_are_not_flagged():
    rows = [{"case_id": "seed-02"}]
    wb = assistant.Workbench(lambda: rows, lambda c: {"case_id": c, "ai_summary": "转入 68,500.00 元，共 33 笔", "qc_issues": []}, lambda c: SEED)
    turn = assistant.converse("概况？", [], {"view": "tasks"}, wb, Scripted([
        {"role": "assistant", "content": "", "tool_calls": [call("s", "case_status", case_id="seed-02")]},
        {"role": "assistant", "content": ""}, answer("共 33 笔、68,500.00 元；另有 99 笔 [A1]", ["A1"])]))
    assert turn["warnings"] == ["以下数字未在工具返回中找到，请核对：99笔"]


def drafted_bench():
    rows = [{"case_id": "seed-02", "ai_recommendation": "report_suspicious", "run_status": "not_run"}]
    status = {"case_id": "seed-02", "ai_recommendation": "report_suspicious", "ai_draft_stale": False, "qc_issues": []}
    return assistant.Workbench(lambda: rows, lambda case_id: status, lambda case_id: SEED)


def test_page_actions_are_validated_against_the_open_case():
    first_tx = SEED["transactions"][0]["transaction_id"]
    turn = assistant.converse("带我看这笔交易", [], {"view": "workspace", "case_id": "seed-02", "stage": "ai"}, bench(), Scripted([
        {"role": "assistant", "content": "", "tool_calls": [call("h", "highlight_transactions", transaction_ids=[first_tx, first_tx]),
                                                             call("x", "highlight_transactions", transaction_ids=["not-in-case"]),
                                                             call("r", "open_report_draft")]},
        {"role": "assistant", "content": ""}, answer("已高亮 [A1]", ["A1"])]))
    assert turn["ui_actions"] == [{"action": "highlight_transactions", "transaction_ids": [first_tx]}]
    assert [t["status"] for t in turn["trace"]] == ["completed", "failed", "failed"]
    assert "没有当前有效的 AI 研判草稿" in turn["trace"][2]["result"]["error"]


def test_going_back_to_the_queue_drops_the_case_tools():
    turn = assistant.converse("回到列表看可快速关闭的", [], {"view": "workspace", "case_id": "seed-02", "stage": "ai"}, bench(), Scripted([
        {"role": "assistant", "content": "", "tool_calls": [call("l", "go_to_list", filter="close")]},
        {"role": "assistant", "content": "", "tool_calls": [call("q", "query_transactions", direction="out")]},
        {"role": "assistant", "content": ""}, answer("已筛选 [A1]", ["A1"])]))
    assert turn["ui_actions"] == [{"action": "go_to_list", "filter": "close"}]
    assert turn["trace"][1]["status"] == "failed"


def test_measures_and_decision_prefill_are_only_proposed_and_never_submitted():
    turn = assistant.converse("帮我直接采纳并勾选持续监测", [], {"view": "workspace", "case_id": "seed-02", "stage": "review"}, drafted_bench(), Scripted([
        {"role": "assistant", "content": "", "tool_calls": [
            call("m", "propose_action", action="select_measures", reason="风险较高", measures=["monitor", "monitor"]),
            call("d", "propose_action", action="prefill_decision", reason="同意草稿", decision="adopt", reason_draft="转出集中且无材料"),
            call("b", "propose_action", action="prefill_decision", reason="改结论", decision="revise")]},
        {"role": "assistant", "content": ""}, answer("已为你准备好，请核对后提交 [A1]", ["A1"])]))
    measures, prefill = turn["ui_actions"]
    assert measures["action"] == "propose" and measures["measures"] == ["monitor"]
    assert prefill == {"action": "propose", "proposal": "prefill_decision", "label": assistant.ACTIONS["prefill_decision"], "reason": "同意草稿",
                       "case_id": "seed-02", "decision": "adopt", "recommendation": None, "reason_draft": "转出集中且无材料"}
    assert turn["trace"][2]["status"] == "failed" and "recommendation" in turn["trace"][2]["result"]["error"]
    assert not any(d["function"]["name"] in ("verdict_review", "submit_decision") for d in assistant.tools(True))
