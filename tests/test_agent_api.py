import json
from pathlib import Path

from fastapi.testclient import TestClient

from aml_qc import agent_api, api
from tests.test_investigation import Scripted, happy_script, call

ROOT = Path(__file__).resolve().parents[1]
CSV = ("transaction_id,direction,amount,timestamp,counterparty_token,counterparty_name\n"
       "t1,转入,200,2026-09-01 09:00:00,p1,付款人一\n"
       "t2,转出,190.50,2026-09-01 15:00:00,s1,某贸易\n")


def client(tmp_path, monkeypatch, scripts=()):
    scripts = list(scripts)
    monkeypatch.setattr(agent_api, "_model", lambda purpose: Scripted(scripts.pop(0)))
    monkeypatch.setenv("AML_QC_SPEND_LOG", str(tmp_path / "spend.jsonl"))
    app = api.create_app(tmp_path / "agent.sqlite3", seed=False)
    c = TestClient(app)
    c.post("/api/cases", json={"package": json.loads((ROOT / "data/synthetic/seed-02.json").read_text())})
    return c


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_investigation_streams_and_is_saved_then_a_human_decides(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, [happy_script()])
    streamed = events(c.post("/api/cases/seed-02/investigate", json={"provider": "deepseek"}))
    assert streamed[-1]["type"] == "done" and streamed[-1]["status"] == "completed"
    assert [e["type"] for e in streamed if e["type"] in ("tool_call", "tool_result")] == ["tool_call", "tool_result"] * 2
    sessions = c.get("/api/cases/seed-02/sessions").json()["sessions"]
    assert len(sessions) == 1 and sessions[0]["verdict"]["recommendation"] == "insufficient_evidence"
    sid = sessions[0]["session_id"]
    bad = c.post("/api/cases/seed-02/verdict-review", json={"session_id": sid, "action": "revise", "reason": "改"})
    assert bad.status_code == 400
    ok = c.post("/api/cases/seed-02/verdict-review", json={"session_id": sid, "action": "revise", "reason": "订单明细已在别处核实",
                                                          "recommendation": "exclude"}).json()
    decision = ok["agent"]["latest_decision"]
    assert decision["final_recommendation"] == "exclude" and decision["ai_recommendation"] == "insufficient_evidence"
    assert ok["audit_integrity"]["valid"] is True


def test_replay_streams_saved_events_without_a_model(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, [happy_script()])
    monkeypatch.setattr(agent_api, "REPLAY_DELAY_SECONDS", 0)
    assert c.post("/api/cases/seed-02/investigate", json={"provider": "replay"}).status_code == 404
    events(c.post("/api/cases/seed-02/investigate", json={"provider": "deepseek"}))
    replayed = events(c.post("/api/cases/seed-02/investigate", json={"provider": "replay"}))
    assert replayed[0]["replay"] is True and any(e["type"] == "verdict" for e in replayed)
    assert len(c.get("/api/cases/seed-02/sessions").json()["sessions"]) == 1


def test_stale_draft_cannot_be_adopted_after_a_source_change(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, [happy_script()])
    events(c.post("/api/cases/seed-02/investigate", json={"provider": "deepseek"}))
    sid = c.get("/api/cases/seed-02/sessions").json()["sessions"][0]["session_id"]
    package = c.get("/api/cases/seed-02").json()["package"]
    package["documents"][0]["text"] += "补充说明。"
    package["documents"][0]["revision"] = "2"
    assert c.post("/api/cases/seed-02/source", json={"package": package, "reason": "补正"}).status_code == 200
    stale = c.post("/api/cases/seed-02/verdict-review", json={"session_id": sid, "action": "adopt", "reason": "同意"})
    assert stale.status_code == 400 and "过期" in stale.json()["detail"]


def test_chat_and_intake_and_rules(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, [[
        {"role": "assistant", "content": "查", "tool_calls": [call("q1", "query_transactions", direction="out")]},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": json.dumps({"answer": "转出 3 笔 [q1]", "citations": ["q1"]}, ensure_ascii=False)}]])
    answer = events(c.post("/api/cases/seed-02/chat", json={"question": "转出几笔？"}))
    assert any(e["type"] == "answer" and e["citations"] == ["q1"] for e in answer)
    created = c.post("/api/intake", json={"case_id": "up-1", "account_id": "a1", "coverage_end": "2026-09-08 00:00:00",
                                          "transactions_csv": CSV, "focuses": ["收付背景"], "synthetic_confirmed": True})
    assert created.status_code == 200 and created.json()["package"]["transactions"][1]["amount"] == "190.50"
    refused = c.post("/api/intake", json={"case_id": "up-2", "account_id": "a1", "coverage_end": "2026-09-08", "transactions_csv": CSV,
                                          "focuses": ["x"], "synthetic_confirmed": False})
    assert refused.status_code == 400
    rules = c.get("/api/cases/seed-02/rules").json()
    assert rules["flow_profile"]["outflow"]["amount"] == "1050.00" and {f["feature_code"] for f in rules["features"]} == {"F1", "F2"}
    status = c.get("/api/agent/status").json()
    assert status["limits"]["tools_per_round"] == 3
