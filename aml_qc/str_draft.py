"""Suspicious-transaction report draft assembled deterministically from the case, the AI draft and the human decision.

Sections follow the usual elements of a suspicious-transaction report (subject, transactions, fund source and
destination, suspicious features and analysis, opinion, measures). Field definitions follow the institution's own
reporting system; this draft is for the screener to edit and is never submitted by the system.
"""
from . import core
from .flows import flow_profile

MEASURES = {
    "raise_risk": "调高客户洗钱风险等级，并相应提高审查频率",
    "monitor": "对该账户开展持续交易监测",
    "edd": "开展强化尽职调查，补充上述材料",
    "limit_channel": "在风险相称的前提下限制非柜面交易渠道或额度",
}
RECOMMENDATION = {"report_suspicious": "建议上报可疑交易", "exclude": "建议排除", "insufficient_evidence": "证据不足，建议补充尽调"}
ACTION = {"verdict_adopt": "采纳 AI 草稿", "verdict_revise": "修改后采纳", "verdict_reject": "驳回 AI 草稿"}
SEVERITY = {"high": "高", "medium": "中", "low": "低"}
DIRECTION = {"in": "转入", "out": "转出"}
MAX_ROWS = 30


def build(package, session, decision=None, measures=()):
    verdict = (session or {}).get("verdict") or {}
    refs = {r: e.get("transaction_ids", []) for f in verdict.get("risk_findings", []) for e in f["evidence"] for r in [e["ref"]]}
    cited = []
    for finding in verdict.get("risk_findings", []):
        for evidence in finding["evidence"]:
            cited += [t for t in evidence.get("transaction_ids", []) if t not in cited]
    by_id = {t["transaction_id"]: t for t in package.get("transactions", [])}
    names = {p["counterparty_token"]: p.get("display_name_masked", p["counterparty_token"]) for p in package.get("counterparties", [])}
    rows = [[t["transaction_id"], t["timestamp"][:16].replace("T", " "), DIRECTION.get(t["direction"], t["direction"]), t["amount"],
             names.get(t["counterparty_token"], t.get("counterparty_name_masked", "")), t.get("memo", "")]
            for t in sorted((by_id[i] for i in cited if i in by_id), key=lambda t: t["timestamp"])]
    profile = flow_profile(package)
    kyc = next((d["text"] for d in package.get("documents", []) if d["document_id"] == "kyc"), "")
    final = (decision or {}).get("final_recommendation")
    unknown = [m for m in measures if m not in MEASURES]
    if unknown:
        raise ValueError("未知措施：" + "、".join(unknown))

    def top(side):
        return [[c["display_name"], str(c["count"]), c["amount"], f"{c['share_percent']}%"] for c in profile[side]["top_counterparties"]]

    sections = [
        {"title": "一、报告主体", "rows": [
            ["账户", package["subject_account_id"]], ["客户类型", (package.get("profile") or {}).get("business_type", "")],
            ["客户资料（未经核实）", kyc], ["检查期间", f"{package['coverage_start'][:10]} 至 {package['coverage_end'][:10]}（不含结束日）"],
            ["流水完整性", {"full": "完整", "partial": "不完整"}.get(core.check_coverage(package)["status"], "未知")]]},
        {"title": "二、可疑交易明细", "note": f"草稿证据引用的交易共 {len(rows)} 笔" + (f"，此处列出前 {MAX_ROWS} 笔" if len(rows) > MAX_ROWS else ""),
         "head": ["交易编号", "时间", "方向", "金额（元）", "对手", "备注"], "table": rows[:MAX_ROWS]},
        {"title": "三、资金来源与去向", "note": f"检查期间转入 {profile['inflow']['count']} 笔、合计 {profile['inflow']['amount']} 元；"
                                          f"转出 {profile['outflow']['count']} 笔、合计 {profile['outflow']['amount']} 元；"
                                          f"转出/转入 {profile['out_to_in_percent']}%。",
         "subtables": [{"caption": "主要资金来源", "head": ["对手", "笔数", "金额（元）", "占比"], "table": top("inflow")},
                       {"caption": "主要资金去向", "head": ["对手", "笔数", "金额（元）", "占比"], "table": top("outflow")}]},
        {"title": "四、可疑特征与疑点分析", "items": [f"【{SEVERITY[f['severity']]}】{f['finding']}" for f in verdict.get("risk_findings", [])]
                                              + [f"已排查的合理解释：{f['finding']}" for f in verdict.get("mitigating_findings", [])]},
        {"title": "五、研判意见", "rows": [
            ["AI 研判结论", RECOMMENDATION.get(verdict.get("recommendation"), "无有效草稿")],
            ["AI 研判意见草稿", verdict.get("draft_opinion", "")],
            ["人工决定", ACTION.get((decision or {}).get("action"), "尚未作出（本稿仅供参考，不得报送）")],
            ["最终结论", RECOMMENDATION.get(final, "—") if decision else "—"],
            ["决定人与理由", f"{decision['actor']}：{decision['reason']}" if decision else "—"]]},
        {"title": "六、已采取或建议采取的措施", "items": [MEASURES[m] for m in measures] or ["（尚未选择）"],
         "note": "措施由甄别人员选择，系统只记录、不执行；须与风险状况相称，并保障客户获得基本必需的金融服务（《反洗钱法》第三十条）。"},
        {"title": "七、尚需补充的尽调事项", "items": verdict.get("information_requests") or ["无"]},
    ]
    ready = bool(decision) and final == "report_suspicious"
    return {"case_id": package["case_id"], "session_id": (session or {}).get("session_id"), "ready_to_submit_for_review": ready,
            "status": "人工决定为上报，可交复核" if ready else "未经人工决定上报，仅供参考",
            "sections": sections, "cited_refs": sorted(refs), "markdown": markdown(package["case_id"], sections, ready)}


def markdown(case_id, sections, ready):
    lines = [f"# 可疑交易报告初稿 · {case_id}", "",
             "> 本稿由工e鉴审依据工具返回与人工决定自动汇编，全部为合成数据；字段口径以机构报送系统为准，系统不报送。"
             + ("" if ready else "当前未经人工决定上报，仅供参考。"), ""]
    for section in sections:
        lines += [f"## {section['title']}", ""]
        if section.get("note"):
            lines += [section["note"], ""]
        for key, value in section.get("rows", []):
            lines.append(f"- **{key}**：{value}")
        for item in section.get("items", []):
            lines.append(f"- {item}")
        for table in ([section] if "table" in section else []) + section.get("subtables", []):
            if table.get("caption"):
                lines += ["", f"**{table['caption']}**", ""]
            lines += ["| " + " | ".join(table["head"]) + " |", "|" + "---|" * len(table["head"])]
            lines += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in table["table"]]
        lines.append("")
    return "\n".join(lines)
