"use strict";
// AML 助手: a side drawer available on every page. It can explain the workbench, open cases and stages,
// look up facts in the open case with read-only tools, and propose actions the user confirms.

const SUGGESTIONS = {
  tasks: ["这个工作台怎么用？", "列出 AI 建议上报的案件", "带我看一个跑分/过渡账户的案例", "怎么上传自己的流水 CSV？"],
  case: ["这个案件 AI 为什么给出这个结论？", "转出最多的对手是谁？占多少？", "这个案件还缺哪些材料？", "下一步我该做什么？"],
};
const ASSISTANT_STAGE = { materials: "① 案件资料", rules: "② 规则检验", ai: "③ AI 研判", qc: "④ 理由质检", review: "⑤ 复核归档" };
const assistantState = { open: false, busy: false, turns: [], live: null, help: null };
try { assistantState.turns = JSON.parse(localStorage.getItem("aml-assistant-turns") || "[]"); } catch (_) { assistantState.turns = []; }

function assistantContext() {
  return { view: state.view, case_id: state.view === "workspace" ? state.caseId : null, stage: state.view === "workspace" ? state.stage : null };
}
function saveTurns() {
  try { localStorage.setItem("aml-assistant-turns", JSON.stringify(assistantState.turns.slice(-20))); } catch (_) {}
}
function toggleAssistant(open = !assistantState.open, prefill = "") {
  assistantState.open = open;
  document.body.classList.toggle("assistant-open", open);
  $("#assistant").hidden = !open;
  $("#assistant-launcher").hidden = open;
  if (open) { renderAssistant(); const input = $("#assistant-input"); if (prefill) input.value = prefill; input.focus(); }
}
function renderAssistant() {
  const panel = $("#assistant");
  const ctx = assistantContext();
  const ready = state.agentStatus?.deepseek_configured ?? true;
  const chip = ctx.case_id ? `当前：${ctx.case_id} · ${ASSISTANT_STAGE[ctx.stage] || ""}` : "当前：案件列表";
  const head = append(node("div", "assistant-head"),
    append(node("div"), append(node("strong"), node("span", "assistant-mark", "✦"), document.createTextNode(" AML 助手")), node("small", "", chip)),
    append(node("div", "assistant-head-actions"),
      Object.assign(button("新对话", "text small", () => { assistantState.turns = []; saveTurns(); renderAssistant(); }), { title: "清空本页对话" }),
      Object.assign(button("×", "icon-button", () => toggleAssistant(false)), { title: "关闭（⌘/Ctrl+J）" })));
  const body = node("div", "assistant-body");
  if (!assistantState.turns.length && !assistantState.live) {
    body.append(append(node("div", "assistant-welcome"),
      node("p", "", "我是工e鉴审的反洗钱助手。可以帮你使用工作台（打开案件、跳到阶段、解释功能），也可以在你打开的案件上查流水、解释 AI 研判和质检问题。我只给建议，决定由你在界面上提交。"),
      ready ? append(node("div", "assistant-chips"), ...(ctx.case_id ? SUGGESTIONS.case : SUGGESTIONS.tasks).map(text => button(text, "chip", () => sendAssistant(text)))) : null));
    if (!ready) body.append(helpFallback());
  }
  assistantState.turns.forEach(turn => body.append(renderTurn(turn.events)));
  if (assistantState.live) body.append(renderTurn(assistantState.live, true));
  const form = node("form", "assistant-composer");
  const input = node("textarea"); input.id = "assistant-input"; input.rows = 2; input.maxLength = 800;
  input.placeholder = ready ? "问我任何关于工作台或当前案件的问题（Enter 发送，Shift+Enter 换行）" : "未配置 DeepSeek API Key，可先看下方使用说明";
  input.disabled = !ready || assistantState.busy;
  input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); } });
  const send = Object.assign(button(assistantState.busy ? "…" : "发送", "primary small"), { type: "submit", disabled: !ready || assistantState.busy });
  form.addEventListener("submit", e => { e.preventDefault(); if (input.value.trim()) sendAssistant(input.value.trim()); });
  append(form, input, send);
  panel.replaceChildren(head, body, form, node("small", "assistant-foot", "助手只读数据、不提交任何决定；涉及费用的操作需要你点击确认。"));
  body.scrollTop = body.scrollHeight;
}
function helpFallback() {
  const box = append(node("div", "assistant-help"), node("strong", "", "使用说明（离线可用）"));
  const load = async () => {
    if (!assistantState.help) { try { assistantState.help = (await api("/api/help")).sections; } catch (_) { assistantState.help = []; } }
    assistantState.help.forEach(section => box.append(append(node("details"), node("summary", "", section.topic), node("p", "", section.text))));
  };
  load();
  return box;
}
function renderTurn(events, live = false) {
  const wrap = node("div", "assistant-turn");
  const steps = node("div", "assistant-steps");
  let stepCount = 0;
  events.forEach(event => {
    if (event.type === "user") wrap.append(append(node("div", "bubble user"), node("p", "", event.text)));
    else if (event.type === "assistant") steps.append(node("div", "assistant-step plan", event.text));
    else if (event.type === "tool_call") { stepCount += 1; steps.append(append(node("div", "assistant-step tool"), node("span", "tool-id", event.ref), node("code", "", `${event.tool}(${formatArgs(event.arguments)})`))); }
    else if (event.type === "tool_result") {
      const line = append(node("div", `assistant-step result ${event.status === "completed" ? "ok" : "fail"}`), node("span", "", (event.status === "completed" ? "✓ " : "✗ ") + event.summary));
      if (event.result && event.status === "completed" && !["read_help"].includes(event.tool)) line.append(append(node("details"), node("summary", "", "返回数据"), pretty(event.result)));
      steps.append(line);
    } else if (event.type === "ui_action") steps.append(actionCard(event, live));
  });
  if (stepCount || steps.children.length) wrap.append(append(node("details", "assistant-trace"), node("summary", "", `Agent 过程 · ${stepCount} 次工具调用`), steps));
  if (live) wrap.querySelector(".assistant-trace")?.setAttribute("open", "");
  const done = events.find(e => e.type === "answer");
  const error = events.find(e => e.type === "error");
  if (done) wrap.append(append(node("div", "bubble agent"), node("span", "who", "AML 助手"), node("p", "", done.text),
    list(done.warnings).length ? node("p", "flag warn", done.warnings.join("；")) : null,
    list(done.errors).length ? node("p", "flag warn", done.errors.join("；")) : null));
  if (error) wrap.append(node("div", "console-error", "✗ " + error.text));
  if (live && !done && !error) wrap.append(node("div", "console-typing", "助手思考中…"));
  return wrap;
}
const QUEUE_LABEL = { all: "全部预警", deep: "优先深挖", evidence: "补充尽调", close: "可快速关闭" };
const SOURCE_LABEL = { narrative: "理由与预警", transactions: "流水", materials: "材料", coverage: "覆盖 / 映射" };
const DECISION_LABEL = { adopt: "采纳 AI 草稿", revise: "修改后采纳", reject: "驳回 AI 草稿" };
function actionCard(event, live) {
  const done = {
    open_case: () => `→ 已打开 ${event.case_id} · ${ASSISTANT_STAGE[event.stage] || ""}`,
    go_to_stage: () => `→ 已切换到 ${ASSISTANT_STAGE[event.stage] || event.stage}`,
    go_to_list: () => `→ 已回到案件列表 · ${QUEUE_LABEL[event.filter] || "全部预警"}`,
    show_source: () => `→ 已打开资料页签「${SOURCE_LABEL[event.tab] || event.tab}」`,
    highlight_transactions: () => `→ 已在流水中高亮 ${event.transaction_ids.length} 笔交易`,
    open_report_draft: () => "→ 已打开可疑交易报告初稿",
    open_intake: () => "→ 已打开上传对话框",
  }[event.action];
  if (done) return node("div", "assistant-step action", done());
  const detail = event.proposal === "select_measures" ? "措施：" + list(event.measures).map(m => strMeasureLabel(m)).join("；")
    : event.proposal === "prefill_decision" ? `${DECISION_LABEL[event.decision]}${event.recommendation ? " → " + (RECOMMENDATION[event.recommendation]?.[0] || "") : ""}${event.reason_draft ? "；理由草稿：" + event.reason_draft : ""}` : "";
  const card = append(node("div", "assistant-proposal"), node("strong", "", "建议操作：" + event.label), node("p", "", event.reason), detail ? node("p", "muted", detail) : null);
  if (!live) {
    card.append(append(node("div", "stage-actions"),
      button("确认执行", "primary small", () => runProposal(event)), button("忽略", "secondary small", () => { card.remove(); })));
  }
  return card;
}
function goStage(stage) {
  if (state.view !== "workspace") { state.stage = stage; setView("workspace"); } else setStage(stage);
}
async function runProposal(event) {
  if (event.case_id && (state.caseId !== event.case_id || state.view !== "workspace")) { state.stage = "ai"; await loadCase(event.case_id, "workspace"); }
  if (event.proposal === "select_measures") { state.strMeasures = list(event.measures); state.strOpen = true; goStage("review"); return; }
  if (event.proposal === "prefill_decision") {
    state.decisionPrefill = { decision: event.decision, recommendation: event.recommendation, reason: event.reason_draft || "" };
    goStage("ai"); document.querySelector(".decision-box")?.scrollIntoView({ block: "center", behavior: "smooth" }); return;
  }
  goStage("ai");
  startInvestigation(event.proposal === "replay_investigation" ? "replay" : "deepseek");
}
// Page actions the assistant may take directly: they change what is shown, never what is recorded.
async function performAction(event) {
  if (event.action === "open_case") { state.stage = event.stage || "ai"; await loadCase(event.case_id, "workspace"); }
  else if (event.action === "go_to_stage" && state.caseId) goStage(event.stage);
  else if (event.action === "go_to_list") { state.triageFilter = event.filter || "all"; setView("tasks"); renderTasks(); }
  else if (event.action === "show_source" && state.caseId) { state.sourceTab = event.tab; state.evidence = null; goStage("materials"); }
  else if (event.action === "highlight_transactions" && state.caseId) { goStage("materials"); locateEvidence({ transaction_ids: event.transaction_ids }); }
  else if (event.action === "open_report_draft" && state.caseId) { state.strOpen = true; goStage("review"); }
  else if (event.action === "open_intake") openIntake();
}
async function sendAssistant(message) {
  if (assistantState.busy) return;
  const history = assistantState.turns.flatMap(turn => {
    const user = turn.events.find(e => e.type === "user"); const answer = turn.events.find(e => e.type === "answer");
    const where = turn.context?.case_id ? `${turn.context.case_id} · ${ASSISTANT_STAGE[turn.context.stage] || ""}` : "案件列表";
    return [user && { role: "user", text: `[当时页面：${where}] ${user.text}` }, answer && { role: "assistant", text: answer.text }].filter(Boolean);
  });
  const context = assistantContext();
  assistantState.busy = true; assistantState.live = [{ type: "user", text: message }];
  renderAssistant();
  try {
    await streamPost("/api/assistant", { message, history, context }, async event => {
      if (event.type === "done") return;
      if (event.type !== "user") assistantState.live.push(event);
      if (event.type === "ui_action" && event.action !== "propose") await performAction(event);
      if (assistantState.open) renderAssistant();
    });
  } catch (error) { assistantState.live.push({ type: "error", text: error.message }); }
  finally {
    assistantState.turns.push({ events: assistantState.live, context }); assistantState.live = null; assistantState.busy = false;
    saveTurns(); renderAssistant();
    try { state.agentStatus = await api("/api/agent/status"); } catch (_) {}
  }
}

document.body.append(
  Object.assign(node("aside", "assistant-drawer"), { id: "assistant", hidden: true }),
  Object.assign(button("✦ AML 助手", "assistant-launcher"), { id: "assistant-launcher", title: "打开 AML 助手（⌘/Ctrl+J）" }));
$("#assistant-launcher").addEventListener("click", () => toggleAssistant(true));
$("#assistant-toggle")?.addEventListener("click", () => toggleAssistant());
document.addEventListener("keydown", e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") { e.preventDefault(); toggleAssistant(); } });
api("/api/agent/status").then(status => { state.agentStatus = status; }).catch(() => {});
