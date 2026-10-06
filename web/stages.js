"use strict";
// Five-stage case workspace: ① 案件资料 ② 规则检验 ③ AI 研判 ④ 理由质检 ⑤ 复核归档.
// Shares helpers and state with app.js (loaded first).

const STAGES = [
  ["materials", "① 案件资料", "经办 · 资料接入"],
  ["rules", "② 规则检验", "系统 · 确定性代码"],
  ["ai", "③ AI 研判", "Agent · 大模型 + 只读工具"],
  ["qc", "④ 理由质检", "系统 + AI · 核验处置理由"],
  ["review", "⑤ 复核归档", "人工 · 最终决定"],
];
const RECOMMENDATION = {
  report_suspicious: ["建议上报可疑交易", "red"], exclude: ["建议排除", "green"], insufficient_evidence: ["证据不足，建议补充尽调", "amber"],
};
const TOOL_LABEL = {
  query_transactions: "查询流水", flow_profile: "资金画像", compute_features: "计算形态特征", check_coverage: "核查资料覆盖",
  resolve_entity: "核实对象", read_material: "读取材料", read_document: "读取客户资料",
};
const VERDICT_ACTION = { verdict_adopt: "采纳", verdict_revise: "修改后采纳", verdict_reject: "驳回" };
Object.assign(state, { stage: state.stage || "ai", rules: {}, sessions: {}, live: null, agentStatus: null, streaming: false });

function setStage(stage) {
  state.stage = stage;
  const hash = `#workspace/${encodeURIComponent(state.caseId)}/${stage}`;
  if (location.hash !== hash) history.replaceState(null, "", hash);
  renderDetail();
  window.scrollTo({ top: 0, behavior: "instant" });
}
function stageDone(stage) {
  const agent = state.detail?.agent || {};
  if (stage === "materials") return true;
  if (stage === "rules") return true;
  if (stage === "ai") return Boolean(agent.latest_investigation?.verdict);
  if (stage === "qc") return Boolean(run().run_id) && !isStale();
  return Boolean(agent.latest_decision) && reviews().some(e => e.target_id === "task" && ["confirm", "reconfirm"].includes(e.action));
}
function stageNav() {
  const bar = node("nav", "stage-nav"); bar.setAttribute("aria-label", "处理阶段");
  STAGES.forEach(([key, title, role]) => {
    const item = button("", `stage-step ${state.stage === key ? "active" : ""} ${stageDone(key) ? "done" : ""}`, () => setStage(key));
    item.setAttribute("aria-current", state.stage === key ? "step" : "false");
    append(item, node("strong", "", title), node("small", "", role));
    bar.append(item);
  });
  return bar;
}
function renderStage(target) {
  const stage = STAGES.some(s => s[0] === state.stage) ? state.stage : "ai";
  if (stage === "materials") renderMaterials(target);
  else if (stage === "rules") renderRules(target);
  else if (stage === "ai") renderAgent(target);
  else if (stage === "qc") {
    target.append(stageIntro("检验经办人员写的处置理由", "系统从理由中抽取次数、金额、对手、时间四类事实，由确定性代码对照流水核验；AI 判断理由是否回应了每个预警关注点、材料能否支撑。质检员逐条确认或驳回。"), statusStrip());
    if (isStale()) target.append(node("div", "stale-banner", `↻ ${state.detail?.engine_changed ? "执行代码已变化" : "来源已变化"}，以下旧运行结果仅供历史参考，请重查。`));
    renderWorkbench(target);
  } else renderArchive(target);
}
function stageIntro(title, text) {
  return append(node("div", "stage-intro"), node("strong", "", title), node("p", "", text));
}

// ① 案件资料
function renderMaterials(target) {
  const p = pkg();
  const focuses = list(p.alert?.focuses);
  const overview = append(node("div", "section-body"),
    append(node("dl", "metadata-grid"),
      ...[["被检账户", p.subject_account_id], ["经营类型", p.profile?.business_type], ["检查期间", `${compact(p.coverage_start).slice(0, 10)} 至 ${compact(p.coverage_end).slice(0, 10)}（不含）`],
        ["流水笔数", list(p.transactions).length], ["对手账户", list(p.counterparties).length], ["材料份数", list(p.materials).length]]
        .flatMap(([k, v]) => [node("dt", "", k), node("dd", "", compact(v))])),
    node("h3", "mini-heading", `预警关注点 ${focuses.length} 条`),
    append(node("ol", "focus-list"), ...focuses.map(f => node("li", "", f.text))),
    append(node("div", "stage-actions"), button("上传资料新建案件", "secondary", openIntake), button("修订本案资料", "secondary", () => openEditor("source")), button("进入规则检验 →", "primary", () => setStage("rules"))));
  const grid = append(node("div", "two-col"), append(node("div", "column"), panel("案件概况", "经办录入 · 资料版本 " + compact(p.data_version), overview)), append(node("div", "column"), sourcePanel()));
  target.append(stageIntro("资料接入", "导入预警、交易流水、客户资料与支持材料，并声明流水是否完整。下游所有判断都只基于这里的资料和版本。"), grid);
}

// ② 规则检验
async function loadRules() {
  try { state.rules[state.caseId] = await api(casePath("/rules")); renderDetail(); }
  catch (error) { info(error.message, true); }
}
function renderRules(target) {
  target.append(stageIntro("系统规则检验", "以下全部由确定性代码按整数分计算，不调用大模型，可复算。F1/F2 为演示口径，用于发现值得调查的资金形态，不是犯罪认定。"));
  const data = state.rules[state.caseId];
  if (!data) { target.append(append(node("div", "panel"), append(node("div", "section-body"), node("div", "loading-indicator"), node("p", "muted center", "正在计算…")))); loadRules(); return; }
  const f = data.flow_profile;
  const tiles = append(node("div", "summary-grid compact"),
    tile("资料覆盖", label(data.coverage.status), data.coverage.status === "full" ? "检查期内流水完整" : "存在缺失时段，结论需保留"),
    tile("转入", `${f.inflow.amount} 元`, `${f.inflow.count} 笔 · ${f.inflow.distinct_counterparties} 个对手`),
    tile("转出", `${f.outflow.amount} 元`, `${f.outflow.count} 笔 · ${f.outflow.distinct_counterparties} 个对手`),
    tile("转出/转入", f.out_to_in_percent === null ? "—" : `${f.out_to_in_percent}%`, `同日收付 ${f.days_with_both_directions.length} 天 · 夜间交易 ${f.night_transaction_ids_00_06.length} 笔`));
  const features = append(node("div", "section-body"), ...list(data.features).map(feature => {
    const row = append(node("div", "feature-row"), append(node("div"), node("strong", "", `${feature.feature_code} ${feature.feature_code === "F1" ? "短时收付集中" : "分散收款集中付款"}`),
      node("small", "", feature.feature_code === "F1" ? "自然日窗口内转出/转入≥80%的完整日≥3个" : "7天窗口内≥10个转入对手、1–2个转出对手且转出/转入≥80%")), badge(feature.result));
    const detail = append(node("details", "source-details"), node("summary", "", "逐窗口指标"), pretty(feature.windows || feature.metrics || feature));
    return append(node("div"), row, detail);
  }));
  const parties = direction => {
    const side = f[direction];
    const table = append(node("table", "small-table"), append(node("thead"), append(node("tr"), node("th", "", "对手"), node("th", "", "笔数"), node("th", "", "金额（元）"), node("th", "", "占比"))));
    const body = node("tbody");
    side.top_counterparties.forEach(p => body.append(append(node("tr"), node("td", "", p.display_name), node("td", "numeric", p.count), node("td", "numeric", p.amount), node("td", "numeric", p.share_percent === null ? "—" : `${p.share_percent}%`))));
    table.append(body);
    return append(node("div", "section-body"), table);
  };
  target.append(tiles, append(node("div", "two-col"),
    append(node("div", "column"), panel("资金形态特征", `规范 ${data.schema_version} · 演示阈值`, features)),
    append(node("div", "column"), panel("转出对手集中度", "前 5 名", parties("outflow")), panel("转入对手集中度", "前 5 名", parties("inflow")))),
    node("p", "scope-note", data.note), append(node("div", "stage-actions"), button("交给 AI 研判 →", "primary", () => setStage("ai"))));
}
function tile(title, value, caption) {
  return append(node("div", "summary-card"), node("div", "summary-label", title), node("div", "summary-value small-value", value), node("small", "", caption));
}

// ③ AI 研判
async function loadSessions() {
  try {
    const [sessions, status] = await Promise.all([api(casePath("/sessions")), api("/api/agent/status")]);
    state.sessions[state.caseId] = sessions.sessions; state.agentStatus = status;
    if (state.stage === "ai" && !state.streaming) renderDetail();
  } catch (error) { info(error.message, true); }
}
function sessionEvents(session) {
  return list(session.events).map(e => ({ ...e, session: session.session_id, kind: session.kind, stale: session.stale }));
}
function renderAgent(target) {
  target.append(stageIntro("AI 研判（Agent）", "Agent 围绕预警关注点自主选择只读工具取证：每一步调用什么、返回了什么都实时显示。金额与笔数只能来自工具返回；草稿由系统自动校验证据引用后，交人工采纳、修改或驳回。"));
  const sessions = state.sessions[state.caseId];
  if (!sessions) { target.append(append(node("div", "panel"), append(node("div", "section-body"), node("div", "loading-indicator")))); loadSessions(); return; }
  const grid = append(node("div", "agent-grid"), consolePanel(sessions), verdictPanel(sessions));
  target.append(grid);
  const scroller = $(".console-stream", target); if (scroller) scroller.scrollTop = scroller.scrollHeight;
}
function consolePanel(sessions) {
  const status = state.agentStatus || {};
  const element = node("section", "panel console");
  const ready = status.deepseek_configured;
  const head = append(node("div", "console-head"),
    append(node("div"), node("strong", "", "研判 Agent"), node("small", "", `deepseek-flash · 每轮≤${status.limits?.tools_per_round ?? 3} 个工具 · ≤${status.limits?.rounds ?? 4} 轮 · 7 个只读工具`)),
    append(node("div", "console-actions"),
      Object.assign(button(state.streaming ? "运行中…" : "开始研判", "primary small", () => startInvestigation("deepseek")), { disabled: state.streaming || !ready, title: ready ? "调用 DeepSeek 实时研判" : "未配置 API Key" }),
      Object.assign(button("回放上次研判", "secondary small", () => startInvestigation("replay")), { disabled: state.streaming || !sessions.some(s => s.kind === "investigation" && s.verdict) })));
  const stream = node("div", "console-stream"); stream.setAttribute("aria-live", "polite");
  const events = state.live?.case === state.caseId ? state.live.events : sessions.flatMap(sessionEvents);
  if (!events.length) stream.append(empty("还没有研判记录", ready ? "点击“开始研判”，Agent 会先读预警和规则检验结果，再决定查什么。" : "未配置 DeepSeek API Key：可在 .env 配置后重启工作台。"));
  renderEvents(stream, events);
  const composer = node("form", "composer");
  const input = node("textarea"); input.rows = 2; input.placeholder = ready ? "追问 Agent，例如：转给丑设备的那笔是什么时间？占转出多少？" : "配置 API Key 后可追问";
  input.disabled = !ready || state.streaming; input.maxLength = 500;
  const send = Object.assign(button("发送", "primary small"), { type: "submit", disabled: !ready || state.streaming });
  input.addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) composer.requestSubmit(); });
  composer.addEventListener("submit", e => { e.preventDefault(); if (input.value.trim()) startChat(input.value.trim()); });
  append(composer, input, send);
  const spend = status.spent_cny !== undefined ? node("small", "console-foot", `本机工作台累计 API 费用约 ${Number(status.spent_cny).toFixed(3)} 元 / 上限 ${status.cap_cny} 元（按高峰价估算）`) : null;
  return append(element, head, stream, composer, spend);
}
function renderEvents(stream, events) {
  const calls = {};
  let lastSession = null;
  events.forEach(event => {
    if (event.session && event.session !== lastSession) {
      lastSession = event.session;
      stream.append(node("div", "console-divider", `${event.kind === "chat" ? "追问" : "研判"}${event.stale ? " · 资料已更新，此记录已过期" : ""}`));
    }
    if (event.type === "stage") stream.append(node("div", "console-stage", "· " + event.text));
    else if (event.type === "context") {
      const card = append(node("details", "console-context"), node("summary", "", `Agent 输入：规则检验结果与 ${list(event.focuses).length} 个预警关注点（不含经办理由）`),
        node("p", "", list(event.rule_results?.features).map(f => `${f.feature_code} ${label(f.result)}`).join(" · ") + ` · 覆盖 ${label(event.rule_results?.coverage)}`),
        append(node("ol"), ...list(event.focuses).map(f => node("li", "", `${f.focus_id}：${f.text}`))));
      stream.append(card);
    } else if (event.type === "assistant") stream.append(append(node("div", "bubble agent"), node("span", "who", "Agent"), node("p", "", event.text)));
    else if (event.type === "user") stream.append(append(node("div", "bubble user"), node("span", "who", "甄别人员"), node("p", "", event.text)));
    else if (event.type === "tool_call") {
      const block = append(node("div", "tool-block running"), append(node("div", "tool-line"), node("span", "tool-name", `${TOOL_LABEL[event.tool] || event.tool}`),
        node("code", "", `${event.tool}(${formatArgs(event.arguments)})`), node("span", "tool-id", event.ref || "")));
      block.id = "call-" + (event.session || "live") + "-" + (event.ref || event.id || "");
      block.dataset.ref = event.ref || "";
      calls[event.ref || event.id] = block; stream.append(block);
    } else if (event.type === "tool_result") {
      const block = calls[event.ref || event.id];
      const result = append(node("div", `tool-result ${event.status === "completed" ? "ok" : "fail"}`), node("span", "", event.status === "completed" ? "✓ " + event.summary : "✗ " + event.summary));
      if (event.result) result.append(append(node("details"), node("summary", "", "查看返回数据"), pretty(event.result)));
      if (block) { block.classList.remove("running"); block.append(result); } else stream.append(result);
    } else if (event.type === "verdict") {
      const [text, tone] = RECOMMENDATION[event.verdict.recommendation] || [event.label, ""];
      stream.append(append(node("div", "bubble agent verdict-bubble"), node("span", "who", "研判草稿"), append(node("p"), node("span", `badge ${tone}`, text), document.createTextNode(" " + event.verdict.summary)),
        list(event.errors).length ? node("p", "flag error", "校验未通过：" + event.errors.join("；")) : null,
        list(event.policy_flags).length ? node("p", "flag error", event.policy_flags.join("；")) : null,
        list(event.warnings).length ? node("p", "flag warn", event.warnings.join("；")) : null));
    } else if (event.type === "answer") stream.append(append(node("div", "bubble agent"), node("span", "who", "Agent"), node("p", "", event.text),
      list(event.citations).length ? append(node("div", "cite-row"), ...event.citations.map(c => citeChip(c))) : null,
      list(event.errors).length ? node("p", "flag warn", event.errors.join("；")) : null));
    else if (event.type === "error") stream.append(node("div", "console-error", "✗ " + event.text));
  });
  if (state.streaming && state.live?.case === state.caseId) stream.append(node("div", "console-typing", "Agent 工作中…"));
}
function formatArgs(args) {
  if (!args || typeof args !== "object") return "";
  return Object.entries(args).map(([k, v]) => `${k}=${String(v).replace(/T00:00:00\+08:00$/, "")}`).join(", ");
}
function citeChip(id, rows) {
  const chip = button(rows && rows.length ? `${id} · ${rows.length} 笔` : id, "evidence-link", () => {
    const blocks = [...document.querySelectorAll(".tool-block")].filter(b => b.dataset.ref === id);
    const target = blocks[blocks.length - 1];
    if (target) { target.scrollIntoView({ behavior: "smooth", block: "center" }); target.classList.add("flash"); setTimeout(() => target.classList.remove("flash"), 1600); }
  });
  chip.title = "定位到该次工具返回";
  return chip;
}
function verdictPanel(sessions) {
  const latest = [...sessions].reverse().find(s => s.kind === "investigation");
  const body = node("div", "section-body verdict");
  if (!latest) { body.append(node("p", "inline-empty", "运行研判后，这里显示研判草稿、证据引用和自动校验结果。")); return panel("研判草稿", "待生成", body); }
  if (!latest.verdict) { body.append(node("p", "flag error", "本次研判未形成有效草稿：" + list(latest.errors).join("；"))); return panel("研判草稿", "未完成", body); }
  const v = latest.verdict; const [text, tone] = RECOMMENDATION[v.recommendation];
  append(body, append(node("div", "verdict-head"), node("span", `badge big ${tone}`, text), latest.stale ? badge("stale") : null), node("p", "verdict-summary", v.summary));
  [["error", latest.errors, "证据引用校验未通过"], ["error", latest.policy_flags, "规则拦截"], ["warn", latest.warnings, "数字核对提示"]].forEach(([kind, items, title]) => {
    if (list(items).length) body.append(append(node("div", `guard ${kind}`), node("strong", "", title), ...items.map(i => node("p", "", i))));
  });
  if (!list(latest.errors).length) body.append(node("div", "guard ok", "✓ 每条发现都引用了实际成功的工具调用，交易 ID 均来自该次返回"));
  const findings = (title, items) => items.length ? append(node("div", "verdict-section"), node("h3", "", title), ...items.map(f => append(node("div", "finding"),
    append(node("div"), f.severity ? badge(f.severity === "high" ? "高风险" : f.severity === "medium" ? "中风险" : "低", "") : null, document.createTextNode(" " + f.finding)),
    append(node("div", "cite-row"), ...f.evidence.map(ev => citeChip(ev.ref, ev.transaction_ids)))))) : null;
  append(body, findings("可疑特征", v.risk_findings), findings("支持合理解释的事实", v.mitigating_findings));
  body.append(append(node("div", "verdict-section"), node("h3", "", "预警关注点逐条回答"), ...v.focus_answers.map(a => append(node("div", "finding"),
    append(node("div"), badge(a.status === "explained" ? "addressed" : a.status === "unexplained" ? "not_addressed" : "pending_judgement"), document.createTextNode(` ${a.focus_id}：${a.answer}`)),
    append(node("div", "cite-row"), ...a.evidence.map(ev => citeChip(ev.ref, ev.transaction_ids)))))));
  if (v.information_requests.length) body.append(append(node("div", "verdict-section"), node("h3", "", "建议补充尽调"), append(node("ul"), ...v.information_requests.map(r => node("li", "", r)))));
  body.append(append(node("div", "verdict-section"), node("h3", "", "研判意见草稿"), node("p", "narrative-text", v.draft_opinion)));
  body.append(decisionForm(latest));
  const stats = latest.stats || {};
  return panel("研判草稿", `${stats.model_calls ?? "—"} 次模型调用 · ${stats.tool_calls ?? "—"} 次工具调用`, body);
}
function decisionForm(session) {
  const agent = state.detail?.agent || {};
  const decision = agent.latest_decision && agent.latest_decision.target_id === "verdict:" + session.session_id ? agent.latest_decision : null;
  const box = node("form", "decision-box");
  box.append(node("h3", "", "人工决定"));
  if (decision) box.append(append(node("div", "decision-done"), node("strong", "", `${decision.actor} ${VERDICT_ACTION[decision.action]}`),
    node("span", "", decision.final_recommendation ? ` · 最终结论：${RECOMMENDATION[decision.final_recommendation][0]}` : " · 不采用 AI 结论"),
    node("p", "", decision.reason), node("small", "", formatDate(decision.created_at))));
  const action = select([["adopt", "采纳 AI 草稿"], ["revise", "修改后采纳"], ["reject", "驳回 AI 草稿"]], "adopt");
  const rec = select(Object.entries(RECOMMENDATION).map(([k, [t]]) => [k, t]), session.verdict.recommendation);
  const recField = field("最终研判结论", rec); recField.hidden = true;
  action.addEventListener("change", () => { recField.hidden = action.value !== "revise"; });
  const reason = node("textarea"); reason.placeholder = "写明采纳、修改或驳回的依据"; reason.rows = 3; reason.required = true;
  const actor = select([["甄别人员", "甄别人员"], ["复核员", "复核员"]], "甄别人员");
  const submit = Object.assign(button(decision ? "追加新的人工决定" : "提交人工决定", "primary run-button"), { type: "submit", disabled: session.stale || state.busy });
  append(box, append(node("div", "runner-fields"), field("人工操作", action), field("角色", actor)), recField, field("理由（必填）", reason), submit,
    node("p", "review-disclaimer", session.stale ? "资料已更新，此草稿已过期，请重新研判。" : "人工决定以追加事件写入审计链；AI 草稿本身不会被修改。"));
  box.addEventListener("submit", async e => {
    e.preventDefault(); if (!reason.value.trim()) return;
    state.busy = true;
    try {
      state.detail = await api(casePath("/verdict-review"), { method: "POST", body: JSON.stringify({ session_id: session.session_id, action: action.value, reason: reason.value.trim(), actor: actor.value, recommendation: action.value === "revise" ? rec.value : null }) });
      info("人工决定已记录。"); renderDetail();
    } catch (error) { info(error.message, true); }
    finally { state.busy = false; }
  });
  return box;
}
async function streamPost(path, payload, onEvent) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  if (!response.ok) { let text = await response.text(); try { text = errorText(JSON.parse(text).detail); } catch (_) {} throw new Error(text || `HTTP ${response.status}`); }
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
  for (;;) {
    const { value, done } = await reader.read(); if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, index); buffer = buffer.slice(index + 2);
      if (chunk.startsWith("data: ")) onEvent(JSON.parse(chunk.slice(6)));
    }
  }
}
async function runLive(path, payload, seed) {
  const caseId = state.caseId;
  state.streaming = true;
  state.live = { case: caseId, events: [...list(state.sessions[caseId]).flatMap(sessionEvents), ...seed] };
  renderDetail();
  try {
    await streamPost(path, payload, event => {
      if (event.type === "done") return;
      state.live.events.push(event);
      if (state.stage === "ai" && state.caseId === caseId) {
        const stream = $(".console-stream"); if (stream) { stream.replaceChildren(); renderEvents(stream, state.live.events); stream.scrollTop = stream.scrollHeight; }
      }
    });
  } catch (error) { info(error.message, true); }
  finally {
    state.streaming = false; state.live = null;
    delete state.sessions[caseId];
    await refresh();
  }
}
function startInvestigation(provider) {
  runLive(casePath("/investigate"), { provider }, provider === "replay" ? [] : [{ type: "stage", text: "启动研判 Agent", session: "live", kind: "investigation" }]);
}
function startChat(question) {
  runLive(casePath("/chat"), { question }, []);
}

// ⑤ 复核归档
function renderArchive(target) {
  const agent = state.detail?.agent || {};
  const latest = agent.latest_investigation; const decision = agent.latest_decision;
  const confirmed = issues().filter(i => reviews().some(e => e.target_id === i.issue_id && ["confirm", "reconfirm"].includes(e.action))).length;
  const rejected = issues().filter(i => reviews().some(e => e.target_id === i.issue_id && e.action === "reject")).length;
  const tiles = append(node("div", "summary-grid compact"),
    tile("AI 研判草稿", latest?.verdict ? RECOMMENDATION[latest.verdict.recommendation][0] : "未研判", latest?.stale ? "资料已更新，需重新研判" : latest ? "已生成" : "请先到③研判"),
    tile("人工最终结论", decision ? (decision.final_recommendation ? RECOMMENDATION[decision.final_recommendation][0] : "已驳回 AI 草稿") : "待决定", decision ? `${decision.actor} · ${VERDICT_ACTION[decision.action]}` : "在③提交人工决定"),
    tile("理由质检问题", `${issues().filter(i => i.type !== "new_lead").length} 条`, `已确认 ${confirmed} · 已驳回 ${rejected}`),
    tile("未决事项", `${openItems().length} 项`, state.detail?.review_status ? label(state.detail.review_status) : "—"));
  target.append(stageIntro("复核与归档", "汇总 AI 研判的人工决定与理由质检结果；资料变化后在这里增量重查，受影响的人工记录会被标为需复核；最后导出带版本的审计包。"), tiles, statusStrip());
  renderDelivery(target);
}

// 上传资料新建案件
function openIntake() {
  let dialog = $("#intake-dialog");
  if (!dialog) {
    dialog = node("dialog", "editor-dialog"); dialog.id = "intake-dialog"; document.body.append(dialog);
  }
  const form = node("form");
  const input = (name, title, placeholder = "", value = "") => { const el = node("input"); el.name = name; el.placeholder = placeholder; el.value = value; return field(title, el); };
  const area = (name, title, placeholder, rows = 3) => { const el = node("textarea"); el.name = name; el.placeholder = placeholder; el.rows = rows; return field(title, el); };
  const csv = node("textarea"); csv.name = "transactions_csv"; csv.rows = 6; csv.className = "code-editor small-editor";
  csv.placeholder = "transaction_id,direction,amount,timestamp,counterparty_token,counterparty_name\nT001,转入,200.00,2026-09-01 09:00:00,P01,付款人甲\nT002,转出,1500.00,2026-09-01 15:20:00,S01,某贸易公司";
  const file = node("input"); file.type = "file"; file.accept = ".csv,text/csv";
  file.addEventListener("change", async () => { if (file.files[0]) csv.value = await file.files[0].text(); });
  const status = select([["full", "完整（检查期内流水齐全）"], ["partial", "不完整（有缺失时段）"]], "full"); status.name = "coverage_status";
  const confirm = node("input"); confirm.type = "checkbox";
  const error = node("p", "form-error"); error.hidden = true;
  append(form, append(node("div", "dialog-heading"), append(node("div"), node("div", "eyebrow", "INTAKE"), node("h2", "", "上传资料新建案件")), Object.assign(button("×", "icon-button", () => dialog.close()), { ariaLabel: "关闭" })),
    node("p", "dialog-description", "上传交易流水 CSV 并填写预警关注点。系统会自动建立案件包、记录资料版本，之后可直接进入规则检验和 AI 研判。"),
    append(node("div", "runner-fields"), input("case_id", "案件编号", "如 case-2026-001"), input("account_id", "被检账户", "如 acct-001")),
    append(node("div", "runner-fields"), input("business_type", "经营类型", "如 社区餐饮店"), field("流水完整性", status)),
    append(node("div", "runner-fields"), input("coverage_start", "检查期起点", "2026-09-01 00:00:00"), input("coverage_end", "检查期终点（不含）", "2026-09-15 00:00:00")),
    append(field("交易流水 CSV（可选择文件或粘贴）", csv), file),
    area("focuses", "预警关注点（每行一条）", "请核实集中收款的资金来源\n请说明向某公司大额转出的用途"),
    area("kyc_text", "客户资料", "经营范围、开户信息、客户自述等"),
    area("narrative_text", "经办处置理由（可选，用于④理由质检）", "如：检查期间仅向某公司付款一次，系采购货款……"),
    append(node("label", "check-label"), confirm, document.createTextNode("我确认这些是合成或已脱敏的演示数据，不含真实客户信息。")),
    error, append(node("div", "dialog-actions"), button("取消", "secondary", () => dialog.close()), Object.assign(button("创建案件", "primary"), { type: "submit" })));
  form.addEventListener("submit", async e => {
    e.preventDefault(); error.hidden = true;
    const data = Object.fromEntries(new FormData(form).entries());
    const payload = { case_id: data.case_id.trim(), account_id: data.account_id.trim(), business_type: data.business_type, coverage_start: data.coverage_start.trim(),
      coverage_end: data.coverage_end.trim(), coverage_status: data.coverage_status, transactions_csv: csv.value, focuses: data.focuses.split("\n"),
      kyc_text: data.kyc_text, narrative_text: data.narrative_text, synthetic_confirmed: confirm.checked };
    try {
      const created = await api("/api/intake", { method: "POST", body: JSON.stringify(payload) });
      dialog.close(); info("案件已创建。"); await refresh(); state.stage = "rules"; await loadCase(created.package.case_id, "workspace");
    } catch (err) { error.textContent = err.message; error.hidden = false; }
  });
  dialog.replaceChildren(form); dialog.showModal();
}
