"use strict";

const state = {
  cases: [], config: {}, detail: null, caseId: null, view: "tasks", sourceTab: "narrative",
  evidence: null, reviewTarget: null, reviewAction: "confirm", provider: "local", mode: "fixed", busy: false,
  editorMode: "source", editorPackage: null, editorSection: "all", loadSerial: 0,
  annotationTarget: null, annotationEvidence: [], resultTab: "issues",
  claimWork: null,
  migration: { open: false, catalog: [], baseHash: "", draft: "", reason: "", actor: "质检员", preview: null, selected: [], error: "", loading: false, requiresRefresh: false },
};
const $ = (selector, root = document) => root.querySelector(selector);
const list = value => Array.isArray(value) ? value : [];
const compact = value => value === undefined || value === null || value === "" ? "—" : String(value);
const json = value => JSON.stringify(value, null, 2);
const first = (...values) => values.find(value => value !== undefined && value !== null && value !== "");
const dictionary = {
  pending: "待处理", not_started: "未运行", not_run: "未运行", running: "运行中", completed: "运行完成", complete: "已完成", partial: "部分完成", failed: "执行失败", computed: "已计算", reused: "已复用",
  return_for_revision: "建议退回修订", return_for_correction: "建议退回修订", revise: "建议退回修订", request_correction: "要求补正", needs_revision: "建议退回修订",
  request_evidence: "建议补证", supplement: "建议补证", needs_evidence: "建议补证", pending_judgement: "待人工判断", manual_review: "待人工判断",
  no_definite_issue: "未发现确定问题", no_definite_issues: "未发现确定问题", no_issues_found: "未发现确定问题", no_issue_found: "未发现确定问题",
  awaiting_review: "待处理", awaiting_evidence: "待补证", needs_correction: "待补正", disputed: "争议中", dispute: "提出争议",
  needs_review: "需重新复核", needs_recheck: "需重新复核", needs_reconfirmation: "需重新复核", recheck_required: "需重新复核", reconfirm_required: "需重新复核", stale: "待重查", reviewed: "已确认完成", confirmed_complete: "已确认完成", audit_invalid: "审计链待核查",
  alert_review: "预警理由复核", annotation_only: "独立标注核验", full: "完整覆盖", unknown: "覆盖未知",
  met: "特征成立", not_met: "特征不成立", undeterminable: "无法判定", supported: "支持", supports: "支持", contradicted: "矛盾", contradiction: "矛盾", insufficient: "证据不足", insufficient_evidence: "证据不足",
  corresponds: "对应", mismatch: "不匹配", addressed: "已回应", not_addressed: "未回应", identity_unresolved: "对象未对齐", extraction_failed: "抽取失败",
  claim_error: "事实矛盾", focus_not_addressed: "关注点未回应", unsupported_explanation: "支持依据不足", material_mismatch: "材料不对应", new_lead: "新增观察线索",
  manual_focus: "回应待人工判断", manual_material: "材料关系待人工判断", manual_extraction: "抽取完整性待核对", material_insufficient: "材料不足", claim_unresolved: "事实待核验", insufficient_coverage: "覆盖不足", execution_failed: "执行失败", missing_alert: "缺少原预警", candidate: "候选待确认", source_changed: "来源已修订",
  confirmed: "问题已确认", rejected: "已驳回", open: "待处理", resolved: "已解决", closed: "已关闭", revoked: "已撤销", high: "高", medium: "中", low: "低", warning: "提示",
  confirm: "确认候选 / 完成任务", reject: "驳回候选", reconfirm: "重新确认", close_item: "裁决未决事项", upgrade_lead: "升级新增线索", close_lead: "关闭新增线索", not_applicable: "不适用",
  fixed: "固定流程", agent: "Agent 动态取证", local: "确定性演示", frozen: "离线冻结响应", deepseek: "DeepSeek", incremental: "增量重查", success: "成功", ok: "成功", passed: "通过", valid: "有效",
  count: "次数", amount_sum: "金额合计", counterparty: "对手对象", time_range: "时间范围", narrative: "处置理由", kyc: "客户资料", in: "转入", out: "转出", incoming: "转入", outgoing: "转出",
  feature: "交易特征", claim: "事实核验", semantic: "疑点回应", material: "材料关系", alert_response: "疑点回应", material_relation: "材料关系", revised: "已人工修订", amended: "已人工修订", abstained: "已弃标", abstain: "弃标 / 无法裁决", amend: "修订标签", manual_override: "人工修订", human: "人工", deterministic: "确定性计算",
  machine_candidate: "运行候选", human_confirmation: "人工确认", human_judgement: "人工语义判断", human_extraction_reverified: "人工抽取修订后重核", human_reviewed: "人工抽取后重核",
  scope_review: "本次范围适用性", no_candidate: "本类无候选",
  human_extraction_correction_pending: "人工抽取修订待复核", correction_pending: "人工抽取修订待复核", pending_source_review: "需补正来源后复核",
  schema_migrated: "规范已迁移",
  claim_proposed: "人工抽取提议已提交", claim_approved: "人工抽取提议已接受", claim_rejected: "人工抽取提议已拒绝", claim_withdrawn: "人工抽取提议已撤回",
};
const label = value => dictionary[value] || compact(value);
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}
function append(parent, ...children) { children.filter(Boolean).forEach(child => parent.append(child)); return parent; }
function button(text, className, action) {
  const element = node("button", `button ${className || "secondary"}`, text);
  element.type = "button";
  if (action) element.addEventListener("click", action);
  return element;
}
function badge(value, type) {
  let tone = "";
  if (["failed", "contradicted", "contradiction", "mismatch", "claim_error", "material_mismatch", "return_for_revision", "return_for_correction", "revise", "needs_revision", "建议退回修订"].includes(value)) tone = "red";
  if (["completed", "complete", "confirmed_complete", "reviewed", "supports", "supported", "corresponds", "addressed", "full", "success", "ok", "passed", "valid", "本次质检范围内通过"].includes(value)) tone = "green";
  if (["partial", "undeterminable", "insufficient", "insufficient_evidence", "needs_evidence", "request_evidence", "awaiting_evidence", "needs_review", "needs_recheck", "needs_reconfirmation", "recheck_required", "reconfirm_required", "stale", "disputed", "pending_judgement", "identity_unresolved", "not_addressed", "建议补证", "待人工判断"].includes(value)) tone = "amber";
  if (["running", "met", "new_lead", "reconfirm", "confirmed", "not_met"].includes(value)) tone = "blue";
  const text = type === "review" && value === "completed" ? "已确认完成" : type === "coverage" && value === "partial" ? "部分覆盖" : label(value);
  return node("span", `badge ${tone}`, text);
}
function panel(title, subtitle, body) {
  const element = node("section", "panel");
  append(element, append(node("div", "section-heading"), node("h2", "", title), subtitle ? node("span", "section-caption", subtitle) : null), body);
  return element;
}
function pretty(value) { return node("pre", "structured-data", json(value)); }
function empty(title, description, action) {
  return append(node("div", "empty-state"), node("div", "empty-icon", "◎"), node("strong", "", title), node("p", "", description), action);
}
function info(message, error = false) {
  const target = $("#notice"); target.textContent = message; target.className = error ? "error" : ""; target.hidden = false;
  clearTimeout(info.timer); info.timer = setTimeout(() => { target.hidden = true; }, error ? 16000 : 8000);
}
function formatDate(value) {
  if (!value) return "未记录时间";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false });
}
function errorText(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map(item => `${list(item.loc).join(".")}: ${item.msg || json(item)}`).join("\n");
  return json(detail);
}
async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options }); }
  catch (_) { throw new Error("无法连接本地服务。请确认后端已启动，再点击刷新。"); }
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; }
  catch (_) { throw new Error(`服务返回了非 JSON 响应（HTTP ${response.status}），请检查后端。`); }
  if (!response.ok) throw new Error(errorText(data.detail || data.error || data.message || `请求失败（HTTP ${response.status}）`));
  return data;
}
const casePath = suffix => `/api/cases/${encodeURIComponent(state.caseId)}${suffix || ""}`;
const pkg = () => state.detail?.package || state.detail?.case || {};
const run = () => state.detail?.latest_run || {};
const issues = () => list(run().issues);
const leadDispositions = () => list(state.detail?.lead_dispositions);
const reviewLead = targetId => leadDispositions().find(lead => lead.lead_id === targetId || lead.candidate?.issue_id === targetId);
const openItems = () => list(first(state.detail?.open_items, run().open_items)).filter(item => !["closed", "resolved", "revoked"].includes(item.status));
const reviews = () => list(state.detail?.review_events);
const annotations = () => list(state.detail?.annotations);
const claimProposals = () => list(state.detail?.claim_proposals);
const claimAmendments = () => list(state.detail?.claim_amendments);
const claimKinds = () => list(pkg().review_scope?.target_labels || ["count", "amount_sum", "counterparty", "time_range"]).filter(kind => ["count", "amount_sum", "counterparty", "time_range"].includes(kind));
const canProposeClaim = () => Boolean(run().run_id && !isStale() && !state.busy && state.detail?.audit_integrity?.valid !== false);
const annotationStatus = item => isStale() ? "needs_review" : item.review?.correction_status === "pending_source_review" ? "correction_pending" : item.review?.status || "candidate";
const annotationStatusText = status => ({ confirmed: "已确认标签", revised: "已修订标签", amended: "已修订标签", candidate: "候选待裁决", rejected: "候选已驳回", needs_review: "需重新裁决" }[status] || label(status));
const annotationValue = value => value === undefined || value === null ? "未形成终值" : typeof value === "object" ? json(value) : label(value);
function statuses() {
  return {
    run_status: isStale() ? "stale" : first(state.detail?.run_status, run().run_status, "not_started"),
    qc_recommendation: first(state.detail?.qc_recommendation, run().qc_recommendation, "pending"),
    review_status: first(state.detail?.review_status, run().review_status, "pending"),
  };
}
function isStale() {
  return Boolean(state.detail?.stale || state.detail?.needs_recheck || state.detail?.is_stale || state.detail?.current_snapshot_valid === false ||
    (state.detail?.source_hash && run().source_hash && state.detail.source_hash !== run().source_hash));
}
function coverLabel(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    if (!value.length) return "unknown";
    const values = value.map(item => first(item.status, item.completeness, item.coverage, "unknown"));
    return values.every(item => item === "full") ? "full" : values.includes("partial") ? "partial" : "unknown";
  }
  return first(value?.status, value?.completeness, "unknown");
}
const completedReviews = ["completed", "confirmed_complete", "reviewed", "本次质检范围内通过"];
const outdatedReviews = ["needs_review", "needs_recheck", "needs_reconfirmation", "recheck_required", "reconfirm_required", "stale"];
function renderTasks() {
  $("#nav-count").textContent = state.cases.length;
  const counts = [
    ["当前合成任务", state.cases.length, "按实际导入案件统计", "▦", false],
    ["待人工处理", state.cases.filter(item => !completedReviews.includes(item.review_status)).length, "问题确认与最终裁决分别记录", "◎", true],
    ["需要重新复核", state.cases.filter(item => outdatedReviews.includes(item.review_status) || item.stale || item.needs_recheck).length, "来源变化后保留原人工记录", "↻", false],
    ["人工已确认完成", state.cases.filter(item => completedReviews.includes(item.review_status)).length, "仅限已完成的当前检查范围", "✓", false],
  ];
  $("#task-summary").replaceChildren(...counts.map(([name, count, subtitle, symbol, accent]) => append(node("div", `summary-card ${accent ? "accent" : ""}`), node("div", "summary-label", name), node("div", "summary-value", count), node("small", "", subtitle), node("span", "summary-mark", symbol))));
  const query = $("#task-search").value.toLowerCase().trim();
  const filter = $("#task-filter").value;
  const filtered = state.cases.filter(item => {
    const matchesText = `${item.case_id} ${item.title || ""}`.toLowerCase().includes(query);
    const reviewState = item.review_status || "pending";
    const complete = completedReviews.includes(reviewState);
    const stale = outdatedReviews.includes(reviewState) || item.stale || item.needs_recheck;
    return matchesText && (filter === "all" || (filter === "completed" && complete) || (filter === "needs_recheck" && stale) || (filter === "pending" && !complete && !stale));
  });
  $("#filtered-count").textContent = `${filtered.length} 个案件`;
  $("#task-rows").replaceChildren(...filtered.map(item => {
    const row = node("tr");
    const identity = append(node("td"), node("span", "case-name", item.title || item.case_id), node("span", "case-id", item.case_id), node("span", "task-type", label(item.task_mode)));
    const ai = item.ai_recommendation ? { report_suspicious: ["建议上报", "red"], exclude: ["建议排除", "green"], insufficient_evidence: ["需补充尽调", "amber"] }[item.ai_recommendation] : null;
    const aiCell = append(node("td"), ai ? node("span", `badge ${item.ai_stale ? "" : ai[1]}`, (item.ai_stale ? "已过期 · " : "") + ai[0] + (item.human_decision ? " · 已人工决定" : "")) : node("span", "badge", "未研判"));
    append(row, identity, append(node("td"), node("span", "version-label", compact(item.data_version))), aiCell, append(node("td"), badge(item.run_status || "not_started")), append(node("td"), badge(item.qc_recommendation || "pending")), append(node("td"), badge(item.review_status || "pending", "review")), append(node("td"), badge(coverLabel(first(item.coverage_summary, item.coverage, item.completeness)), "coverage")), append(node("td"), button("审查 →", "text", () => loadCase(item.case_id, "workspace"))));
    return row;
  }));
  const blank = $("#task-empty"); blank.hidden = filtered.length > 0;
  if (!filtered.length) blank.replaceChildren(node("strong", "", state.cases.length ? "没有匹配的任务" : "还没有可审查的案件"), node("p", "", state.cases.length ? "试试其他名称、编号或处理状态。" : "导入合成案件包，或检查本地服务是否已加载种子案例。"));
}
function setView(view) {
  if (view === "delivery") { state.stage = "review"; view = "workspace"; }
  const changed = state.view !== view;
  state.view = view;
  document.querySelectorAll(".view").forEach(element => { element.hidden = element.id !== `view-${view}`; });
  document.querySelectorAll(".nav-item").forEach(element => { element.classList.toggle("active", element.dataset.view === view); element.setAttribute("aria-current", element.dataset.view === view ? "page" : "false"); });
  $("#view-label").textContent = { tasks: "案件列表", workspace: "甄别工作台" }[view];
  if (view !== "tasks") renderDetail();
  if (changed) window.scrollTo({ top: 0, behavior: "instant" });
  const hash = view !== "tasks" && state.caseId ? `#${view}/${encodeURIComponent(state.caseId)}/${state.stage}` : "#tasks";
  if (location.hash !== hash) history.replaceState(null, "", hash);
}
async function loadCase(caseId, view = state.view) {
  if (state.busy) return;
  const serial = ++state.loadSerial;
  const previousId = state.caseId;
  try {
    const detail = await api(`/api/cases/${encodeURIComponent(caseId)}`);
    if (serial !== state.loadSerial) return;
    state.caseId = caseId; state.detail = detail;
    if (previousId !== caseId) { state.evidence = null; state.sourceTab = "narrative"; state.reviewTarget = null; }
    setView(view); renderDetail();
  } catch (error) { info(error.message, true); }
}
async function refresh() {
  $("#refresh-button").disabled = true;
  const results = await Promise.allSettled([api("/api/cases"), api("/api/config")]);
  const connection = $("#connection-status");
  if (results[0].status === "fulfilled") {
    state.cases = list(results[0].value.cases || results[0].value);
    connection.className = "connection online"; connection.replaceChildren(node("i"), document.createTextNode("本地服务已连接"));
    renderTasks();
  } else {
    connection.className = "connection offline"; connection.replaceChildren(node("i"), document.createTextNode("本地服务未连接"));
    info(results[0].reason.message, true); renderTasks();
  }
  if (results[1].status === "fulfilled") state.config = results[1].value;
  else state.config = { deepseek_configured: false, deepseek_governed_ready: false, config_error: results[1].reason.message };
  $("#refresh-button").disabled = false;
  if (state.caseId && results[0].status === "fulfilled") await loadCase(state.caseId);
}
function caseHeading(delivery = false) {
  const p = pkg();
  const title = append(node("div"), button("← 返回任务列表", "text back-link", () => setView("tasks")), node("h1", "", delivery ? "重查与交付" : p.title || state.caseId));
  const subtitle = append(node("div", "case-subtitle"), node("span", "case-id", state.caseId), node("span", "", delivery ? p.title || "当前案件" : label(p.task_mode)), node("span", "version-label", `来源 ${compact(p.data_version)}`), badge("合成案例", ""));
  title.append(subtitle);
  const chooser = node("select", "case-select"); chooser.setAttribute("aria-label", "切换案件");
  state.cases.forEach(item => { const option = node("option", "", item.title || item.case_id); option.value = item.case_id; chooser.append(option); });
  chooser.value = state.caseId; chooser.addEventListener("change", () => loadCase(chooser.value));
  return append(node("div", "case-heading"), title, chooser);
}
function statusStrip() {
  const s = statuses();
  return append(node("div", "status-strip"), ...[["执行状态", s.run_status], ["质检建议", s.qc_recommendation], ["人工处理", s.review_status]].map(([title, value]) => append(node("div", "status-cell"), node("span", "", title), badge(value, title === "人工处理" ? "review" : ""))));
}
function renderDetail() {
  const target = $("#workspace-content");
  if (!state.detail) { target.replaceChildren(empty("选择一个案件", "从案件列表打开案件，按五个阶段完成资料接入、规则检验、AI 研判、理由质检与复核归档。", button("前往案件列表", "primary", () => setView("tasks")))); return; }
  target.replaceChildren(caseHeading(), stageNav());
  renderStage(target);
}
function documents() {
  const p = pkg();
  if (Array.isArray(p.documents)) return [...p.documents].sort((a, b) => Number(b.type === "narrative" || b.document_id === "narrative") - Number(a.type === "narrative" || a.document_id === "narrative"));
  return [p.narrative ? { document_id: "narrative", type: "narrative", text: typeof p.narrative === "string" ? p.narrative : p.narrative.text } : null, p.kyc ? { document_id: "kyc", type: "kyc", text: typeof p.kyc === "string" ? p.kyc : p.kyc.text } : null].filter(Boolean);
}
function documentText(document) { return first(document.text, document.content, ""); }
function textWithSpan(text, sourceDocument) {
  const paragraph = node("p", "narrative-text");
  const characters = Array.from(text);
  const evidence = state.evidence;
  const source = evidence?.source || evidence;
  const span = source?.span;
  const start = Array.isArray(span) ? span[0] : first(span?.start, source?.start);
  const end = Array.isArray(span) ? span[1] : first(span?.end, source?.end);
  const sameDocument = source?.document_id === sourceDocument.document_id;
  const sameRevision = source?.revision === undefined || String(source.revision) === String(sourceDocument.revision);
  const sameText = source?.text === undefined || source.text === characters.slice(start, end).join("");
  if (sameDocument && sameRevision && sameText && Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start && end <= characters.length) append(paragraph, document.createTextNode(characters.slice(0, start).join("")), node("mark", "text-highlight", characters.slice(start, end).join("")), document.createTextNode(characters.slice(end).join("")));
  else paragraph.textContent = text;
  return paragraph;
}
function responseRequirementsBlock(focus) {
  const kinds = { explanation: "解释业务关系", verification_result: "报告核对结果", unresolved_item: "回应已知未决项" };
  const items = node("ul", "checklist");
  list(focus.response_requirements).forEach(item => items.append(node("li", "", `${kinds[item.kind] || item.kind}：${item.text}`)));
  return append(node("div", "source-block"), node("div", "field-caption", `作者需回应 · ${compact(focus.focus_id)}`), items);
}
function sourcePanel() {
  const body = node("div");
  const tabs = node("div", "tab-bar");
  [["narrative", "理由与预警"], ["transactions", "流水"], ["materials", "材料"], ["coverage", "覆盖 / 映射"]].forEach(([value, text]) => {
    const tab = button(text, "tab-button", () => { state.sourceTab = value; renderDetail(); });
    tab.className = `tab-button ${state.sourceTab === value ? "active" : ""}`; tab.setAttribute("aria-pressed", String(state.sourceTab === value)); tabs.append(tab);
  });
  body.append(tabs);
  const content = node("div", "section-body");
  if (state.evidence) {
    const details = node("details", "source-details"); details.open = true;
    append(details, node("summary", "", "已选择的证据引用"), pretty(state.evidence)); content.append(details);
    content.append(node("p", "source-count", "来源引用按其声明的版本定位；历史修订请在交付包中核对。"));
  }
  const p = pkg();
  if (state.sourceTab === "narrative") {
    const a = p.alert;
    const alert = typeof a === "string" ? a : first(a?.original_focus, a?.focus_text, a?.text, a?.description, a?.original_text, a?.focus, a?.focus_points);
    const alertText = typeof alert === "string" ? alert : alert ? json(alert) : a ? json(a) : "未提供原始预警。预警理由复核任务需补充预警后完成相关检查。";
    content.append(append(node("div", "source-block"), node("div", "field-caption", `原始预警${a?.alert_id ? ` · ${a.alert_id}` : ""}`), node("div", "alert-box", alertText)));
    // Every focus the narrative must answer stays visible, not only the first one.
    list(a?.focuses).filter(focus => focus.text && focus.text !== alert).forEach(focus =>
      content.append(append(node("div", "source-block"), node("div", "field-caption", `预警关注点 · ${focus.focus_id}`), node("div", "alert-box", focus.text))));
    list(a?.focuses).filter(focus => list(focus.response_requirements).length).forEach(focus => content.append(responseRequirementsBlock(focus)));
    const upgraded = list(p.review_scope?.upgraded_leads);
    upgraded.forEach(focus => {
      const block = node("div", `source-block ${state.evidence?.type === "upgraded_focus" && state.evidence.focus_id === focus.focus_id ? "lead-focus-highlight" : ""}`);
      append(block, node("div", "field-caption", `人工升级关注点 · ${compact(focus.focus_id)}`), node("div", "alert-box", focus.text || focus.question || "请核对完整升级记录"));
      const details = node("details", "source-details"); append(details, node("summary", "", "查看升级来源与处置依据"), pretty(focus)); block.append(details); content.append(block);
      if (list(focus.response_requirements).length) content.append(responseRequirementsBlock(focus));
    });
    if (state.evidence?.type === "upgraded_focus" && !upgraded.some(focus => focus.focus_id === state.evidence.focus_id)) content.append(node("p", "lead-boundary", "引用的升级关注点已不在当前应回应范围；请在导出包的历史来源中核对。"));
    documents().forEach(document => {
      const block = node("div", "source-block");
      append(block, node("div", "field-caption", `${label(document.type || document.document_id || "文档")} · ${document.document_id || "未标识"} / ${compact(document.revision)}`), textWithSpan(String(documentText(document)), document)); content.append(block);
    });
    if (!documents().length) content.append(node("div", "inline-empty", "暂无处置理由或客户资料。"));
    const metadata = node("dl", "metadata-grid");
    [["被检账户", p.subject_account_id], ["币种 / 时区", `${compact(p.currency)} / ${compact(p.timezone)}`], ["覆盖起点（包含）", p.coverage_start], ["覆盖终点（不含）", p.coverage_end]].forEach(([title, value]) => metadata.append(append(node("div"), node("dt", "", title), node("dd", "", compact(value)))));
    content.append(metadata);
  } else if (state.sourceTab === "transactions") {
    const transactions = list(p.transactions); content.append(node("div", "source-count", `${transactions.length} 条实际导入记录 · 金额单位：${p.currency || "CNY"}`));
    const table = node("table", "small-table"); const head = append(node("thead"), append(node("tr"), ...["时间 / 编号", "对手", "方向", "金额"].map(value => node("th", "", value))));
    const tbody = node("tbody");
    const selectedIds = evidenceTransactionIds(state.evidence);
    transactions.forEach(transaction => {
      const row = node("tr", `transaction-row ${selectedIds.includes(transaction.transaction_id) ? "highlight" : ""}`);
      row.dataset.transactionId = transaction.transaction_id;
      append(row, append(node("td"), node("span", "", compact(transaction.timestamp).replace("T", " ").slice(0, 16)), node("span", "transaction-cell-id", transaction.transaction_id)), append(node("td"), node("span", "", first(transaction.counterparty_name_masked, transaction.counterparty_token, "—")), node("span", "transaction-cell-id", transaction.counterparty_token)), node("td", "", label(transaction.direction)), node("td", "numeric", compact(transaction.amount))); tbody.append(row);
    });
    append(table, head, tbody); content.append(append(node("div", "table-scroll"), table));
    if (!transactions.length) content.append(node("div", "inline-empty", "当前来源没有交易记录。空流水不代表已验证不存在交易。"));
  } else if (state.sourceTab === "materials") {
    const materials = list(p.materials); content.append(node("div", "source-count", `${materials.length} 份支持材料 · 不核验材料真伪`));
    materials.forEach(material => {
      const block = node("div", `document-card ${state.evidence?.material_id === material.material_id ? "highlight" : ""}`);
      append(block, node("h4", "", first(material.title, material.material_id, "支持材料")), node("p", "", `${compact(material.type || material.material_type)} · 修订 ${compact(material.revision)}`), pretty(material)); content.append(block);
    });
    const links = node("details", "source-details"); append(links, node("summary", "", `材料支持关系（${list(p.material_links).length}）`), pretty(p.material_links || [])); content.append(links);
    if (!materials.length) content.append(node("div", "inline-empty", "未导入支持材料，补充后须重查相关候选。"));
  } else {
    content.append(append(node("div", "source-block"), node("div", "field-caption", "逐来源、字段、账户与窗口的覆盖声明"), pretty(p.coverage || [])));
    content.append(append(node("div", "source-block"), node("div", "field-caption", "对象对应关系 · 名称相似不等于已确认"), pretty(p.entity_mappings || [])));
  }
  body.append(content);
  const result = panel("来源与材料", "只读证据", body);
  $(".section-heading", result).append(button("修订", "text", () => openEditor("source")));
  return result;
}
function evidenceTransactionIds(evidence) {
  if (!evidence) return [];
  if (typeof evidence === "string") return [evidence];
  return [evidence.transaction_id, ...list(evidence.transaction_ids), ...list(evidence.query_result?.transaction_ids)].filter(Boolean);
}
function locateEvidence(evidence) {
  state.evidence = evidence;
  if (evidenceTransactionIds(evidence).length) state.sourceTab = "transactions";
  else if (evidence?.material_id || evidence?.type === "material") state.sourceTab = "materials";
  else if (evidence?.document_id || evidence?.source?.document_id || ["document", "upgraded_focus"].includes(evidence?.type)) state.sourceTab = "narrative";
  else state.sourceTab = "coverage";
  if (state.view === "delivery") setView("workspace"); else renderDetail();
  const highlight = $(".transaction-row.highlight, .document-card.highlight, .text-highlight, .lead-focus-highlight");
  highlight?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}
function evidenceLinks(value) {
  const refs = Array.isArray(value) ? value : value ? [value] : [];
  return append(node("div", "issue-evidence"), ...refs.map((reference, index) => {
    const title = evidenceTitle(reference, index);
    const result = button(`↗ ${title}`, "", () => locateEvidence(reference)); result.className = "evidence-link"; result.title = "定位来源并查看完整证据引用"; return result;
  }));
}
function evidenceTitle(reference, index = 0) {
  if (typeof reference === "string") return reference;
  if (reference?.type === "query_scope" && reference.scope?.start && reference.scope?.end) return `流水 ${reference.scope.start.slice(5, 10)} → ${reference.scope.end.slice(5, 10)}`;
  if (reference?.type === "tool_result") return `工具返回 · ${compact(reference.result_ref)}`;
  if (reference?.type === "upgraded_focus") return `人工升级关注点 · ${compact(reference.focus_id)}`;
  return first(reference?.transaction_id, reference?.material_id, reference?.document_id, reference?.query_id, reference?.query_scope_id, reference?.node_id, reference?.ref, reference?.type, `证据 ${index + 1}`);
}
function claimComparisonText(issue) {
  // Plain-language discrepancy for a contradicted/unresolved fact, from the deterministic comparison only.
  if (!String(issue.target_id || "").startsWith("claim:")) return null;
  const result = list(run().claim_results).find(item => "claim:" + item.claim_id === issue.target_id);
  const c = result?.comparison;
  if (!c) return null;
  const scope = result.coverage?.status === "full" ? "完整流水" : "现有可见流水";
  if (c.unit === "transactions") return `理由陈述 ${c.expected} 笔；${scope}核得 ${c.actual} 笔。`;
  if (c.unit === "CNY_fen") return `理由陈述合计 ${(c.expected / 100).toFixed(2)} 元；${scope}核得 ${(c.actual / 100).toFixed(2)} 元（整数分计算）。`;
  const names = Object.fromEntries(list(state.detail?.package?.counterparties).map(p => [p.counterparty_token, p.display_name_masked || p.counterparty_token]));
  const show = tokens => tokens.map(t => names[t] || t).join("、") || "无";
  if (Array.isArray(c.expected)) return `理由陈述对手：${show(c.expected)}；${scope}实际对手：${show(c.actual)}。`;
  if ("inside_count" in c) return `陈述期间内 ${c.inside_count} 笔，期间外 ${c.outside_count} 笔（${scope}）。`;
  return null;
}
function resultPanel() {
  const body = node("div", "section-body");
  if (!run().run_id) { body.append(empty("等待开始质检", "运行固定流程或已配置的 DeepSeek Agent，候选与证据将显示在这里。")); return panel("问题与证据", "尚未运行", body); }
  const warnings = list(run().warnings);
  warnings.forEach(warning => body.append(node("div", "stale-banner", typeof warning === "string" ? warning : json(warning))));
  const issueList = node("div", "issue-list");
  const ordinaryIssues = issues().filter(issue => issue.type !== "new_lead");
  ordinaryIssues.forEach((issue, index) => {
    const item = node("article", `issue-card ${state.reviewTarget === issue.issue_id ? "selected" : ""}`);
    append(item, append(node("div", "issue-top"), badge(issue.type || "待核实问题"), node("span", "issue-number", `#${String(index + 1).padStart(2, "0")}`)), node("h3", "", issue.title || label(issue.type)), node("p", "issue-description", claimComparisonText(issue) || label(first(issue.description, issue.reason, "请结合引用证据人工核对。"))), evidenceLinks(issue.evidence));
    if (issue.type === "unsupported_explanation") item.append(node("p", "review-disclaimer", "模型候选，需人工核对。材料字段是否对应与解释的支持依据是否充分，须分别判断；确定性核验结果见下方明细。"));
    const decision = !isStale() ? [...reviews()].reverse().find(event => event.target_id === issue.issue_id && event.snapshot_id === run().snapshot_id) : null;
    const resolutionLabel = issue.type === "manual_extraction" && decision?.resolution === "addressed" ? "抽取已核对" : label(decision?.resolution);
    const decisionLabel = decision ? ({ confirm: "问题已确认", reconfirm: "已重新确认", reject: "已驳回", close_item: `人工裁决：${resolutionLabel}`, dispute: "争议中", request_correction: "待补正" }[decision.action] || label(decision.action)) : null;
    append(item, append(node("div", "issue-actions"), decisionLabel ? badge(decisionLabel) : issue.status ? badge(issue.status) : node("span", "muted", "候选 · 待人工判断"), button("处理此项 →", "text", () => { state.reviewTarget = issue.issue_id; renderDetail(); $("#review-action")?.focus(); })));
    issueList.append(item);
  });
  body.append(issueList);
  if (!ordinaryIssues.length) body.append(node("div", "inline-empty", "本次运行未生成阻断问题候选。仍需核对必需检查、未决事项及资料覆盖，不自动表示质检通过。"));
  const features = list(run().features);
  if (features.length) {
    const section = append(node("div", "result-section"), node("h3", "", "交易形态 · 不是犯罪认定"));
    features.forEach(feature => {
      const row = node("div", "feature-row");
      const name = first(feature.feature_code, feature.code, feature.feature_result_id, "交易特征");
      const left = append(node("div"), node("strong", "", `${name}${name === "F1" ? " 短时收付集中" : name === "F2" ? " 分散收款集中付款" : ""}`), node("small", "", "使用版本化演示阈值，非监管标准"));
      append(row, left, badge(first(feature.result, feature.status))); section.append(row);
      const detail = node("details", "source-details"); append(detail, node("summary", "", "查看计算指标与依据"), pretty(first(feature.metrics, feature)), evidenceLinks(feature.evidence)); section.append(detail);
    }); body.append(section);
  }
  const claimResults = list(run().claim_results);
  if (claimResults.length || list(run().material_results).length) {
    const section = append(node("div", "result-section"), node("h3", "", "事实与材料核验明细"));
    [...claimResults, ...list(run().material_results)].forEach(result => {
      const claim = list(run().claims).find(item => item.claim_id === result.claim_id);
      const detail = node("details", "source-details");
      append(detail, node("summary", "", `${claim?.origin === "human_reviewed" ? "人工抽取后重核 · " : ""}${label(result.result || result.execution_status)} · ${first(claim?.text, result.claim_id, result.material_id, result.link_id, "支持关系")}`), pretty(result), evidenceLinks(result.evidence), claim ? claimProposalButtons(claim) : null); section.append(detail);
    }); body.append(section);
  }
  return panel("问题与证据", `${ordinaryIssues.length} 条问题候选`, body);
}
function leadPanel() {
  const records = leadDispositions();
  if (!records.length) return null;
  const body = append(node("div", "section-body"), node("p", "annotation-intro", "新观察先提示，未升级时不新增应回应义务。升级后须重查并核对回应；关闭理由与历史观察仍保留在审查记录中。"));
  records.forEach(lead => {
    const names = { candidate: "观察提示 · 尚未升级", upgraded: "已升级 · 本案需回应", closed: "已关闭 · 保留记录", needs_review: "旧处置待复核" };
    const card = node("article", `lead-card ${lead.status === "upgraded" ? "upgraded" : ""}`);
    card.dataset.leadId = lead.lead_id;
    const originNote = lead.candidate?.origin === "model_candidate" ? "观察与问题为模型草稿。引用绑定可核对，但是否属于新疑点及其业务意义尚未经人工确认。" : "历史观察的原始出处请展开核对；本记录本身不证明观察成立或业务含义已确认。";
    append(card, append(node("div", "issue-top"), node("h3", "", lead.question || lead.candidate?.question || "待人工研判的新观察"), badge(names[lead.status] || label(lead.status))), node("p", "issue-description", lead.observation || lead.candidate?.observation || "请结合来源核对该观察。"), node("p", "review-disclaimer", originNote), evidenceLinks(lead.evidence || lead.candidate?.evidence));
    if (isStale()) card.append(node("p", "lead-boundary", "当前来源或执行版本待重查，以下观察与处置仅供历史核对。"));
    else if (lead.status === "needs_review") card.append(node("p", "lead-boundary", "来源依据已变化，原处置不可直接代表当前结果。请核对新依据后重新处置。"));
    else if (lead.status === "upgraded") card.append(node("p", "lead-boundary", "该观察已进入本案应回应范围。是否已回应，以重查后的必需检查和人工裁决为准。"));
    else if (lead.status === "closed") card.append(node("p", "lead-boundary", "本次已关闭该观察，不因此认定客户不存在风险。"));
    if (!isStale() && lead.disposition && lead.current_valid === false) card.append(node("p", "lead-boundary", "先前处置所依据的内容或执行版本已变化，原理由不能直接作为当前有效裁决；已升级事项仍须按当前范围处理。"));
    if (lead.candidate_current === false) card.append(node("p", "annotation-meta", "本轮未再次提出此观察；历史出处与处置继续保留，当前义务以处置状态和必需检查为准。"));
    if (lead.source_current === false) card.append(node("p", "annotation-meta", "观察引用来自历史来源，不作为当前资料证据；原文与原流水请在导出包历史来源中核对。"));
    if (lead.disposition) card.append(node("p", "lead-decision", `${label(lead.disposition.action)} · ${compact(lead.disposition.actor)}：${lead.disposition.reason || "未记录理由"}`));
    const details = node("details", "source-details");
    append(details, node("summary", "", `查看观察出处与处置历史（${list(lead.history).length}）`), pretty({ lead_id: lead.lead_id, candidate_current: lead.candidate_current, current_valid: lead.current_valid, candidate: lead.candidate, history: lead.history })); card.append(details);
    const controls = node("div", "lead-actions");
    list(lead.allowed_actions).forEach(action => {
      if (!["upgrade_lead", "close_lead"].includes(action)) return;
      const title = action === "upgrade_lead" ? "升级为应回应 →" : lead.status === "needs_review" ? "重新核对并关闭 →" : "说明理由并关闭 →";
      const control = button(title, "text", () => { state.reviewTarget = lead.lead_id; state.reviewAction = action; setView("workspace"); $("#review-reason")?.focus(); });
      control.disabled = state.busy || isStale(); controls.append(control);
    });
    card.append(controls); body.append(card);
  });
  return panel("新增观察线索", `${records.length} 条观察 / 处置记录`, body);
}
function field(text, input) { const result = node("label", "field-label", text); result.append(input); return result; }
function claimProposalButtons(claim) {
  const actions = node("div", "claim-work-actions");
  if (!claim) return actions;
  const amendment = claimAmendments().find(item => list(item.allowed_actions).length && (item.proposed_claim?.claim_id === claim.claim_id || item.target_claim_id === claim.claim_id));
  if (amendment) {
    actions.append(node("p", "review-disclaimer", "该事实关联人工抽取版本，可在下方人工抽取作业区重新提议或请求撤销。"));
    return actions;
  }
  [["replace", "提出抽取修订 →"], ["retire", "请求撤除等价重复候选 →"]].forEach(([operation, title]) => {
    if (operation === "retire" && !hasRetainedDuplicate(claim)) return;
    const control = button(title, "text", () => openClaimWork({ operation, original: claim, targetClaimId: claim.claim_id }));
    control.disabled = !canProposeClaim(); actions.append(control);
  });
  return actions;
}
function hasRetainedDuplicate(claim) {
  function signature(value) {
    const kept = Object.fromEntries(Object.entries(value).filter(([key]) => !["claim_id", "origin", "amendment_id", "replaces_claim_id", "unit"].includes(key)));
    kept.operator ||= "exact";
    if (kept.kind === "amount_sum") kept.value = String(kept.value).replace(/\.0+$/, "").replace(/(\.\d*?[1-9])0+$/, "$1");
    const ordered = object => Array.isArray(object) ? object.map(ordered) : object && typeof object === "object" ? Object.fromEntries(Object.keys(object).sort().map(key => [key, ordered(object[key])])) : object;
    return JSON.stringify(ordered(kept));
  }
  return list(run().claims).some(other => other.claim_id !== claim.claim_id && signature(other) === signature(claim));
}
function claimWorkError(message) { $("#claim-work-error").textContent = message; $("#claim-work-error").hidden = !message; }
function claimWorkInput(name, title, value = "", multiline = false) {
  const input = node(multiline ? "textarea" : "input"); input.dataset.proposedClaim = name; input.value = value ?? "";
  if (!multiline) input.type = "text";
  return field(title, input);
}
function renderProposedClaimFields(kind, claim = {}) {
  const root = $("#proposed-claim-fields"); root.replaceChildren();
  root.append(claimWorkInput("text", "原文引用（复制当前理由中的原话）", claim.text || "", true));
  const occurrence = select([], ""); occurrence.id = "claim-quote-occurrence";
  root.append(field("原文位置（重复出现时须明确选择）", occurrence));
  const direction = select([["", "全部方向"], ["in", "转入"], ["out", "转出"]], claim.direction || ""); direction.dataset.proposedClaim = "direction";
  const operators = ["count", "amount_sum"].includes(kind) ? ["exact", "only", "at_least", "at_most", "exists", "none"] : ["exact", "only", "exists", "none"];
  const names = { exact: "恰好 / 精确", only: "仅限", at_least: "至少", at_most: "至多", exists: "存在", none: "不存在" };
  const operator = select(operators.map(value => [value, names[value]]), operators.includes(claim.operator) ? claim.operator : "exact"); operator.dataset.proposedClaim = "operator";
  root.append(append(node("div", "runner-fields"), field("查询交易方向", direction), field("原文限定词", operator)), claimWorkInput("counterparty_ref", "查询对手筛选（可留空，不是陈述对手集合）", claim.counterparty_ref || claim.counterparty_token || ""), claimWorkInput("start", "查询起点（含，ISO 时间，可与终点同时留空）", claim.start || ""), claimWorkInput("end", "查询终点（不含，ISO 时间）", claim.end || ""));
  if (kind === "count") root.append(claimWorkInput("value", "原文陈述次数（非负整数）", claim.value ?? ""));
  if (kind === "amount_sum") root.append(claimWorkInput("value", "原文陈述金额（元，最多两位小数）", claim.value ?? (claim.value_cents === undefined ? "" : (claim.value_cents / 100).toFixed(2))));
  if (kind === "counterparty") root.append(claimWorkInput("value", "原文陈述对手集合（用逗号分隔）", Array.isArray(claim.value) ? claim.value.join("，") : claim.value || ""));
  if (kind === "time_range") root.append(claimWorkInput("value.start", "原文陈述期间起点（含，ISO 时间）", claim.value?.start || ""), claimWorkInput("value.end", "原文陈述期间终点（不含，ISO 时间）", claim.value?.end || ""));
  root.append(node("p", "review-disclaimer", "查询范围与原文断言范围分别填写；不要把待核验的对手或期间同时当成筛选条件，从而排除反例。查询留空表示本案范围。金额以元填写，由后端按整数分核验。"));
  $("[data-proposed-claim='text']", root).addEventListener("input", () => updateClaimOccurrences());
  updateClaimOccurrences(claim.source?.span);
}
function updateClaimOccurrences(preferredSpan) {
  const source = state.claimWork?.document;
  const quote = $("[data-proposed-claim='text']", $("#proposed-claim-fields"))?.value || "";
  const positions = [];
  if (source && quote) {
    let offset = 0;
    while ((offset = source.text.indexOf(quote, offset)) !== -1) {
      const start = Array.from(source.text.slice(0, offset)).length;
      positions.push([start, start + Array.from(quote).length]); offset += 1;
    }
  }
  const choices = $("#claim-quote-occurrence");
  const options = positions.map((span, index) => [JSON.stringify(span), `第 ${index + 1} 处 · 字符 ${span[0]}–${span[1]}`]);
  if (positions.length !== 1) options.unshift(["", positions.length ? "请选择原文中的具体位置" : "原文未找到该引用"]);
  choices.replaceChildren(...options.map(([value, text]) => { const option = node("option", "", text); option.value = value; return option; }));
  choices.value = positions.length === 1 ? JSON.stringify(positions[0]) : positions.some(span => json(span) === json(preferredSpan)) ? JSON.stringify(preferredSpan) : "";
}
function openClaimWork({ operation, original = null, targetClaimId = null, kind = null, amendment = null }) {
  if (!canProposeClaim()) return;
  const doc = documents().find(item => item.document_id === "narrative");
  state.claimWork = { mode: "propose", operation, original, targetClaimId, amendment, snapshotId: run().snapshot_id, document: doc ? JSON.parse(json(doc)) : null };
  $("#claim-work-title").textContent = { replace: "提出抽取修订", add: "补录原文中的事实", retire: "请求撤除等价重复候选", revoke: "请求撤销人工抽取版本" }[operation];
  const context = $("#claim-work-context"); context.replaceChildren();
  if (doc) context.append(node("p", "field-caption", `当前原文只读 · ${doc.document_id} / 修订 ${doc.revision}`), textWithSpan(String(documentText(doc)), doc));
  if (original) context.append(append(node("details", "source-details"), node("summary", "", "本轮原事实结构（保留原记录）"), pretty(original)));
  if (amendment) context.append(node("p", "stale-banner", `本提议关联人工抽取版本 ${amendment.amendment_id}；另人批准前，该版本不会被撤销或替代。`));
  const fields = $("#claim-work-fields"); fields.replaceChildren();
  if (["replace", "retire"].includes(operation) && amendment) {
      const machine = list(run().machine_claims);
      const options = [["", "请选择当前机器事实"], ...machine.map(claim => [claim.claim_id, `${claim.claim_id} · ${label(claim.kind)} · ${claim.text}`])];
      const target = select(options, machine.some(claim => claim.claim_id === targetClaimId) ? targetClaimId : ""); target.id = "claim-replacement-target"; fields.append(field("当前机器事实目标", target));
  }
  if (["replace", "add"].includes(operation)) {
    const draft = amendment?.proposed_claim || original || {};
    const kindSelect = select(claimKinds().map(value => [value, label(value)]), kind || draft.kind || claimKinds()[0]); kindSelect.id = "proposed-claim-kind";
    fields.append(field("事实类型", kindSelect), node("div", "review-fields", null)); fields.lastElementChild.id = "proposed-claim-fields";
    kindSelect.addEventListener("change", () => renderProposedClaimFields(kindSelect.value, { text: $("[data-proposed-claim='text']", fields)?.value || draft.text }));
    renderProposedClaimFields(kindSelect.value, draft);
  } else fields.append(node("p", "annotation-intro", operation === "retire" ? "仅支持撤除重复抽取，须保留另一条同原文、对象、数值和期间的等价陈述。另一操作人须独立核对。暂不支持凭“并非事实”删除候选；错抽应提出修订。" : "请求撤销上述人工抽取版本。批准后仅恢复由原文产生的正常抽取路径，不修改业务资料；仍须重查。"));
  $("#claim-work-actor").value = ""; $("#claim-work-reason").value = "";
  $("#save-claim-work").textContent = "提交人工提议"; $("#save-claim-work").disabled = false;
  claimWorkError(""); $("#claim-work-dialog").showModal();
}
function readProposedClaim() {
  const values = {};
  $("#proposed-claim-fields").querySelectorAll("[data-proposed-claim]").forEach(input => { values[input.dataset.proposedClaim] = input.dataset.proposedClaim === "text" ? input.value : input.value.trim(); });
  const work = state.claimWork, kind = $("#proposed-claim-kind").value;
  const spanText = $("#claim-quote-occurrence").value;
  if (!work.document || !spanText) throw new Error("请引用当前理由原文，并选择明确的原文位置。");
  const span = JSON.parse(spanText);
  if (Array.from(work.document.text).slice(...span).join("") !== values.text) throw new Error("原文引用或位置已改变，请重新选择。");
  function validRange(start, end, title) {
    if (!/T.*(Z|[+-]\d\d:\d\d)$/.test(start) || !/T.*(Z|[+-]\d\d:\d\d)$/.test(end) || !Number.isFinite(Date.parse(start)) || !Number.isFinite(Date.parse(end)) || Date.parse(start) >= Date.parse(end)) throw new Error(`${title}须为带时区的 ISO 时间，起点早于终点。`);
  }
  const claim = { kind, operator: values.operator, text: values.text, source: { document_id: "narrative", revision: work.document.revision, span } };
  if (values.direction) claim.direction = values.direction;
  if (values.counterparty_ref) claim.counterparty_ref = values.counterparty_ref;
  if (values.start || values.end) { validRange(values.start, values.end, "查询期间"); claim.start = values.start; claim.end = values.end; }
  if (kind === "count") {
    const value = values.value || (["exists", "none"].includes(values.operator) ? "0" : "");
    if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value))) throw new Error("陈述次数须为可精确表示的非负整数。"); claim.value = Number(value);
  } else if (kind === "amount_sum") {
    const value = values.value || (["exists", "none"].includes(values.operator) ? "0" : "");
    if (!/^\d+(\.\d{1,2})?$/.test(value)) throw new Error("陈述金额须为非负元数，最多两位小数。"); claim.value = value;
  } else if (kind === "counterparty") {
    claim.value = values.value.split(/[,，]/).map(value => value.trim()).filter(Boolean);
    if (!claim.value.length || new Set(claim.value).size !== claim.value.length) throw new Error("陈述对手集合须非空且不能重复。");
  } else if (kind === "time_range") {
    validRange(values["value.start"], values["value.end"], "陈述期间"); claim.value = { start: values["value.start"], end: values["value.end"] };
  }
  return claim;
}
function openClaimProposalReview(proposal) {
  if (state.busy || !list(proposal.allowed_actions).length) return;
  state.claimWork = { mode: "review", proposal: JSON.parse(json(proposal)), snapshotId: run().snapshot_id };
  $("#claim-work-title").textContent = "核对人工抽取提议";
  const context = $("#claim-work-context"); context.replaceChildren();
  const doc = documents().find(item => item.document_id === "narrative");
  if (doc) context.append(node("p", "field-caption", `当前原文只读 · ${doc.document_id} / 修订 ${doc.revision}`), node("p", "narrative-text", documentText(doc)));
  context.append(node("p", "annotation-meta", `提议 ${proposal.proposal_id} · 提议人 ${proposal.proposer} · ${proposal.reason || ""}`));
  if (proposal.original_claim) context.append(append(node("details", "source-details"), node("summary", "", "提议时原事实结构"), pretty(proposal.original_claim)));
  if (proposal.proposed_claim) context.append(append(node("details", "source-details"), node("summary", "", "待接受的人工事实结构"), pretty(proposal.proposed_claim)));
  context.append(node("p", "review-disclaimer", "流水与修改后的数值吻合，不证明修改忠实原文。接受只批准人工抽取作业；业务核验与标签裁决随后单独完成。"));
  if (proposal.preview_verification || proposal.fidelity_assessment) context.append(append(node("details", "source-details"), node("summary", "", "结构检查与核验预览（不是人工终值）"), pretty({ fidelity_assessment: proposal.fidelity_assessment, preview_verification: proposal.preview_verification })));
  const fields = $("#claim-work-fields"); fields.replaceChildren();
  const names = { approve: "接受此提议", reject: "拒绝此提议", withdraw: "提议人撤回" };
  const action = select(list(proposal.allowed_actions).map(value => [value, names[value] || value]), proposal.allowed_actions[0]); action.id = "claim-work-review-action";
  fields.append(field("审核操作", action));
  const fidelity = node("div"); fidelity.id = "claim-work-fidelity";
  if (["replace", "add"].includes(proposal.operation)) {
    const check = node("input"); check.type = "checkbox"; check.id = "claim-faithful";
    fidelity.append(append(node("label", "check-label"), check, node("span", "", "我已逐项核对当前原文，确认拟采用的事实类型、对象、期间、数值和限定词忠实于原文")));
  } else if (proposal.operation === "retire") {
    const choice = select([["", "请核对仍保留的等价陈述"], ["duplicate", "重复抽取，另一条等价陈述仍保留"]], ""); choice.id = "claim-retire-fidelity"; fidelity.append(field("撤除依据", choice));
  }
  fields.append(fidelity, node("p", "review-disclaimer", "接受或拒绝须由提议人以外的操作人提交；只有提议人本人可撤回。过期提议只能由原提议人撤回。"));
  action.addEventListener("change", updateClaimReviewAction);
  $("#claim-work-actor").value = ""; $("#claim-work-reason").value = "";
  $("#save-claim-work").textContent = "提交审核决定"; claimWorkError(""); updateClaimReviewAction(); $("#claim-work-dialog").showModal();
}
function updateClaimReviewAction(clearError = true) {
  if (state.claimWork?.mode !== "review") return;
  const action = $("#claim-work-review-action").value;
  $("#claim-work-fidelity").hidden = action !== "approve";
  const actor = $("#claim-work-actor").value.trim().toLocaleLowerCase(), proposer = state.claimWork.proposal.proposer.trim().toLocaleLowerCase();
  const wrongActor = actor && (action === "withdraw" ? actor !== proposer : actor === proposer);
  $("#save-claim-work").disabled = state.busy || !action || Boolean(wrongActor);
  if (clearError) claimWorkError(wrongActor ? action === "withdraw" ? "只有原提议人可以撤回。" : "接受或拒绝必须由另一操作人完成；请填写实际审核人的个人 ID。" : "");
}
function claimWorkPanel() {
  if (!run().run_id || !claimKinds().length) return null;
  const body = node("div", "section-body");
  append(body, node("p", "annotation-intro", "发现错抽或漏抽时，在原文不变的前提下提出人工作业。另一操作人接受后更新抽取版本，随后重查并逐项裁决。工具结果不能代替原文忠实性审核，人工修订不能计作模型准确率提升。"));
  const add = button("＋ 补录原文中的事实", "secondary small", () => openClaimWork({ operation: "add" })); add.disabled = !canProposeClaim(); body.append(add);
  const machine = list(run().machine_claims);
  if (machine.length) {
    const details = node("details", "source-details"); append(details, node("summary", "", `本轮机器抽取原件（${machine.length}）`), pretty(machine)); body.append(details);
  }
  claimProposals().forEach(proposal => {
    const statusNames = { pending: "待另一操作人审核", approved: "提议已接受", rejected: "提议已拒绝", withdrawn: "提议已撤回", stale: "旧提议已过期" };
    const operationNames = { replace: "修订抽取", add: "补录事实", retire: "撤除等价重复候选", revoke: "撤销人工抽取版本" };
    const card = append(node("article", "claim-work-card"), append(node("div", "issue-top"), node("h3", "", operationNames[proposal.operation] || proposal.operation), badge(statusNames[proposal.status] || proposal.status)), node("p", "annotation-meta", `提议人 ${proposal.proposer} · ${proposal.proposal_id}`), node("p", "issue-description", proposal.reason));
    if (proposal.proposed_claim?.text) card.append(node("p", "narrative-text", proposal.proposed_claim.text));
    const errors = list(proposal.fidelity_assessment?.blocking_errors);
    if (errors.length) card.append(node("p", "stale-banner", `存在不能接受的结构或原文问题：${errors.map(item => typeof item === "string" ? item : json(item)).join("；")}`));
    const details = node("details", "source-details"); append(details, node("summary", "", "原事实、人工提议、预览与审核历史"), pretty(proposal)); card.append(details);
    if (list(proposal.allowed_actions).length) card.append(button("核对 / 处理提议 →", "text", () => openClaimProposalReview(proposal)));
    body.append(card);
  });
  claimAmendments().forEach(amendment => {
    const statusNames = { applied: "人工抽取版本已采用", needs_review: "人工抽取版本待复核", revoked: "人工抽取版本已撤销", superseded: "人工抽取版本已替代" };
    const card = append(node("article", "claim-work-card"), node("h3", "", statusNames[amendment.status] || amendment.status), node("p", "annotation-meta", `${amendment.amendment_id} · 提议 ${compact(amendment.proposer)} / 审核 ${compact(amendment.reviewer)}`), node("p", "issue-description", amendment.reason || ""));
    const details = node("details", "source-details"); append(details, node("summary", "", "人工版本、原事实及适用依据"), pretty(amendment)); card.append(details);
    const controls = node("div", "claim-work-actions");
    if (list(amendment.allowed_actions).includes("supersede")) { const control = button("重新核对并提出替代 →", "text", () => openClaimWork({ operation: amendment.operation, original: amendment.original_claim, targetClaimId: amendment.target_claim_id, amendment })); control.disabled = !canProposeClaim(); controls.append(control); }
    if (list(amendment.allowed_actions).includes("revoke")) { const control = button("请求撤销此人工版本 →", "text", () => openClaimWork({ operation: "revoke", amendment })); control.disabled = !canProposeClaim(); controls.append(control); }
    card.append(controls); body.append(card);
  });
  if (!claimProposals().length && !claimAmendments().length) body.append(node("p", "inline-empty", "尚无人工抽取提议。已有事实可提出修订，四类事实均可补录漏项。撤除仅支持仍有等价陈述保留的重复候选，暂不支持凭“并非事实”删除。"));
  return panel("人工抽取作业", "原文核对 / 重查 / 标签分别记录", body);
}
function annotationValues(item) {
  const review = item.review || {};
  const values = node("div", "annotation-values");
  [[item.origin === "human_reviewed" ? "人工抽取后重核" : "机器候选", item.candidate_value ?? item.value], ["人工终值", review.final_value]].forEach(([title, value]) => {
    const text = title === "人工终值" && review.applicability === "not_applicable" ? "不适用（未赋标签值）" : title === "机器候选" && item.kind === "scope_review" ? "本类无候选，适用性待核对" : annotationValue(value);
    values.append(append(node("div"), node("span", "", title), node("strong", "", text)));
  });
  return values;
}
function annotationCorrectionNote(item) {
  if (item.review?.correction_status !== "pending_source_review") return null;
  return node("div", "stale-banner", `旧人工抽取修订尚未完成独立复核。工具建议：${annotationValue(item.review.proposed_value)}，不证明修订忠实原文，也未形成有效标签。请通过“提出抽取修订”重新提交给另一操作人，不要为匹配流水改写原文。`);
}
function annotationPanel() {
  const body = node("div", "section-body");
  const records = annotations();
  const adjudicated = records.filter(item => !isStale() && item.review?.adjudicated).length;
  const valid = records.filter(item => !isStale() && item.review?.valid).length;
  const pending = state.detail?.annotation_pending_count;
  const scopeReviews = records.filter(item => item.kind === "scope_review").length;
  append(body, node("p", "annotation-intro", "每个标签单独裁决。标签确认不等于问题解决或任务通过；无法裁决时保留未决状态。"), node("div", "annotation-counts", `候选标签 ${records.length - scopeReviews} · 适用性核对 ${scopeReviews} · 当前已裁决 ${adjudicated} · 有效 ${valid} · 必需待裁决 ${compact(pending)}`));
  records.forEach(item => {
    const status = annotationStatus(item);
    const statusBadge = badge(status); statusBadge.textContent = annotationStatusText(status);
    const card = node("article", "annotation-card");
    const title = `${label(item.label)} · ${compact(item.target_id)}`;
    append(card, append(node("div", "issue-top"), node("h3", "", title), statusBadge), node("p", "annotation-meta", `${label(item.kind)} · ${item.required ? "必需标签" : "范围内可选标签"} · ${label(item.origin || "候选来源见明细")}`), annotationValues(item), annotationCorrectionNote(item), evidenceLinks(item.evidence));
    if (item.review?.reason) card.append(node("p", "issue-description", `人工依据：${item.review.reason}`));
    const detail = node("details", "source-details");
    append(detail, node("summary", "", "对象、覆盖、版本与裁决记录"), pretty(item)); card.append(detail);
    const process = button(status === "needs_review" ? "比较并重新裁决 →" : "裁决此标签 →", "text", () => openAnnotation(item.annotation_id));
    process.disabled = state.busy || isStale();
    append(card, append(node("div", "issue-actions"), node("span", "muted", status === "needs_review" ? "历史人工记录保留，不计当前交付" : item.review?.valid ? "有效裁决，仍须满足任务通过门禁" : "尚未形成当前有效交付标签"), process));
    if (item.kind === "claim") card.append(claimProposalButtons(item.claim));
    if (item.kind === "scope_review" && claimKinds().includes(item.label)) {
      const add = button("补录原文中的事实 →", "text", () => openClaimWork({ operation: "add", kind: item.label })); add.disabled = !canProposeClaim(); card.append(add);
    }
    body.append(card);
  });
  if (!records.length) body.append(node("div", "inline-empty", run().run_id ? "当前运行尚无统一标签记录。候选问题仍可在下方处理，旧运行需按最新执行版本重查。" : "运行完成后展示服务端生成的候选标签。"));
  return panel("标签裁决", "候选 / 人工 / 通过分别记录", body);
}
function annotationError(message) { $("#annotation-error").textContent = message; $("#annotation-error").hidden = !message; }
function openAnnotation(annotationId) {
  if (state.busy || isStale()) return;
  const item = annotations().find(record => record.annotation_id === annotationId);
  if (!item) return;
  state.annotationTarget = annotationId;
  $("#annotation-title").textContent = `裁决标签 · ${label(item.label)}`;
  const context = $("#annotation-context");
  context.replaceChildren(node("p", "dialog-description", `${compact(item.target_id)} · 快照 ${compact(item.snapshot_id)} · 规范 ${compact(item.schema_version)}`), annotationValues(item));
  const correctionNote = annotationCorrectionNote(item); if (correctionNote) context.append(correctionNote);
  if (item.prior_review) context.append(append(node("div", "annotation-prior"), node("strong", "", "上次快照的人工裁决 · 当前无效"), node("p", "", `${annotationValue(item.prior_review.final_value)} · ${compact(item.prior_review.actor)} · ${item.prior_review.reason || ""}`), pretty(item.prior_review.evidence)));
  if (item.review?.reason) context.append(node("p", "dialog-description", `当前人工记录：${item.review.reason}`));
  const actions = list(item.allowed_actions).filter(action => !item.claim || !["revise", "amend"].includes(action));
  const names = { confirm: "确认当前候选", revise: "修订标签终值", amend: "修订标签终值", abstain: "弃标 / 无法裁决", dispute: "提交争议", reconfirm: "重新确认当前标签", reject: "驳回候选标签", not_applicable: "不适用（确认原文没有该类事实；发现漏抽请人工补录）" };
  $("#annotation-action").replaceChildren(...actions.map(action => { const option = node("option", "", names[action] || label(action)); option.value = action; return option; }));
  $("#annotation-value").replaceChildren(...list(item.allowed_values).filter(value => !["undeterminable", "insufficient_evidence", "pending_judgement"].includes(value)).map(value => { const option = node("option", "", annotationValue(value)); option.value = JSON.stringify(value); return option; }));
  if (list(item.allowed_values).includes(item.value)) $("#annotation-value").value = JSON.stringify(item.value);
  if (!$("#annotation-value").value) $("#annotation-value").selectedIndex = 0;
  const primary = list(item.evidence).filter(ref => ref && typeof ref === "object" && !Array.isArray(ref));
  state.annotationEvidence = [...primary, ...annotations().flatMap(record => list(record.evidence))].filter((ref, index, all) => ref && typeof ref === "object" && !Array.isArray(ref) && all.findIndex(other => json(other) === json(ref)) === index);
  const choices = $("#annotation-evidence-options"); choices.replaceChildren();
  state.annotationEvidence.forEach((ref, index) => {
    const input = node("input"); input.type = "checkbox"; input.value = String(index); input.checked = primary.some(value => json(value) === json(ref));
    const choice = append(node("div", "annotation-evidence-option"), append(node("label", "check-label"), input, node("span", "", `${evidenceTitle(ref, index)}${input.checked ? " · 本标签候选引用" : " · 当前运行其他引用"}`)));
    const details = node("details", "source-details"); append(details, node("summary", "", "查看引用内容"), pretty(ref)); choice.append(details); choices.append(choice);
  });
  if (!state.annotationEvidence.length) choices.append(node("p", "inline-empty", "当前没有可验证引用。请补证后重查，或记录弃标 / 争议原因。"));
  $("#annotation-reason").value = ""; $("#save-annotation").disabled = !actions.length;
  annotationError(""); updateAnnotationAction(); $("#annotation-dialog").showModal();
}
function updateAnnotationAction() {
  const item = annotations().find(record => record.annotation_id === state.annotationTarget);
  const revising = ["revise", "amend"].includes($("#annotation-action").value);
  $("#annotation-value-field").hidden = !revising || Boolean(item?.claim);
}
function select(options, value) {
  const result = node("select"); options.forEach(([key, text, disabled]) => { const option = node("option", "", text); option.value = key; option.disabled = Boolean(disabled); result.append(option); }); result.value = value; return result;
}
function runnerPanel() {
  const body = node("div", "section-body");
  const paidReady = state.config.deepseek_governed_ready === true;
  const provider = select([["local", "确定性演示（非 Agent）"], ["deepseek", "DeepSeek API", !paidReady]], state.provider);
  provider.id = "run-provider";
  const mode = select([["fixed", "固定流程"], ["agent", "Agent 动态取证", state.provider === "local"]], state.mode); mode.id = "run-mode";
  provider.addEventListener("change", () => { state.provider = provider.value; state.mode = provider.value === "deepseek" ? "agent" : "fixed"; renderDetail(); });
  mode.addEventListener("change", () => { state.mode = mode.value; });
  const controls = append(node("div", "runner-fields"), field("执行提供方", provider), field("运行方式", mode)); body.append(controls);
  const runButton = button(state.busy ? "正在运行，请稍候…" : "开始质检", "primary run-button", () => executeRun("full")); runButton.disabled = state.busy || (state.provider === "deepseek" && !paidReady); body.append(runButton);
  const providerNote = state.provider === "local" ? "下次运行将使用确定性代码演示，不调用模型，也不替代 Agent 验收。当前结果的执行方式见下方轨迹。" : paidReady ? `下次运行将实际调用 ${state.config.model || "DeepSeek"}。工具调用与耗时按真实执行记录，模型候选仍须人工复核。` : "当前 DeepSeek 选择不可运行；可切换到确定性演示。";
  body.append(node("p", "runner-note", providerNote));
  if (!paidReady) body.append(node("p", "runner-note", state.config.config_error ? "模型配置状态读取失败。修复服务后刷新。" : state.config.deepseek_disabled_reason || "付费入口不可用，请刷新后核对累计费用预算配置。"));
  return panel("下次运行配置", paidReady ? "累计预算入口可用" : state.config.deepseek_configured ? "API 已配置 · 付费入口禁用" : "本地演示可用", body);
}
function tracePanel() {
  const body = node("div", "section-body");
  const trace = list(first(run().trace, run().tool_trace));
  const provenance = run().agent_verified === true ? "真实 Agent 执行" : run().provider === "local" ? "确定性流程 · 非 Agent" : run().mode === "agent" ? "DeepSeek Agent · 验收未确认" : `${label(run().provider)} · 固定流程`;
  if (run().run_id) body.append(append(node("div", "trace-header"), node("span", "", provenance), node("span", "", `${trace.length} 条实际记录`)));
  if (run().provider === "deepseek") body.append(node("p", "source-count", `模型 ${compact(run().execution?.model)} · ${compact(run().stats?.model_calls)} 次模型调用 · ${compact(run().stats?.adaptive_tool_calls)} 次追加工具调用`));
  const budget = run().governed_development?.budget;
  if (budget) body.append(node("p", "source-count", `本次返回的累计费用（${compact(budget.currency)} 估算，含已知失败费用）：已知支出 ${compact(budget.spent)}${budget.conservative_spent ? ` · 未知用量保守占用 ${compact(budget.conservative_spent)}（预留上限，非实际费用）` : ""} · 未定费用预留 ${compact(budget.held)} · 剩余 ${compact(budget.remaining)} · 原总额度 ${compact(budget.total)}。账本状态：${budget.status === "ready" ? "可继续" : budget.status === "stopped" ? "已停止" : "未知"}${budget.stop_reason ? `（${compact(budget.stop_reason)}）` : ""}。预留或未知费用仍待确认。`));
  const timeline = node("ol", "trace-list");
  trace.forEach(entry => {
    const item = node("li", `trace-item ${["completed", "success", "ok", "computed", "reused"].includes(entry.status) ? "success" : entry.status === "failed" ? "failed" : ""}`);
    append(item, node("strong", "", entry.tool || entry.tool_name || entry.name || "执行步骤"), node("p", "", entry.purpose || entry.description || ""));
    const time = entry.duration_ms === undefined ? null : node("span", "", `${Number(entry.duration_ms).toFixed(0)} ms`);
    append(item, append(node("div", "trace-meta"), node("span", "", label(entry.status || "unknown")), entry.round ? node("span", "", `第 ${entry.round} 轮`) : null, time, entry.result_ref ? node("span", "", compact(entry.result_ref)) : null));
    const detail = node("details"); append(detail, node("summary", "", "参数与返回结果"), pretty({ arguments: entry.arguments || entry.args, result_ref: entry.result_ref, result: entry.result, error: entry.error })); item.append(detail); timeline.append(item);
  }); body.append(timeline);
  if (!trace.length) body.append(node("div", "inline-empty", "运行后展示真实工具调用；尚无执行轨迹。"));
  return panel("工具调用轨迹", "可展开核对", body);
}
function reviewPanel() {
  const form = node("form", "section-body"); form.id = "review-form";
  const targetOptions = [["task", "整个任务（通过由服务端条件判断）"]];
  issues().filter(issue => issue.type !== "new_lead").forEach(issue => targetOptions.push([issue.issue_id, `问题：${issue.title || label(issue.type)}`]));
  leadDispositions().forEach(lead => targetOptions.push([lead.lead_id, `观察：${lead.question || lead.candidate?.question || lead.lead_id}`]));
  openItems().filter(item => item.target_id !== "task").forEach(item => targetOptions.push([item.item_id, `未决：${item.title || label(item.kind)}`]));
  const uniqueOptions = targetOptions.filter((option, index, all) => option[0] && all.findIndex(item => item[0] === option[0]) === index);
  if (!uniqueOptions.some(item => item[0] === state.reviewTarget)) state.reviewTarget = issues().find(issue => issue.type !== "new_lead")?.issue_id || leadDispositions()[0]?.lead_id || "task";
  const target = select(uniqueOptions, state.reviewTarget); target.id = "review-target";
  const action = select([], ""); action.id = "review-action";
  const actor = select([["质检员", "质检员"], ["复核员", "复核员"], ["经办人", "经办人"]], "质检员");
  const resolution = select([], "");
  const resolutionField = field("明确裁决结果", resolution); resolutionField.hidden = true;
  const reason = node("textarea"); reason.id = "review-reason"; reason.placeholder = "说明判断依据、补正要求或争议原因"; reason.required = true; reason.maxLength = 4000;
  const leadNote = node("p", "lead-boundary"); leadNote.hidden = true;
  const submit = button("追加人工记录", "primary run-button"); submit.type = "submit";
  function updateResolution() {
    resolutionField.hidden = action.value !== "close_item";
    const type = issues().find(issue => issue.issue_id === target.value)?.type;
    const options = target.value === "task" ? [["addressed", "任务争议 / 补正已处理"], ["not_applicable", "任务争议 / 补正不适用"]] : type === "manual_focus" ? [["addressed", "已回应"], ["not_addressed", "未回应"]] : type === "manual_extraction" ? [["addressed", "已完整核对抽取结果"]] : type === "manual_material" ? [["corresponds", "对应"], ["mismatch", "不匹配"], ["not_applicable", "不适用"]] : [["addressed", "已回应"], ["not_addressed", "未回应"], ["corresponds", "对应"], ["mismatch", "不匹配"], ["not_applicable", "不适用"]];
    resolution.replaceChildren(...options.map(([value, text]) => { const option = node("option", "", text); option.value = value; return option; }));
  }
  function updateActions() {
    const lead = reviewLead(target.value);
    const actions = lead ? list(lead.allowed_actions).filter(value => ["upgrade_lead", "close_lead"].includes(value)) : ["confirm", "reject", "request_correction", "dispute", "reconfirm", "close_item"];
    action.replaceChildren(...actions.map(value => { const option = node("option", "", label(value)); option.value = value; return option; }));
    action.value = actions.includes(state.reviewAction) ? state.reviewAction : actions[0] || "";
    state.reviewAction = action.value;
    action.disabled = !actions.length;
    submit.disabled = state.busy || !run().run_id || isStale() || !actions.length;
    leadNote.hidden = !lead;
    leadNote.textContent = lead ? actions.length ? "升级或关闭会新增来源版本并要求重查。处置该观察不会移除原预警或其他必需问题。" : "当前观察没有可执行处置。请先重查待更新来源，或查看已关闭记录。" : "";
    reason.placeholder = lead ? "说明升级为应回应事项的依据，或关闭该观察的理由" : "说明判断依据、补正要求或争议原因";
    updateResolution();
  }
  action.addEventListener("change", () => { state.reviewAction = action.value; updateResolution(); });
  target.addEventListener("change", () => { state.reviewTarget = target.value; updateActions(); });
  updateActions();
  const fields = append(node("div", "review-fields"), field("处理对象", target), append(node("div", "runner-fields"), field("人工操作", action), field("演示角色", actor)), resolutionField, leadNote, field("处理理由（必填）", reason));
  append(form, fields, submit, node("p", "review-disclaimer", "人工记录只追加。确认问题并不解决问题；未决事项须显式裁决或补证。全案通过需满足后端必需检查与当前快照条件。"));
  form.addEventListener("submit", async event => {
    event.preventDefault(); if (!reason.value.trim()) return;
    const prior = [...reviews()].reverse().find(record => record.source_hash === state.detail.source_hash && record.target_id === target.value
      && ["confirm", "reject", "request_correction", "dispute", "reconfirm", "close_item"].includes(record.action));
    const payload = { action: action.value, target_id: target.value, actor: actor.value, reason: reason.value.trim(),
      snapshot_id: run().snapshot_id, expected_event_id: prior?.event_id || null };
    const lead = reviewLead(target.value);
    if (lead) {
      if (!list(lead.allowed_actions).includes(action.value) || isStale()) { info("该观察当前不可执行此操作，请刷新后核对。", true); return; }
      payload.expected_event_id = lead.expected_event_id || null;
    }
    if (action.value === "close_item") payload.resolution = resolution.value;
    await mutate(casePath("/reviews"), payload, lead ? "观察处置已追加并更新来源。请重查当前范围，再核对回应与处置状态。" : "人工处理记录已追加，当前状态已刷新。");
  });
  return panel("问题与任务处理", "演示角色 / 无生产权限", form);
}
function renderWorkbench(target) {
  const grid = node("div", "workbench");
  const centerTabs = node("div", "center-tabs");
  [["issues", `问题与核验 ${issues().length}`], ["annotations", `标签裁决 ${annotations().length}`]].forEach(([value, title]) => {
    const tab = button(title, "", () => { state.resultTab = value; renderDetail(); }); tab.className = `tab-button ${state.resultTab === value ? "active" : ""}`; tab.setAttribute("aria-pressed", String(state.resultTab === value)); centerTabs.append(tab);
  });
  append(grid, append(node("div", "column"), sourcePanel()), append(node("div", "column"), centerTabs, state.resultTab === "annotations" ? annotationPanel() : resultPanel(), claimWorkPanel(), leadPanel(), requiredPanel()), append(node("div", "column"), runnerPanel(), tracePanel(), reviewPanel())); target.append(grid);
}
function requiredPanel() {
  const checks = list(run().required_checks);
  const body = node("div", "section-body"); const checklist = node("ul", "checklist");
  checks.forEach(check => {
    const manuallyResolved = !isStale() && check.status === "pending_judgement" && Array.isArray(state.detail?.pending_checks) && !state.detail.pending_checks.some(pending => pending.check_id === check.check_id);
    checklist.append(append(node("li"), node("span", "", check.label || check.check_id), badge(manuallyResolved ? "已人工裁决" : check.status)));
  }); body.append(checklist);
  const upgraded = list(run().semantic_results).filter(result => result.type === "upgraded_lead" && result.required);
  if (upgraded.length) {
    const section = append(node("div", "result-section"), node("h3", "", "人工升级后应回应事项"), node("p", "annotation-intro", "运行完成仅表示核验已执行，不代表事项已经回应。下列为本次回应候选，人工终值在标签裁决中记录。"));
    const focuses = node("ul", "checklist");
    upgraded.forEach(result => {
      const focus = list(pkg().review_scope?.upgraded_leads).find(item => item.focus_id === result.focus_id);
      const lead = leadDispositions().find(item => item.focus_id === result.focus_id);
      focuses.append(append(node("li"), node("span", "", focus?.text || lead?.question || result.focus_id), badge(result.status)));
    });
    section.append(focuses); body.append(section);
  }
  if (!checks.length) body.append(node("div", "inline-empty", "运行后列出本任务的必需检查；无检查清单时不能自动通过。"));
  return panel("必需检查清单", `${checks.length} 项`, body);
}
async function mutate(path, payload, message) {
  if (state.busy) return false;
  state.busy = true; renderDetail();
  try {
    await api(path, { method: "POST", body: JSON.stringify(payload) });
    state.busy = false; await refresh(); info(message); return true;
  } catch (error) { state.busy = false; renderDetail(); info(error.message, true); return false; }
}
async function executeRun(strategy) {
  if (state.provider === "deepseek" && state.config.deepseek_governed_ready !== true) { info(state.config.deepseek_disabled_reason || "累计费用预算入口不可用，不能运行 DeepSeek。", true); return; }
  if (state.provider === "local" && state.mode === "agent") { info("确定性演示不是真实 Agent。请先配置并选择 DeepSeek API。", true); return; }
  await mutate(casePath("/run"), { mode: state.mode, provider: state.provider, strategy }, "本次运行已返回。请检查执行状态、未决事项与真实调用轨迹。");
}
function renderDelivery(target) {
  const actions = node("div", "delivery-top-actions");
  const disabled = state.busy || (state.provider === "deepseek" && state.config.deepseek_governed_ready !== true);
  const incremental = button(state.busy ? "正在重查…" : "执行增量重查", "primary", () => executeRun("incremental")); incremental.disabled = disabled;
  const full = button("全量重查", "secondary", () => executeRun("full")); full.disabled = disabled;
  append(actions, incremental, full, button("修订来源", "secondary", () => openEditor("source")), node("span", "muted", `当前：${label(state.provider)} / ${label(state.mode)}，可在质检工作区切换`));
  target.append(actions, node("div", "scope-note", "来源变更先使旧候选失效；重算完成后，仍需人工处理受影响记录。下方数字仅展示后端实际返回的统计。"));
  target.append(migrationPanel());
  const grid = node("div", "delivery-grid");
  const left = node("div", "delivery-stack"); const right = node("div", "delivery-stack");
  const stats = run().stats || {};
  const statGrid = node("dl", "metric-grid");
  [["潜在影响节点", stats.potentially_affected], ["实际重算节点", stats.recomputed], ["复用节点", stats.reused], ["撤销候选", stats.revoked], ["实际工具调用", stats.tool_calls], ["实际模型调用", stats.model_calls]].forEach(([name, value]) => statGrid.append(append(node("div", "metric-block"), node("dt", "", name), node("dd", "", compact(value)))));
  const metricsBody = append(node("div", "section-body"), statGrid, node("p", "delivery-explain", `输入 tokens：${compact(stats.input_tokens)} · 输出 tokens：${compact(stats.output_tokens)} · 总耗时：${stats.duration_ms === undefined ? "未记录" : `${(stats.duration_ms / 1000).toFixed(2)} 秒`}。节点数不能直接换算为人工或算力节省比例。`));
  const budget = run().governed_development?.budget;
  if (budget) metricsBody.append(node("p", "delivery-explain", `本次返回的累计费用（${compact(budget.currency)} 估算，含已知失败费用）：已知支出 ${compact(budget.spent)}${budget.conservative_spent ? ` · 未知用量保守占用 ${compact(budget.conservative_spent)}（预留上限，非实际费用）` : ""} · 未定费用预留 ${compact(budget.held)} · 剩余 ${compact(budget.remaining)} · 原总额度 ${compact(budget.total)}。账本状态：${budget.status === "ready" ? "可继续" : budget.status === "stopped" ? "已停止" : "未知"}${budget.stop_reason ? `（${compact(budget.stop_reason)}）` : ""}。预留或未知费用仍待确认。`));
  left.append(panel(isStale() ? "上次执行统计（已失效）" : "本次执行统计", run().strategy === "full" ? "全量重查" : run().strategy ? label(run().strategy) : "运行记录", metricsBody));
  const feeRuns = list(state.detail?.governed_fee_history).filter(record => record.governed_development?.budget);
  if (feeRuns.length) {
    const feeHistory = append(node("div", "section-body"), node("p", "delivery-explain", "以下累计值以各次运行回执保存时点为准。"));
    [...feeRuns].reverse().forEach(record => {
      const receipt = record.governed_development; const saved = receipt.budget;
      feeHistory.append(append(node("article", "review-event"), node("strong", "", `${label(record.mode)} · ${label(record.run_status)}`),
        node("p", "", `累计已知支出 ${compact(saved.spent)} ${compact(saved.currency)} · 未知用量保守占用 ${compact(saved.conservative_spent || "0")}（预留上限，非实际费用） · 未定费用预留 ${compact(saved.held)} · 剩余 ${compact(saved.remaining)} / ${compact(saved.total)}`),
        node("small", "", `回执：${compact(receipt.attempt_id)} · 保存于 ${formatDate(record.created_at)}`)));
    });
    left.append(panel("费用回执历史", `${feeRuns.length} 次受预算约束运行`, feeHistory));
  }
  const events = state.detail?.change_events ? list(state.detail.change_events) : reviews().filter(event => event.action === "source_changed" || event.action === "needs_review");
  const changesBody = node("div", "section-body"); const changes = node("div", "event-list");
  [...events].reverse().forEach(event => {
    const card = node("article", "change-event");
    append(card, append(node("header"), node("h3", "", first(event.title, label(event.action), event.event_type, event.type, "来源或快照变更")), event.data_version ? node("span", "version-label", event.data_version) : null), node("p", "", typeof event.changed_nodes === "object" ? `变更对象：${json(event.changed_nodes)}` : first(event.reason, event.description, event.summary, "")), node("small", "", formatDate(first(event.created_at, event.timestamp, event.time))));
    const details = node("details", "source-details"); append(details, node("summary", "", "查看完整变更记录"), pretty(event)); card.append(details); changes.append(card);
  }); changesBody.append(changes);
  if (!events.length) changesBody.append(node("div", "inline-empty", "尚无来源变更记录。修订理由、材料、覆盖声明或对象映射后，在这里核对新旧版本。"));
  left.append(panel("来源变化与版本", `${events.length} 条记录`, changesBody));
  const reviewBody = node("div", "section-body"); const reviewList = node("div", "review-events");
  [...reviews()].reverse().forEach(event => {
    const annotationAction = { confirm: "确认标签", reconfirm: "重新确认标签", revise: "修订标签", reject: "驳回标签" }[event.action] || label(event.action);
    const card = append(node("article", "review-event"), node("strong", "", `${event.annotation_id ? `标签 · ${annotationAction}` : label(event.action)} · ${compact(event.actor)}`), node("p", "", event.reason || "未记录理由"), node("small", "", `${compact(event.target_id)} · ${formatDate(first(event.at, event.created_at, event.timestamp, event.time))}`), event.resolution ? badge(event.resolution) : null);
    if (event.annotation_id) card.append(node("p", "", `上次人工终值：${annotationValue(event.previous_value)} → 本次人工终值：${event.applicability === "not_applicable" ? "不适用（未赋标签值）" : annotationValue(event.final_value)}`));
    if (event.correction_status === "pending_source_review") card.append(node("p", "", `人工抽取修订待复核 · 修订命题的工具建议：${annotationValue(event.proposed_value)}，未计入有效标签。`));
    const detail = node("details", "source-details"); append(detail, node("summary", "", "查看完整事件与证据"), pretty(event)); card.append(detail); reviewList.append(card);
  }); reviewBody.append(reviewList);
  if (!reviews().length) reviewBody.append(node("div", "inline-empty", "尚未追加人工操作记录。"));
  left.append(panel("人工裁决历史", `${reviews().length} 条追加记录`, reviewBody));
  const unresolvedBody = node("div", "section-body");
  openItems().forEach(item => {
    const card = append(node("div", "open-item"), node("h3", "", item.title || label(item.kind)), node("p", "", label(item.reason || item.description || "需要人工处理")), node("small", "", `${compact(item.item_id)} · ${compact(item.target_id)}`), button("前往处理 →", "text", () => { state.reviewTarget = item.target_id === "task" ? "task" : item.item_id; setView("workspace"); $("#review-action")?.focus(); })); unresolvedBody.append(card);
  });
  if (!openItems().length) unresolvedBody.append(node("div", "inline-empty", "当前 API 未返回未决事项。是否可通过仍由完整检查清单、来源有效性和人工确认共同决定。"));
  list(state.detail?.annotation_pending).forEach(item => {
    const annotation = annotations().find(record => record.annotation_id === item.annotation_id);
    const action = button("前往标签裁决 →", "text", () => { state.resultTab = "annotations"; setView("workspace"); if (annotation) openAnnotation(annotation.annotation_id); }); action.disabled = isStale();
    unresolvedBody.append(append(node("div", "open-item"), node("h3", "", annotation ? `标签：${label(annotation.label)}` : "必需标签待裁决"), node("p", "", item.reason || label(item.status)), action));
  });
  right.append(panel("未决 / 需重新复核", `${openItems().length} 项业务未决 · ${compact(state.detail?.annotation_pending_count)} 项标签未决`, unresolvedBody), requiredPanel());
  left.append(annotationPanel());
  const claimWork = claimWorkPanel(); if (claimWork) left.append(claimWork);
  const leads = leadPanel(); if (leads) left.append(leads);
  const exportBody = append(node("div", "section-body"), node("p", "export-intro", "导出当前检查范围、来源与规范版本、证据、人工记录及未决事项，包含未升级、已升级及已关闭观察的出处和处置历史。历史或待复核记录不得冒充当前已通过结果。"));
  const exportButton = button("↓ 导出审查记录 JSON", "primary export-button", downloadExport); exportButton.disabled = state.busy; exportBody.append(exportButton);
  exportBody.append(node("p", "delivery-explain", "是否进入可交付集合由服务端按当前快照与裁决状态判断。本地哈希链仅辅助验证链条一致性，不单独保证不可篡改。")); right.append(panel("版本化交付", "保留检查边界", exportBody));
  append(grid, left, right); target.append(grid);
}
async function openMigration() {
  const m = state.migration;
  m.open = !m.open;
  if (!m.open || m.catalog.length) { renderDetail(); return; }
  m.loading = true; m.error = ""; renderDetail();
  try {
    const data = await api("/api/schemas");
    m.catalog = list(data.schemas);
    const base = m.catalog.find(item => list(item.case_ids).includes(state.caseId)) || m.catalog[0];
    m.baseHash = base?.schema_hash || ""; m.draft = json(base?.schema || data.default_schema || pkg().schema || {});
  } catch (error) { m.error = error.message; }
  finally { m.loading = false; renderDetail(); }
}
function invalidateMigration() {
  const m = state.migration;
  m.preview = null; m.selected = []; m.error = ""; m.requiresRefresh = false;
  renderMigrationResults();
}
function migrationPanel() {
  const m = state.migration;
  const body = node("div", "section-body");
  body.append(node("p", "annotation-intro", "完整规范先预览、明确选择范围，再逐案迁移。未迁移案件保留原规范；迁移后必须另行重查与人工复核。阈值仅为合成演示规则。"));
  const toggle = button(m.open ? "收起规范迁移" : "预览跨案件规范迁移", "secondary", openMigration); toggle.disabled = m.loading || state.busy; body.append(toggle);
  if (!m.open) return panel("跨案件规范迁移", "不自动重查", body);
  if (m.loading) { body.append(node("p", "inline-empty", "正在读取各案件实际采用的完整规范…")); return panel("跨案件规范迁移", "只读载入", body); }
  const form = node("form", "migration-editor");
  const base = node("select"); base.id = "migration-base"; base.disabled = state.busy;
  m.catalog.forEach(item => { const option = node("option", "", `${item.schema_version} · ${item.schema_hash.slice(0, 10)}`); option.value = item.schema_hash; base.append(option); }); base.value = m.baseHash;
  base.addEventListener("change", () => {
    m.baseHash = base.value; m.draft = json(m.catalog.find(item => item.schema_hash === base.value)?.schema || {}); invalidateMigration(); renderDetail();
  });
  const actor = node("select"); actor.id = "migration-actor"; actor.disabled = state.busy;
  ["质检员", "复核员"].forEach(value => { const option = node("option", "", value); option.value = value; actor.append(option); }); actor.value = m.actor;
  actor.addEventListener("change", () => { m.actor = actor.value; invalidateMigration(); });
  form.append(append(node("div", "runner-fields"), append(node("label", "field-label", "待迁移的原规范"), base), append(node("label", "field-label", "演示操作角色"), actor)));
  const draft = node("textarea", "code-editor"); draft.id = "migration-schema"; draft.value = m.draft; draft.required = true; draft.spellcheck = false; draft.disabled = state.busy;
  draft.addEventListener("input", () => { m.draft = draft.value; invalidateMigration(); });
  form.append(append(node("label", "field-label", "目标完整 Schema JSON（填写新的 schema_version）"), draft));
  form.append(node("p", "review-disclaimer", "修改阈值后，请同时核对标签定义、正反例与弃标条件。结构校验通过不代表文字说明与规则含义一致。首次预览后版本内容已登记；再改内容须另用新版本名。"));
  const reason = node("input"); reason.id = "migration-reason"; reason.value = m.reason; reason.required = true; reason.maxLength = 1000; reason.disabled = state.busy; reason.placeholder = "说明规范变化的依据与本次迁移目的";
  reason.addEventListener("input", () => { m.reason = reason.value; invalidateMigration(); });
  form.append(append(node("label", "field-label", "迁移依据与理由（必填）"), reason));
  const preview = button("① 预览相关案件与字段差异", "secondary"); preview.type = "submit"; preview.disabled = state.busy || !m.baseHash;
  form.append(preview); form.addEventListener("submit", event => { event.preventDefault(); previewMigration([]); }); body.append(form);
  const results = node("div"); results.id = "migration-results"; body.append(results);
  fillMigrationResults(results);
  return panel("跨案件规范迁移", "选择范围 → 逐案执行 → 显式重查", body);
}
function migrationSelectionConfirmed() {
  const m = state.migration;
  const frozen = list(m.preview?.cases).filter(item => item.selected).map(item => item.case_id).sort();
  return !m.requiresRefresh && frozen.length > 0 && json(frozen) === json([...m.selected].sort());
}
function renderMigrationResults() {
  const host = $("#migration-results"); if (host) { host.replaceChildren(); fillMigrationResults(host); }
}
function migrationDetails(title, content) {
  const details = node("details", "source-details migration-details");
  return append(details, node("summary", "", title), content);
}
function migrationDiff(changes) {
  const table = node("table", "small-table migration-diff");
  table.append(append(node("thead"), append(node("tr"), ...["字段 / 操作", "原值", "目标值"].map(title => node("th", "", title)))));
  const rows = node("tbody");
  list(changes).forEach(change => rows.append(append(node("tr"), append(node("td"), node("code", "", change.path), node("small", "muted", change.op)), node("td", "", change.before_exists === false || change.before === undefined ? "（不存在）" : json(change.before)), node("td", "", change.after_exists === false || change.after === undefined ? "（移除）" : json(change.after)))));
  table.append(rows); return append(node("div", "table-scroll"), table);
}
function fillMigrationResults(host) {
  const m = state.migration;
  if (m.error) host.append(node("p", "form-error", m.error));
  if (!m.preview) return;
  const p = m.preview; const confirmed = migrationSelectionConfirmed();
  host.append(node("p", "migration-summary", `目标规范 ${compact(p.target_schema?.schema_version)} · ${list(p.cases).length} 个相关案件 · 已选择 ${m.selected.length} 案 · ${m.requiresRefresh ? "预览待刷新，请重新生成" : confirmed ? "范围已冻结，逐案状态见下方" : "先勾选，再生成确认预览"}`));
  host.append(migrationDetails(`完整目标规范 · ${compact(p.target_schema_hash).slice(0, 12)}`, pretty(p.target_schema)), migrationDetails(`字段差异 · ${list(p.changes).length} 项`, migrationDiff(p.changes)));
  const freeze = button("② 按所选范围生成确认预览", "primary", () => previewMigration([...m.selected])); freeze.disabled = state.busy || !m.selected.length; host.append(freeze);
  const records = node("div", "migration-cases");
  list(p.cases).forEach(item => {
    const receipt = list(p.receipts).find(entry => entry.case_id === item.case_id);
    const migrated = Boolean(receipt || item.status === "migrated" || item.status === "applied");
    const card = node("article", "migration-case"); card.dataset.caseId = item.case_id;
    const check = node("input"); check.type = "checkbox"; check.checked = m.selected.includes(item.case_id); check.disabled = state.busy || !item.selectable || migrated;
    check.addEventListener("change", () => { m.selected = check.checked ? [...m.selected, item.case_id] : m.selected.filter(id => id !== item.case_id); renderMigrationResults(); });
    const heading = append(node("label", "check-label"), check, node("strong", "", `${item.title || item.case_id} · ${item.case_id}`));
    card.append(append(node("div", "migration-case-heading"), heading, badge(migrated ? "已迁移 · 回执不含重查" : m.requiresRefresh || item.status === "stale" ? "预览待刷新" : item.selected ? "已纳入确认范围" : "未迁移")));
    card.append(node("p", "annotation-meta", `规范 ${compact(item.current_schema_version)} → ${compact(p.target_schema?.schema_version)} · 来源 ${compact(item.current_source_hash).slice(0, 12)} · 规则 ${list(item.affected_rule_codes).join("、") || "未列出规则变化"}`));
    card.append(node("p", "annotation-intro", item.requires_full_recheck ? "旧依赖图缺失或已失效：迁移后需全量重查。" : `预览影响 ${list(item.affected_nodes).length} 个节点；实际重算与复用以重查结果为准。`));
    card.append(migrationDetails(`查看本案差异与影响节点（${list(item.affected_nodes).length}）`, append(node("div"), migrationDiff(item.changes), pretty({ affected_nodes: item.affected_nodes, requires_full_recheck: item.requires_full_recheck }))));
    const reviewRecords = list(item.review_records);
    const reviewList = node("ul", "migration-review-list");
    reviewRecords.forEach(record => {
      const action = record.annotation_id ? { confirm: "确认标签", reconfirm: "重新确认标签", revise: "修订标签", reject: "驳回标签" }[record.action] || label(record.action) : label(record.action);
      reviewList.append(append(node("li"), node("strong", "", `${compact(record.target_id)} · ${action} · ${compact(record.actor)}`), node("small", "muted", `记录 ${compact(record.event_id)} / 快照 ${compact(record.snapshot_id)}`)));
    });
    card.append(migrationDetails(`旧人工需重新复核清单（${reviewRecords.length}）`, reviewRecords.length ? append(node("div"), reviewList, pretty(reviewRecords)) : node("p", "annotation-intro", "预览未列出旧人工记录；不表示任务已经通过。")));
    if (list(item.material_link_changes).length || list(item.material_link_warnings).length) {
      const warnings = append(node("div"), node("p", "annotation-intro", "仅原规范一致的材料绑定可随规范迁移；已有版本冲突保留，须补正并重查，不能自动视为材料对应。"));
      list(item.material_link_warnings).forEach(warning => warnings.append(node("p", "annotation-prior", `${compact(warning.link_id)}：${warning.message || warning.code}`)));
      warnings.append(pretty({ changes: item.material_link_changes, warnings: item.material_link_warnings }));
      const details = migrationDetails("材料规范绑定与冲突", warnings); details.open = list(item.material_link_warnings).length > 0; card.append(details);
    }
    if (migrated) {
      card.append(node("p", "migration-result", "本回执只确认来源迁移。请另行重查并重新裁决，实际完成状态以案件当前记录为准。"));
      const open = button("打开案件（离线固定流程） →", "secondary", async () => { state.provider = "local"; state.mode = "fixed"; await loadCase(item.case_id, "delivery"); info(item.requires_full_recheck ? "已切换为离线固定流程；本次迁移要求全量重查。请核对当前记录后显式运行，未自动重查。" : "已切换为离线固定流程。请核对当前记录，再显式重查与重新裁决；未自动运行。"); }); open.disabled = state.busy; card.append(open);
      if (receipt) card.append(migrationDetails("迁移回执", pretty(receipt)));
    } else {
      const apply = button("③ 仅迁移此案", "secondary", () => applyMigration(item.case_id));
      apply.disabled = state.busy || !confirmed || !m.selected.includes(item.case_id) || !item.can_apply; card.append(apply);
      if (!item.selectable) card.append(node("p", "annotation-intro", `服务端不允许迁移：${compact(item.status)}。请核对完整预览。`));
    }
    records.append(card);
  });
  host.append(records, node("p", "review-disclaimer", "未选择或未执行的案件保留原规范。任何来源、运行、人工裁决或执行版本变化均可能使预览过期；收到拒绝后必须刷新预览，不自动重试。"));
}
async function previewMigration(selected) {
  if (state.busy) return;
  const m = state.migration;
  let schema;
  try {
    schema = JSON.parse(m.draft);
    if (!schema || typeof schema !== "object" || Array.isArray(schema)) throw new Error("目标规范必须为完整 JSON 对象。");
    if (!m.reason.trim()) throw new Error("请填写本次规范迁移的依据与理由。");
  } catch (error) { m.error = error.message; renderMigrationResults(); return; }
  state.busy = true; m.error = ""; renderDetail();
  try {
    m.preview = await api("/api/migrations/preview", { method: "POST", body: JSON.stringify({ base_schema_hash: m.baseHash, target_schema: schema, selected_case_ids: selected, reason: m.reason.trim(), actor: m.actor }) });
    m.requiresRefresh = false;
    m.selected = list(m.preview.cases).filter(item => item.selected).map(item => item.case_id);
  } catch (error) { m.error = error.message; m.preview = null; }
  finally { state.busy = false; renderDetail(); }
}
async function applyMigration(caseId) {
  const m = state.migration;
  if (state.busy || !migrationSelectionConfirmed() || !m.selected.includes(caseId)) return;
  const preview = m.preview;
  state.busy = true; m.error = ""; renderDetail();
  try {
    const result = await api(`/api/migrations/${encodeURIComponent(preview.preview_id)}/cases/${encodeURIComponent(caseId)}`, { method: "POST", body: JSON.stringify({ preview_hash: preview.preview_hash, actor: m.actor, reason: m.reason.trim() }) });
    m.preview = result.preview;
    state.provider = "local"; state.mode = "fixed";
    state.busy = false; await refresh();
    info(`${caseId} 已迁移。尚未重查；其余案件仅在逐案执行后改变规范。`);
  } catch (error) {
    m.error = `迁移未完成：${error.message}。若预览已过期，请重新生成预览并核对选择范围。`;
    // Rejected previews must be reviewed again, never silently refreshed and retried.
    m.selected = []; m.requiresRefresh = true;
  } finally { state.busy = false; renderDetail(); }
}
async function downloadExport() {
  try {
    const data = await api(casePath("/export"));
    const objectUrl = URL.createObjectURL(new Blob([json(data)], { type: "application/json;charset=utf-8" }));
    const link = node("a"); link.href = objectUrl; link.download = `${state.caseId}-review-export.json`; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    info("已导出服务端当前审查记录。请核对交付状态与未决事项。");
  } catch (error) { info(error.message, true); }
}
function sectionValue(section) {
  const p = state.editorPackage;
  if (section === "all") return p;
  if (section === "narrative" && Array.isArray(p.documents)) return p.documents.filter(document => document.type === "narrative" || document.document_id === "narrative");
  return p[section] ?? (section === "narrative" ? "" : []);
}
function mergeEditorSection(value) {
  if (state.editorSection === "all") {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("完整案件包必须是 JSON 对象。");
    state.editorPackage = value;
  } else if (state.editorSection === "narrative" && Array.isArray(state.editorPackage.documents)) {
    if (!Array.isArray(value)) throw new Error("处置理由文档应为 JSON 数组，请保留 document_id、revision、type 与 text。");
    let inserted = false;
    state.editorPackage.documents = state.editorPackage.documents.flatMap(document => {
      if (document.type !== "narrative" && document.document_id !== "narrative") return [document];
      if (inserted) return [];
      inserted = true; return value;
    });
    if (!inserted) state.editorPackage.documents.push(...value);
  } else state.editorPackage[state.editorSection] = value;
}
function editorError(message) { $("#editor-error").textContent = message; $("#editor-error").hidden = !message; }
function openEditor(mode) {
  if (state.busy) return;
  state.editorMode = mode; state.editorSection = "all"; state.editorPackage = mode === "import" ? {} : JSON.parse(json(pkg()));
  $("#editor-title").textContent = mode === "import" ? "导入合成案件" : "修订案件来源";
  $("#editor-description").textContent = mode === "import" ? "载入符合项目规范的 JSON 案件包。导入时执行校验，发现重复冲突或缺少必要字段时会返回具体错误。" : "来源变更会立即使旧候选失效，并保留原人工裁决。请在修订后重查与重新确认。";
  $("#editor-section-label").hidden = mode === "import"; $("#source-reason-label").hidden = mode === "import"; $("#source-reason").required = mode !== "import";
  $("#save-source").textContent = mode === "import" ? "校验并导入案件" : "保存新来源版本";
  $("#save-source").disabled = false; $("#source-json").value = json(state.editorPackage); $("#source-reason").value = ""; $("#editor-section").value = "all"; $("#synthetic-confirm").checked = false; $("#import-file").value = ""; editorError("");
  $("#source-dialog").showModal();
}
$("#editor-section").addEventListener("change", event => {
  try {
    mergeEditorSection(JSON.parse($("#source-json").value)); state.editorSection = event.target.value;
    $("#source-json").value = json(sectionValue(state.editorSection)); editorError("");
  } catch (error) { event.target.value = state.editorSection; editorError(`请先修正当前 JSON：${error.message}`); }
});
$("#import-file").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 4_900_000) { editorError("当前原型请求体上限为 5 MB，请使用小于 4.9 MB 的 JSON 文件。"); return; }
  try { const value = JSON.parse(await file.text()); $("#source-json").value = json(value); editorError(""); }
  catch (error) { editorError(`无法读取 JSON 文件：${error.message}`); }
});
$("#source-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!$("#synthetic-confirm").checked) { editorError("请确认这些是合成数据。"); return; }
  try {
    mergeEditorSection(JSON.parse($("#source-json").value));
    const reason = $("#source-reason").value.trim();
    if (state.editorMode !== "import" && !reason) throw new Error("保存来源变更前必须填写具体理由。");
    $("#save-source").disabled = true; editorError("");
    const payload = state.editorMode === "import" ? { package: state.editorPackage } : { package: state.editorPackage, reason };
    await api(state.editorMode === "import" ? "/api/cases" : casePath("/source"), { method: "POST", body: JSON.stringify(payload) });
    $("#source-dialog").close();
    await refresh();
    if (state.editorMode === "import" && state.editorPackage.case_id) await loadCase(state.editorPackage.case_id, "workspace");
    else if (state.editorMode !== "import") setView("delivery");
    info(state.editorMode === "import" ? "合成案件已导入，请开始质检。" : "新来源已保存，旧候选已失效。请执行重查并处理需要重新确认的人工记录。");
  } catch (error) { editorError(error.message); }
  finally { $("#save-source").disabled = false; }
});
$("#close-editor").addEventListener("click", () => $("#source-dialog").close());
$("#cancel-editor").addEventListener("click", () => $("#source-dialog").close());
$("#close-annotation").addEventListener("click", () => $("#annotation-dialog").close());
$("#cancel-annotation").addEventListener("click", () => $("#annotation-dialog").close());
$("#close-claim-work").addEventListener("click", () => $("#claim-work-dialog").close());
$("#cancel-claim-work").addEventListener("click", () => $("#claim-work-dialog").close());
$("#claim-work-actor").addEventListener("input", updateClaimReviewAction);
$("#claim-work-form").addEventListener("submit", async event => {
  event.preventDefault();
  const work = state.claimWork;
  if (!work || state.busy) return;
  const actor = $("#claim-work-actor").value.trim(), reason = $("#claim-work-reason").value.trim();
  if (!actor || !reason) { claimWorkError("请填写明确的操作人 ID 和作业理由。"); return; }
  let path = casePath("/claim-proposals"), payload;
  try {
    if (work.mode === "propose") {
      payload = { operation: work.operation, actor, reason, snapshot_id: work.snapshotId };
      if (["replace", "retire"].includes(work.operation)) {
        payload.target_claim_id = $("#claim-replacement-target")?.value ?? work.targetClaimId;
        if (!payload.target_claim_id) throw new Error("请选择当前抽取结果中的目标事实。");
      }
      if (["replace", "add"].includes(work.operation)) payload.proposed_claim = readProposedClaim();
      if (work.amendment) payload.supersedes_amendment_id = work.amendment.amendment_id;
    } else {
      const proposal = work.proposal, action = $("#claim-work-review-action").value;
      if (!list(proposal.allowed_actions).includes(action)) throw new Error("当前提议不允许该操作，请刷新后核对。");
      const sameActor = actor.toLocaleLowerCase() === proposal.proposer.trim().toLocaleLowerCase();
      if (action === "withdraw" ? !sameActor : sameActor) throw new Error(action === "withdraw" ? "只有原提议人可以撤回。" : "接受或拒绝须由另一操作人完成。");
      payload = { action, actor, reason, snapshot_id: work.snapshotId, expected_event_id: proposal.expected_event_id || null };
      if (action === "approve" && ["replace", "add"].includes(proposal.operation)) {
        if (!$("#claim-faithful").checked) throw new Error("请先逐项核对原文，并明确确认拟采用结构忠实原文。"); payload.fidelity = "faithful";
      }
      if (action === "approve" && proposal.operation === "retire") {
        payload.fidelity = $("#claim-retire-fidelity").value;
        if (!payload.fidelity) throw new Error("请确认仍保留另一条等价陈述。");
      }
      path += `/${encodeURIComponent(proposal.proposal_id)}/review`;
    }
    state.busy = true; $("#save-claim-work").disabled = true; claimWorkError("");
    await api(path, { method: "POST", body: JSON.stringify(payload) });
    $("#claim-work-dialog").close(); state.busy = false; await refresh();
    info(work.mode === "propose" ? "人工提议已记录，等待独立原文复核；尚未形成有效标签。" : payload.action === "approve" ? "提议已接受。请按新抽取版本重查，再裁决当前核验标签。" : "审核决定已追加，原文和流水保持原样。");
  } catch (error) { claimWorkError(error.message); }
  finally { state.busy = false; $("#save-claim-work").disabled = false; if (work.mode === "review") updateClaimReviewAction(false); }
});
$("#annotation-action").addEventListener("change", updateAnnotationAction);
$("#annotation-form").addEventListener("submit", async event => {
  event.preventDefault();
  const item = annotations().find(record => record.annotation_id === state.annotationTarget);
  if (!item || state.busy) return;
  const action = $("#annotation-action").value;
  const reason = $("#annotation-reason").value.trim();
  const evidence = [...document.querySelectorAll("#annotation-evidence-options input:checked")].map(input => state.annotationEvidence[Number(input.value)]);
  if (!reason) { annotationError("请填写本次裁决的具体依据或无法裁决的原因。"); return; }
  if (["confirm", "reconfirm", "revise", "amend", "not_applicable"].includes(action) && !evidence.length) { annotationError("确认、修订或判定不适用必须选择可验证证据。资料不足时请补证或记录弃标 / 争议。"); return; }
  const payload = { action, target_id: item.annotation_id, actor: $("#annotation-actor").value, reason, evidence,
    snapshot_id: item.snapshot_id, expected_event_id: item.review?.event_id || null };
  if (["revise", "amend"].includes(action)) {
    try {
      payload.new_value = JSON.parse($("#annotation-value").value);
    } catch (error) { annotationError(error.message); return; }
  }
  if (action === "reconfirm") payload.previous_event_id = item.prior_review?.event_id || null;
  state.busy = true; $("#save-annotation").disabled = true; annotationError("");
  try {
    await api(casePath("/reviews"), { method: "POST", body: JSON.stringify(payload) });
    $("#annotation-dialog").close(); state.busy = false; await refresh();
    info("标签裁决已追加。抽取与核验记录保留，是否可交付仍由当前任务通过条件决定。");
  } catch (error) { annotationError(error.message); }
  finally { state.busy = false; $("#save-annotation").disabled = false; }
});
$("#refresh-button").addEventListener("click", refresh);
$("#import-button").addEventListener("click", () => openEditor("import"));
$("#task-search").addEventListener("input", renderTasks);
$("#task-filter").addEventListener("change", renderTasks);
document.querySelectorAll(".nav-item").forEach(item => item.addEventListener("click", () => setView(item.dataset.view)));
$(".brand").addEventListener("click", event => { event.preventDefault(); setView("tasks"); });
$("#intake-button").addEventListener("click", () => openIntake());
// Shareable deep link: #workspace/<case_id>[/<stage>] or #delivery/<case_id>.
const [linkedView, linkedCase, linkedStage] = location.hash.slice(1).split("/");
if (linkedCase && ["workspace", "delivery"].includes(linkedView)) {
  state.caseId = decodeURIComponent(linkedCase); state.view = "workspace";
  state.stage = linkedView === "delivery" ? "review" : linkedStage || state.stage;
}
window.addEventListener("hashchange", () => {
  const [view, caseId, stage] = location.hash.slice(1).split("/");
  if (view === "tasks") { if (state.view !== "tasks") setView("tasks"); return; }
  if (caseId && ["workspace", "delivery"].includes(view)) {
    const target = decodeURIComponent(caseId), nextStage = view === "delivery" ? "review" : stage || state.stage;
    if (target !== state.caseId || state.view !== "workspace") { state.stage = nextStage; loadCase(target, "workspace"); }
    else if (nextStage !== state.stage) setStage(nextStage);
  }
});
refresh();
