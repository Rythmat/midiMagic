// DOM layer: renders from the pure state in ui-state.js and talks to the API.
import {
  analyzeBlockers, buildGenerateBody, buildOnBlockers, canAnalyze, canBuildOn, canDownload, canGenerate,
  canInterpret, canRegenerateSection, canRegenerateTrack, critiqueKey, currentCandidate, generateBlockers,
  initialState, interpretBlockers, reduce, regenerateSectionBlockers, regenerateTrackBlockers,
} from "./ui-state.js";
import { FORM_TIPS, INPUT_ROLE_LABELS, PART_GROUPS, promptHighlights, promptSegments } from "./choices.js";
import { drawRoll, trackColor } from "./pianoroll.js";
import { play, stopPlayback } from "./audio.js";

// API base: same origin by default; `?api=http://host:port` when the UI is served separately.
const API = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
let state = initialState();

function dispatch(action) {
  state = reduce(state, action);
  render();
  maybeCritique();
}

class SessionExpired extends Error {}

// Every request carries this page's session id; 410 means the server has dropped that session.
async function request(path, opts = {}) {
  const sid = state.sessionId;
  const headers = { ...(opts.headers || {}) };
  if (sid) headers["X-Session-Id"] = sid;
  const res = await fetch(API + path, { ...opts, headers });
  if (res.status === 410) {
    if (sid === state.sessionId) expireSession();
    throw new SessionExpired("This session has ended.");
  }
  if (!res.ok) {
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON */ }
    const detail = body && (body.detail || body.error);
    throw new Error(typeof detail === "string" ? detail : (detail ? JSON.stringify(detail) : `HTTP ${res.status}`));
  }
  return res;
}

async function api(path, opts = {}) {
  const res = await request(path, opts);
  try { return await res.json(); } catch (_) { return null; }
}

// ------------------------------------------------------------------ session lifetime
// Uploads and results exist only in server memory, for as long as this page's session lives.
const DEFAULT_HEARTBEAT_MS = 30000;
let heartbeatTimer = null;
let connecting = null;

function setServerStatus(ok) {
  $("server-status").textContent = ok ? "server ok" : "server unreachable";
  $("server-status").className = ok ? "pill ok" : "pill bad";
}

function expireSession() {
  stopPlayback();
  document.body.classList.remove("playing");
  $("file").value = "";
  dispatch({ type: "sessionExpired", notice: "This page lost contact with the server (a long sleep or a server "
    + "restart), so its uploads and results were discarded. Choose your file or generate again to continue." });
  connect();
}

async function openSession() {
  const s = await api("/session", { method: "POST" });
  dispatch({ type: "sessionReady", sessionId: s.session_id });
  clearInterval(heartbeatTimer);
  heartbeatTimer = setInterval(heartbeat, s.heartbeat_seconds * 1000);
}

// Runs at page load, then again from the heartbeat until the server answers.
function connect() {
  connecting = connecting || (async () => {
    if (!state.options) await loadOptions();
    if (!state.options || state.sessionId) return;
    try {
      await openSession();
      setServerStatus(true);
    } catch (_) {
      setServerStatus(false);
    }
  })().finally(() => { connecting = null; });
  return connecting;
}

async function heartbeat() {
  if (!state.options || !state.sessionId) return connect();
  try {
    await api(`/session/${state.sessionId}/heartbeat`, { method: "POST" });
    setServerStatus(true);
  } catch (e) {
    if (!(e instanceof SessionExpired)) setServerStatus(false);
  }
}

// ------------------------------------------------------------------ setup
function fillSelect(el, values, { labels, keep } = {}) {
  const kept = keep ? [...el.querySelectorAll("option")].filter((o) => o.value === "") : [];
  el.innerHTML = "";
  for (const o of kept) el.appendChild(o);
  values.forEach((v, i) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = labels ? labels[i] : v.replace(/_/g, " ");
    el.appendChild(o);
  });
}

async function loadOptions() {
  try {
    const o = await api("/options");
    fillSelect($("genre"), o.genres);
    fillSelect($("emotion"), o.emotions);
    fillSelect($("key"), o.keys);
    fillSelect($("rootOverride"), o.keys.flatMap((k) => [k, `${k} minor`]), { keep: true });
    fillSelect($("targetKey"), o.keys, { keep: true });
    fillSelect($("scaleMode"), o.modes, { keep: true });
    fillSelect($("inputRole"), o.input_roles);
    fillSelect($("form"), o.forms);
    fillSelect($("engine"), Object.keys(o.engines), {
      labels: Object.entries(o.engines).map(([k, ok]) => (ok ? k : `${k} (no checkpoint)`)),
    });
    [...$("engine").options].forEach((opt) => { opt.disabled = !o.engines[opt.value]; });
    $("presets").innerHTML = o.instrumentation_presets.map((p) => `<option value="${p}">`).join("");
    buildChoices("genre", o.genres);
    buildChoices("emotion", o.emotions);
    buildChoices("key", o.keys);
    buildChoices("form", o.forms, { tips: FORM_TIPS });
    buildChoices("inputRole", o.input_roles, { labels: INPUT_ROLE_LABELS });
    buildChoices("targetKey", ["", ...o.keys], { labels: { "": "Same as file" } });
    buildParts(o.generated_roles);
    $("genre").value = state.form.genre; $("emotion").value = state.form.emotion; $("key").value = state.form.key;
    $("inputRole").value = state.form.inputRole; $("form").value = state.form.form; $("engine").value = state.form.engine;
    setServerStatus(true);
    const nm = o.neural_model;
    $("engine-status").textContent = nm ? `neural: ${nm.model_version} — ${nm.honesty_note}` : "neural: no checkpoint (rule-based engine only)";
    dispatch({ type: "optionsLoaded", options: o });
  } catch (e) {
    setServerStatus(false);
    dispatch({ type: "optionsFailed", error: e.message });
  }
}

// Button groups replace the dropdowns: one button per value, the state holds the choice.
const CHOICE_FIELDS = ["genre", "emotion", "key", "form", "inputRole", "targetKey"];

function buildChoices(field, values, { labels = {}, tips = {} } = {}) {
  const box = $(`${field}-choices`);
  box.innerHTML = "";
  for (const v of values) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "choice"; b.dataset.value = v;
    const label = labels[v] ?? String(v).replace(/_/g, " ");
    b.textContent = label;
    if (tips[v]) { b.dataset.tip = tips[v]; b.setAttribute("aria-label", `${label}: ${tips[v]}`); }
    b.onclick = () => dispatch({ type: "setField", field, value: v });
    box.appendChild(b);
  }
}

// "Parts to write": one box per group, one choice per box (or Skip).
function buildParts(available) {
  const has = new Set(available);
  const opt = (group, role, name, desc) => `<button type="button" class="opt" data-group="${group}" data-role="${role}">
      <span class="radio" aria-hidden="true"></span><span><b>${escapeHtml(name)}</b><small>${escapeHtml(desc)}</small></span></button>`;
  $("roles").innerHTML = PART_GROUPS.map((g) => {
    const roles = g.roles.filter(([r]) => has.has(r));
    if (!roles.length) return "";
    return `<div class="pcard" role="radiogroup" aria-label="${g.name}"><div class="phead"><b>${g.name}</b><small>${g.what}</small></div>
      ${roles.map(([r, n, d]) => opt(g.id, r, n, d)).join("")}${opt(g.id, "", "Skip", "Leave this part out")}</div>`;
  }).join("");
  $("roles").querySelectorAll(".opt").forEach((b) => {
    b.onclick = () => dispatch({ type: "setGroupRole", group: b.dataset.group, role: b.dataset.role || null });
  });
}

function renderChoices() {
  for (const field of CHOICE_FIELDS) {
    const cur = String(state.form[field] ?? "");
    $(`${field}-choices`).querySelectorAll(".choice").forEach((b) => {
      const on = b.dataset.value === cur;
      b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on));
    });
  }
  const chosen = new Set(state.form.generatedRoles);
  $("roles").querySelectorAll(".pcard").forEach((card) => {
    const opts = [...card.querySelectorAll(".opt")];
    const any = opts.some((b) => b.dataset.role && chosen.has(b.dataset.role));
    for (const b of opts) {
      const on = b.dataset.role ? chosen.has(b.dataset.role) : !any;
      b.classList.toggle("on", on); b.setAttribute("aria-checked", String(on)); b.setAttribute("role", "radio");
    }
  });
}

function bindField(id, field, transform = (v) => v) {
  const el = $(id);
  const ev = el.tagName === "SELECT" ? "change" : "input";
  el.addEventListener(ev, () => dispatch({ type: "setField", field, value: transform(el.value) }));
}

// ------------------------------------------------------------------ actions
async function analyze() {
  if (!canAnalyze(state)) return;
  const file = $("file").files[0];
  dispatch({ type: "analyzeStart" });
  const fd = new FormData();
  fd.append("file", file);
  if (state.form.rootOverride) fd.append("root_override", state.form.rootOverride);
  if (state.form.inputRole) fd.append("input_role", state.form.inputRole);
  try {
    const r = await api("/analyze-midi", { method: "POST", body: fd });
    dispatch({ type: "analyzeSuccess", analysis: r.analysis, uploadId: r.upload_id });
    syncFormFields();                    // the detected emotion/genre now drive generation
  } catch (e) {
    if (!(e instanceof SessionExpired)) dispatch({ type: "analyzeError", error: e.message });
  }
}

async function pollJob(jobId) {
  const session = state.sessionId;
  for (;;) {
    let j;
    try {
      j = await api(`/generation/${jobId}`);
    } catch (e) {
      if (!(e instanceof SessionExpired) && state.sessionId === session) dispatch({ type: "jobError", error: e.message });
      return;
    }
    if (state.sessionId !== session) return;     // the session ended while this request was out
    if (j.status === "done") { dispatch({ type: "jobDone", jobId, result: j.result }); return; }
    if (j.status === "error") { dispatch({ type: "jobError", error: j.error }); return; }
    dispatch({ type: "jobUpdate", status: j.status, progress: j.progress });
    await new Promise((r) => setTimeout(r, 600));
  }
}

async function startJob(path, body) {
  dispatch({ type: "generateStart" });
  try {
    const r = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    dispatch({ type: "jobAccepted", jobId: r.job_id });
    await pollJob(r.job_id);
  } catch (e) {
    if (!(e instanceof SessionExpired)) dispatch({ type: "jobError", error: e.message });
  }
}

// The settings panel is tall, so bring the progress and then the result into view.
async function generate() {
  if (!canGenerate(state)) return;
  const done = startJob("/generate", buildGenerateBody(state));
  $("job-box").scrollIntoView({ behavior: "smooth", block: "center" });
  await done;
  if (state.result && !state.jobError) $("result-box").scrollIntoView({ behavior: "smooth", block: "start" });
}

function regenTrack() {
  if (!canRegenerateTrack(state)) return;
  startJob("/regenerate-track", { job_id: state.jobId, candidate: state.selectedCandidate, track_id: state.selectedTrack,
    seed: Math.floor(Math.random() * 1e6) });
}

function regenSection() {
  if (!canRegenerateSection(state)) return;
  startJob("/regenerate-section", { job_id: state.jobId, candidate: state.selectedCandidate,
    section_index: state.selectedSection, seed: Math.floor(Math.random() * 1e6) });
}

async function toggleLock(trackId, locked) {
  try {
    await api(`/generation/${state.jobId}/lock`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_id: trackId, locked, candidate: state.selectedCandidate }) });
    dispatch({ type: "trackLocked", trackId, locked });
  } catch (e) {
    if (!(e instanceof SessionExpired)) dispatch({ type: "jobError", error: e.message });
  }
}

// Text planner: the prompt becomes generation settings, which then show (editable) in the form below.
async function interpret() {
  if (!canInterpret(state)) return;
  const prompt = state.prompt;
  // Same prompt as last time: go straight back to step 2 and keep any changes made there.
  if (state.interpretation && prompt.trim() === state.interpretedPrompt) { dispatch({ type: "setPromptStep", step: 2 }); return; }
  dispatch({ type: "interpretStart" });
  try {
    const plan = await api("/plan-from-text", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }) });
    dispatch({ type: "interpretSuccess", plan, prompt });
    syncFormFields();
  } catch (e) {
    if (!(e instanceof SessionExpired)) dispatch({ type: "interpretError", error: e.message });
  }
}

// Critic: fetched whenever a different job/candidate is shown (or its locks change).
function maybeCritique() {
  const key = critiqueKey(state);
  if (!key || key === state.critiqueKey || state.jobStatus !== "done" || state.generating) return;
  const [jobId, candidate] = key.split(":");
  dispatch({ type: "critiqueStart", key });
  api(`/generation/${jobId}/critique?candidate=${candidate}`)
    .then((critique) => dispatch({ type: "critiqueReady", key, critique }))
    .catch((e) => { if (!(e instanceof SessionExpired)) dispatch({ type: "critiqueError", key, error: e.message }); });
}

function runCriticAction(action) {
  if (action.type === "regenerate_track") {
    dispatch({ type: "selectTrack", trackId: action.track_id });
    regenTrack();
  } else if (action.type === "regenerate_section") {
    dispatch({ type: "selectSection", index: action.section_index });
    regenSection();
  }
}

// Fetched as a blob so that saving a file never navigates, which could look like leaving the page.
function download() {
  if (!canDownload(state)) return;
  const index = state.selectedCandidate;
  const name = `${state.jobId}_candidate_${index + 1}_seed${currentCandidate(state).seed}.mid`;
  return saveFile(`/generation/${state.jobId}/download?candidate=${index}`, name);
}

function downloadStems() {
  if (!canDownload(state)) return;
  const index = state.selectedCandidate;
  const name = `${state.jobId}_candidate_${index + 1}_seed${currentCandidate(state).seed}_stems.zip`;
  return saveFile(`/generation/${state.jobId}/stems?candidate=${index}`, name);
}

// "Build on this": the server turns the selected result into a new upload; its tracks become locked anchors.
async function buildOn() {
  if (!canBuildOn(state)) return;
  stopPlayback();
  playStatus("");
  const jobId = state.jobId;
  const candidate = state.selectedCandidate;
  dispatch({ type: "buildOnStart" });
  try {
    const r = await api(`/generation/${jobId}/as-input`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate }) });
    $("file").value = "";
    dispatch({ type: "buildOnReady", jobId, candidate, uploadId: r.upload_id, analysis: r.analysis,
      file: { name: r.filename, size: r.size } });
    syncFormFields();
    document.querySelector(".controls").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    if (!(e instanceof SessionExpired)) dispatch({ type: "buildOnError", error: e.message });
  }
}

async function saveFile(path, name) {
  try {
    const res = await request(path);
    const url = URL.createObjectURL(await res.blob());
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (e) {
    if (!(e instanceof SessionExpired)) dispatch({ type: "jobError", error: e.message });
  }
}

// ------------------------------------------------------------------ rendering
function setButton(el, enabled, reasons) {
  el.disabled = !enabled;
  el.title = enabled ? "" : reasons.join("\n");
}

function renderAnalysis() {
  const box = $("analysis-box");
  box.hidden = state.mode !== "midi" || (!state.analysis && !state.analysisError && !state.analyzing);
  $("analysis-error").hidden = !state.analysisError;
  $("analysis-error").textContent = state.analysisError || "";
  const a = state.analysis;
  const facts = $("analysis-facts");
  if (!a) { facts.innerHTML = state.analyzing ? "<div><dd>Analyzing…</dd></div>" : ""; $("analysis-tracks").tBodies[0].innerHTML = ""; $("root-warning").hidden = true; return; }
  const r = a.root;
  const conf = r.root_confidence != null ? `${Math.round(r.root_confidence * 100)}%` : "—";
  const items = [
    ...(state.file ? [["Input", escapeHtml(state.file.name)]] : []),
    ["Detected root", r.root_name ?? "undetermined"], ["Root confidence", conf], ["Key / mode", r.key_name ?? "—"],
    ["Method", r.root_detection_method], ["Tempo", `${a.tempo_bpm} BPM`], ["Meter", a.meter],
    ["Bars", a.n_bars], ["Duration", `${a.duration_seconds}s`],
  ];
  const style = a.style;
  const pct = (l) => l.map((x) => `${escapeHtml(x.label.replace(/_/g, " "))} ${Math.round(x.score * 100)}%`).join(" · ");
  if (style && style.emotion && style.emotion.length) items.push(["Detected emotion", pct(style.emotion)]);
  if (style && style.genre && style.genre.length) items.push(["Detected genre", pct(style.genre)]);
  facts.innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  const warn = $("root-warning");
  const applied = style && style.reliable !== false;
  const usedStyle = style && style.emotion && style.emotion.length
    ? `<div class="wide"><dt>Used for generation</dt><dd>emotion: ${escapeHtml(state.form.emotion)}, genre: ${escapeHtml(state.form.genre)}
       <span class="muted">— ${applied ? "from the detected mood" : "your settings (the detected mood was not applied)"}.
       ${escapeHtml(style.note || "")} To use a different mood, pick it in the Settings tab.</span></dd></div>` : "";
  facts.insertAdjacentHTML("beforeend", usedStyle);
  if (r.root_pitch_class === null) {
    warn.hidden = false; warn.textContent = "The key of your file couldn't be found. Choose it in the MIDI tab before generating.";
  } else if (r.ambiguous) {
    warn.hidden = false;
    warn.textContent = `The key is unclear — it could be ${r.top_candidates.slice(0, 4).map((c) => `${c.root_name} ${c.mode}`).join(", ")}. You can choose it in the MIDI tab.`;
  } else { warn.hidden = true; }
  const roleOpts = ["lead", "bass", "chordal", "harmony", "high_texture", "low_texture", "counterline", "pad", "full_mix"];
  $("analysis-tracks").tBodies[0].innerHTML = a.tracks.map((t) => `<tr>
    <td>${t.track_id}</td><td>${escapeHtml(t.name)}</td><td>${t.instrument}</td><td>${t.n_notes}</td>
    <td>${t.detected_role} <span class="muted">(${Math.round(t.role_confidence * 100)}%)</span></td>
    <td>${t.is_drum ? "drums" : `<select data-track="${t.track_id}">${roleOpts.map((o) =>
      `<option ${((state.trackRoles[t.track_id] || t.assigned_role) === o) ? "selected" : ""}>${o}</option>`).join("")}</select>`}</td>
  </tr>`).join("");
  $("analysis-tracks").querySelectorAll("select").forEach((sel) => {
    sel.onchange = () => dispatch({ type: "setTrackRole", trackId: Number(sel.dataset.track), role: sel.value });
  });
}

function renderResult() {
  const jb = $("job-box");
  jb.hidden = !state.jobStatus && !state.jobError;
  jb.classList.toggle("running", Boolean(state.generating));   // shows the animated progress bar
  $("job-status").textContent = state.generating ? (state.jobStatus || "starting") : (state.jobStatus || "");
  $("job-error").hidden = !state.jobError;
  $("job-error").textContent = state.jobError || "";
  $("job-progress").textContent = state.generating ? (state.jobProgress.slice(-1)[0] || "working…") : "";
  const res = state.result;
  $("result-box").hidden = !res;
  if (!res) return;
  const sel = $("candidate");
  const want = res.candidates.map((c, i) => `#${i + 1} — score ${c.scores.total.toFixed(3)} (seed ${c.seed})`).join("|");
  if (sel.dataset.sig !== want) {
    sel.innerHTML = res.candidates.map((c, i) => `<option value="${i}">#${i + 1} — score ${c.scores.total.toFixed(3)} (seed ${c.seed})</option>`).join("");
    sel.dataset.sig = want;
  }
  sel.value = String(state.selectedCandidate);
  const c = currentCandidate(state);
  $("manifest-link").href = `${API}/generation/${state.jobId}/manifest`;
  const sc = c.scores;
  const comp = { total: sc.total, quality: sc.quality, genre: sc.genre, emotion: sc.emotion, ...sc.components };
  $("scores").innerHTML = Object.entries(comp).map(([k, v]) => `<div>${k.replace(/_/g, " ")}: <b>${Number(v).toFixed(2)}</b></div>`).join("");
  $("result-tracks").tBodies[0].innerHTML = c.tracks.map((t, i) => `<tr data-track="${t.track_id}" class="${state.selectedTrack === t.track_id ? "sel" : ""}">
    <td><span class="swatch" style="background:${trackColor(i)}"></span></td><td>${escapeHtml(t.name)}</td><td>${t.role}</td>
    <td>${t.instrument}</td><td>${t.n_notes}</td><td>${t.source}</td>
    <td><input type="checkbox" data-lock="${t.track_id}" ${t.locked ? "checked" : ""} ${t.source === "input" || state.generating ? "disabled" : ""}
      aria-label="lock track ${t.track_id}"></td></tr>`).join("");
  $("result-tracks").querySelectorAll("tr[data-track]").forEach((tr) => {
    tr.onclick = (e) => { if (e.target.tagName !== "INPUT") dispatch({ type: "selectTrack", trackId: Number(tr.dataset.track) }); };
  });
  $("result-tracks").querySelectorAll("input[data-lock]").forEach((cb) => {
    cb.onchange = () => toggleLock(Number(cb.dataset.lock), cb.checked);
  });
  const sections = res.plan.sections || [];
  $("sections").innerHTML = sections.map((s, i) => `<span class="chip ${state.selectedSection === i ? "on" : ""}" data-sec="${i}">
    ${escapeHtml(s.name)} · bars ${s.start_bar + 1}-${s.end_bar}</span>`).join("");
  $("sections").querySelectorAll(".chip").forEach((ch) => { ch.onclick = () => dispatch({ type: "selectSection", index: Number(ch.dataset.sec) }); });
  const issues = (c.validation.issues || []).map((i) => `<li class="${i.severity === "error" ? "err" : i.severity === "warning" ? "wrn" : ""}">[${i.severity}] ${escapeHtml(i.message)}${i.track_id != null ? ` (track ${i.track_id})` : ""}</li>`);
  const warns = (res.warnings || []).map((w) => `<li class="wrn">${escapeHtml(w)}</li>`);
  const reps = (c.validation.repairs || []).map((r) => `<li>repair: ${escapeHtml(r)}</li>`);
  const all = [...issues, ...warns, ...reps];
  $("warnings").innerHTML = all.length ? all.join("") : `<li>Validation passed with no warnings.</li>`;
  const barBeats = res.meter ? (Number(res.meter.split("/")[0]) * 4) / Number(res.meter.split("/")[1]) : 4;
  drawRoll($("roll"), c, { totalBeats: res.total_beats, barBeats, sections, selectedTrack: state.selectedTrack });
  renderCritic();
}

function render() {
  const midi = state.mode === "midi";
  $("session-notice").hidden = !state.sessionNotice;
  $("session-notice").textContent = state.sessionNotice || "";
  $("empty-state").hidden = !!(state.result || state.jobStatus || state.jobError || state.analysis || state.analyzing
    || state.analysisError);
  const promptMode = state.mode === "prompt";
  const promptStep1 = promptMode && state.promptStep !== 2;
  for (const t of document.querySelectorAll(".tab")) {
    t.classList.toggle("active", t.dataset.mode === state.mode);
    t.setAttribute("aria-selected", String(t.dataset.mode === state.mode));
  }
  // Prompt: step 1 is only the text box; step 2 shows what was found as editable buttons.
  $("prompt-controls").hidden = !promptMode;
  $("prompt-step1").hidden = !promptStep1;
  $("prompt-step2").hidden = !promptMode || promptStep1;
  $("midi-controls").hidden = !midi;
  $("metadata-controls").hidden = midi || promptStep1;
  $("shared-controls").hidden = promptStep1;
  $("edit-prompt").hidden = !promptMode;
  const r = state.analysis && state.analysis.root;
  $("root-field").hidden = !(midi && ((r && (r.root_pitch_class === null || r.ambiguous)) || state.form.rootOverride));
  $("drop-sub").textContent = state.file ? `${state.file.name} · choose another file to replace it`
    : ".mid or .midi · it is analyzed as soon as you choose it";
  renderPrompt();
  renderChoices();
  $("density-val").textContent = Number(state.form.density).toFixed(2);
  const ab = analyzeBlockers(state);
  setButton($("analyze"), canAnalyze(state), ab);
  $("analyze-hint").textContent = midi && ab.length && !state.analyzing ? ab[0] : (state.analyzing ? "Analyzing…" : "");
  const gb = generateBlockers(state);
  setButton($("generate"), canGenerate(state), gb);
  $("generate").textContent = state.generating ? "Generating…" : (promptMode ? "Accept and generate" : "Generate");
  $("generate-hint").innerHTML = gb.map((m) => `<li>${escapeHtml(m)}</li>`).join("");
  const bf = state.buildingFrom;
  $("building-from").hidden = !(midi && bf);
  $("building-from").textContent = bf
    ? `Building on candidate #${bf.candidate + 1} of job ${bf.jobId}: its tracks are locked. Pick the new roles to add, then Generate.`
    : "";
  setButton($("download"), canDownload(state), ["Finish a generation first."]);
  setButton($("download-stems"), canDownload(state), ["Finish a generation first."]);
  setButton($("build-on"), canBuildOn(state), buildOnBlockers(state));
  $("build-on").textContent = state.preparingInput ? "Preparing…" : "Build on this →";
  setButton($("play"), canDownload(state), ["Finish a generation first."]);
  setButton($("stop"), canDownload(state), ["Nothing is playing."]);
  setButton($("regen-track"), canRegenerateTrack(state), regenerateTrackBlockers(state));
  $("regen-track-hint").textContent = regenerateTrackBlockers(state)[0] || "";
  setButton($("regen-section"), canRegenerateSection(state), regenerateSectionBlockers(state));
  $("regen-section-hint").textContent = regenerateSectionBlockers(state)[0] || "";
  renderAnalysis();
  renderResult();
}

function renderPrompt() {
  const ib = interpretBlockers(state);
  setButton($("interpret"), canInterpret(state), ib);
  $("interpret").textContent = state.interpreting ? "Reading your prompt…" : "Next →";
  $("interpret-hint").textContent = state.interpretError ? `Couldn't read the prompt: ${state.interpretError}` : "";
  const p = state.interpretation;
  const box = $("interpretation");
  box.hidden = !p;
  if (!p) { $("prompt-quote").textContent = ""; return; }
  // The prompt, with the words that became settings coloured by setting.
  const text = state.interpretedPrompt || "";
  const segs = promptSegments(text, promptHighlights(text, p.settings, p.explanation));
  $("prompt-quote").innerHTML = `“${segs.map((s) => (s.cat ? `<mark class="hl ${s.cat}">${escapeHtml(s.text)}</mark>`
    : escapeHtml(s.text))).join("")}”`;
  // One chip per setting; solid = found in the prompt, dashed = kept from before or left to the app.
  const S = p.settings, f = state.form;
  const cap = (v) => String(v).charAt(0).toUpperCase() + String(v).slice(1);
  const chips = [
    ["genre", "Genre", f.genre && cap(f.genre), S.genre, "genre-choices"],
    ["mood", "Mood", f.emotion && cap(f.emotion), S.emotion, "emotion-choices"],
    ["key", "Key", f.key && `${f.key}${f.scaleMode ? ` ${f.scaleMode}` : ""}`, S.key, "key-choices"],
    ["tempo", "Tempo", f.tempo && `${f.tempo} BPM`, S.tempo, "tempo"],
    ["length", "Length", f.bars && `${f.bars} bars`, S.bars, "bars"],
    ["instruments", "Instruments", f.instrumentation && cap(f.instrumentation), S.instrumentation, null],
    ["form", "Form", f.form && f.form !== "auto" ? f.form : null, S.form, "form-choices"],
  ];
  const chip = ([cat, label, value, found, target]) => {
    const shown = value || "App will choose";
    const note = found ? "" : (value ? " · not in your prompt" : "");
    const tag = target ? "button" : "span";
    return `<${tag} class="heard ${cat} ${found ? "" : "auto"}" ${target ? `type="button" data-target="${target}"` : ""}
      ${target ? "" : `title="Change the instruments by editing your prompt."`}>
      <i aria-hidden="true"></i><small>${label}</small> ${escapeHtml(shown)}<small>${note}</small></${tag}>`;
  };
  box.innerHTML = `<div>${chips.filter((c) => c[3] != null).map(chip).join("")}</div>`
    + `<div>${chips.filter((c) => c[3] == null).map(chip).join("")}</div>`;
  box.querySelectorAll("button[data-target]").forEach((b) => { b.onclick = () => jumpTo(b.dataset.target); });
}

// Scroll to a control and flash it, so a chip leads straight to the setting it names.
function jumpTo(id) {
  const el = $(id);
  const target = el.closest(".group") || el.closest(".field") || el;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.remove("flash"); void target.offsetWidth; target.classList.add("flash");
  if (el.tagName === "INPUT") setTimeout(() => el.focus({ preventScroll: true }), 350);
}

function renderCritic() {
  const box = $("critic");
  const key = critiqueKey(state);
  const fresh = key && key === state.critiqueKey;
  $("critic-source").textContent = fresh && state.critique ? `(${state.critique.source})` : "";
  if (!fresh || state.critiqueLoading) { box.innerHTML = `<p class="hint">Listening to this candidate…</p>`; return; }
  if (state.critiqueError) { box.innerHTML = `<p class="hint">Critic unavailable: ${escapeHtml(state.critiqueError)}</p>`; return; }
  const c = state.critique;
  if (!c) { box.innerHTML = ""; return; }
  box.innerHTML = `<p class="small">${escapeHtml(c.summary)}</p>` + c.notes.map((n, i) => `<div class="note">
      <span class="tag ${n.severity}">${escapeHtml(n.kind)}</span>
      <span class="msg">${escapeHtml(n.message)}</span>
      ${n.action ? `<button class="secondary" data-note="${i}" ${state.generating ? "disabled" : ""}>${escapeHtml(n.action.label)}</button>` : ""}
    </div>`).join("");
  box.querySelectorAll("button[data-note]").forEach((b) => {
    b.onclick = () => runCriticAction(c.notes[Number(b.dataset.note)].action);
  });
}

// Push state.form into the inputs (after the prompt interpreter or the emotion reader changed it).
function syncFormFields() {
  for (const [id, field] of FIELDS) {
    const v = state.form[field];
    $(id).value = v === null || v === undefined ? "" : v;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

// ------------------------------------------------------------------ wiring
// [element id, form field] for every plain input/select.
const FIELDS = [["inputRole", "inputRole"], ["rootOverride", "rootOverride"], ["targetKey", "targetKey"],
  ["genre", "genre"], ["emotion", "emotion"], ["key", "key"], ["form", "form"], ["scaleMode", "scaleMode"],
  ["meter", "meter"], ["tempo", "tempo"], ["bars", "bars"], ["density", "density"], ["instrumentation", "instrumentation"],
  ["variations", "variations"], ["seed", "seed"], ["engine", "engine"]];
document.querySelectorAll(".tab").forEach((t) => { t.onclick = () => dispatch({ type: "setMode", mode: t.dataset.mode }); });
$("file").addEventListener("change", () => {
  const f = $("file").files[0];
  dispatch({ type: "fileSelected", file: f ? { name: f.name, size: f.size } : null });
  if (f) analyze();                      // analyze immediately; the button allows re-analysis (e.g. after a root override)
});
$("analyze").onclick = analyze;
$("generate").onclick = generate;
$("download").onclick = download;
$("download-stems").onclick = downloadStems;
$("build-on").onclick = buildOn;
$("regen-track").onclick = regenTrack;
$("regen-section").onclick = regenSection;
$("candidate").onchange = () => dispatch({ type: "selectCandidate", index: Number($("candidate").value) });
const playStatus = (text) => {
  $("play-status").textContent = text;
  document.body.classList.toggle("playing", text.startsWith("Playing"));   // animates the header equalizer
};
$("play").onclick = () => {
  const c = currentCandidate(state);
  if (c) play(c, state.result.tempo, { onStatus: playStatus }).catch((e) => playStatus(`Playback failed: ${e.message}`));
};
$("stop").onclick = () => { stopPlayback(); playStatus(""); };
for (const [id, field] of FIELDS) bindField(id, field);
$("prompt").addEventListener("input", () => dispatch({ type: "setPrompt", value: $("prompt").value }));
$("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) interpret(); });
$("interpret").onclick = interpret;
$("edit-prompt").onclick = () => { dispatch({ type: "setPromptStep", step: 1 }); $("prompt").focus(); };
window.addEventListener("resize", () => renderResult());

// Closing or refreshing the page disposes of everything it uploaded or generated on the server.
window.addEventListener("pagehide", () => {
  stopPlayback();
  if (state.sessionId) navigator.sendBeacon(`${API}/session/${state.sessionId}/close`);
});
// A page restored from the browser's back/forward cache has already closed its session.
window.addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") heartbeat(); });

// The Prompt tab opens first; `?mode=midi` or `?mode=metadata` opens another tab.
const startMode = new URLSearchParams(location.search).get("mode") || "prompt";
if (["metadata", "prompt", "midi"].includes(startMode)) state = reduce(state, { type: "setMode", mode: startMode });
render();
heartbeatTimer = setInterval(heartbeat, DEFAULT_HEARTBEAT_MS);
connect();
