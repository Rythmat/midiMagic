// DOM layer: renders from the pure state in ui-state.js and talks to the API.
import {
  analyzeBlockers, buildGenerateBody, canAnalyze, canDownload, canGenerate, canRegenerateSection,
  canRegenerateTrack, currentCandidate, generateBlockers, initialState, reduce, regenerateSectionBlockers,
  regenerateTrackBlockers,
} from "./ui-state.js";
import { drawRoll, play, stopPlayback, trackColor } from "./pianoroll.js";

// API base: same origin by default; `?api=http://host:port` when the UI is served separately.
const API = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
let state = initialState();

function dispatch(action) {
  state = reduce(state, action);
  render();
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
    const roles = $("roles");
    roles.innerHTML = "";
    for (const r of o.generated_roles) {
      const c = document.createElement("span");
      c.className = "chip"; c.dataset.role = r; c.textContent = r.replace(/_/g, " ");
      c.setAttribute("role", "checkbox"); c.tabIndex = 0;
      c.onclick = () => dispatch({ type: "toggleRole", role: r });
      c.onkeydown = (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); c.click(); } };
      roles.appendChild(c);
    }
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

const generate = () => canGenerate(state) && startJob("/generate", buildGenerateBody(state));

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

// Fetched as a blob so that saving a file never navigates, which could look like leaving the page.
async function download() {
  if (!canDownload(state)) return;
  const index = state.selectedCandidate;
  const name = `${state.jobId}_candidate_${index + 1}_seed${currentCandidate(state).seed}.mid`;
  try {
    const res = await request(`/generation/${state.jobId}/download?candidate=${index}`);
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
    ["Detected root", r.root_name ?? "undetermined"], ["Root confidence", conf], ["Key / mode", r.key_name ?? "—"],
    ["Method", r.root_detection_method], ["Tempo", `${a.tempo_bpm} BPM`], ["Meter", a.meter],
    ["Bars", a.n_bars], ["Duration", `${a.duration_seconds}s`],
  ];
  facts.innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  const warn = $("root-warning");
  if (r.root_pitch_class === null) {
    warn.hidden = false; warn.textContent = "Root could not be determined. Choose a manual root and re-analyze (or just generate with the override).";
  } else if (r.ambiguous) {
    warn.hidden = false;
    warn.textContent = `Root is ambiguous — candidates: ${r.top_candidates.slice(0, 4).map((c) => `${c.root_name} ${c.mode}`).join(", ")}. Consider a manual root.`;
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
}

function render() {
  const midi = state.mode === "midi";
  $("session-notice").hidden = !state.sessionNotice;
  $("session-notice").textContent = state.sessionNotice || "";
  $("empty-state").hidden = !!(state.result || state.jobStatus || state.jobError || state.analysis || state.analyzing
    || state.analysisError);
  $("tab-midi").classList.toggle("active", midi);
  $("tab-metadata").classList.toggle("active", !midi);
  $("midi-controls").hidden = !midi;
  $("metadata-controls").hidden = midi;
  document.querySelectorAll("#roles .chip").forEach((c) => {
    const on = state.form.generatedRoles.includes(c.dataset.role);
    c.classList.toggle("on", on); c.setAttribute("aria-checked", String(on));
  });
  $("density-val").textContent = Number(state.form.density).toFixed(2);
  const ab = analyzeBlockers(state);
  setButton($("analyze"), canAnalyze(state), ab);
  $("analyze-hint").textContent = midi && ab.length && !state.analyzing ? ab[0] : (state.analyzing ? "Analyzing…" : "");
  const gb = generateBlockers(state);
  setButton($("generate"), canGenerate(state), gb);
  $("generate").textContent = state.generating ? "Generating…" : "Generate";
  $("generate-hint").innerHTML = gb.map((m) => `<li>${escapeHtml(m)}</li>`).join("");
  setButton($("download"), canDownload(state), ["Finish a generation first."]);
  setButton($("play"), canDownload(state), ["Finish a generation first."]);
  setButton($("stop"), canDownload(state), ["Nothing is playing."]);
  setButton($("regen-track"), canRegenerateTrack(state), regenerateTrackBlockers(state));
  $("regen-track-hint").textContent = regenerateTrackBlockers(state)[0] || "";
  setButton($("regen-section"), canRegenerateSection(state), regenerateSectionBlockers(state));
  $("regen-section-hint").textContent = regenerateSectionBlockers(state)[0] || "";
  renderAnalysis();
  renderResult();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

// ------------------------------------------------------------------ wiring
document.querySelectorAll(".tab").forEach((t) => { t.onclick = () => dispatch({ type: "setMode", mode: t.dataset.mode }); });
$("file").addEventListener("change", () => {
  const f = $("file").files[0];
  dispatch({ type: "fileSelected", file: f ? { name: f.name, size: f.size } : null });
  if (f) analyze();                      // analyze immediately; the button allows re-analysis (e.g. after a root override)
});
$("analyze").onclick = analyze;
$("generate").onclick = generate;
$("download").onclick = download;
$("regen-track").onclick = regenTrack;
$("regen-section").onclick = regenSection;
$("candidate").onchange = () => dispatch({ type: "selectCandidate", index: Number($("candidate").value) });
$("play").onclick = () => { const c = currentCandidate(state); if (c) play(c, state.result.tempo); };
$("stop").onclick = stopPlayback;
for (const [id, field] of [["inputRole", "inputRole"], ["rootOverride", "rootOverride"], ["targetKey", "targetKey"],
  ["genre", "genre"], ["emotion", "emotion"], ["key", "key"], ["form", "form"], ["scaleMode", "scaleMode"],
  ["meter", "meter"], ["tempo", "tempo"], ["bars", "bars"], ["density", "density"], ["instrumentation", "instrumentation"],
  ["variations", "variations"], ["seed", "seed"], ["engine", "engine"]]) bindField(id, field);
window.addEventListener("resize", () => renderResult());

// Closing or refreshing the page disposes of everything it uploaded or generated on the server.
window.addEventListener("pagehide", () => {
  stopPlayback();
  if (state.sessionId) navigator.sendBeacon(`${API}/session/${state.sessionId}/close`);
});
// A page restored from the browser's back/forward cache has already closed its session.
window.addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") heartbeat(); });

if (new URLSearchParams(location.search).get("mode") === "metadata") state = reduce(state, { type: "setMode", mode: "metadata" });
render();
heartbeatTimer = setInterval(heartbeat, DEFAULT_HEARTBEAT_MS);
connect();
