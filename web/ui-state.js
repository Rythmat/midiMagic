// Pure UI state logic (no DOM). Shared by app.js and the Node test-suite (web/tests).
// Buttons are enabled only when these predicates say so, and every disabled state has a reason.

export const MAX_VARIATIONS = 8;
export const MAX_BARS = 256;

export function initialState() {
  return {
    mode: "midi",                 // "midi" (mode 1) | "metadata" (mode 2)
    options: null,                // from GET /options
    optionsError: null,
    sessionId: null,              // held only in this page, so a refresh starts a new session
    sessionNotice: null,          // why the previous session's data is gone
    file: null,                   // {name, size}
    analyzing: false,
    analysis: null,
    uploadId: null,
    analysisError: null,
    form: {
      inputRole: "lead",
      generatedRoles: ["chord_bed", "bass_line", "drums"],
      genre: "pop", emotion: "happy", key: "C", scaleMode: "", targetKey: "", rootOverride: "",
      tempo: "", meter: "4/4", bars: "", density: 0.5, instrumentation: "", form: "auto",
      variations: 3, seed: 0, engine: "rule",
    },
    trackRoles: {},               // track_id -> role override for the uploaded file
    generating: false,
    jobId: null,
    jobStatus: null,              // queued | running | done | error
    jobProgress: [],
    jobError: null,
    result: null,
    selectedCandidate: 0,
    selectedTrack: null,
    selectedSection: null,
  };
}

const isInt = (v) => /^-?\d+$/.test(String(v).trim());

export function fieldErrors(s) {
  const f = s.form;
  const errs = {};
  if (f.tempo !== "" && f.tempo !== null) {
    const t = Number(f.tempo);
    if (!Number.isFinite(t) || t < 20 || t > 300) errs.tempo = "Tempo must be 20-300 BPM (or empty for automatic).";
  }
  if (f.bars !== "" && f.bars !== null) {
    if (!isInt(f.bars) || Number(f.bars) < 1 || Number(f.bars) > MAX_BARS) errs.bars = `Length must be 1-${MAX_BARS} bars (or empty).`;
  }
  if (!isInt(f.variations) || Number(f.variations) < 1 || Number(f.variations) > MAX_VARIATIONS) {
    errs.variations = `Variations must be 1-${MAX_VARIATIONS}.`;
  }
  if (!isInt(f.seed)) errs.seed = "Seed must be an integer.";
  if (!/^\s*\d+\s*\/\s*\d+\s*$/.test(f.meter || "")) errs.meter = "Meter must look like 4/4 or 6/8.";
  const d = Number(f.density);
  if (!Number.isFinite(d) || d < 0 || d > 1) errs.density = "Density must be between 0 and 1.";
  return errs;
}

export function analyzeBlockers(s) {
  const out = [];
  if (s.mode !== "midi") out.push("Analysis is only used in MIDI mode.");
  if (!s.sessionId) out.push("Connecting to the server.");
  if (!s.file) out.push("Choose a MIDI file.");
  if (s.analyzing) out.push("Analysis in progress.");
  if (s.generating) out.push("Wait for the current generation to finish.");
  return out;
}

export const canAnalyze = (s) => analyzeBlockers(s).length === 0;

export function generateBlockers(s) {
  const out = [];
  if (!s.options) out.push(s.optionsError ? "Server unavailable: " + s.optionsError : "Loading options from the server.");
  else if (!s.sessionId) out.push("Connecting to the server.");
  if (s.generating) out.push("A generation is already running.");
  if (s.mode === "midi") {
    if (!s.file) out.push("Upload a MIDI file.");
    else if (s.analyzing) out.push("Wait for the analysis to finish.");
    else if (!s.uploadId || !s.analysis) out.push("Analyze the uploaded MIDI first.");
    if (s.analysis && s.analysis.root && s.analysis.root.root_pitch_class === null && !s.form.rootOverride) {
      out.push("The root could not be detected: choose a manual root override.");
    }
    if (!s.form.inputRole) out.push("Choose the input role.");
  } else {
    if (!s.form.genre) out.push("Choose a genre.");
    if (!s.form.emotion) out.push("Choose an emotion.");
    if (!s.form.key) out.push("Choose a key/root.");
  }
  if (!s.form.generatedRoles || s.form.generatedRoles.length === 0) out.push("Select at least one role to generate.");
  for (const msg of Object.values(fieldErrors(s))) out.push(msg);
  return out;
}

export const canGenerate = (s) => generateBlockers(s).length === 0;

export function currentCandidate(s) {
  if (!s.result || !s.result.candidates) return null;
  return s.result.candidates[s.selectedCandidate] || null;
}

export function canDownload(s) {
  return s.jobStatus === "done" && !s.generating && !!currentCandidate(s);
}

export function regenerateTrackBlockers(s) {
  const out = [];
  const c = currentCandidate(s);
  if (!c || s.jobStatus !== "done") out.push("Generate something first.");
  if (s.generating) out.push("A job is running.");
  if (c) {
    const t = c.tracks.find((x) => x.track_id === s.selectedTrack);
    if (!t) out.push("Select a generated track.");
    else if (t.source !== "generated") out.push("Input tracks are locked and cannot be regenerated.");
    else if (t.locked) out.push("Unlock the track to regenerate it.");
  }
  return out;
}

export const canRegenerateTrack = (s) => regenerateTrackBlockers(s).length === 0;

export function regenerateSectionBlockers(s) {
  const out = [];
  const c = currentCandidate(s);
  if (!c || s.jobStatus !== "done") out.push("Generate something first.");
  if (s.generating) out.push("A job is running.");
  if (s.selectedSection === null || s.selectedSection === undefined) out.push("Select a section.");
  if (c && !c.tracks.some((t) => t.source === "generated" && !t.locked)) out.push("All generated tracks are locked.");
  return out;
}

export const canRegenerateSection = (s) => regenerateSectionBlockers(s).length === 0;

// Build the POST /generate body from the state.
export function buildGenerateBody(s) {
  const f = s.form;
  const body = {
    mode: s.mode, genre: f.genre || null, emotion: f.emotion || null, key: f.key || null,
    scale_mode: f.scaleMode || null, tempo: f.tempo === "" ? null : Number(f.tempo), meter: f.meter,
    bars: f.bars === "" ? null : Number(f.bars), density: Number(f.density),
    instrumentation: f.instrumentation || null, form: f.form === "auto" ? null : f.form,
    generated_roles: [...f.generatedRoles], n_variations: Number(f.variations), seed: Number(f.seed),
    engine: f.engine,
  };
  if (s.mode === "midi") {
    body.upload_id = s.uploadId;
    body.input_role = f.inputRole;
    body.target_key = f.targetKey || null;
    body.root_override = f.rootOverride || null;
    if (Object.keys(s.trackRoles).length) body.track_roles = { ...s.trackRoles };
  }
  return body;
}

// Reducer: every UI event goes through here so behaviour is testable.
export function reduce(s, action) {
  const n = { ...s, form: { ...s.form } };
  switch (action.type) {
    case "optionsLoaded":
      n.options = action.options; n.optionsError = null; return n;
    case "optionsFailed":
      n.options = null; n.optionsError = action.error; return n;
    case "sessionReady":
      n.sessionId = action.sessionId; return n;
    case "sessionExpired":
      // The server has dropped everything: keep only the user's settings and say what happened.
      return { ...initialState(), mode: s.mode, options: s.options, form: n.form, sessionNotice: action.notice };
    case "setMode":
      n.mode = action.mode; return n;
    case "fileSelected":
      // A new file invalidates the previous analysis and upload.
      n.file = action.file; n.analysis = null; n.uploadId = null; n.analysisError = null; n.trackRoles = {};
      n.sessionNotice = null;
      return n;
    case "analyzeStart":
      n.analyzing = true; n.analysisError = null; return n;
    case "analyzeSuccess":
      n.analyzing = false; n.analysis = action.analysis; n.uploadId = action.uploadId; n.analysisError = null;
      return n;
    case "analyzeError":
      n.analyzing = false; n.analysis = null; n.uploadId = null; n.analysisError = action.error; return n;
    case "setField":
      n.form[action.field] = action.value; return n;
    case "toggleRole": {
      const roles = new Set(n.form.generatedRoles);
      if (roles.has(action.role)) roles.delete(action.role); else roles.add(action.role);
      n.form.generatedRoles = [...roles]; return n;
    }
    case "setTrackRole":
      n.trackRoles = { ...s.trackRoles, [action.trackId]: action.role }; return n;
    case "generateStart":
      n.generating = true; n.jobError = null; n.jobStatus = "queued"; n.jobProgress = []; n.sessionNotice = null;
      return n;
    case "jobAccepted":
      n.jobId = action.jobId; return n;
    case "jobUpdate":
      n.jobStatus = action.status; n.jobProgress = action.progress || []; return n;
    case "jobDone":
      n.generating = false; n.jobStatus = "done"; n.result = action.result; n.jobId = action.jobId;
      n.selectedCandidate = 0; n.selectedTrack = null; n.selectedSection = null; return n;
    case "jobError":
      // Keep the previous result visible; just report the failure.
      n.generating = false; n.jobStatus = s.result ? "done" : "error"; n.jobError = action.error; return n;
    case "selectCandidate":
      n.selectedCandidate = action.index; n.selectedTrack = null; return n;
    case "selectTrack":
      n.selectedTrack = action.trackId; return n;
    case "selectSection":
      n.selectedSection = action.index; return n;
    case "trackLocked": {
      const c = currentCandidate(n);
      if (c) {
        n.result = { ...n.result, candidates: n.result.candidates.map((cand, i) => i !== n.selectedCandidate ? cand : {
          ...cand, tracks: cand.tracks.map((t) => t.track_id === action.trackId ? { ...t, locked: action.locked } : t),
        }) };
      }
      return n;
    }
    default:
      return s;
  }
}
