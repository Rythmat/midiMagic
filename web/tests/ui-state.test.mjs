// Run with:  node --test web/tests
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildGenerateBody, canAnalyze, canDownload, canGenerate, canRegenerateSection, canRegenerateTrack,
  generateBlockers, initialState, reduce,
} from "../ui-state.js";

const OPTIONS = { genres: ["pop"], emotions: ["happy"] };
const ANALYSIS = { root: { root_pitch_class: 2, root_name: "D", root_confidence: 0.9 }, tracks: [] };
const RESULT = {
  candidates: [{ tracks: [
    { track_id: 0, source: "input", locked: true },
    { track_id: 1, source: "generated", locked: false },
  ] }],
};

const connected = () => [
  { type: "optionsLoaded", options: OPTIONS },
  { type: "sessionReady", sessionId: "s1" },
].reduce(reduce, initialState());

const ready = () => [
  { type: "fileSelected", file: { name: "a.mid", size: 10 } },
  { type: "analyzeStart" },
  { type: "analyzeSuccess", analysis: ANALYSIS, uploadId: "u1" },
].reduce(reduce, connected());

test("generate is disabled until connected, a file is uploaded and analyzed", () => {
  let s = initialState();
  assert.equal(canGenerate(s), false);
  s = reduce(s, { type: "optionsLoaded", options: OPTIONS });
  assert.ok(generateBlockers(s).some((m) => m.includes("Connecting")));
  s = reduce(s, { type: "sessionReady", sessionId: "s1" });
  assert.equal(canGenerate(s), false);
  assert.ok(generateBlockers(s).some((m) => m.includes("Upload")));
  s = reduce(s, { type: "fileSelected", file: { name: "a.mid", size: 10 } });
  assert.equal(canAnalyze(s), true);
  assert.equal(canGenerate(s), false);
  s = reduce(s, { type: "analyzeStart" });
  assert.equal(canAnalyze(s), false);
  assert.equal(canGenerate(s), false);
  s = reduce(s, { type: "analyzeSuccess", analysis: ANALYSIS, uploadId: "u1" });
  assert.equal(canGenerate(s), true);
});

test("selecting a new file invalidates the analysis", () => {
  let s = ready();
  s = reduce(s, { type: "fileSelected", file: { name: "b.mid", size: 5 } });
  assert.equal(s.uploadId, null);
  assert.equal(canGenerate(s), false);
});

test("analysis errors disable generation and are reported", () => {
  let s = reduce(ready(), { type: "fileSelected", file: { name: "bad.mid", size: 3 } });
  s = reduce(s, { type: "analyzeStart" });
  s = reduce(s, { type: "analyzeError", error: "Corrupted MIDI data" });
  assert.equal(s.analysisError, "Corrupted MIDI data");
  assert.equal(canGenerate(s), false);
});

test("undetected root requires a manual override", () => {
  let s = reduce(ready(), { type: "analyzeSuccess", uploadId: "u2",
    analysis: { root: { root_pitch_class: null }, tracks: [] } });
  assert.equal(canGenerate(s), false);
  assert.ok(generateBlockers(s).some((m) => m.includes("manual root")));
  s = reduce(s, { type: "setField", field: "rootOverride", value: "E minor" });
  assert.equal(canGenerate(s), true);
});

test("no roles or invalid fields block generation with reasons", () => {
  let s = ready();
  for (const r of [...s.form.generatedRoles]) s = reduce(s, { type: "toggleRole", role: r });
  assert.ok(generateBlockers(s).some((m) => m.includes("at least one role")));
  s = reduce(s, { type: "toggleRole", role: "pad" });
  s = reduce(s, { type: "setField", field: "tempo", value: "999" });
  assert.equal(canGenerate(s), false);
  s = reduce(s, { type: "setField", field: "tempo", value: "" });
  s = reduce(s, { type: "setField", field: "variations", value: "0" });
  assert.equal(canGenerate(s), false);
  s = reduce(s, { type: "setField", field: "variations", value: "2" });
  assert.equal(canGenerate(s), true);
});

test("metadata mode needs genre, emotion and key but no file", () => {
  let s = reduce(connected(), { type: "setMode", mode: "metadata" });
  assert.equal(canGenerate(s), true);
  s = reduce(s, { type: "setField", field: "key", value: "" });
  assert.equal(canGenerate(s), false);
});

test("generate and download buttons follow the job lifecycle", () => {
  let s = reduce(ready(), { type: "generateStart" });
  assert.equal(canGenerate(s), false);
  assert.equal(canDownload(s), false);
  s = reduce(s, { type: "jobAccepted", jobId: "j1" });
  s = reduce(s, { type: "jobUpdate", status: "running", progress: ["x"] });
  assert.equal(canDownload(s), false);
  s = reduce(s, { type: "jobDone", jobId: "j1", result: RESULT });
  assert.equal(canDownload(s), true);
  assert.equal(canGenerate(s), true);
});

test("job errors are surfaced and re-enable generation", () => {
  let s = reduce(ready(), { type: "generateStart" });
  s = reduce(s, { type: "jobError", error: "boom" });
  assert.equal(s.jobError, "boom");
  assert.equal(canGenerate(s), true);
  assert.equal(canDownload(s), false);
});

test("regenerate track/section respect locks and input tracks", () => {
  let s = reduce(ready(), { type: "jobDone", jobId: "j1", result: RESULT });
  assert.equal(canRegenerateTrack(s), false);
  s = reduce(s, { type: "selectTrack", trackId: 0 });
  assert.equal(canRegenerateTrack(s), false);            // input track
  s = reduce(s, { type: "selectTrack", trackId: 1 });
  assert.equal(canRegenerateTrack(s), true);
  s = reduce(s, { type: "trackLocked", trackId: 1, locked: true });
  assert.equal(canRegenerateTrack(s), false);
  s = reduce(s, { type: "selectSection", index: 0 });
  assert.equal(canRegenerateSection(s), false);          // every generated track locked
  s = reduce(s, { type: "trackLocked", trackId: 1, locked: false });
  assert.equal(canRegenerateSection(s), true);
});

test("request body reflects the form", () => {
  let s = reduce(ready(), { type: "setField", field: "targetKey", value: "A" });
  s = reduce(s, { type: "setTrackRole", trackId: 0, role: "lead" });
  const b = buildGenerateBody(s);
  assert.equal(b.mode, "midi");
  assert.equal(b.upload_id, "u1");
  assert.equal(b.target_key, "A");
  assert.deepEqual(b.track_roles, { 0: "lead" });
  assert.equal(b.tempo, null);
});

test("analysis and generation need a session", () => {
  let s = reduce(initialState(), { type: "optionsLoaded", options: OPTIONS });
  s = reduce(s, { type: "fileSelected", file: { name: "a.mid", size: 10 } });
  assert.equal(canAnalyze(s), false);
  s = reduce(s, { type: "setMode", mode: "metadata" });
  assert.equal(canGenerate(s), false);
  s = reduce(s, { type: "sessionReady", sessionId: "s1" });
  assert.equal(canGenerate(s), true);
});

test("an expired session clears results and disables every action that used them", () => {
  let s = reduce(ready(), { type: "setField", field: "seed", value: "42" });
  s = reduce(s, { type: "generateStart" });
  s = reduce(s, { type: "jobDone", jobId: "j1", result: RESULT });
  s = reduce(s, { type: "selectTrack", trackId: 1 });
  s = reduce(s, { type: "selectSection", index: 0 });
  assert.equal(canDownload(s), true);
  assert.equal(canRegenerateTrack(s), true);

  s = reduce(s, { type: "sessionExpired", notice: "gone" });
  assert.equal(s.sessionNotice, "gone");
  assert.equal(s.sessionId, null);
  assert.equal(s.result, null);
  assert.equal(s.uploadId, null);
  assert.equal(s.file, null);
  assert.equal(canDownload(s), false);
  assert.equal(canRegenerateTrack(s), false);
  assert.equal(canRegenerateSection(s), false);
  assert.equal(canGenerate(s), false);
  assert.equal(s.form.seed, "42");                       // the user's settings survive

  // Reconnected: generation works again once there is a new file, and the notice goes away.
  s = reduce(s, { type: "sessionReady", sessionId: "s2" });
  assert.ok(generateBlockers(s).some((m) => m.includes("Upload")));
  assert.equal(s.sessionNotice, "gone");
  s = reduce(s, { type: "fileSelected", file: { name: "a.mid", size: 10 } });
  assert.equal(s.sessionNotice, null);
  assert.equal(canAnalyze(s), true);
});

test("an expired session interrupts a running job", () => {
  let s = reduce(ready(), { type: "generateStart" });
  s = reduce(s, { type: "sessionExpired", notice: "gone" });
  assert.equal(s.generating, false);
  assert.equal(s.jobStatus, null);
  s = reduce(reduce(s, { type: "setMode", mode: "metadata" }), { type: "sessionReady", sessionId: "s2" });
  assert.equal(canGenerate(s), true);
  s = reduce(s, { type: "generateStart" });
  assert.equal(s.sessionNotice, null);
});
