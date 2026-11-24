// Run with:  node --test web/tests
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildGenerateBody, canAnalyze, canBuildOn, canDownload, canGenerate, canRegenerateSection, canRegenerateTrack, critiqueKey,
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
  assert.ok(generateBlockers(s).some((m) => m.includes("at least one part")));
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

test("build on this turns a result into a locked MIDI input and asks for new roles", () => {
  let s = reduce(connected(), { type: "setMode", mode: "metadata" });
  assert.equal(canBuildOn(s), false);                    // nothing generated yet
  s = reduce(s, { type: "generateStart" });
  s = reduce(s, { type: "jobDone", jobId: "j1", result: RESULT });
  assert.equal(canBuildOn(s), true);
  s = reduce(s, { type: "buildOnStart" });
  assert.equal(canBuildOn(s), false);
  s = reduce(s, { type: "buildOnReady", jobId: "j1", candidate: 0, uploadId: "u9",
    file: { name: "j1_candidate_1.mid", size: 100 }, analysis: ANALYSIS });
  assert.equal(s.mode, "midi");
  assert.equal(s.form.inputRole, "full_mix");
  assert.deepEqual(s.buildingFrom, { jobId: "j1", candidate: 0 });
  assert.ok(generateBlockers(s).some((m) => m.includes("at least one part")));
  s = reduce(s, { type: "toggleRole", role: "pad" });
  assert.equal(canGenerate(s), true);
  const b = buildGenerateBody(s);
  assert.equal(b.upload_id, "u9");
  assert.equal(b.input_role, "full_mix");
  assert.deepEqual(b.generated_roles, ["pad"]);
  s = reduce(s, { type: "fileSelected", file: { name: "other.mid", size: 5 } });
  assert.equal(s.buildingFrom, null);                    // a new file replaces the built-on result
});

test("a failed build-on request is reported and can be retried", () => {
  let s = reduce(ready(), { type: "jobDone", jobId: "j1", result: RESULT });
  s = reduce(s, { type: "buildOnStart" });
  s = reduce(s, { type: "buildOnError", error: "nope" });
  assert.equal(s.jobError, "nope");
  assert.equal(canBuildOn(s), true);
});

test("prompt mode needs an interpretation of the current prompt, then sends metadata settings", () => {
  const opts = { genres: ["pop", "classical"], emotions: ["happy", "sad"] };
  let s = reduce(reduce(initialState(), { type: "optionsLoaded", options: opts }), { type: "sessionReady", sessionId: "s1" });
  s = reduce(s, { type: "setMode", mode: "prompt" });
  assert.ok(generateBlockers(s).some((m) => m.includes("Describe")));
  s = reduce(s, { type: "setPrompt", value: "sad piano in D minor" });
  assert.equal(canGenerate(s), false);
  s = reduce(s, { type: "interpretStart" });
  s = reduce(s, { type: "interpretSuccess", prompt: "sad piano in D minor ", plan: { settings: {
    genre: "jazz", emotion: "sad", key: "D", scale_mode: "minor", tempo: 76, bars: 16, form: null,
    instrumentation: "piano", density: null, generated_roles: ["full_composition"] } } });
  assert.equal(canGenerate(s), true);
  assert.equal(s.form.emotion, "sad");
  assert.equal(s.form.genre, "pop");                     // unknown label "jazz" is not applied
  assert.equal(s.form.tempo, "76");
  const b = buildGenerateBody(s);
  assert.equal(b.mode, "metadata");
  assert.equal(b.scale_mode, "minor");
  assert.equal(b.bars, 16);
  s = reduce(s, { type: "setPrompt", value: "happy pop" });
  assert.ok(generateBlockers(s).some((m) => m.includes("Interpret again")));
});

test("the detected emotion and genre of an upload drive generation", () => {
  const opts = { genres: ["pop", "jazz"], emotions: ["happy", "sad"] };
  let s = reduce(connected(), { type: "optionsLoaded", options: opts });
  s = reduce(s, { type: "fileSelected", file: { name: "a.mid", size: 1 } });
  s = reduce(s, { type: "analyzeSuccess", uploadId: "u1", analysis: { ...ANALYSIS,
    style: { source: "classifier", emotion: [{ label: "sad", score: 0.6 }], genre: [{ label: "jazz", score: 0.5 }] } } });
  assert.equal(s.form.emotion, "sad");
  assert.equal(s.form.genre, "jazz");
  assert.equal(buildGenerateBody(s).emotion, "sad");
});

test("critique answers are keyed to the job and candidate they were asked for", () => {
  let s = reduce(ready(), { type: "jobDone", jobId: "j1", result: RESULT });
  assert.equal(critiqueKey(s), "j1:0");
  s = reduce(s, { type: "critiqueStart", key: "j1:0" });
  s = reduce(s, { type: "critiqueReady", key: "old:0", critique: { notes: [] } });
  assert.equal(s.critique, null);                        // stale answer ignored
  s = reduce(s, { type: "critiqueReady", key: "j1:0", critique: { notes: [1] } });
  assert.deepEqual(s.critique, { notes: [1] });
  s = reduce(s, { type: "trackLocked", trackId: 1, locked: true });
  assert.equal(s.critiqueKey, null);                     // locks change the critic's actions: ask again
});

test("an unreliable emotion reading is shown but does not change the settings", () => {
  const opts = { genres: ["pop", "jazz"], emotions: ["happy", "sad"] };
  let s = reduce(connected(), { type: "optionsLoaded", options: opts });
  s = reduce(s, { type: "fileSelected", file: { name: "a.mid", size: 1 } });
  s = reduce(s, { type: "analyzeSuccess", uploadId: "u1", analysis: { ...ANALYSIS,
    style: { source: "classifier", reliable: false, emotion: [{ label: "sad", score: 0.3 }], genre: [] } } });
  assert.equal(s.form.emotion, "happy");                 // the default stays
  assert.equal(s.analysis.style.emotion[0].label, "sad"); // but the reading is still available to show
});

// ------------------------------------------------------------------ tab workflow (grouped parts, prompt steps)
import { normalizeRoles, setGroupRole, partName } from "../choices.js";

test("parts: one role per group, in group order; full composition expands to the core parts", () => {
  assert.deepEqual(normalizeRoles(["drums", "chord_bed", "pad", "bass_line"]), ["chord_bed", "bass_line", "drums"]);
  assert.deepEqual(normalizeRoles(["full_composition"]), ["lead_melody", "chord_bed", "bass_line", "drums"]);
  assert.deepEqual(normalizeRoles([]), []);
  assert.equal(partName("chord_bed"), "Block chords (chord bed)");
});

test("parts: choosing in a group replaces that group's part; Skip removes it", () => {
  let roles = ["chord_bed", "bass_line", "drums"];
  roles = setGroupRole(roles, "chords", "arpeggio");
  assert.deepEqual(roles, ["arpeggio", "bass_line", "drums"]);
  roles = setGroupRole(roles, "drums", null);
  assert.deepEqual(roles, ["arpeggio", "bass_line"]);
  roles = setGroupRole(roles, "melody", "lead_melody");
  assert.deepEqual(roles, ["lead_melody", "arpeggio", "bass_line"]);
  assert.deepEqual(setGroupRole(roles, "bass", "drums"), ["lead_melody", "arpeggio"]);   // a role from another group is ignored
});

test("setGroupRole action updates the form", () => {
  const s = reduce(initialState(), { type: "setGroupRole", group: "bass", role: "walking_bass" });
  assert.deepEqual(s.form.generatedRoles, ["chord_bed", "walking_bass", "drums"]);
});

test("prompt steps: interpreting moves to step 2 with one part per group; Edit prompt goes back", () => {
  let s = reduce(initialState(), { type: "setMode", mode: "prompt" });
  assert.equal(s.promptStep, 1);
  assert.equal(reduce(s, { type: "setPromptStep", step: 2 }).promptStep, 1);     // nothing interpreted yet
  s = reduce(s, { type: "setPrompt", value: "full band jazz" });
  s = reduce(s, { type: "interpretSuccess", prompt: "full band jazz",
    plan: { settings: { generated_roles: ["full_composition", "pad"] }, explanation: [], unmatched: [], source: "keywords" } });
  assert.equal(s.promptStep, 2);
  assert.deepEqual(s.form.generatedRoles, ["lead_melody", "chord_bed", "bass_line", "drums"]);
  s = reduce(s, { type: "setPromptStep", step: 1 });
  assert.equal(s.promptStep, 1);
  assert.equal(reduce(s, { type: "setPromptStep", step: 2 }).promptStep, 2);
});

test("major/minor from a prompt is not sent from the Settings or MIDI tabs", () => {
  let s = initialState();
  s = reduce(s, { type: "setField", field: "scaleMode", value: "minor" });
  s = reduce(s, { type: "setMode", mode: "prompt" });
  assert.equal(buildGenerateBody(s).scale_mode, "minor");
  s = reduce(s, { type: "setMode", mode: "metadata" });
  assert.equal(buildGenerateBody(s).scale_mode, null);
  s = reduce(s, { type: "setMode", mode: "midi" });
  assert.equal(buildGenerateBody(s).scale_mode, null);
});

import { promptHighlights, promptSegments } from "../choices.js";

test("prompt highlighting finds the words behind each setting", () => {
  const p = "A sad slow jazz piece with piano in D minor, 16 bars at 80 bpm";
  const h = promptHighlights(p, { genre: "jazz", emotion: "sad", key: "D", scale_mode: "minor", tempo: 80, bars: 16,
    instrumentation: "piano" }, ["genre 'jazz' from \"jazz\""]);
  assert.deepEqual(h.map((x) => [p.slice(x.start, x.end), x.cat]),
    [["sad", "mood"], ["jazz", "genre"], ["piano", "instruments"], ["D minor", "key"], ["16 bars", "length"], ["80 bpm", "tempo"]]);
  assert.equal(promptSegments(p, h).map((s) => s.text).join(""), p);
});

test("prompt highlighting: genre from its source word; a lone article is not a key", () => {
  const p = "A cheerful bouncy bluegrass hoedown in A, 140bpm";
  const h = promptHighlights(p, { genre: "country", key: "A", tempo: 140 }, ["genre 'country' from \"bluegrass\""]);
  assert.deepEqual(h.map((x) => [p.slice(x.start, x.end), x.cat]), [["bluegrass", "genre"], ["A", "key"], ["140bpm", "tempo"]]);
  assert.equal(h.find((x) => x.cat === "key").start, p.indexOf("in A") + 3);
});
