// In-browser playback of a candidate with real General MIDI instrument samples.
//
// Pitched tracks use soundfont-player (loaded from a CDN in index.html) with the FluidR3 GM soundfont, so each track
// sounds like the instrument its MIDI program names. Drums are synthesized (that soundfont has no drum kit).
// If the samples can't be loaded (offline, CDN blocked) playback falls back to a simple oscillator synth.

// General MIDI program number -> soundfont instrument name (MIDI.js naming).
export const GM_INSTRUMENTS = [
  "acoustic_grand_piano", "bright_acoustic_piano", "electric_grand_piano", "honkytonk_piano", "electric_piano_1",
  "electric_piano_2", "harpsichord", "clavinet", "celesta", "glockenspiel", "music_box", "vibraphone", "marimba",
  "xylophone", "tubular_bells", "dulcimer", "drawbar_organ", "percussive_organ", "rock_organ", "church_organ",
  "reed_organ", "accordion", "harmonica", "tango_accordion", "acoustic_guitar_nylon", "acoustic_guitar_steel",
  "electric_guitar_jazz", "electric_guitar_clean", "electric_guitar_muted", "overdriven_guitar", "distortion_guitar",
  "guitar_harmonics", "acoustic_bass", "electric_bass_finger", "electric_bass_pick", "fretless_bass", "slap_bass_1",
  "slap_bass_2", "synth_bass_1", "synth_bass_2", "violin", "viola", "cello", "contrabass", "tremolo_strings",
  "pizzicato_strings", "orchestral_harp", "timpani", "string_ensemble_1", "string_ensemble_2", "synth_strings_1",
  "synth_strings_2", "choir_aahs", "voice_oohs", "synth_choir", "orchestra_hit", "trumpet", "trombone", "tuba",
  "muted_trumpet", "french_horn", "brass_section", "synth_brass_1", "synth_brass_2", "soprano_sax", "alto_sax",
  "tenor_sax", "baritone_sax", "oboe", "english_horn", "bassoon", "clarinet", "piccolo", "flute", "recorder",
  "pan_flute", "blown_bottle", "shakuhachi", "whistle", "ocarina", "lead_1_square", "lead_2_sawtooth",
  "lead_3_calliope", "lead_4_chiff", "lead_5_charang", "lead_6_voice", "lead_7_fifths", "lead_8_bass__lead",
  "pad_1_new_age", "pad_2_warm", "pad_3_polysynth", "pad_4_choir", "pad_5_bowed", "pad_6_metallic", "pad_7_halo",
  "pad_8_sweep", "fx_1_rain", "fx_2_soundtrack", "fx_3_crystal", "fx_4_atmosphere", "fx_5_brightness",
  "fx_6_goblins", "fx_7_echoes", "fx_8_scifi", "sitar", "banjo", "shamisen", "koto", "kalimba", "bagpipe", "fiddle",
  "shanai", "tinkle_bell", "agogo", "steel_drums", "woodblock", "taiko_drum", "melodic_tom", "synth_drum",
  "reverse_cymbal", "guitar_fret_noise", "breath_noise", "seashore", "bird_tweet", "telephone_ring", "helicopter",
  "applause", "gunshot",
];
const SOUNDFONT = "FluidR3_GM";

let ctx = null;
let out = null;                    // master gain every sound goes through
let noise = null;                  // shared white-noise buffer for drums
const instruments = new Map();     // instrument name -> Promise<player>
let voices = [];                   // oscillator/buffer sources of the current playback
let players = [];                  // soundfont players used by the current playback
let token = 0;                     // bumps on every play/stop, so a slow sample load can't start stale playback
let endTimer = null;

function audio() {
  if (!ctx) {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    out = ctx.createGain();
    out.connect(ctx.destination);
  }
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

const samplesAvailable = () => typeof window !== "undefined" && !!window.Soundfont;

function loadInstrument(program) {
  const name = GM_INSTRUMENTS[program] || GM_INSTRUMENTS[0];
  if (!instruments.has(name)) {
    const p = window.Soundfont.instrument(audio(), name, { soundfont: SOUNDFONT, format: "mp3", destination: out });
    p.catch(() => instruments.delete(name));          // let a later play retry
    instruments.set(name, p);
  }
  return instruments.get(name);
}

export function stopPlayback() {
  token += 1;
  clearTimeout(endTimer);
  for (const v of voices) { try { v.stop(); } catch (_) { /* already stopped */ } }
  for (const p of players) { try { p.stop(); } catch (_) { /* nothing scheduled */ } }
  voices = [];
  players = [];
}

/**
 * Play a candidate ({tracks: [{program, is_drum, notes: [[startBeat, durBeats, pitch, velocity]]}]}).
 * onStatus(text) reports loading/playing/finished; resolves once playback has been scheduled.
 */
export async function play(candidate, bpm, { mutedIds = new Set(), onStatus = () => {} } = {}) {
  stopPlayback();
  const mine = token;
  const ac = audio();
  const tracks = candidate.tracks.filter((t) => !mutedIds.has(t.track_id) && t.notes.length);
  const byProgram = new Map();
  let sampled = false;
  if (samplesAvailable()) {
    onStatus("Loading instrument sounds…");
    const programs = [...new Set(tracks.filter((t) => !t.is_drum).map((t) => t.program))];
    try {
      const loaded = await Promise.all(programs.map(loadInstrument));
      programs.forEach((p, i) => byProgram.set(p, loaded[i]));
      sampled = true;
    } catch (_) {
      onStatus("Couldn't load instrument sounds, using the simple synth.");
    }
  }
  if (mine !== token) return;                         // stopped (or restarted) while samples were loading
  out.gain.value = sampled ? 0.7 : 0.18;
  players = [...byProgram.values()];

  const spb = 60 / (bpm || 120);
  const t0 = ac.currentTime + 0.15;
  let end = t0;
  for (const t of tracks) {
    for (const [s, d, p, v] of t.notes) {
      const start = t0 + s * spb;
      const dur = Math.min(8, Math.max(0.05, d * spb));
      end = Math.max(end, start + dur);
      if (t.is_drum) drumHit(ac, p, v, start);
      else if (sampled) byProgram.get(t.program).play(p, start, { duration: dur, gain: 0.25 + (v / 127) * 0.75 });
      else synthNote(ac, p, v, start, dur);
    }
  }
  onStatus(sampled ? "Playing (sampled instruments)" : "Playing (simple synth)");
  endTimer = setTimeout(() => { if (mine === token) onStatus(""); }, (end - ac.currentTime + 0.5) * 1000);
}

function envelope(ac, start, peak, decay) {
  const g = ac.createGain();
  g.gain.setValueAtTime(0, start);
  g.gain.linearRampToValueAtTime(peak, start + 0.005);
  g.gain.exponentialRampToValueAtTime(0.0008, start + decay);
  g.connect(out);
  return g;
}

function synthNote(ac, pitch, vel, start, dur) {
  const osc = ac.createOscillator();
  osc.type = "triangle";
  osc.frequency.value = 440 * Math.pow(2, (pitch - 69) / 12);
  osc.connect(envelope(ac, start, (vel / 127) * 0.35, dur));
  osc.start(start);
  osc.stop(start + dur + 0.05);
  voices.push(osc);
}

function noiseSource(ac) {
  if (!noise) {
    noise = ac.createBuffer(1, ac.sampleRate, ac.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  }
  const src = ac.createBufferSource();
  src.buffer = noise;
  return src;
}

// General MIDI drum map, roughly: kicks, snares/claps, hats and cymbals, everything else as toms/percussion.
function drumHit(ac, note, vel, start) {
  const amp = (vel / 127) * (out.gain.value > 0.5 ? 1.0 : 2.5);
  if (note === 35 || note === 36) {                                  // kick: falling sine
    const osc = ac.createOscillator();
    osc.frequency.setValueAtTime(140, start);
    osc.frequency.exponentialRampToValueAtTime(42, start + 0.12);
    osc.connect(envelope(ac, start, amp * 0.9, 0.25));
    osc.start(start); osc.stop(start + 0.3);
    voices.push(osc);
    return;
  }
  const hat = [42, 44, 46].includes(note);
  const cymbal = [49, 51, 52, 53, 55, 57, 59].includes(note);
  const snare = [37, 38, 39, 40].includes(note);
  if (hat || cymbal || snare) {                                      // filtered noise
    const src = noiseSource(ac);
    const filt = ac.createBiquadFilter();
    filt.type = "highpass";
    filt.frequency.value = snare ? 1200 : 7000;
    const decay = note === 42 || note === 44 ? 0.05 : snare ? 0.16 : 0.45;
    src.connect(filt);
    filt.connect(envelope(ac, start, amp * (snare ? 0.6 : 0.25), decay));
    src.start(start); src.stop(start + decay + 0.05);
    voices.push(src);
    if (snare) {                                                     // body of the snare
      const body = ac.createOscillator();
      body.frequency.value = 190;
      body.connect(envelope(ac, start, amp * 0.3, 0.08));
      body.start(start); body.stop(start + 0.1);
      voices.push(body);
    }
    return;
  }
  const osc = ac.createOscillator();                                // toms and other percussion
  const f = 80 + (note - 41) * 18;
  osc.frequency.setValueAtTime(Math.max(60, f), start);
  osc.frequency.exponentialRampToValueAtTime(Math.max(40, f * 0.6), start + 0.18);
  osc.connect(envelope(ac, start, amp * 0.5, 0.22));
  osc.start(start); osc.stop(start + 0.25);
  voices.push(osc);
}
