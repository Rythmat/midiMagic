// Plain-language labels for the button groups (pure data + helpers; shared by app.js and the tests).

// "Parts to write": one choice per box. The musician's term is in parentheses.
export const PART_GROUPS = [
  { id: "melody", name: "Melody", what: "The tune you would hum", roles: [
    ["lead_melody", "Lead melody", "The main tune"],
    ["counterline", "Counter-melody (counterline)", "A second tune that answers the main one"],
  ] },
  { id: "chords", name: "Chords", what: "The harmony behind the tune", roles: [
    ["chord_bed", "Block chords (chord bed)", "Chords played together on the beat"],
    ["arpeggio", "Arpeggio", "Chords played one note at a time"],
    ["pad", "Pad", "Long, soft chords in the background"],
    ["harmony", "Harmony voice", "A second voice that follows the tune"],
  ] },
  { id: "bass", name: "Bass", what: "The low notes that hold it together", roles: [
    ["bass_line", "Bass line", "Steady bass notes on the chords"],
    ["walking_bass", "Walking bass", "A moving bass line, jazz style"],
  ] },
  { id: "drums", name: "Drums", what: "The beat", roles: [
    ["drums", "Drum kit", "Kick, snare and hi-hats"],
  ] },
  { id: "extras", name: "Extras", what: "Optional colour on top", roles: [
    ["high_texture", "High sparkle (high texture)", "Light, high notes for shine"],
    ["low_texture", "Low rumble (low texture)", "Deep held notes for weight"],
    ["octave_doubles", "Octave double (octave doubles)", "The tune copied an octave up or down"],
  ] },
];

// What "full composition" means in the grouped picker.
export const FULL_COMPOSITION = ["lead_melody", "chord_bed", "bass_line", "drums"];

export const groupOfRole = (role) => (PART_GROUPS.find((g) => g.roles.some(([r]) => r === role)) || {}).id || null;

export const partName = (role) => {
  for (const g of PART_GROUPS) for (const [r, name] of g.roles) if (r === role) return name;
  return role.replace(/_/g, " ");
};

// At most one role per group, in group order; "full_composition" expands to the four core parts.
export function normalizeRoles(roles) {
  const list = (roles || []).includes("full_composition")
    ? [...FULL_COMPOSITION, ...roles.filter((r) => r !== "full_composition")] : [...(roles || [])];
  const picked = {};
  for (const r of list) {
    const g = groupOfRole(r);
    if (g && !(g in picked)) picked[g] = r;
  }
  return PART_GROUPS.map((g) => picked[g.id]).filter(Boolean);
}

// Replace whatever this group had with `role` (null = skip the group).
export function setGroupRole(roles, groupId, role) {
  const group = PART_GROUPS.find((g) => g.id === groupId);
  if (!group) return normalizeRoles(roles);
  const inGroup = new Set(group.roles.map(([r]) => r));
  const rest = normalizeRoles(roles).filter((r) => !inGroup.has(r));
  return normalizeRoles(role && inGroup.has(role) ? [...rest, role] : rest);
}

// "Your file is the …" (MIDI tab).
export const INPUT_ROLE_LABELS = {
  lead: "Melody (lead)", bass: "Bass line", chordal: "Chords (chordal)", highs: "High part (highs)",
  lows: "Low part (lows)", full_mix: "Full song (full mix)",
};

// Hover tooltips for the song forms.
export const FORM_TIPS = {
  "auto": "Let the app pick a form that suits the genre.",
  "verse-chorus": "Verses alternate with a repeated chorus, as in most pop songs.",
  "AABA": "One section played three times with a contrasting bridge (B) before the last; the classic jazz-standard form.",
  "ABA": "A section, a contrasting middle, then the first section again (ternary form).",
  "AABB": "Two sections, each played twice (binary form), common in folk dances.",
  "rondo": "A main theme that keeps coming back between different episodes (ABACA).",
  "strophic": "The same music repeated for every verse, like a hymn or folk song.",
  "through-composed": "New music all the way through, with no repeated sections.",
  "12-bar-blues": "The 12-bar blues chord pattern, played round and round.",
  "head-solo-head": "Jazz form: play the tune (the head), solos over its chords, then the tune again.",
  "build-drop": "Dance-music form: the energy builds up, then the full beat drops in.",
};
