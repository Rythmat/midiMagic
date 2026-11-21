# Musical review of `examples/review/`

Bars are 1-based; "b12.3" is bar 12, beat 3. Regenerate the 10 files with the settings below, e.g.:

```
python -m midicomposer.cli generate --seed 0 --genre rock --emotion angry --key E --mode minor --meter 4/4 --engine hybrid --out examples/review/02.mid
python -m midicomposer.cli generate --seed 0 --input evaluation/data/melodies/amazing_grace.mid --input-role lead --roles full_composition --target-key F --engine rule --out examples/review/10.mid
```

| # | genre | emotion | key/mode | meter | engine |
|---|---|---|---|---|---|
| 01 | pop | happy | C major | 4/4 | rule |
| 02 | rock | angry | E minor | 4/4 | hybrid |
| 03 | jazz | lazy | Bb major | 4/4 | rule |
| 04 | classical | sad | A minor | 3/4 | rule |
| 05 | country | warm | G major | 4/4 | hybrid |
| 06 | traditional | magnificent | D major | 4/4 | rule |
| 07 | pop | fear | F# minor | 4/4 | rule |
| 08 | jazz | funny | F major | 4/4 | hybrid |
| 09 | rock | exciting | A major | 4/4 | neural |
| 10 | from MIDI (pop/happy) | Amazing Grace lead, G→F | F major | 3/4 | rule |

The #10 lead passes the check: all 32 notes keep their onset, duration and velocity and move by exactly −2 semitones.

Coverage limits: in 02, 05 and 08 the top-ranked hybrid candidate used the rule engine for every role, so this is
9 rule pieces and 1 neural piece. Quiet and romantic are not covered, each genre is paired with one emotion (genre
and mood effects can't be separated), and there is one seed per setting.

## Issues

| # | Bars | Likely audible problem (from the notes) | Category | Severity |
|---|---|---|---|---|
| 01 | 12, 20, 28, 36, 44, 52 (b.2–3) | The cadence note C5 is held for 2–3 beats over the G chord, a semitone above the strings' B4 and the piano's B3 | harmony clash | jarring |
| 02 | 12, 28, 44 (b.2–3) | The lead's tonic (E4 or E5) is held over B major, against D#4 in the guitar and D#5 in the strings | harmony clash | jarring |
| 06 | 24.2 | The lead's D5 is held over the A chord, against C#5 in the strings and C#4 in the koto | harmony clash | jarring |
| 07 | 12, 20, 28, 36, 44, 52 (b.2–3) | The lead's F#4 is held directly above the piano's E# (F4) on C# major; in bars 28 and 36 it is F#5 against the strings' F5 | harmony clash | jarring |
| 02, 08 | throughout | The string pad's top voice is above the lead for 179/246 (02) and 125/136 (08) lead notes, by a median of 9 and 11 semitones (e.g. 02 b13.1: lead G4, pad top G5), so the pad, not the melody, is the top line | register/voicing | noticeable |
| 01, 03–07 | throughout | The pad's top voice is above the lead for 86/230 (01), 60/136 (03), 25/80 (04), 90/145 (05), 67/140 (06) and 130/246 (07) lead notes; the lead is ~26 velocity louder, so the melody is not the top voice but stays audible | register/voicing | nitpick |
| 01 | about 40 bar ends | The bass's short, softer chromatic approach on 4& rubs a 7th/9th against the chord it leads into; worst case b3.4.5: F#3 against pad F3 and piano F4 | harmony clash | nitpick |
| 07 | 1, 5, 9, 13 … (4&) | The bass's C3 approach into B2 sits a minor 9th under C#4 in the piano and strings (b1.4.5, b5.4.5 also against lead C#5) | harmony clash | nitpick |
| 06 | 2, 3, 6, 7, 10 … (4&) | The bass's approach notes A#2 and D#3 sound against A3 and E3 in the koto and strings (23 rubs) | harmony clash | nitpick |
| 03 | 8, 16, 23, 24, 27, 31, 32 | The lead's Bb4 (the root) sits a minor 9th above the piano's A3 (the maj7) on Bbmaj7; the piano is quiet and overlaps briefly | harmony clash | nitpick |
| 08 | 3, 8, 11, 15, 16, 23, 24, 28, 31, 32 | The lead holds F4 (the root) a semitone above the piano's E4 (the maj7) on Fmaj7 | harmony clash | noticeable |
| 08 | 32.4 (last note) | The walking bass ends on a chromatic "approach" F#3 that leads nowhere, under the pad's F3 and the piano's Fmaj7 | harmony clash | jarring |
| 08 | all | The walking bass stays in G2–G3, so on the Fmaj7 downbeats it plays F3 in unison with the pad's and piano's F3. There is no low end | register/voicing | nitpick |
| 10 | all, e.g. b2.3.5, b6.1.5 | The chord bed (E3–D4), the pad (E3–D4) and the counterline (C3–D4) share one octave. The tenor sax doubles them 87 times, rubs a 2nd against them (Bb3 against A3 at b2.3.5, b10.3.5, b11.3.5 and b14.3.5) and drops under the bass (C3 under D3 at b6 and b14) | register/voicing | noticeable |
| 10 | 2, 3, 5, 6, 10, 11, 13 (b.3.5) | The bass's approach notes C#3 and D#3 rub against the hymn's chords, e.g. b3.3.5: C#3 against the pad's D4 | harmony clash | nitpick |
| 04 | 1→2, 4→5, 5→6, 19→20, 20→21, 21→22 | The violin and cello move in parallel octaves on the downbeats (A2/A4 → F2/F4), which sounds hollow in a two-voice texture | harmony clash | nitpick |
| 09 | 2, 3, 6, 10–20, 26, 28, 33–40 … | The bass's downbeat note misses the planned chord root in 40 of the 55 bars where it plays, and the parts don't agree. b3: C4-E4-G#4 over bass A2 in A major. b13: D2 under A3-C#4. b44.3: lead C#5 over C4 in the guitar | harmony clash | jarring |
| 09 | 5–8, 23–24, 33–40, 47–50, 52–56 | The lead is missing for 4 verse bars, the first half of the solo (37–40) and most of the final chorus and outro. Where it does play, it alternates 16th-note runs (17–21, 41–45) with near-empty bars | form/transitions | jarring |
| 09 | changes at 9, 17, 25, 33, 41, 49 | The groove and dynamics change every 8 bars starting at bar 1, not at the section starts (5, 13, 21 …). Bar 8 has no drums. Bars 9–16 have side-stick only, through the start of the chorus. Velocities jump 104→72→120→88, with a drop at b49 in the middle of the final chorus | form/transitions | jarring |
| 09 | 17–24 | The kick plays almost every 8th note with no hats or cymbals, and in bars 41–48 the pedal hat keeps time. It doesn't sound like a rock beat | drums | noticeable |
| 09 | 52–56 | There is no ending: the lead stops at bar 51, and the piece stops mid-figure (piano D3 in 16ths, bass E2–F#2–A2) | form/transitions | noticeable |
| 01, 02, 05, 07 | last bar (56, 56, 40, 56) | A drum fill on beat 4 of the final bar (e.g. 01: snare at 3.5 and 3.75) leads into silence, and the comp's last hit is a 0.4-beat stab on 4&. The song stops instead of ending | form/transitions | noticeable |
| 01, 05, 07 (02 mostly) | verse→chorus (13, 29, 45) | Apart from its progression (01, 05), the chorus differs from the verse only in velocity (+15–20). Note counts, comp, bass and drum rhythms are identical from intro to outro (01: 12 chord notes and 13 drum hits in every bar). There is no lift in arrangement | form/transitions | noticeable |
| 03, 08 | 17 (B section) | The B section of AABA repeats ii7–V7–Imaj7–Imaj7 unchanged, so all 32 bars are one 4-bar loop | form/transitions | noticeable |
| 04 | 9 (B) | The A and B sections share i–VI–iv–V, so the same 4-bar loop runs 6 times | form/transitions | noticeable |
| 07, 01 | 13 (07 chorus), 37 (01 bridge) | 07: the verse and chorus share i–iv–VI–V. 01: the bridge reuses the chorus's I–IV–vi–V. There is no harmonic contrast | form/transitions | nitpick |
| 05 | 3–4, 6, 8, 10–11, 14–15, 18–19, 22, 26, 29–30, 33, 35, 37 | The timekeeper switches randomly between hi-hat and maracas from bar to bar inside a section | drums | noticeable |
| 06 | all 32 | Hand percussion only (toms on 1 and 3, shaker on the offbeats) in one pattern for all 32 bars, with no fills and no build into B (bar 17) | drums | noticeable |
| 03, 08 | 03: 2, 3, 8, 9, 10, 15, 24–27; 08: about 9 bars | The comping snare lands off the triplet grid (beat + 1/6, e.g. 1.17 and 3.17), which sounds like a flam against the ride | drums | nitpick |
| 02, 09 | all | The rock band is a trumpet lead, a steel-string acoustic guitar comp, an upright acoustic bass and a string pad. There is no electric or distorted guitar and no electric bass | instrument choice | noticeable |
| 07 | all | "Fear" has the same arrangement as 01 "happy": flute, piano and strings, 115 vs 117 BPM, the same pop beat with crashes and fills, the same pulse-8 comp. Only the mode differs | mood mismatch | noticeable |
| 03 | all | "Lazy" is played at 110 BPM, a medium swing, with a steady ride and crash accents | mood mismatch | noticeable |
| 06 | all | "Magnificent" is a flute lead with block koto chords twice a bar, shaker and toms, with no low brass, timpani or dynamic growth (velocity +6 from A to B) | mood mismatch | noticeable |
| 03, 08 | all | A sustained string-ensemble pad (whole notes or 2-bar notes) is used in a jazz combo | instrument choice | nitpick |

## Top issues

- **Neural engine output has no shape (#09, the one neural sample).** It ignores section boundaries and the planned
  chords: grooves, dynamics and lead presence change in 8-bar blocks offset 4 bars from the form, the lead is missing
  for 23 bars, and the bass's downbeat misses the planned root in 40 of 55 bars. Cause:
  `generation/neural.py::NeuralTrackGenerator.generate`. Its windows `range(0, n_bars, self.window)` start at bar 0,
  not at sections; the section token comes from each window's first bar; the chord scaffold only conditions.
- **The tonic is held over the dominant at phrase ends, a semitone against the leading tone.** Found in 01 (6×),
  07 (6×), 02 (3×) and 06 (1×), starting on beat 2 or 3 and held 2–3 beats. Cause:
  `generation/rule_based.py::gen_lead_melody` forces the cadence note to `pcs=[0]` without checking the chord, while
  `planner.py::progression_to_chords` (half cadence) and the progressions put V in that bar.
- **The string pad is often the top voice instead of the lead.** Its top voice is above the melody for 31–92% of
  lead notes in 8 of 10 files; in 02 and 08 (73%, 92%, by 9–11 semitones) the pad becomes the top line. In #10,
  chord bed, pad and counterline share one octave. Cause: `planner.py::ROLE_REGISTERS` (pad 48–79 overlaps lead
  62–84) with `rule_based.py::gen_pad`'s open voicing centred in that window; for #10,
  `planner.py::relative_registers`, whose anchor-avoid branch gives counterline, chord_bed and pad the same window.
- **08 ends on a wrong bass note.** The walking bass's last note is F#3 under the pad's F3 and the Fmaj7, loud and
  unresolved. Cause: `rule_based.py::gen_walking_bass` approaches the next chord's root even on the last span, where
  there is no next chord.
- **The form and energy are flat.** Some contrasting sections reuse the same progression (03, 04, 07, 08; 01's
  bridge). The chorus keeps the verse's rhythms and differs mainly in loudness (01, 02, 05, 07). Endings are a fill
  into silence (01, 02, 05, 07). Causes: `planner.py::plan_from_metadata` calls `choose_progression` with replacement
  for each section name; the per-candidate patterns (`ctx.pick`) in `rule_based.py::_comp_offsets`, `gen_bass_line`
  and `gen_drums` ignore section energy; `gen_drums` fills the last bar of every section, the final one included,
  with no closing hit.
- **Separately, fear, lazy and magnificent are arranged like the genre default** (07, 03, 06): the same instruments and
  groove as other moods, with only small mode/tempo/loudness changes (07 fear at 115 BPM vs 01 happy at 117).
