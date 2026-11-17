// Piano-roll rendering. Playback lives in audio.js.

const PALETTE = ["#2f6fde", "#e8590c", "#2b8a3e", "#ae3ec9", "#c92a2a", "#1098ad", "#f08c00", "#5c7cfa", "#d6336c",
  "#37b24d", "#845ef7", "#868e96"];

export const trackColor = (i) => PALETTE[i % PALETTE.length];

export function drawRoll(canvas, candidate, opts) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 300;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  if (!candidate) return;
  const css = getComputedStyle(document.documentElement);
  const pitched = candidate.tracks.filter((t) => !t.is_drum).flatMap((t) => t.notes.map((n) => n[2]));
  const lo = Math.min(...pitched, 48) - 2;
  const hi = Math.max(...pitched, 72) + 2;
  const drumBand = candidate.tracks.some((t) => t.is_drum) ? 34 : 0;
  const total = opts.totalBeats || Math.max(4, ...candidate.tracks.flatMap((t) => t.notes.map((n) => n[0] + n[1])));
  const x = (b) => (b / total) * w;
  const y = (p) => (h - drumBand) - ((p - lo) / (hi - lo)) * (h - drumBand);
  // Bar lines and section boundaries.
  g.strokeStyle = css.getPropertyValue("--grid") || "#ddd";
  g.lineWidth = 1;
  const barBeats = opts.barBeats || 4;
  for (let b = 0; b <= total; b += barBeats) {
    g.beginPath(); g.moveTo(x(b), 0); g.lineTo(x(b), h); g.stroke();
  }
  g.strokeStyle = css.getPropertyValue("--muted") || "#888";
  for (const s of opts.sections || []) {
    const sx = x(s.start_bar * barBeats);
    g.beginPath(); g.moveTo(sx, 0); g.lineTo(sx, h); g.stroke();
    g.fillStyle = g.strokeStyle; g.font = "10px system-ui"; g.fillText(s.name || s.label || "", sx + 3, 11);
  }
  candidate.tracks.forEach((t, i) => {
    g.fillStyle = trackColor(i);
    g.globalAlpha = opts.selectedTrack === null || opts.selectedTrack === undefined || opts.selectedTrack === t.track_id ? 0.9 : 0.25;
    for (const [s, d, p] of t.notes) {
      if (t.is_drum) {
        g.fillRect(x(s), h - drumBand + ((p - 35) % 12) * 2.6, Math.max(1.5, x(d)), 2.4);
      } else {
        const hh = Math.max(2, (h - drumBand) / (hi - lo));
        g.fillRect(x(s), y(p) - hh / 2, Math.max(1.5, x(s + d) - x(s) - 0.5), hh);
      }
    }
    if (t.source === "input") {   // outline anchor tracks
      g.globalAlpha = 0.9; g.strokeStyle = trackColor(i);
    }
  });
  g.globalAlpha = 1;
}
