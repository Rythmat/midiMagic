// Piano-roll rendering. Playback lives in audio.js.

const PALETTE = ["#ff5c9a", "#8b6cff", "#2ee6c6", "#ffcf6b", "#5ab8ff", "#ff8f6b", "#b98cff", "#7cf29a", "#ff6bd6",
  "#4fd1ff", "#ffa94d", "#a3a0c8"];

export const trackColor = (i) => PALETTE[i % PALETTE.length];

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
const KEYS_W = 26;      // keyboard strip on the left

function roundRect(g, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  g.beginPath();
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

export function drawRoll(canvas, candidate, opts) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 300;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const cssVar = (name, dflt) => (css.getPropertyValue(name) || "").trim() || dflt;
  g.fillStyle = cssVar("--roll-bg", "#0f0c20");
  g.fillRect(0, 0, w, h);
  if (!candidate) return;

  const pitched = candidate.tracks.filter((t) => !t.is_drum).flatMap((t) => t.notes.map((n) => n[2]));
  const lo = Math.min(...pitched, 48) - 2;
  const hi = Math.max(...pitched, 72) + 2;
  const drumBand = candidate.tracks.some((t) => t.is_drum) ? 40 : 0;
  const rollH = h - drumBand;
  const total = opts.totalBeats || Math.max(4, ...candidate.tracks.flatMap((t) => t.notes.map((n) => n[0] + n[1])));
  const x = (b) => KEYS_W + (b / total) * (w - KEYS_W);
  const rowH = rollH / (hi - lo);
  const y = (p) => rollH - ((p - lo) / (hi - lo)) * rollH;

  // Pitch lanes: black-key rows darker, C rows marked, plus a small keyboard on the left.
  for (let p = Math.floor(lo); p <= hi; p++) {
    const top = y(p + 0.5);
    const black = BLACK_KEYS.has(((p % 12) + 12) % 12);
    if (black) { g.fillStyle = cssVar("--roll-black", "#0b0918"); g.fillRect(KEYS_W, top, w - KEYS_W, rowH); }
    g.fillStyle = black ? "#2a2545" : "#d9d4f2";
    g.fillRect(0, top + 0.5, black ? KEYS_W * 0.62 : KEYS_W - 2, Math.max(1, rowH - 1));
    if (p % 12 === 0 && rowH >= 5) {
      g.fillStyle = cssVar("--roll-label", "#6f68a0"); g.font = "9px Inter, system-ui";
      g.fillText(`C${p / 12 - 1}`, KEYS_W + 3, top + rowH - 1);
    }
  }

  // Bar lines (every bar faint, every 4 bars stronger) and bar numbers.
  const barBeats = opts.barBeats || 4;
  const gridCol = cssVar("--grid", "#241f42");
  let bar = 0;
  for (let b = 0; b <= total; b += barBeats, bar++) {
    g.strokeStyle = gridCol; g.globalAlpha = bar % 4 === 0 ? 1 : 0.55; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x(b) + 0.5, 0); g.lineTo(x(b) + 0.5, h); g.stroke();
  }
  g.globalAlpha = 1;

  // Section boundaries with labels.
  for (const s of opts.sections || []) {
    const sx = x(s.start_bar * barBeats);
    g.strokeStyle = "rgba(255, 92, 154, .55)"; g.setLineDash([4, 4]);
    g.beginPath(); g.moveTo(sx + 0.5, 0); g.lineTo(sx + 0.5, h); g.stroke(); g.setLineDash([]);
    const label = s.name || s.label || "";
    if (label) {
      g.font = "600 10px Inter, system-ui";
      const tw = g.measureText(label).width + 10;
      g.fillStyle = "rgba(255, 92, 154, .18)"; roundRect(g, sx + 3, 4, tw, 16, 8); g.fill();
      g.fillStyle = "#ffb3cf"; g.fillText(label, sx + 8, 15);
    }
  }

  // Drum lane.
  if (drumBand) {
    g.fillStyle = "rgba(255, 207, 107, .06)"; g.fillRect(KEYS_W, rollH, w - KEYS_W, drumBand);
    g.strokeStyle = gridCol; g.beginPath(); g.moveTo(0, rollH + 0.5); g.lineTo(w, rollH + 0.5); g.stroke();
    g.save(); g.fillStyle = cssVar("--roll-label", "#6f68a0"); g.font = "600 8px Inter, system-ui";
    g.translate(15, rollH + drumBand / 2 + 13); g.rotate(-Math.PI / 2); g.fillText("DRUMS", 0, 0); g.restore();
  }

  // Notes, with a soft glow; the selected track stays bright, the others fade.
  candidate.tracks.forEach((t, i) => {
    const col = trackColor(i);
    const dim = !(opts.selectedTrack === null || opts.selectedTrack === undefined || opts.selectedTrack === t.track_id);
    g.globalAlpha = dim ? 0.22 : 0.95;
    g.fillStyle = col;
    g.shadowColor = col; g.shadowBlur = dim ? 0 : 6;
    for (const [s, d, p] of t.notes) {
      if (t.is_drum) {
        const dy = rollH + 4 + ((((p - 35) % 12) + 12) % 12) * ((drumBand - 8) / 12);
        g.fillRect(x(s), dy, Math.max(2, x(s + Math.min(d, 0.25)) - x(s)), 2.6);
      } else {
        const nh = Math.max(2, rowH - 1);
        roundRect(g, x(s), y(p) - nh / 2, Math.max(2, x(s + d) - x(s) - 1), nh, Math.min(3, nh / 2));
        g.fill();
      }
    }
    g.shadowBlur = 0;
    if (t.source === "input" && !dim) {   // outline locked input tracks so they read as the anchor
      g.strokeStyle = "#ffffff"; g.globalAlpha = 0.5; g.lineWidth = 1;
      for (const [s, d, p] of t.notes) {
        if (t.is_drum) continue;
        const nh = Math.max(2, rowH - 1);
        roundRect(g, x(s) + 0.5, y(p) - nh / 2 + 0.5, Math.max(2, x(s + d) - x(s) - 2), nh - 1, Math.min(3, nh / 2));
        g.stroke();
      }
    }
  });
  g.globalAlpha = 1;
}
