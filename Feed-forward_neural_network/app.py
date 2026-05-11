"""
FastAPI + React (CDN) web server.
Picks raw screw recordings, runs them through the full preprocessing +
tsfresh feature-extraction pipeline, then predicts with the trained FNN.

Run:  uvicorn app:app --reload --port 5000
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

CLASS_LABELS = {0: "N", 1: "NS", 2: "OT", 3: "P", 4: "UT"}

CLASS_META = {
    "N":  {"full": "Normal",          "color": "#22c55e"},
    "NS": {"full": "Not Screwed",     "color": "#6b7280"},
    "OT": {"full": "Over Tightened",  "color": "#ef4444"},
    "P":  {"full": "Partial",         "color": "#f59e0b"},
    "UT": {"full": "Under Tightened", "color": "#3b82f6"},
}

MODEL_PATH    = r"C:\github\VT2\Feed-forward_neural_network\trained_model.keras"
FEATURES_PATH = r"C:\github\VT2\Feature_engineering\features_selected.csv"

# ── Lazy singletons ───────────────────────────────────────────────────────────

_model           = None
_df              = None
_scaler          = None
_raw_pairs       = None
_task_kind_fc    = None
_intr_kind_fc    = None
_selected_cols   = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def get_df_and_scaler():
    global _df, _scaler
    if _df is None:
        _df = pd.read_csv(FEATURES_PATH, index_col=0)
        _scaler = StandardScaler().fit(_df.values)
    return _df, _scaler


def get_inference_assets():
    global _raw_pairs, _task_kind_fc, _intr_kind_fc, _selected_cols
    if _raw_pairs is None:
        import inference_pipeline as ip
        _raw_pairs = ip.list_test_pairs()
        df, _ = get_df_and_scaler()
        _selected_cols = list(df.columns)
        _task_kind_fc, _intr_kind_fc = ip.build_kind_fc_parameters(_selected_cols)
        print(f">> inference assets ready: {len(_raw_pairs)} test-set pairs, "
              f"{len(_selected_cols)} features")
    return _raw_pairs, _task_kind_fc, _intr_kind_fc, _selected_cols


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Screw Fault Visualiser")


class ScrewResult(BaseModel):
    screw:       int
    label:       str
    full:        str
    color:       str
    confidence:  float
    true_label:  str
    true_full:   str
    correct:     bool
    base_id:     str


@app.on_event("startup")
async def startup():
    print(">> startup: loading model...")
    get_model()
    print(">> startup: model loaded. loading features CSV + scaler...")
    get_df_and_scaler()
    print(">> startup: scaler ready. Server is up — inference assets will load on first request.")


def _predict_one_instance(screw_number: int) -> ScrewResult:
    """Pick one random raw pair, run pipeline, return prediction."""
    import inference_pipeline as ip

    t0 = time.perf_counter()
    raw_pairs, task_kind_fc, intr_kind_fc, selected_cols = get_inference_assets()
    _, scaler = get_df_and_scaler()
    model = get_model()
    t_assets = time.perf_counter()
    print(f"[screw {screw_number}] assets ready:     {t_assets - t0:.3f}s")

    max_attempts = 16
    for attempt in range(max_attempts):
        label, base_id, task_path, intr_path = random.choice(raw_pairs)

        t_pipe0 = time.perf_counter()
        feat_series = ip.pipeline_one(
            task_path, intr_path, sample_id=0,
            task_kind_to_fc=task_kind_fc,
            intr_kind_to_fc=intr_kind_fc,
            selected_columns=selected_cols,
        )
        t_pipe1 = time.perf_counter()
        print(f"[screw {screw_number}] pipeline (attempt {attempt+1}): {t_pipe1 - t_pipe0:.3f}s  id={base_id}")

        if feat_series is None:
            print(f"[screw {screw_number}]   -> no plateau, retrying...")
            continue

        t_inf0 = time.perf_counter()
        X = scaler.transform(feat_series.values.reshape(1, -1))
        probs = model.predict(X, verbose=0)[0]
        t_inf1 = time.perf_counter()
        print(f"[screw {screw_number}] model inference:  {t_inf1 - t_inf0:.3f}s")
        print(f"[screw {screw_number}] TOTAL:            {t_inf1 - t0:.3f}s")

        cls = int(np.argmax(probs))
        pred_label = CLASS_LABELS[cls]

        return ScrewResult(
            screw      = screw_number,
            label      = pred_label,
            full       = CLASS_META[pred_label]["full"],
            color      = CLASS_META[pred_label]["color"],
            confidence = round(float(probs[cls]) * 100, 1),
            true_label = label,
            true_full  = CLASS_META[label]["full"],
            correct    = pred_label == label,
            base_id    = str(base_id),
        )

    raise HTTPException(status_code=500,
                        detail="Could not find a valid screw instance after 16 attempts")


@app.get("/api/predict_one", response_model=ScrewResult)
def predict_one(screw: int = 1):
    return _predict_one_instance(screw)


# ── React SPA (served inline, no build step) ──────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Screw Fault Visualiser</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    @keyframes screw-spin {
      from { transform: rotate(0deg); }
      to   { transform: rotate(360deg); }
    }
    @keyframes screw-pulse {
      0%, 100% { opacity: 0.55; }
      50%       { opacity: 1;    }
    }
  </style>
</head>
<body>
  <div id="root"></div>

  <script>
    const { useState, useCallback, useRef } = React;
    const R = React.createElement;

    const N = 8;
    const SVG_W = 560;

    /* Screw positions inside the 560x400 scene (2 per corner, 36x36 px each) */
    const POSITIONS = [
      { x: 10,  y:  8  },  // top-left  A
      { x: 10,  y: 46  },  // top-left  B
      { x: 514, y:  8  },  // top-right A
      { x: 514, y: 46  },  // top-right B
      { x: 10,  y: 316 },  // bot-left  A
      { x: 10,  y: 354 },  // bot-left  B
      { x: 514, y: 316 },  // bot-right A
      { x: 514, y: 354 },  // bot-right B
    ];

    const CLASS_META = {
      N:  { full: "Normal",          color: "#22c55e" },
      NS: { full: "Not Screwed",     color: "#6b7280" },
      OT: { full: "Over Tightened",  color: "#ef4444" },
      P:  { full: "Partial",         color: "#f59e0b" },
      UT: { full: "Under Tightened", color: "#3b82f6" },
    };

    function Screw({ x, y, status, result, showTrue, tooltipBelow }) {
      const [hovered, setHovered] = useState(false);
      const size = 36;
      const cx   = size / 2;
      const cy   = size / 2;

      const isEmpty   = status === "empty";
      const isWorking = status === "working";
      const isDone    = status === "done";

      const faceColor = isEmpty   ? "#334155"
                      : isWorking ? "#475569"
                      : result.color;

      const ttW  = 180;
      const ttX  = Math.max(4 - x, Math.min(SVG_W - 4 - x - ttW, cx - ttW / 2));
      const ttCx = ttX + ttW / 2;
      const ttTop = tooltipBelow ? size + 10 : -82;
      const lineY  = offset => ttTop + offset;

      return R("g", {
        transform: `translate(${x}, ${y})`,
        style: { cursor: isEmpty ? "default" : "pointer" },
        onMouseEnter: () => !isEmpty && setHovered(true),
        onMouseLeave: () => setHovered(false),
      },
        R("circle", { cx, cy, r: cx, fill: "#1e293b" }),
        R("circle", { cx, cy, r: cx - 3, fill: faceColor,
          style: isWorking ? { animation: "screw-pulse 1s ease-in-out infinite" } : {} }),
        R("circle", { cx: cx - 6, cy: cy - 6, r: 5, fill: "rgba(255,255,255,0.25)" }),
        R("g", {
          style: isWorking
            ? { transformOrigin: `${cx}px ${cy}px`, animation: "screw-spin 1s linear infinite" }
            : {}
        },
          R("rect", { x: cx - 9, y: cy - 2, width: 18, height: 4, rx: 1, fill: "rgba(0,0,0,0.45)" }),
          R("rect", { x: cx - 2, y: cy - 9, width: 4,  height: 18, rx: 1, fill: "rgba(0,0,0,0.45)" })
        ),
        hovered && isDone && result && R("g", null,
          R("rect", { x: ttX, y: lineY(0), width: ttW, height: showTrue ? 54 : 36,
            rx: 6, fill: "#0f172a", stroke: "#334155", strokeWidth: 1 }),
          R("text", { x: ttCx, y: lineY(14), textAnchor: "middle",
            fontSize: "11", fontWeight: "700", fill: result.color,
            fontFamily: "Segoe UI, system-ui, sans-serif" },
            `${result.full} (${result.confidence}%)`
          ),
          showTrue && R("text", { x: ttCx, y: lineY(30), textAnchor: "middle",
            fontSize: "10", fill: result.correct ? "#86efac" : "#fca5a5",
            fontFamily: "Segoe UI, system-ui, sans-serif" },
            `true: ${result.true_full} ${result.correct ? "[OK]" : "[WRONG]"}`
          ),
          showTrue && R("text", { x: ttCx, y: lineY(46), textAnchor: "middle",
            fontSize: "9", fill: "#64748b",
            fontFamily: "Segoe UI, system-ui, sans-serif" },
            `id: ${result.base_id}`
          )
        )
      );
    }

    function WindowScene({ slots, showTrue }) {
      const W = SVG_W, H = 400;
      return R("svg", { width: W, height: H, style: { display: "block", overflow: "visible" } },
        R("rect", { x: 0, y: 0, width: W, height: H, rx: 12,
          fill: "#374151", stroke: "#4b5563", strokeWidth: 3 }),
        R("rect", { x: 42, y: 42, width: W-84, height: H-84, rx: 4,
          fill: "rgba(147,197,253,0.12)", stroke: "rgba(148,163,184,0.3)", strokeWidth: 2 }),
        R("line", { x1: W/2, y1: 42, x2: W/2, y2: H-42,
          stroke: "rgba(148,163,184,0.2)", strokeWidth: 3 }),
        R("line", { x1: 42, y1: H/2, x2: W-42, y2: H/2,
          stroke: "rgba(148,163,184,0.2)", strokeWidth: 3 }),
        ...slots.map((slot, i) => R(Screw, {
          key: i,
          x: POSITIONS[i].x, y: POSITIONS[i].y,
          status: slot.status, result: slot.result,
          showTrue, tooltipBelow: i < 4,
        }))
      );
    }

    function Legend() {
      return R("div", {
        style: { display: "flex", flexWrap: "wrap", gap: 10,
          justifyContent: "center", marginBottom: 24 }
      },
        ...Object.entries(CLASS_META).map(([code, { full, color }]) =>
          R("div", { key: code, style: {
            display: "flex", alignItems: "center", gap: 8,
            background: "#1e293b", borderRadius: 8, padding: "6px 14px", fontSize: 13,
          }},
            R("div", { style: { width: 14, height: 14, borderRadius: "50%",
              background: color, flexShrink: 0 } }),
            R("span", null, R("strong", null, code), ` – ${full}`)
          )
        )
      );
    }

    function Summary({ slots, showTrue }) {
      const done = slots.filter(s => s.status === "done" && s.result);
      if (done.length === 0) return null;
      return R("div", {
        style: { display: "flex", flexWrap: "wrap", gap: 8,
          justifyContent: "center", marginTop: 20 }
      },
        ...done.map((s, i) => {
          const idx  = slots.indexOf(s);
          const text = showTrue
            ? (s.result.correct
                ? `#${idx+1} ${s.result.label} [OK]`
                : `#${idx+1} ${s.result.true_label} → ${s.result.label}`)
            : `#${idx+1} ${s.result.label}`;
          return R("span", { key: i, style: {
            background: s.result.color, color: "#fff",
            borderRadius: 99, padding: "4px 14px", fontSize: 12, fontWeight: 700,
          }}, text);
        })
      );
    }

    function Dashboard({ history }) {
      if (history.length === 0) return null;
      const all   = history.flat();
      const total = all.length;
      const counts = {};
      Object.keys(CLASS_META).forEach(k => { counts[k] = 0; });
      all.forEach(s => { counts[s.label] = (counts[s.label] || 0) + 1; });
      const BAR_W = 420;

      return R("div", {
        style: { background: "#1e293b", borderRadius: 12,
          padding: "20px 28px", marginTop: 28, width: SVG_W }
      },
        R("div", {
          style: { display: "flex", justifyContent: "space-between",
            alignItems: "center", marginBottom: 16 }
        },
          R("span", { style: { fontWeight: 700, fontSize: 15 } }, "Class Distribution"),
          R("span", { style: { color: "#94a3b8", fontSize: 12 } },
            `${history.length} round${history.length !== 1 ? "s" : ""} · ${total} screws`)
        ),
        ...Object.entries(CLASS_META).map(([code, { full, color }]) => {
          const count = counts[code] || 0;
          const pct   = total > 0 ? (count / total) * 100 : 0;
          return R("div", { key: code, style: { marginBottom: 12 } },
            R("div", { style: { display: "flex", justifyContent: "space-between",
              marginBottom: 4, fontSize: 12 } },
              R("span", { style: { fontWeight: 700, color } },
                code, R("span", { style: { color: "#94a3b8", fontWeight: 400 } }, ` – ${full}`)
              ),
              R("span", { style: { color: "#e2e8f0", fontWeight: 600 } },
                `${count} (${pct.toFixed(1)}%)`)
            ),
            R("div", { style: { background: "#0f172a", borderRadius: 99,
              height: 10, width: BAR_W, overflow: "hidden" } },
              R("div", { style: { height: "100%", width: (pct / 100) * BAR_W,
                background: color, borderRadius: 99, transition: "width 0.4s ease" } })
            )
          );
        })
      );
    }

    /* ── Root App ── */
    function App() {
      const emptySlots = () => Array.from({ length: N }, () => ({ status: "empty", result: null }));

      const [slots,    setSlots]    = useState(emptySlots());
      const [history,  setHistory]  = useState([]);
      const [running,  setRunning]  = useState(false);
      const [error,    setError]    = useState(null);
      const [showTrue, setShowTrue] = useState(false);
      const abortRef = useRef(false);

      const runPredictions = useCallback(async (freshSlots) => {
        setRunning(true);
        setError(null);
        abortRef.current = false;
        const roundResults = [];

        for (let i = 0; i < N; i++) {
          if (abortRef.current) break;

          // Mark this screw as working
          setSlots(prev => {
            const next = [...prev];
            next[i] = { status: "working", result: null };
            return next;
          });

          // Run pipeline and 4-second screw timer in parallel — result is
          // ready by the time the physical process finishes.
          const delay   = new Promise(r => setTimeout(r, 4000));
          const fetchFn = fetch(`/api/predict_one?screw=${i + 1}`)
            .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); });

          if (abortRef.current) break;

          try {
            const [, data] = await Promise.all([delay, fetchFn]);

            setSlots(prev => {
              const next = [...prev];
              next[i] = { status: "done", result: data };
              return next;
            });
            roundResults.push(data);
          } catch (e) {
            setError(e.message);
            setSlots(prev => {
              const next = [...prev];
              next[i] = { status: "empty", result: null };
              return next;
            });
            break;
          }
        }

        if (roundResults.length > 0) {
          setHistory(prev => [...prev, roundResults]);
        }
        setRunning(false);
      }, []);

      const handleStart = useCallback(() => {
        const fresh = emptySlots();
        setSlots(fresh);
        runPredictions(fresh);
      }, [runPredictions]);

      const handleReset = useCallback(() => {
        abortRef.current = true;
        setSlots(emptySlots());
        setHistory([]);
        setError(null);
        setRunning(false);
      }, []);

      const btnBase = {
        padding: "10px 28px", border: "none", borderRadius: 8,
        color: "#fff", fontSize: 14, fontWeight: 700,
        cursor: "pointer", transition: "background 0.15s",
      };

      return R("div", {
        style: { display: "flex", flexDirection: "column",
          alignItems: "center", padding: "40px 16px" }
      },
        R("h1", { style: { fontSize: "1.6rem", marginBottom: 4, letterSpacing: "0.05em" } },
          "Window Screw Fault Visualiser"),
        R("p", { style: { color: "#94a3b8", fontSize: 13, marginBottom: 28 } },
          "FNN · 5-class fault detection · full pipeline per screw"),
        R(Legend, null),
        R("div", { style: { borderRadius: 16, boxShadow: "0 20px 60px rgba(0,0,0,0.6)", overflow: "visible" } },
          R(WindowScene, { slots, showTrue })
        ),
        error && R("div", { style: { color: "#ef4444", fontSize: 13, marginTop: 12 } },
          `Error: ${error}`),
        R(Summary, { slots, showTrue }),
        R("div", { style: { display: "flex", gap: 12, marginTop: 28 } },
          R("button", {
            onClick: handleStart, disabled: running,
            style: { ...btnBase, background: running ? "#334155" : "#2563eb",
              cursor: running ? "not-allowed" : "pointer" }
          }, running ? "Running pipeline…" : "⟳ Resample & Predict"),
          R("button", {
            onClick: () => setShowTrue(v => !v),
            style: { ...btnBase, background: showTrue ? "#0f766e" : "#334155" }
          }, showTrue ? "Hide true label" : "Show true label"),
          R("button", {
            onClick: handleReset,
            style: { ...btnBase, background: "#7c3aed" }
          }, "✕ Reset")
        ),
        R(Dashboard, { history })
      );
    }

    ReactDOM.createRoot(document.getElementById("root")).render(R(App, null));
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


# ── Dev entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
