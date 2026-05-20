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

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

CLASS_LABELS = {0: "N", 1: "NS", 2: "OT", 3: "NE", 4: "UT"}

CLASS_META = {
    "N":  {"full": "Normal",          "color": "#22c55e"},
    "NS": {"full": "No Screw",     "color": "#6b7280"},
    "OT": {"full": "Over Tightened",  "color": "#ef4444"},
    "NE":  {"full": "No Engage",         "color": "#f59e0b"},
    "UT": {"full": "Under Tightened", "color": "#3b82f6"},
}

MODEL_PATH    = r"C:\github\VT2\Feed-forward_neural_network\trained_model.keras"
SVM_MODEL_PATH = r"C:\github\VT2\SVM\trained_svm.joblib"
RF_MODEL_PATH  = r"C:\github\VT2\RandomForest\trained_rf_tsfresh_uden_lyd.joblib" #New model
FEATURES_PATH = r"C:\github\VT2\Feature_engineering\features_selected.csv"

# ── With-audio models — set paths here when trained ──────────────────────────
AUDIO_MODEL_PATH     = r"C:\github\VT2\Feed-forward_neural_network\trained_model.keras"
AUDIO_SVM_MODEL_PATH = r"C:\github\VT2\SVM\trained_svm.joblib"
AUDIO_RF_MODEL_PATH  = r"C:\github\VT2\RandomForest\RandomForrest_tsfresh_med_lyd.joblib" #New model
AUDIO_FEATURES_PATH  = r"C:\github\VT2\Feature_engineering\features_selected_audio.csv"

# ── Lazy singletons ───────────────────────────────────────────────────────────

_model           = None
_svm_model       = None
_rf_model        = None
_df              = None
_scaler          = None
_raw_pairs_by_model = {}
_task_kind_fc    = None
_intr_kind_fc    = None
_selected_cols   = None
_training_means  = None

# Audio-mode singletons
_audio_model              = None
_audio_svm_model          = None
_audio_rf_model           = None
_audio_df                 = None
_audio_scaler             = None
_audio_raw_pairs_by_model = {}
_audio_task_kind_fc       = None
_audio_intr_kind_fc       = None
_audio_audio_kind_fc      = None
_audio_selected_cols      = None
_audio_training_means     = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def get_svm_model():
    global _svm_model
    if _svm_model is None:
        _svm_model = joblib.load(SVM_MODEL_PATH)
    return _svm_model


def get_rf_model():
    global _rf_model
    if _rf_model is None:
        _rf_model = joblib.load(RF_MODEL_PATH)
    return _rf_model


def get_df_and_scaler():
    global _df, _scaler
    if _df is None:
        _df = pd.read_csv(FEATURES_PATH, index_col=0)
        _scaler = StandardScaler().fit(_df.values)
    return _df, _scaler


def get_inference_assets(model_name="fnn"):
    global _task_kind_fc, _intr_kind_fc, _selected_cols, _training_means
    import inference_pipeline as ip

    # Shared assets (features list, tsfresh params) — build once
    if _selected_cols is None:
        df, _ = get_df_and_scaler()
        _selected_cols = list(df.columns)
        _training_means = df.mean()
        _task_kind_fc, _intr_kind_fc = ip.build_kind_fc_parameters(_selected_cols)
        print(f">> shared inference assets ready: {len(_selected_cols)} features")

    # Per-model test pairs — each model has its own held-out split
    if model_name not in _raw_pairs_by_model:
        _raw_pairs_by_model[model_name] = ip.list_test_pairs(model_name)
        print(f">> [{model_name}] test pairs ready: "
              f"{len(_raw_pairs_by_model[model_name])} samples")

    return (_raw_pairs_by_model[model_name],
            _task_kind_fc, _intr_kind_fc, _selected_cols, _training_means)


# ── Audio-mode getters ────────────────────────────────────────────────────────

def _require_path(path: str, label: str):
    if not path:
        raise HTTPException(503, detail=f"{label} model path is not configured")
    if not Path(path).exists():
        raise HTTPException(503, detail=f"{label} model file not found: {path}")


def get_audio_model():
    global _audio_model
    _require_path(AUDIO_MODEL_PATH, "Audio FNN")
    if _audio_model is None:
        _audio_model = keras.models.load_model(AUDIO_MODEL_PATH)
    return _audio_model


def get_audio_svm_model():
    global _audio_svm_model
    _require_path(AUDIO_SVM_MODEL_PATH, "Audio SVM")
    if _audio_svm_model is None:
        _audio_svm_model = joblib.load(AUDIO_SVM_MODEL_PATH)
    return _audio_svm_model


def get_audio_rf_model():
    global _audio_rf_model
    _require_path(AUDIO_RF_MODEL_PATH, "Audio RF")
    if _audio_rf_model is None:
        _audio_rf_model = joblib.load(AUDIO_RF_MODEL_PATH)
    return _audio_rf_model


def get_audio_df_and_scaler():
    global _audio_df, _audio_scaler
    if _audio_df is None:
        _audio_df    = pd.read_csv(AUDIO_FEATURES_PATH, index_col=0)
        _audio_scaler = StandardScaler().fit(_audio_df.values)
    return _audio_df, _audio_scaler


def get_audio_inference_assets(model_name="fnn"):
    global _audio_task_kind_fc, _audio_intr_kind_fc, _audio_audio_kind_fc
    global _audio_selected_cols, _audio_training_means
    import inference_pipeline as ip

    if _audio_selected_cols is None:
        df, _ = get_audio_df_and_scaler()
        _audio_selected_cols  = list(df.columns)
        _audio_training_means = df.mean()
        (_audio_task_kind_fc,
         _audio_intr_kind_fc,
         _audio_audio_kind_fc) = ip.build_kind_fc_parameters_audio(_audio_selected_cols)
        print(f">> audio inference assets ready: {len(_audio_selected_cols)} features")

    if model_name not in _audio_raw_pairs_by_model:
        _audio_raw_pairs_by_model[model_name] = ip.list_test_pairs_audio(model_name)
        print(f">> [audio/{model_name}] test pairs ready: "
              f"{len(_audio_raw_pairs_by_model[model_name])} samples")

    return (_audio_raw_pairs_by_model[model_name],
            _audio_task_kind_fc, _audio_intr_kind_fc, _audio_audio_kind_fc,
            _audio_selected_cols, _audio_training_means)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Screw Fault Visualiser")


class ScrewResult(BaseModel):
    screw:       int
    label:       str
    full:        str
    color:       str
    confidence:  Optional[float]
    true_label:  str
    true_full:   str
    correct:     bool
    base_id:     str


@app.get("/api/models")
def list_models():
    return {"models": ["fnn", "svm", "rf"]}


@app.get("/api/model_status")
def model_status():
    return {
        "no_audio": {"fnn": True, "svm": True, "rf": True},
        "audio": {
            "fnn": bool(AUDIO_MODEL_PATH),
            "svm": bool(AUDIO_SVM_MODEL_PATH),
            "rf":  bool(AUDIO_RF_MODEL_PATH),
        },
    }


@app.on_event("startup")
async def startup():
    print(">> startup: loading FNN model...")
    get_model()
    print(">> startup: loading SVM model...")
    get_svm_model()
    print(">> startup: loading RF model...")
    get_rf_model()
    print(">> startup: all models loaded. loading features CSV + scaler...")
    get_df_and_scaler()
    print(">> startup: scaler ready. Server is up — inference assets will load on first request.")


def _predict_one_instance(screw_number, model_name="fnn", audio_mode=False):
    """Pick one random raw pair, run pipeline, return prediction."""
    import inference_pipeline as ip

    t0 = time.perf_counter()

    if audio_mode:
        assets = get_audio_inference_assets(model_name)
        raw_pairs, task_kind_fc, intr_kind_fc, audio_kind_fc, selected_cols, training_means = assets
        _, scaler = get_audio_df_and_scaler()
    else:
        raw_pairs, task_kind_fc, intr_kind_fc, selected_cols, training_means = get_inference_assets(model_name)
        _, scaler = get_df_and_scaler()

    t_assets = time.perf_counter()
    print(f"[screw {screw_number}] assets ready:     {t_assets - t0:.3f}s")

    max_attempts = 16
    for attempt in range(max_attempts):
        pair = random.choice(raw_pairs)

        if audio_mode:
            label, base_id, task_path, intr_path, audio_wav_path = pair
        else:
            label, base_id, task_path, intr_path = pair

        t_pipe0 = time.perf_counter()
        if audio_mode:
            feat_series = ip.pipeline_one_audio(
                task_path, intr_path, audio_wav_path, sample_id=0,
                task_kind_to_fc=task_kind_fc,
                intr_kind_to_fc=intr_kind_fc,
                audio_kind_to_fc=audio_kind_fc,
                selected_columns=selected_cols,
                training_means=training_means,
            )
        else:
            feat_series = ip.pipeline_one(
                task_path, intr_path, sample_id=0,
                task_kind_to_fc=task_kind_fc,
                intr_kind_to_fc=intr_kind_fc,
                selected_columns=selected_cols,
                training_means=training_means,
            )
        t_pipe1 = time.perf_counter()
        print(f"[screw {screw_number}] pipeline (attempt {attempt+1}): {t_pipe1 - t_pipe0:.3f}s  id={base_id}")

        if feat_series is None:
            print(f"[screw {screw_number}]   -> no plateau, retrying...")
            continue

        t_inf0 = time.perf_counter()
        X = scaler.transform(feat_series.values.reshape(1, -1))

        if model_name == "svm":
            model_fn = get_audio_svm_model if audio_mode else get_svm_model
            cls  = int(model_fn().predict(X)[0])
            conf = None
        elif model_name == "rf":
            model_fn = get_audio_rf_model if audio_mode else get_rf_model
            rf     = model_fn()
            cls    = int(rf.predict(X)[0])
            probs  = rf.predict_proba(X)[0]
            conf   = round(float(probs[cls]) * 100, 1)
        else:  # fnn
            model_fn = get_audio_model if audio_mode else get_model
            probs  = model_fn().predict(X, verbose=0)[0]
            cls    = int(np.argmax(probs))
            conf   = round(float(probs[cls]) * 100, 1)

        t_inf1 = time.perf_counter()
        print(f"[screw {screw_number}] model inference:  {t_inf1 - t_inf0:.3f}s")
        print(f"[screw {screw_number}] TOTAL:            {t_inf1 - t0:.3f}s")

        pred_label = CLASS_LABELS[cls]

        return ScrewResult(
            screw      = screw_number,
            label      = pred_label,
            full       = CLASS_META[pred_label]["full"],
            color      = CLASS_META[pred_label]["color"],
            confidence = conf,
            true_label = label,
            true_full  = CLASS_META[label]["full"],
            correct    = pred_label == label,
            base_id    = str(base_id),
        )

    raise HTTPException(status_code=500,
                        detail="Could not find a valid screw instance after 16 attempts")


@app.get("/api/predict_one", response_model=ScrewResult)
def predict_one(screw: int = 1, model: str = "fnn", audio: bool = False):
    return _predict_one_instance(screw, model, audio)


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
    const SVG_W = 600;

    /* Screw positions — one per frame arm at each corner (36x36 px each).
       Each screw is centered in the 62px frame bar (perp. axis = 31px).
       The two screws at each corner sit on opposite sides of the 45° joint line. */
    /* Processing order: top screws (x asc), then bottom screws (x asc) */
    const POSITIONS = [
      { x: 13,  y: 50  },  // 1 top-left  – left bar  (x=13, top)
      { x: 50,  y: 13  },  // 2 top-left  – top bar   (x=50, top)
      { x: 514, y: 13  },  // 3 top-right – top bar   (x=514, top)
      { x: 551, y: 50  },  // 4 top-right – right bar (x=551, top)
      { x: 13,  y: 354 },  // 5 bot-left  – left bar  (x=13, bot)
      { x: 50,  y: 391 },  // 6 bot-left  – bot bar   (x=50, bot)
      { x: 514, y: 391 },  // 7 bot-right – bot bar   (x=514, bot)
      { x: 551, y: 354 },  // 8 bot-right – right bar (x=551, bot)
    ];

    const CLASS_META = {
      N:  { full: "Normal",          color: "#22c55e" },
      OT: { full: "Over Tightened",  color: "#ef4444" },
      UT: { full: "Under Tightened", color: "#3b82f6" },
      NE: { full: "No Engage",       color: "#f59e0b" },
      NS: { full: "No Screw",        color: "#6b7280" },
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
            `${result.full}${result.confidence != null ? ` (${result.confidence}%)` : ""}`
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
      const W = SVG_W, H = 440;
      return R("svg", { width: W, height: H, style: { display: "block", overflow: "visible" } },
        R("rect", { x: 0, y: 0, width: W, height: H, rx: 12,
          fill: "#374151", stroke: "#4b5563", strokeWidth: 3 }),
        R("rect", { x: 62, y: 62, width: W-124, height: H-124, rx: 4,
          fill: "rgba(147,197,253,0.12)", stroke: "rgba(148,163,184,0.3)", strokeWidth: 2 }),
        R("line", { x1: W/2, y1: 62, x2: W/2, y2: H-62,
          stroke: "rgba(148,163,184,0.2)", strokeWidth: 3 }),
        R("line", { x1: 62, y1: H/2, x2: W-62, y2: H/2,
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

    function PredictionsTable({ history, showTrue }) {
      if (history.length === 0) return null;

      const cellBase = {
        padding: "8px 12px", fontSize: 12, textAlign: "left",
        borderBottom: "1px solid #334155",
      };
      const headerCell = { ...cellBase, color: "#94a3b8", fontWeight: 700,
        textTransform: "uppercase", letterSpacing: "0.05em", fontSize: 11 };

      // Flatten history into rows: [{round, screw, result}, ...]
      const rows = [];
      history.forEach((round, rIdx) => {
        round.forEach((result, sIdx) => {
          rows.push({ round: rIdx + 1, screw: sIdx + 1, result });
        });
      });

      const correctCount = rows.filter(r => r.result.correct).length;
      const accuracy     = rows.length > 0 ? (correctCount / rows.length) * 100 : 0;

      return R("div", {
        style: { background: "#1e293b", borderRadius: 12,
          padding: "20px 28px", marginTop: 20, width: SVG_W }
      },
        R("div", {
          style: { display: "flex", justifyContent: "space-between",
            alignItems: "center", marginBottom: 16 }
        },
          R("span", { style: { fontWeight: 700, fontSize: 15 } }, "Predictions"),
          showTrue && R("span", { style: { color: "#94a3b8", fontSize: 12 } },
            `${correctCount}/${rows.length} correct (${accuracy.toFixed(1)}%)`)
        ),
        R("div", { style: { overflowX: "auto" } },
          R("table", { style: { width: "100%", borderCollapse: "collapse" } },
            R("thead", null,
              R("tr", null,
                R("th", { style: headerCell }, "Round"),
                R("th", { style: headerCell }, "Screw"),
                R("th", { style: headerCell }, "Predicted"),
                showTrue && R("th", { style: headerCell }, "True"),
                showTrue && R("th", { style: headerCell }, "Correct"),
                R("th", { style: headerCell }, "Confidence"),
                R("th", { style: headerCell }, "ID")
              )
            ),
            R("tbody", null,
              ...rows.map((row, i) => {
                const r = row.result;
                return R("tr", { key: i },
                  R("td", { style: cellBase }, row.round),
                  R("td", { style: cellBase }, `#${row.screw}`),
                  R("td", { style: { ...cellBase, color: r.color, fontWeight: 700 } },
                    `${r.label} – ${r.full}`),
                  showTrue && R("td", { style: cellBase }, r.true_label),
                  showTrue && R("td", { style: { ...cellBase,
                    color: r.correct ? "#86efac" : "#fca5a5", fontWeight: 700 } },
                    r.correct ? "✓" : "✗"),
                  R("td", { style: cellBase },
                    r.confidence != null ? `${r.confidence}%` : "—"),
                  R("td", { style: { ...cellBase, color: "#64748b", fontFamily: "monospace" } },
                    r.base_id)
                );
              })
            )
          )
        )
      );
    }

    /* ── Root App ── */
    function App() {
      const emptySlots = () => Array.from({ length: N }, () => ({ status: "empty", result: null }));

      const [slots,     setSlots]     = useState(emptySlots());
      const [history,   setHistory]   = useState([]);
      const [running,   setRunning]   = useState(false);
      const [error,     setError]     = useState(null);
      const [showTrue,  setShowTrue]  = useState(false);
      const [modelName, setModelName] = useState("fnn");
      const [audioMode, setAudioMode] = useState(false);
      // Session ID prevents stale in-flight loops from updating state after
      // Reset/Restart. Each run captures its own ID; any setSlots call from
      // a session whose ID no longer matches sessionRef.current is ignored.
      const sessionRef = useRef(0);

      const runPredictions = useCallback(async () => {
        const mySession = ++sessionRef.current;
        setRunning(true);
        setError(null);
        const roundResults = [];

        for (let i = 0; i < N; i++) {
          if (mySession !== sessionRef.current) return;

          // Mark this screw as working
          setSlots(prev => {
            if (mySession !== sessionRef.current) return prev;
            const next = [...prev];
            next[i] = { status: "working", result: null };
            return next;
          });

          // Run pipeline and 4-second screw timer in parallel — result is
          // ready by the time the physical process finishes.
          const delay   = new Promise(r => setTimeout(r, 4000));
          const fetchFn = fetch(`/api/predict_one?screw=${i + 1}&model=${modelName}&audio=${audioMode}`)
            .then(res => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); });

          try {
            const [, data] = await Promise.all([delay, fetchFn]);

            if (mySession !== sessionRef.current) return;

            setSlots(prev => {
              if (mySession !== sessionRef.current) return prev;
              const next = [...prev];
              next[i] = { status: "done", result: data };
              return next;
            });
            roundResults.push(data);
          } catch (e) {
            if (mySession !== sessionRef.current) return;
            setError(e.message);
            setSlots(prev => {
              if (mySession !== sessionRef.current) return prev;
              const next = [...prev];
              next[i] = { status: "empty", result: null };
              return next;
            });
            break;
          }
        }

        if (mySession !== sessionRef.current) return;
        if (roundResults.length > 0) {
          setHistory(prev => [...prev, roundResults]);
        }
        setRunning(false);
      }, [modelName, audioMode]);

      const handleStart = useCallback(() => {
        sessionRef.current++;   // invalidate any in-flight run before starting
        setSlots(emptySlots());
        runPredictions();
      }, [runPredictions]);

      const handleReset = useCallback(() => {
        sessionRef.current++;   // invalidate any in-flight run
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
          "Dashboard"),
        R("p", { style: { color: "#94a3b8", fontSize: 13, marginBottom: 16 } },
          `${audioMode ? "Audio" : "No Audio"} · ${modelName.toUpperCase()} · 5-class fault detection`),
        R("div", { style: { display: "flex", alignItems: "center", gap: 12, marginBottom: 16 } },
          R("span", { style: { fontSize: 13, fontWeight: 700,
            color: !audioMode ? "#e2e8f0" : "#94a3b8" } }, "No Audio"),
          R("button", {
            onClick: () => { if (!running) setAudioMode(v => !v); },
            style: {
              position: "relative", width: 48, height: 26, border: "none",
              borderRadius: 99, cursor: running ? "not-allowed" : "pointer",
              background: audioMode ? "#2563eb" : "#334155",
              transition: "background 0.2s", padding: 0, flexShrink: 0,
            }
          },
            R("div", { style: {
              position: "absolute", top: 3, left: audioMode ? 25 : 3,
              width: 20, height: 20, borderRadius: "50%", background: "#fff",
              transition: "left 0.2s",
            }})
          ),
          R("span", { style: { fontSize: 13, fontWeight: 700,
            color: audioMode ? "#e2e8f0" : "#94a3b8" } }, "Audio")
        ),
        R("div", { style: { display: "flex", gap: 8, marginBottom: 24 } },
          ...[ ["fnn", "FNN"], ["svm", "SVM"], ["rf", "RF"] ].map(([key, label]) =>
            R("button", {
              key,
              disabled: running,
              onClick: () => setModelName(key),
              style: {
                padding: "6px 18px", border: "none", borderRadius: 6, fontSize: 13,
                fontWeight: 700, cursor: running ? "not-allowed" : "pointer",
                background: modelName === key ? "#2563eb" : "#1e293b",
                color: modelName === key ? "#fff" : "#94a3b8",
                transition: "background 0.15s",
              }
            }, label)
          )
        ),
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
          }, running ? "Running pipeline…" : "⟳ Start Prediction"),
          R("button", {
            onClick: () => setShowTrue(v => !v),
            style: { ...btnBase, background: showTrue ? "#0f766e" : "#334155" }
          }, showTrue ? "Hide true label" : "Show true label"),
          R("button", {
            onClick: handleReset,
            style: { ...btnBase, background: "#7c3aed" }
          }, "✕ Reset")
        ),
        R(Dashboard, { history }),
        R(PredictionsTable, { history, showTrue })
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
