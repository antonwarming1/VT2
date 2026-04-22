"""
FastAPI + React (CDN) web server.
Samples 8 random screw instances, runs them through the trained FNN,
and returns predictions for the React frontend to visualise.

Run:  uvicorn app:app --reload --port 5000
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
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

_model  = None
_df     = None
_scaler = None


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


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Screw Fault Visualiser")


class ScrewResult(BaseModel):
    screw: int
    label: str
    full: str
    color: str
    confidence: float


@app.on_event("startup")
async def startup():
    get_model()
    get_df_and_scaler()


@app.get("/api/predict", response_model=List[ScrewResult])
def predict():
    df, scaler = get_df_and_scaler()
    model      = get_model()

    sample   = df.sample(n=8)
    X_scaled = scaler.transform(sample.values)
    probs    = model.predict(X_scaled, verbose=0)
    classes  = np.argmax(probs, axis=1)

    results = []
    for i, (cls, prob_row) in enumerate(zip(classes, probs)):
        label = CLASS_LABELS[int(cls)]
        results.append(ScrewResult(
            screw      = i + 1,
            label      = label,
            full       = CLASS_META[label]["full"],
            color      = CLASS_META[label]["color"],
            confidence = round(float(prob_row[cls]) * 100, 1),
        ))
    return results


# ── React SPA (served inline, no build step) ──────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Screw Fault Visualiser</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
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
  </style>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel">
    const { useState, useEffect, useCallback } = React;

    /* Screw positions inside the 560×400 scene (2 per corner, 36×36 px each) */
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

    /* ── Screw SVG component ── */
    function Screw({ x, y, label, full, color, confidence }) {
      const [hovered, setHovered] = useState(false);
      const size = 36;

      return (
        <g
          transform={`translate(${x}, ${y})`}
          style={{ cursor: "pointer" }}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
        >
          {/* outer ring */}
          <circle cx={size/2} cy={size/2} r={size/2} fill="#1e293b" />
          {/* coloured face */}
          <circle cx={size/2} cy={size/2} r={size/2 - 3} fill={color} />
          {/* highlight */}
          <circle cx={size/2 - 6} cy={size/2 - 6} r={5}
                  fill="rgba(255,255,255,0.25)" />
          {/* cross slot */}
          <rect x={size/2 - 9} y={size/2 - 2} width={18} height={4}
                rx={1} fill="rgba(0,0,0,0.45)" />
          <rect x={size/2 - 2} y={size/2 - 9} width={4} height={18}
                rx={1} fill="rgba(0,0,0,0.45)" />

          {/* label below */}
          <text x={size/2} y={size + 14} textAnchor="middle"
                fontSize="10" fontWeight="700" fill="#cbd5e1"
                fontFamily="Segoe UI, system-ui, sans-serif">
            {label}
          </text>

          {/* tooltip on hover */}
          {hovered && (
            <g>
              <rect x={size/2 - 60} y={-44} width={120} height={34}
                    rx={6} fill="#0f172a" stroke="#334155" strokeWidth={1} />
              <text x={size/2} y={-26} textAnchor="middle"
                    fontSize="11" fontWeight="700" fill={color}
                    fontFamily="Segoe UI, system-ui, sans-serif">
                {full}
              </text>
              <text x={size/2} y={-12} textAnchor="middle"
                    fontSize="10" fill="#94a3b8"
                    fontFamily="Segoe UI, system-ui, sans-serif">
                Confidence: {confidence}%
              </text>
            </g>
          )}
        </g>
      );
    }

    /* ── Window SVG ── */
    function WindowScene({ screws }) {
      const W = 560, H = 400;
      return (
        <svg width={W} height={H} style={{ display: "block" }}>
          {/* frame */}
          <rect x={0} y={0} width={W} height={H} rx={12}
                fill="#374151" stroke="#4b5563" strokeWidth={3} />
          {/* glass */}
          <rect x={42} y={42} width={W-84} height={H-84} rx={4}
                fill="rgba(147,197,253,0.12)"
                stroke="rgba(148,163,184,0.3)" strokeWidth={2} />
          {/* cross bars */}
          <line x1={W/2} y1={42} x2={W/2} y2={H-42}
                stroke="rgba(148,163,184,0.2)" strokeWidth={3} />
          <line x1={42} y1={H/2} x2={W-42} y2={H/2}
                stroke="rgba(148,163,184,0.2)" strokeWidth={3} />

          {/* screws */}
          {screws.map((s, i) => (
            <Screw key={i} {...s} {...POSITIONS[i]} />
          ))}
        </svg>
      );
    }

    /* ── Legend ── */
    function Legend() {
      return (
        <div style={{
          display: "flex", flexWrap: "wrap", gap: 10,
          justifyContent: "center", marginBottom: 24,
        }}>
          {Object.entries(CLASS_META).map(([code, { full, color }]) => (
            <div key={code} style={{
              display: "flex", alignItems: "center", gap: 8,
              background: "#1e293b", borderRadius: 8,
              padding: "6px 14px", fontSize: 13,
            }}>
              <div style={{
                width: 14, height: 14, borderRadius: "50%",
                background: color, flexShrink: 0,
              }} />
              <span><strong>{code}</strong> – {full}</span>
            </div>
          ))}
        </div>
      );
    }

    /* ── Summary chips ── */
    function Summary({ screws }) {
      return (
        <div style={{
          display: "flex", flexWrap: "wrap", gap: 8,
          justifyContent: "center", marginTop: 20,
        }}>
          {screws.map((s, i) => (
            <span key={i} style={{
              background: s.color, color: "#fff",
              borderRadius: 99, padding: "4px 14px",
              fontSize: 12, fontWeight: 700,
            }}>
              #{i+1} {s.label}
            </span>
          ))}
        </div>
      );
    }

    /* ── Dashboard ── */
    function Dashboard({ history }) {
      if (history.length === 0) return null;

      // Flatten all predictions across all rounds
      const all = history.flat();
      const total = all.length;
      const counts = {};
      Object.keys(CLASS_META).forEach(k => { counts[k] = 0; });
      all.forEach(s => { counts[s.label] = (counts[s.label] || 0) + 1; });

      const BAR_W = 420;

      return (
        <div style={{
          background: "#1e293b", borderRadius: 12,
          padding: "20px 28px", marginTop: 28,
          width: 560,
        }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            alignItems: "center", marginBottom: 16,
          }}>
            <span style={{ fontWeight: 700, fontSize: 15 }}>
              Class Distribution
            </span>
            <span style={{ color: "#94a3b8", fontSize: 12 }}>
              {history.length} sample{history.length !== 1 ? "s" : ""} · {total} screws total
            </span>
          </div>

          {Object.entries(CLASS_META).map(([code, { full, color }]) => {
            const count = counts[code] || 0;
            const pct   = total > 0 ? (count / total) * 100 : 0;
            const barPx = (pct / 100) * BAR_W;
            return (
              <div key={code} style={{ marginBottom: 12 }}>
                <div style={{
                  display: "flex", justifyContent: "space-between",
                  marginBottom: 4, fontSize: 12,
                }}>
                  <span style={{ fontWeight: 700, color }}>
                    {code} <span style={{ color: "#94a3b8", fontWeight: 400 }}>– {full}</span>
                  </span>
                  <span style={{ color: "#e2e8f0", fontWeight: 600 }}>
                    {count} ({pct.toFixed(1)}%)
                  </span>
                </div>
                <div style={{
                  background: "#0f172a", borderRadius: 99,
                  height: 10, width: BAR_W, overflow: "hidden",
                }}>
                  <div style={{
                    height: "100%", width: barPx,
                    background: color, borderRadius: 99,
                    transition: "width 0.4s ease",
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      );
    }

    /* ── Root App ── */
    function App() {
      const [screws,  setScrews]  = useState([]);
      const [history, setHistory] = useState([]);
      const [loading, setLoading] = useState(true);
      const [error,   setError]   = useState(null);

      const fetchPredictions = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
          const res  = await fetch("/api/predict");
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          setScrews(data);
          setHistory(prev => [...prev, data]);
        } catch (e) {
          setError(e.message);
        } finally {
          setLoading(false);
        }
      }, []);

      const reset = useCallback(() => {
        setScrews([]);
        setHistory([]);
        setError(null);
        fetchPredictions();
      }, [fetchPredictions]);

      useEffect(() => { fetchPredictions(); }, [fetchPredictions]);

      const btnBase = {
        padding: "10px 28px", border: "none", borderRadius: 8,
        color: "#fff", fontSize: 14, fontWeight: 700,
        cursor: "pointer", transition: "background 0.15s",
      };

      return (
        <div style={{
          display: "flex", flexDirection: "column",
          alignItems: "center", padding: "40px 16px",
        }}>
          <h1 style={{ fontSize: "1.6rem", marginBottom: 4, letterSpacing: "0.05em" }}>
            Window Screw Fault Visualiser
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 13, marginBottom: 28 }}>
            FNN · 5-class fault detection · 8 screws (2 per corner)
          </p>

          <Legend />

          <div style={{
            borderRadius: 16,
            boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
            overflow: "visible",
          }}>
            {loading ? (
              <div style={{
                width: 560, height: 400, display: "flex",
                alignItems: "center", justifyContent: "center",
                background: "#1e293b", borderRadius: 12,
                color: "#94a3b8", fontSize: 16,
              }}>
                Running model…
              </div>
            ) : error ? (
              <div style={{
                width: 560, height: 400, display: "flex",
                alignItems: "center", justifyContent: "center",
                background: "#1e293b", borderRadius: 12,
                color: "#ef4444", fontSize: 14, padding: 32, textAlign: "center",
              }}>
                Error: {error}
              </div>
            ) : (
              <WindowScene screws={screws} />
            )}
          </div>

          {!loading && !error && <Summary screws={screws} />}

          <div style={{ display: "flex", gap: 12, marginTop: 28 }}>
            <button
              onClick={fetchPredictions}
              disabled={loading}
              style={{
                ...btnBase,
                background: loading ? "#334155" : "#2563eb",
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              {loading ? "Predicting…" : "⟳ Resample & Predict"}
            </button>
            <button
              onClick={reset}
              disabled={loading}
              style={{
                ...btnBase,
                background: loading ? "#334155" : "#7c3aed",
                cursor: loading ? "not-allowed" : "pointer",
              }}
            >
              ✕ Reset
            </button>
          </div>

          <Dashboard history={history} />
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById("root")).render(<App />);
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
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
