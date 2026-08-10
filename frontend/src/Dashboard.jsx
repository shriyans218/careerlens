import React, { useMemo } from "react";
import ApertureMark from "./ApertureMark.jsx";

/**
 * Milestone 2 Dashboard — model comparison, top-5 recommendations,
 * and a t-SNE skill-embedding scatter plot. Pure SVG/CSS, no chart
 * libraries, consistent with the rest of the app.
 */

const MODEL_SCORES = [
  { name: "Logistic Regression", f1: 0.65, met: false },
  { name: "Random Forest", f1: 0.75, met: false },
  { name: "XGBoost", f1: 0.82, met: true },
];

const THRESHOLD = 0.8;

const TOP5 = [
  { rank: 1, name: "ML Engineer", confidence: 94 },
  { rank: 2, name: "Data Scientist", confidence: 89 },
  { rank: 3, name: "Backend Dev", confidence: 85 },
  { rank: 4, name: "Product Manager", confidence: 81 },
  { rank: 5, name: "AI Researcher", confidence: 78 },
];

const CLUSTER_COLORS = {
  "Software Engineering": "#3fa0ff",
  "Data Science": "#2f6fed",
  "Product Management": "#e8a33d",
  "AI/ML": "#e05d5d",
  Research: "#8a6fd6",
  Finance: "#d7c93e",
};

// Deterministic pseudo-random cluster generator so the scatter looks
// the same on every render without needing real embedding data.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateClusters() {
  const rand = mulberry32(42);
  const centers = [
    { label: "Software Engineering", cx: -12, cy: 14, n: 140 },
    { label: "Data Science", cx: 6, cy: 14, n: 130 },
    { label: "Product Management", cx: 14, cy: -14, n: 110 },
    { label: "AI/ML", cx: -2, cy: -4, n: 90 },
    { label: "Research", cx: -14, cy: -16, n: 100 },
    { label: "Finance", cx: 18, cy: -2, n: 90 },
  ];
  const points = [];
  centers.forEach((c) => {
    for (let i = 0; i < c.n; i++) {
      const angle = rand() * Math.PI * 2;
      const r = Math.pow(rand(), 0.5) * 9;
      points.push({
        label: c.label,
        x: c.cx + Math.cos(angle) * r + (rand() - 0.5) * 3,
        y: c.cy + Math.sin(angle) * r + (rand() - 0.5) * 3,
      });
    }
  });
  return points;
}

function ModelComparisonChart() {
  const width = 560;
  const height = 260;
  const padL = 40;
  const padB = 32;
  const padT = 16;
  const chartW = width - padL - 20;
  const chartH = height - padT - padB;
  const barGap = 40;
  const barW = (chartW - barGap * (MODEL_SCORES.length - 1)) / MODEL_SCORES.length;
  const maxVal = 1;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="Model comparison macro F1 score">
      {[0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => {
        const y = padT + chartH - (v / maxVal) * chartH;
        return (
          <g key={v}>
            <line x1={padL} x2={width - 20} y1={y} y2={y} stroke="var(--hairline)" strokeWidth="1" />
            <text x={padL - 8} y={y + 4} textAnchor="end" fontSize="11" fill="var(--slate)" fontFamily="var(--mono)">
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}

      {MODEL_SCORES.map((m, i) => {
        const x = padL + i * (barW + barGap);
        const barH = (m.f1 / maxVal) * chartH;
        const y = padT + chartH - barH;
        return (
          <g key={m.name}>
            {m.met && (
              <rect
                x={x - 8}
                y={y - 8}
                width={barW + 16}
                height={barH + 16}
                rx="10"
                fill="var(--amber)"
                opacity="0.15"
              />
            )}
            <rect
              x={x}
              y={y}
              width={barW}
              height={barH}
              rx="4"
              fill={m.met ? "var(--amber)" : "rgba(224,168,76,0.35)"}
            />
            <text
              x={x + barW / 2}
              y={y - 12}
              textAnchor="middle"
              fontSize="13"
              fontWeight="600"
              fill={m.met ? "var(--amber)" : "var(--paper-dim)"}
              fontFamily="var(--mono)"
            >
              {m.f1.toFixed(2)}
            </text>
            <text
              x={x + barW / 2}
              y={height - padB + 18}
              textAnchor="middle"
              fontSize="12"
              fill="var(--paper-dim)"
            >
              {m.name}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function TSNEScatter() {
  const points = useMemo(() => generateClusters(), []);
  const width = 520;
  const height = 340;
  const padding = 36;
  const domain = 32; // -32..32 on both axes

  function project(v) {
    return ((v + domain) / (domain * 2));
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="t-SNE visualization of skill embeddings">
      {[-30, -20, -10, 0, 10, 20, 30].map((v) => {
        const x = padding + project(v) * (width - padding * 2 - 90);
        const y = padding + (1 - project(v)) * (height - padding * 2);
        return (
          <g key={v}>
            <line
              x1={x}
              x2={x}
              y1={padding}
              y2={height - padding}
              stroke="var(--hairline)"
              strokeWidth="0.5"
              opacity="0.5"
            />
            <line
              x1={padding}
              x2={width - padding - 90}
              y1={y}
              y2={y}
              stroke="var(--hairline)"
              strokeWidth="0.5"
              opacity="0.5"
            />
            <text x={x} y={height - padding + 16} fontSize="9" fill="var(--slate)" textAnchor="middle" fontFamily="var(--mono)">
              {v}
            </text>
            <text x={padding - 8} y={y + 3} fontSize="9" fill="var(--slate)" textAnchor="end" fontFamily="var(--mono)">
              {v}
            </text>
          </g>
        );
      })}

      {points.map((p, i) => {
        const x = padding + project(p.x) * (width - padding * 2 - 90);
        const y = padding + (1 - project(p.y)) * (height - padding * 2);
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r="2.4"
            fill={CLUSTER_COLORS[p.label]}
            opacity="0.75"
          />
        );
      })}

      <g transform={`translate(${width - 84}, ${padding})`}>
        {Object.entries(CLUSTER_COLORS).map(([label, color], i) => (
          <g key={label} transform={`translate(0, ${i * 18})`}>
            <circle cx="4" cy="0" r="4" fill={color} />
            <text x="12" y="4" fontSize="9.5" fill="var(--paper-dim)">
              {label}
            </text>
          </g>
        ))}
      </g>

      <text
        x={(width - 90) / 2 + padding / 2}
        y={height - 4}
        fontSize="10"
        fill="var(--slate)"
        textAnchor="middle"
      >
        t-SNE Component 1
      </text>
      <text
        x={12}
        y={height / 2}
        fontSize="10"
        fill="var(--slate)"
        textAnchor="middle"
        transform={`rotate(-90, 12, ${height / 2})`}
      >
        t-SNE Component 2
      </text>
    </svg>
  );
}

export default function Dashboard({ onBack }) {
  return (
    <div className="dash">
      <header className="dash-header">
        <div className="brand">
          <ApertureMark size={26} />
          <span className="wordmark">CareerLens</span>
        </div>
        <div className="dash-title">CareerLens Analytics Dashboard</div>
        {onBack && (
          <button className="secondary-btn dash-back" onClick={onBack}>
            ← Back
          </button>
        )}
      </header>

      <main className="dash-grid">
        <section className="dash-card dash-card-wide">
          <div className="dash-card-head">
            <h3>Model Comparison – Macro F1-Score</h3>
            <span className="dash-badge">
              Threshold {THRESHOLD.toFixed(2)}
            </span>
          </div>
          <ModelComparisonChart />
        </section>

        <section className="dash-card">
          <div className="dash-card-head">
            <h3>Top-5 Career Recommendations</h3>
          </div>
          <table className="dash-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Name</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {TOP5.map((r) => (
                <tr key={r.rank} className={r.rank === 1 ? "dash-row-top" : ""}>
                  <td>{r.rank}</td>
                  <td>{r.name}</td>
                  <td className="dash-conf">{r.confidence}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="dash-card">
          <div className="dash-card-head">
            <h3>t-SNE Visualization: Skill Embeddings (SBERT)</h3>
          </div>
          <TSNEScatter />
        </section>
      </main>
    </div>
  );
}
