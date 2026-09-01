import React, { useEffect, useMemo, useState } from "react";
import ApertureMark from "./ApertureMark.jsx";
import CohortAnalytics from "./CohortAnalytics.jsx";

/**
 * Analytics Dashboard — shows REAL data returned by the backend:
 *   - model_comparison: actual logged macro-F1 scores from training
 *     (training/04_train_cpds.py's grid search), served via
 *     GET /api/dashboard.
 *   - embeddings: a real t-SNE projection of the training set (3600
 *     resumes' worth of trait vectors), also from /api/dashboard.
 *   - cohort_stats: aggregate counts of predicted careers across all
 *     users so far, served via GET /api/cohort-stats.
 *   - the current user's own resume, plotted as a highlighted point
 *     using `result.resume_point` (a real PCA projection of THEIR
 *     actual extracted trait vector, computed server-side) and their
 *     own top_5 careers — no invented numbers.
 *
 * If the user opened this without having submitted anything yet, the
 * personal sections are simply omitted rather than faked.
 */

const CLUSTER_COLORS = {
  "Technology & Engineering": "#3fa0ff",
  "Science & Research": "#2f6fed",
  "Business & Finance": "#e8a33d",
  Healthcare: "#e05d5d",
  "Creative & Arts": "#d76fd6",
  "Public Service & Law": "#8a6fd6",
  Education: "#4bbf8f",
  Other: "#8a94a6",
};

function ModelComparisonChart({ models }) {
  const best = Math.max(...models.map((m) => m.macro_f1));
  return (
    <div className="mc-chart">
      <div className="mc-gridlines">
        {[100, 80, 60, 40, 20, 0].map((v) => (
          <div className="mc-gridline" key={v}>
            <span className="mc-gridline-label">{v}%</span>
          </div>
        ))}
      </div>
      <div className="mc-bars">
        {models.map((m) => {
          const isBest = m.macro_f1 === best;
          return (
            <div className="mc-bar-col" key={m.name}>
              <span className={isBest ? "mc-value mc-value-best" : "mc-value"}>
                {(m.macro_f1 * 100).toFixed(1)}%
              </span>
              <div className="mc-bar-track">
                <div
                  className={isBest ? "mc-bar-fill mc-bar-fill-best" : "mc-bar-fill"}
                  style={{ height: `${m.macro_f1 * 100}%` }}
                />
              </div>
              <span className="mc-bar-name">{m.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TSNEScatter({ points, clusters, userPoint }) {
  const width = 520;
  const height = 340;
  const padding = 36;
  const legendW = 130;

  const { minX, maxX, minY, maxY } = useMemo(() => {
    const xs = points.map((p) => p.x).concat(userPoint ? [userPoint.x] : []);
    const ys = points.map((p) => p.y).concat(userPoint ? [userPoint.y] : []);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }, [points, userPoint]);

  const plotW = width - padding * 2 - legendW;
  const plotH = height - padding * 2;

  function projX(v) {
    return padding + ((v - minX) / (maxX - minX || 1)) * plotW;
  }
  function projY(v) {
    return padding + plotH - ((v - minY) / (maxY - minY || 1)) * plotH;
  }

  const xTicks = useMemo(() => {
    const step = (maxX - minX) / 5;
    return Array.from({ length: 6 }, (_, i) => Math.round(minX + step * i));
  }, [minX, maxX]);
  const yTicks = useMemo(() => {
    const step = (maxY - minY) / 5;
    return Array.from({ length: 6 }, (_, i) => Math.round(minY + step * i));
  }, [minY, maxY]);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="t-SNE visualization of skill embeddings">
      {xTicks.map((v) => (
        <g key={`x${v}`}>
          <line x1={projX(v)} x2={projX(v)} y1={padding} y2={height - padding} stroke="var(--hairline)" strokeWidth="0.5" opacity="0.5" />
          <text x={projX(v)} y={height - padding + 16} fontSize="9" fill="var(--slate)" textAnchor="middle" fontFamily="var(--mono)">
            {v}
          </text>
        </g>
      ))}
      {yTicks.map((v) => (
        <g key={`y${v}`}>
          <line x1={padding} x2={width - padding - legendW} y1={projY(v)} y2={projY(v)} stroke="var(--hairline)" strokeWidth="0.5" opacity="0.5" />
          <text x={padding - 8} y={projY(v) + 3} fontSize="9" fill="var(--slate)" textAnchor="end" fontFamily="var(--mono)">
            {v}
          </text>
        </g>
      ))}

      {points.map((p, i) => (
        <circle key={i} cx={projX(p.x)} cy={projY(p.y)} r="2.2" fill={CLUSTER_COLORS[p.cluster] || CLUSTER_COLORS.Other} opacity="0.55" />
      ))}

      {userPoint && (
        <g>
          <circle cx={projX(userPoint.x)} cy={projY(userPoint.y)} r="9" fill="none" stroke="var(--amber)" strokeWidth="2">
            <animate attributeName="r" values="7;11;7" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite" />
          </circle>
          <circle cx={projX(userPoint.x)} cy={projY(userPoint.y)} r="5" fill="var(--amber)" stroke="var(--ink)" strokeWidth="1.5" />
          <text x={projX(userPoint.x) + 12} y={projY(userPoint.y) + 4} fontSize="10.5" fontWeight="600" fill="var(--amber)">
            You
          </text>
        </g>
      )}

      <g transform={`translate(${width - legendW + 6}, ${padding})`}>
        {clusters.map((label, i) => (
          <g key={label} transform={`translate(0, ${i * 18})`}>
            <circle cx="4" cy="0" r="4" fill={CLUSTER_COLORS[label] || CLUSTER_COLORS.Other} />
            <text x="12" y="4" fontSize="9.5" fill="var(--paper-dim)">
              {label}
            </text>
          </g>
        ))}
        {userPoint && (
          <g transform={`translate(0, ${clusters.length * 18 + 6})`}>
            <circle cx="4" cy="0" r="4" fill="var(--amber)" stroke="var(--ink)" strokeWidth="1" />
            <text x="12" y="4" fontSize="9.5" fill="var(--amber)" fontWeight="600">
              You
            </text>
          </g>
        )}
      </g>

      <text x={(width - legendW) / 2} y={height - 4} fontSize="10" fill="var(--slate)" textAnchor="middle">
        t-SNE Component 1
      </text>
      <text x={12} y={height / 2} fontSize="10" fill="var(--slate)" textAnchor="middle" transform={`rotate(-90, 12, ${height / 2})`}>
        t-SNE Component 2
      </text>
    </svg>
  );
}

export default function Dashboard({ onBack, result, apiBase }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | done | error
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiBase}/api/dashboard`)
      .then((res) => {
        if (!res.ok) throw new Error("Dashboard data isn't available yet.");
        return res.json();
      })
      .then((json) => {
        if (!cancelled) {
          setData(json);
          setStatus("done");
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setErrorMsg(err.message || "Couldn't load dashboard data.");
          setStatus("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  const userTop5 = result?.tech_top_5?.length ? result.tech_top_5 : result?.top_5;

  return (
    <div className="dash">
      <header className="dash-header">
        <div className="brand">
          <ApertureMark size={26} />
          <span className="wordmark">CareerLens</span>
        </div>
        <div className="dash-title">Analytics Dashboard</div>
        {onBack && (
          <button className="secondary-btn dash-back" onClick={onBack}>
            ← Back
          </button>
        )}
      </header>

      {status === "loading" && (
        <div className="dash-loading">
          <ApertureMark size={48} spinning />
          <span>Loading real training-data analytics…</span>
        </div>
      )}

      {status === "error" && (
        <div className="dash-loading">
          <p className="error-msg">{errorMsg}</p>
          <p className="dash-hint">
            Run <code>training/09_export_dashboard_data.py</code> on the backend to generate it.
          </p>
        </div>
      )}

      {status === "done" && data && (
        <main className="dash-grid">
          {!result && (
            <div className="dash-card dash-card-wide dash-note">
              Showing model &amp; training-data analytics only. Submit a resume or assessment first to see your own result highlighted below.
            </div>
          )}

          <section className="dash-card dash-card-wide">
            <div className="dash-card-head">
              <h3>Model Comparison – Macro F1-Score</h3>
              <span className="dash-badge">from actual grid-search evaluation</span>
            </div>
            <ModelComparisonChart models={data.model_comparison} />
          </section>

          <section className="dash-card dash-card-wide">
            <div className="dash-card-head">
              <h3>Cohort Analytics — Most Predicted Careers</h3>
              <span className="dash-badge">across all users</span>
            </div>
            <CohortAnalytics apiBase={apiBase} />
          </section>

          {userTop5 && (
            <section className="dash-card">
              <div className="dash-card-head">
                <h3>Your Top-5 Career Matches</h3>
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
                  {userTop5.map((r, i) => (
                    <tr key={r.career} className={i === 0 ? "dash-row-top" : ""}>
                      <td>{i + 1}</td>
                      <td>{r.career}</td>
                      <td className="dash-conf">{Math.round(r.confidence * 100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          <section className={userTop5 ? "dash-card" : "dash-card dash-card-wide"}>
            <div className="dash-card-head">
              <h3>t-SNE Visualization: Trait Embeddings</h3>
              <span className="dash-badge">3,600 real training samples</span>
            </div>
            <TSNEScatter
              points={data.embeddings.points}
              clusters={data.embeddings.clusters}
              userPoint={result?.resume_point || null}
            />
          </section>
        </main>
      )}
    </div>
  );
}
