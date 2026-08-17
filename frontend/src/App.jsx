import React, { useState, useCallback, useEffect } from "react";
import ApertureMark from "./ApertureMark.jsx";
import Dashboard from "./Dashboard.jsx";
import "./App.css";
import "./Dashboard.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const SLIDER_FIELDS = [
  { key: "Linguistic", label: "Words & language", hint: "writing, explaining, storytelling" },
  { key: "Logical_Mathematical", label: "Logic & numbers", hint: "problem-solving, data, systems" },
  { key: "Spatial_Visualization", label: "Visual & spatial", hint: "design, layout, mental imagery" },
  { key: "Bodily", label: "Hands-on & physical", hint: "building, movement, craft" },
  { key: "Musical", label: "Sound & rhythm", hint: "music, audio, pattern in sound" },
  { key: "Interpersonal", label: "People & teams", hint: "leading, persuading, connecting" },
  { key: "Intrapersonal", label: "Self-direction", hint: "independence, reflection, focus" },
  { key: "Naturalist", label: "Nature & environment", hint: "outdoors, ecology, living systems" },
];

function HighlightedResumeText({ text, entities }) {
  if (!text) return null;
  if (!entities || entities.length === 0) {
    return <div className="parse-text">{text}</div>;
  }
  const pieces = [];
  let cursor = 0;
  entities.forEach((ent, i) => {
    if (ent.start > cursor) pieces.push(text.slice(cursor, ent.start));
    pieces.push(
      <span
        key={i}
        className={ent.label === "ROLE" ? "entity-role" : "entity-skill"}
      >
        {text.slice(ent.start, ent.end)}
      </span>
    );
    cursor = ent.end;
  });
  if (cursor < text.length) pieces.push(text.slice(cursor));
  return (
    <>
      <div className="parse-legend">
        <span className="legend-item">
          <span className="legend-swatch entity-skill" /> Skill
        </span>
        <span className="legend-item">
          <span className="legend-swatch entity-role" /> Role / title
        </span>
      </div>
      <div className="parse-text">{pieces}</div>
    </>
  );
}

export default function App() {
  const [view, setView] = useState("app"); // "app" | "dashboard"
  const [mode, setMode] = useState("resume"); // "resume" | "assessment"
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [scores, setScores] = useState(
    Object.fromEntries(SLIDER_FIELDS.map((f) => [f.key, 12]))
  );
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [gapReport, setGapReport] = useState(null);
  const [gapStatus, setGapStatus] = useState("idle"); // idle | loading | done | error
  const [roleInput, setRoleInput] = useState("");
  const [careerOptions, setCareerOptions] = useState([]);
  const [gapErrorMsg, setGapErrorMsg] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/careers`)
      .then((res) => (res.ok ? res.json() : { careers: [] }))
      .then((data) => setCareerOptions(data.careers || []))
      .catch(() => setCareerOptions([]));
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  async function submitResume() {
    if (!file) return;
    setStatus("loading");
    setErrorMsg("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/api/predict-resume`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Prediction failed");
      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message || "Something went wrong reading that file.");
      setStatus("error");
    }
  }

  async function submitAssessment() {
    setStatus("loading");
    setErrorMsg("");
    try {
      const res = await fetch(`${API_BASE}/api/predict-scores`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scores),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message || "Something went wrong.");
      setStatus("error");
    }
  }

  function reset() {
    setStatus("idle");
    setResult(null);
    setFile(null);
    setErrorMsg("");
    setGapReport(null);
    setGapStatus("idle");
    setRoleInput("");
  }

  async function fetchGapReport(career) {
    if (!result) return;
    // trait_scores comes from /api/predict-resume; the assessment
    // mode already has the raw scores in `scores` state.
    const traitScores = result.trait_scores || scores;
    const targetCareer = career || "";
    setGapStatus("loading");
    try {
      const res = await fetch(`${API_BASE}/api/gap-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...traitScores, career: targetCareer }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Gap report failed");
      const data = await res.json();
      setGapReport(data);
      setGapStatus("done");
    } catch (err) {
      setGapStatus("error");
      setGapErrorMsg(err.message || "Couldn't generate a gap report for that role.");
    }
  }

  if (view === "dashboard" && result) {
    return <Dashboard onBack={() => setView("app")} result={result} apiBase={API_BASE} />;
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <ApertureMark size={30} />
          <span className="wordmark">CareerLens</span>
        </div>
        {status === "done" && result && (
          <button className="dash-link" onClick={() => setView("dashboard")}>
            View Analytics Dashboard
          </button>
        )}
      </header>

      <main className={status === "done" ? "hero hero-wide" : "hero"}>
        {status !== "done" && (
          <>
            <h1 className="headline">
              Bring your profile into <em>focus.</em>
            </h1>
            <p className="sub">
              Drop in your resume and see which careers actually match your
              skills and experience.
            </p>

            {mode === "resume" && (
              <div className="panel">
                <label
                  className={dragActive ? "dropzone drag" : "dropzone"}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    accept=".pdf,.docx,.txt"
                    hidden
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                  />
                  {file ? (
                    <span className="filename">{file.name}</span>
                  ) : (
                    <span>
                      Drop your resume here, or <u>browse</u>
                      <br />
                      <span className="hint">PDF, DOCX, or TXT</span>
                    </span>
                  )}
                </label>
                <button
                  className="primary-btn"
                  disabled={!file || status === "loading"}
                  onClick={submitResume}
                >
                  {status === "loading" ? "Analyzing…" : "Find my fit"}
                </button>
                <p className="disclaimer">
                  Resume scoring is an approximate keyword-based read on your
                  background, not a formal assessment — treat results as a
                  starting point for exploration, not a verdict.
                </p>
              </div>
            )}

            {status === "loading" && (
              <div className="loading-mark">
                <ApertureMark size={56} spinning />
                <span>Focusing…</span>
              </div>
            )}

            {status === "error" && <p className="error-msg">{errorMsg}</p>}
          </>
        )}

        {status === "done" && result && (
          <div className="result">
            <ApertureMark size={64} />

            {result.resume_text && (
              <div className="parse-panel">
                <p className="parse-panel-title">Resume Parsing (spaCy NER)</p>
                <HighlightedResumeText
                  text={result.resume_text}
                  entities={result.parsed_entities}
                />
              </div>
            )}

            <div className={result.tech_top_5 ? "result-columns" : "result-columns single"}>
              {result.tech_top_5 && (
                <div className="result-col">
                  <p className="result-eyebrow">Closest tech role (skill-based)</p>
                  <h2 className="result-career">{result.tech_top_5[0].career}</h2>
                  <p className="result-confidence">
                    {Math.round(result.tech_top_5[0].confidence * 100)}% match
                  </p>
                  <div className="result-chart">
                    {result.tech_top_5.map((r, i) => {
                      const pct = Math.round(r.confidence * 100);
                      const maxPct = Math.round(result.tech_top_5[0].confidence * 100);
                      const widthPct = maxPct > 0 ? (pct / maxPct) * 100 : 0;
                      return (
                        <div className="bar-row" key={r.career}>
                          <span className="bar-label">{r.career}</span>
                          <div className="bar-track">
                            <div
                              className={i === 0 ? "bar-fill bar-fill-top" : "bar-fill"}
                              style={{ width: `${widthPct}%` }}
                            />
                          </div>
                          <span className="bar-pct">{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="result-col">
                <p className="result-eyebrow">
                  {result.tech_top_5 ? "Broader fit (trait-based, 72 careers)" : "Your closest fit"}
                </p>
                <h2 className="result-career">{result.top_prediction}</h2>
                {result.top_5?.[0] && (
                  <p className="result-confidence">
                    {Math.round(result.top_5[0].confidence * 100)}% match
                  </p>
                )}
                {result.top_5?.length > 0 && (
                  <div className="result-chart">
                    {result.top_5.map((r, i) => {
                      const pct = Math.round(r.confidence * 100);
                      const maxPct = Math.round(result.top_5[0].confidence * 100);
                      const widthPct = maxPct > 0 ? (pct / maxPct) * 100 : 0;
                      return (
                        <div className="bar-row" key={r.career}>
                          <span className="bar-label">{r.career}</span>
                          <div className="bar-track">
                            <div
                              className={i === 0 ? "bar-fill bar-fill-top" : "bar-fill"}
                              style={{ width: `${widthPct}%` }}
                            />
                          </div>
                          <span className="bar-pct">{pct}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            <div className="gap-section">
              <div className="gap-role-box">
                <p className="hint">Skill Gap Analysis</p>
                <p className="gap-role-subhint">
                  Enter the role you're aiming for and see which of your
                  skills match it, and which ones are missing.
                </p>
                <form
                  className="gap-role-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (!roleInput.trim()) return;
                    setGapErrorMsg("");
                    fetchGapReport(roleInput.trim());
                  }}
                >
                  <input
                    className="gap-role-input"
                    type="text"
                    list="gap-career-options"
                    placeholder="e.g. Data Scientist, UX Designer, Mechanical Engineer…"
                    value={roleInput}
                    onChange={(e) => setRoleInput(e.target.value)}
                  />
                  <datalist id="gap-career-options">
                    {careerOptions.map((c) => (
                      <option key={c} value={c} />
                    ))}
                  </datalist>
                  <button
                    type="submit"
                    className="secondary-btn"
                    disabled={!roleInput.trim() || gapStatus === "loading"}
                  >
                    {gapStatus === "loading" ? "Analyzing…" : "Analyze skill gap"}
                  </button>
                </form>
                {result.top_prediction && (
                  <p className="gap-role-suggestion">
                    Predicted fit:{" "}
                    <button
                      type="button"
                      className="gap-role-suggestion-btn"
                      onClick={() => {
                        setRoleInput(result.top_prediction);
                        setGapErrorMsg("");
                        fetchGapReport(result.top_prediction);
                      }}
                    >
                      {result.top_prediction}
                    </button>
                  </p>
                )}
              </div>

              {gapStatus === "error" && (
                <p className="error-msg">
                  {gapErrorMsg || "Couldn't generate a gap report for that role."}
                </p>
              )}

              {gapStatus === "done" && gapReport && (
                <div className="gap-panel">
                  <p className="parse-panel-title">
                    Skill Gap Report — {gapReport.career}
                  </p>
                  <p className="result-confidence">
                    Overall readiness: {gapReport.overall_readiness}%
                  </p>
                  <p className="gap-source-note">
                    {gapReport.source === "tech_resumes"
                      ? "Target profile derived from real resumes in this tech category."
                      : "Target profile derived from trait-survey averages for this career."}
                  </p>

                  {(() => {
                    const missing = gapReport.gaps.filter(
                      (g) => g.severity === "high" || g.severity === "medium"
                    );
                    const present = gapReport.gaps.filter(
                      (g) => g.severity === "on_track" || g.severity === "strength"
                    );
                    return (
                      <>
                        <div className="gap-group">
                          <p className="gap-group-title gap-group-missing">
                            Missing / underdeveloped skills
                          </p>
                          {missing.length === 0 && (
                            <p className="gap-source-note">
                              No significant gaps found — you're well matched on every trait.
                            </p>
                          )}
                          <div className="result-chart">
                            {missing.map((g) => (
                              <div className="gap-row" key={g.trait}>
                                <div className="bar-row">
                                  <span className="bar-label">{g.trait}</span>
                                  <span className={`gap-badge gap-${g.severity}`}>
                                    {g.severity.replace("_", " ")}
                                  </span>
                                  <span className="bar-pct">
                                    {g.resume_score} / {g.target_score}
                                  </span>
                                </div>
                                {g.suggestion && (
                                  <p className="gap-suggestion">{g.suggestion}</p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>

                        <div className="gap-group">
                          <p className="gap-group-title gap-group-present">
                            Skills you already have
                          </p>
                          {present.length === 0 && (
                            <p className="gap-source-note">
                              None of the traits for this role are fully covered yet.
                            </p>
                          )}
                          <div className="result-chart">
                            {present.map((g) => (
                              <div className="gap-row" key={g.trait}>
                                <div className="bar-row">
                                  <span className="bar-label">{g.trait}</span>
                                  <span className={`gap-badge gap-${g.severity}`}>
                                    {g.severity.replace("_", " ")}
                                  </span>
                                  <span className="bar-pct">
                                    {g.resume_score} / {g.target_score}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </>
                    );
                  })()}

                  <button
                    className="secondary-btn"
                    onClick={() => {
                      setGapStatus("idle");
                      setGapReport(null);
                      setRoleInput("");
                    }}
                  >
                    Check a different role
                  </button>
                </div>
              )}
            </div>

            <button className="secondary-btn" onClick={reset}>
              Try again
            </button>
          </div>
        )}
      </main>
    </div>

  );
}
