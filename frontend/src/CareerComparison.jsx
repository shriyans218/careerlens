import React, { useState } from "react";

export default function CareerComparison({ apiBase, traitScores, resumeText, careerOptions, onClose }) {
  const [careerA, setCareerA] = useState("");
  const [careerB, setCareerB] = useState("");
  const [reportA, setReportA] = useState(null);
  const [reportB, setReportB] = useState(null);
  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function fetchOne(career) {
    const res = await fetch(`${apiBase}/api/gap-report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...traitScores, career, resume_text: resumeText || "" }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `Couldn't analyze ${career}`);
    return res.json();
  }

  async function compare() {
    if (!careerA.trim() || !careerB.trim()) return;
    setStatus("loading");
    setErrorMsg("");
    try {
      const [a, b] = await Promise.all([fetchOne(careerA.trim()), fetchOne(careerB.trim())]);
      setReportA(a);
      setReportB(b);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message || "Comparison failed.");
      setStatus("error");
    }
  }

  return (
    <div className="compare-panel">
      <div className="compare-header">
        <p className="parse-panel-title">Compare Two Careers</p>
        <button className="secondary-btn" onClick={onClose}>Close</button>
      </div>

      <div className="compare-inputs">
        <input
          className="gap-role-input"
          list="gap-career-options"
          placeholder="First career, e.g. Data Scientist"
          value={careerA}
          onChange={(e) => setCareerA(e.target.value)}
        />
        <input
          className="gap-role-input"
          list="gap-career-options"
          placeholder="Second career, e.g. UX Designer"
          value={careerB}
          onChange={(e) => setCareerB(e.target.value)}
        />
        <datalist id="gap-career-options">
          {careerOptions.map((c) => <option key={c} value={c} />)}
        </datalist>
        <button
          className="primary-btn"
          disabled={!careerA.trim() || !careerB.trim() || status === "loading"}
          onClick={compare}
        >
          {status === "loading" ? "Comparing…" : "Compare"}
        </button>
      </div>

      {status === "error" && <p className="error-msg">{errorMsg}</p>}

      {status === "done" && reportA && reportB && (
        <div className="compare-columns">
          {[reportA, reportB].map((report) => (
            <div className="compare-col" key={report.career}>
              <h3 className="result-career">{report.career}</h3>
              <p className="result-confidence">Readiness: {report.overall_readiness}%</p>
              <div className="result-chart">
                {report.gaps.map((g) => (
                  <div className="bar-row" key={g.trait}>
                    <span className="bar-label">{g.trait}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${(g.resume_score / g.target_score) * 100}%` }} />
                    </div>
                    <span className="bar-pct">{g.resume_score}/{g.target_score}</span>
                  </div>
                ))}
              </div>
              {report.technical_gaps && (
                <p className="dash-hint">
                  {report.technical_gaps.filter((s) => s.resume_has_it).length} /{" "}
                  {report.technical_gaps.length} technical skills matched
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
