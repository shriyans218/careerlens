import React, { useEffect, useState } from "react";

export default function CohortAnalytics({ apiBase }) {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    fetch(`${apiBase}/api/cohort-stats`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((json) => {
        setData(json);
        setStatus("done");
      })
      .catch(() => setStatus("error"));
  }, [apiBase]);

  if (status === "loading") return <p className="dash-hint">Loading cohort data…</p>;
  if (status === "error" || !data) return <p className="dash-hint">Cohort data unavailable.</p>;
  if (data.total === 0) return <p className="dash-hint">No predictions logged yet.</p>;

  const maxCount = Math.max(...data.top_careers.map((c) => c.count));

  return (
    <div className="cohort-panel">
      <p className="dash-hint">
        {data.total} predictions so far · {data.by_source.resume || 0} from resumes,{" "}
        {data.by_source.assessment || 0} from self-assessment
      </p>
      <div className="result-chart">
        {data.top_careers.map((c) => (
          <div className="bar-row" key={c.career}>
            <span className="bar-label">{c.career}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(c.count / maxCount) * 100}%` }} />
            </div>
            <span className="bar-pct">{c.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
