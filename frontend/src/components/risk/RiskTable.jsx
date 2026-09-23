import React from "react";

const severityClass = { High: "risk-high", Medium: "risk-medium", Low: "risk-low" };

export default function RiskTable({ riskReport }) {
  if (!riskReport || riskReport.length === 0) {
    return <p className="text-muted">No risk analysis available yet.</p>;
  }

  return (
    <div className="d-flex flex-column gap-2">
      {riskReport.map((entry, i) => (
        <div key={i} className={`card p-3 ${severityClass[entry.severity] || ""}`}>
          <div className="d-flex justify-content-between">
            <strong>{entry.stage}</strong>
            <span className="badge bg-secondary">{entry.severity}</span>
          </div>
          <div className="small text-muted mt-1">{entry.risk}</div>
          <div className="small mt-1">Mitigation: {entry.mitigation}</div>
        </div>
      ))}
    </div>
  );
}
