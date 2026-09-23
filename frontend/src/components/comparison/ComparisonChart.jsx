import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

export default function ComparisonChart({ comparisonSummary }) {
  if (!comparisonSummary || comparisonSummary.length === 0) return null;

  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer>
        <BarChart data={comparisonSummary}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="model" />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Bar dataKey="overall_score" fill="#0d6efd" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
