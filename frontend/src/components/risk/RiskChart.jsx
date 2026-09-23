import React from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";

const COLORS = { High: "#dc3545", Medium: "#ffc107", Low: "#198754" };

export default function RiskChart({ riskReport }) {
  if (!riskReport || riskReport.length === 0) return null;

  const counts = riskReport.reduce((acc, entry) => {
    acc[entry.severity] = (acc[entry.severity] || 0) + 1;
    return acc;
  }, {});

  const data = Object.entries(counts).map(([severity, count]) => ({ name: severity, value: count }));

  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" outerRadius={90} label>
            {data.map((entry, i) => (
              <Cell key={i} fill={COLORS[entry.name] || "#6c757d"} />
            ))}
          </Pie>
          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
