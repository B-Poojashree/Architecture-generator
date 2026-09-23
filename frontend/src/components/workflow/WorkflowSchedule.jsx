import React, { useState } from "react";

export default function WorkflowSchedule({ workflowJson }) {
  const [weeks, setWeeks] = useState(4);
  const [startDate, setStartDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [schedule, setSchedule] = useState(null);

  if (!workflowJson || !workflowJson.nodes) return null;

  const tasks = workflowJson.nodes.map((n) => n.name || n.id);

  function generateSchedule(e) {
    e.preventDefault();
    const w = Math.max(1, parseInt(weeks, 10) || 1);
    const out = Array.from({ length: w }, (_, index) => ({ week: index + 1, tasks: [] }));

    if (tasks.length === 0) {
      setSchedule(out);
      return;
    }

    tasks.forEach((task, index) => {
      const weekIndex = index % w;
      out[weekIndex].tasks.push(task);
    });

    setSchedule(out);
  }

  return (
    <div className="mt-4">
      <h5>Task schedule</h5>
      <form className="row g-2 align-items-end" onSubmit={generateSchedule}>
        <div className="col-auto">
          <label className="form-label small">Start date</label>
          <input type="date" className="form-control" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
        </div>
        <div className="col-auto">
          <label className="form-label small">Weeks</label>
          <input type="number" min="1" className="form-control" value={weeks} onChange={(e) => setWeeks(e.target.value)} />
        </div>
        <div className="col-auto">
          <button className="btn btn-primary">Generate weekly schedule</button>
        </div>
      </form>

      {schedule && (
        <div className="mt-3">
          {schedule.map((s) => (
            <div key={s.week} className="card mb-2">
              <div className="card-body">
                <h6 className="card-title">Week {s.week}</h6>
                {s.tasks.length ? (
                  <ul className="mb-0">
                    {s.tasks.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-muted mb-0">No tasks assigned</p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
