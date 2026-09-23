import React from "react";
import { Alert } from "react-bootstrap";
import WorkflowSchedule from "../../components/workflow/WorkflowSchedule";
import { useAppContext } from "../../context/AppContext";

export default function TaskSchedule() {
  const { result } = useAppContext();

  if (!result) {
    return <Alert variant="info">Generate a workflow from the Home page first.</Alert>;
  }

  return (
    <div>
      <h2 className="mb-3">Task schedule</h2>
      <p className="text-muted">Weekly delivery plan generated from the current workflow.</p>
      <WorkflowSchedule workflowJson={result.workflow_json} />
    </div>
  );
}
