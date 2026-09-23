import React from "react";

export default function WorkflowDiagram({ workflowJson }) {
  if (!workflowJson || !workflowJson.nodes || workflowJson.nodes.length === 0) {
    return <p className="text-muted">No workflow generated yet.</p>;
  }

  return (
    <div className="workflow-diagram-shell">
      <div className="workflow-diagram-caption text-muted mb-3">Workflow steps</div>
      <div className="d-flex flex-column align-items-center gap-3 workflow-flow-vertical">
        {workflowJson.nodes.map((node, index) => (
          <React.Fragment key={node.id || `workflow-node-${index}`}>
            <div className="workflow-node-wrapper">
              <div className="workflow-node">
                <div className="workflow-node-title">{node.name}</div>
              </div>
            </div>
            {index < workflowJson.nodes.length - 1 && <div className="workflow-arrow-vertical">↓</div>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
