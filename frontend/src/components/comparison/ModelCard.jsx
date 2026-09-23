import React from "react";
import { Card, Badge } from "react-bootstrap";

export default function ModelCard({ model, isBest }) {
  return (
    <Card className={isBest ? "best-model-card" : ""}>
      <Card.Body>
        {isBest && <Badge bg="primary" className="mb-2">Best model</Badge>}
        <Card.Title className="text-capitalize">{model.model}</Card.Title>
        <div className="small text-muted mb-2">Overall score</div>
        <h3>{model.overall_score.toFixed(1)}</h3>
        <hr />
        <div className="small">
          <div>Requirement coverage: {model.requirement_coverage.toFixed(1)}</div>
          <div>Workflow completeness: {model.workflow_completeness.toFixed(1)}</div>
          <div>Workflow correctness: {model.workflow_correctness.toFixed(1)}</div>
          <div>Risk quality: {model.risk_quality.toFixed(1)}</div>
          <div>Response time: {model.response_time_seconds}s</div>
        </div>
      </Card.Body>
    </Card>
  );
}
