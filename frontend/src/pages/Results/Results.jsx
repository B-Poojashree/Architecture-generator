import React from "react";
import { Alert, Row, Col } from "react-bootstrap";
import SummaryCard from "../../components/results/SummaryCard";
import ExportPanel from "../../components/results/ExportPanel";
import WorkflowDiagram from "../../components/workflow/WorkflowDiagram";
import RiskTable from "../../components/risk/RiskTable";
import { useAppContext } from "../../context/AppContext";

export default function Results() {
  const { result } = useAppContext();

  if (!result) {
    return <Alert variant="info">Generate a workflow from the Home page first.</Alert>;
  }

  return (
    <div className="results-page">
      <div className="results-header d-flex justify-content-between align-items-center mb-3">
        <div>
          <h2 className="results-title mb-1">Results dashboard</h2>
        </div>
        <ExportPanel result={result} />
      </div>

      <SummaryCard result={result} />

      <Row className="results-content-grid">
        <Col md={7}>
          <h5 className="results-section-title">Workflow architecture</h5>
          <WorkflowDiagram workflowJson={result.workflow_json} diagramPath={result.diagram_path} />
        </Col>
        <Col md={5}>
          <h5 className="results-section-title">Risk analysis and mitigation</h5>
          <RiskTable riskReport={result.risk_report} />
        </Col>
      </Row>

      <div className="results-validation mt-4">
        <h5 className="results-section-title">Validation report</h5>
        <ul className="mb-0">
          <li>Workflow completeness: {result.validation_report.workflow_completeness}</li>
          <li>Risk coverage: {result.validation_report.risk_coverage}</li>
          <li>Duplicate tasks: {result.validation_report.duplicate_tasks.join(", ") || "None"}</li>
          <li>Isolated nodes: {result.validation_report.isolated_nodes.join(", ") || "None"}</li>
        </ul>
      </div>
    </div>
  );
}
