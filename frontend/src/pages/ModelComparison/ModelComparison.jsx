import React from "react";
import { Alert, Row, Col } from "react-bootstrap";
import ModelCard from "../../components/comparison/ModelCard";
import ComparisonChart from "../../components/comparison/ComparisonChart";
import WorkflowDiagram from "../../components/workflow/WorkflowDiagram";
import { useAppContext } from "../../context/AppContext";

export default function ModelComparison() {
  const { result } = useAppContext();

  if (!result) {
    return <Alert variant="info">Generate a workflow from the Home page first.</Alert>;
  }

  return (
    <div>
      <h2 className="mb-3">Model comparison</h2>
      <p className="text-muted">
        Every run automatically benchmarks Gemini, DeepSeek, Llama, Qwen, and Mistral.
        Only the best-scoring model's detailed workflow is shown below.
      </p>

      <ComparisonChart comparisonSummary={result.comparison_summary} />

      <Row className="g-3 mt-2">
        {result.comparison_summary.map((model) => (
          <Col md={4} key={model.model}>
            <ModelCard model={model} isBest={model.model === result.best_model} />
          </Col>
        ))}
      </Row>

      <h5 className="mt-4">Detailed workflow ({result.best_model})</h5>
      <WorkflowDiagram workflowJson={result.workflow_json} />
    </div>
  );
}
