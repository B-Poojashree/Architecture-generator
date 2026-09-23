import React from "react";
import { Card, Row, Col } from "react-bootstrap";

export default function SummaryCard({ result }) {
  return (
    <Card className="mb-4 result-summary-card">
      <Card.Body>
        <Row className="align-items-center">
          <Col md={8}>
            <div className="text-muted small text-uppercase fw-semibold letter-spacing">Project description</div>
            <p className="mb-0 mt-2 summary-project-description">{result.project_description}</p>
          </Col>
          <Col md={4} className="text-md-end mt-3 mt-md-0">
            <div className="text-muted small text-uppercase fw-semibold letter-spacing">Best LLM</div>
            <h4 className="text-capitalize mb-3">{result.best_model}</h4>
            <div className="text-muted small text-uppercase fw-semibold letter-spacing">Overall score</div>
            <h4 className="mb-3">{result.overall_score.toFixed(1)}</h4>
            <div className="text-muted small text-uppercase fw-semibold letter-spacing">Validation score</div>
            <h5 className="mb-0">{result.validation_report.validation_score}</h5>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );
}
