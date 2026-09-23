import React from "react";
import { Alert, Row, Col } from "react-bootstrap";
import RiskTable from "../../components/risk/RiskTable";
import RiskChart from "../../components/risk/RiskChart";
import { useAppContext } from "../../context/AppContext";

export default function RiskAnalysis() {
  const { result } = useAppContext();

  if (!result) {
    return <Alert variant="info">Generate a workflow from the Home page first.</Alert>;
  }

  return (
    <div>
      <h2 className="mb-3">Risk analysis</h2>
      <Row>
        <Col md={7}>
          <RiskTable riskReport={result.risk_report} />
        </Col>
        <Col md={5}>
          <RiskChart riskReport={result.risk_report} />
        </Col>
      </Row>
    </div>
  );
}
