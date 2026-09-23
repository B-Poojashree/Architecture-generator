import React from "react";
import { Table } from "react-bootstrap";

export default function WorkflowTable({ workflowJson }) {
  if (!workflowJson || !workflowJson.edges) return null;

  return (
    <Table striped bordered hover size="sm" className="mt-3">
      <thead>
        <tr>
          <th>From task</th>
          <th>To task</th>
        </tr>
      </thead>
      <tbody>
        {workflowJson.edges.map((edge, i) => (
          <tr key={i}>
            <td>{edge.from}</td>
            <td>{edge.to}</td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}
