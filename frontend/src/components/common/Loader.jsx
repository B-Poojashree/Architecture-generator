import React from "react";
import { Spinner } from "react-bootstrap";

export default function Loader({ label = "Generating..." }) {
  return (
    <div className="loading-spinner-wrap flex-column align-items-center">
      <Spinner animation="border" role="status" variant="primary" />
      <div className="mt-2 text-muted">{label}</div>
    </div>
  );
}
