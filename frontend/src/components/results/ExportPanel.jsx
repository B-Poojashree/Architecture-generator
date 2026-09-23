import React from "react";
import { Button, ButtonGroup } from "react-bootstrap";
import { getBackendAssetUrl } from "../../services/api";

export default function ExportPanel({ result }) {
  const exportJson = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "workflow_result.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportImage = () => {
    if (!result.diagram_path) return;
    const a = document.createElement("a");
    a.href = getBackendAssetUrl(result.diagram_path);
    a.download = "workflow_diagram.png";
    a.click();
  };

  const exportPdf = () => {
    window.print();
  };

  return (
    <ButtonGroup>
      <Button variant="outline-secondary" onClick={exportJson}>Export JSON</Button>
      <Button variant="outline-secondary" onClick={exportImage}>Export workflow image</Button>
      <Button variant="outline-secondary" onClick={exportPdf}>Export PDF</Button>
    </ButtonGroup>
  );
}
