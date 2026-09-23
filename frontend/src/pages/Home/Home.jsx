import React, { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Form, Button, ListGroup, InputGroup } from "react-bootstrap";
import Loader from "../../components/common/Loader";
import { useAppContext } from "../../context/AppContext";
import { startRequirements, applyRequirementAction, generateWorkflow } from "../../services/api";

export default function Home() {
  const navigate = useNavigate();
  const {
    projectDescription, setProjectDescription,
    sessionId, setSessionId,
    requirements, setRequirements,
    setResult, loading, setLoading,
  } = useAppContext();

  const [newReqText, setNewReqText] = useState("");
  const [editingRequirement, setEditingRequirement] = useState(null);
  const [error, setError] = useState("");

  const normalizedRequirements = useMemo(() => {
    const functionalRequirements = (requirements.functional_requirements || []).map((req, index) => ({
      bucket: "functional",
      index,
      text: req,
    }));
    const nonFunctionalRequirements = (requirements.non_functional_requirements || []).map((req, index) => ({
      bucket: "non_functional",
      index,
      text: req,
    }));

    return [...functionalRequirements, ...nonFunctionalRequirements];
  }, [requirements]);

  const handleGenerateRequirements = async () => {
    if (!projectDescription.trim()) {
      setError("Enter a project description first.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const data = await startRequirements(projectDescription);
      setSessionId(data.session_id);
      setRequirements(data);
    } catch (err) {
      setError("Couldn't reach the backend. Check the API is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleAction = async (action) => {
    try {
      const data = await applyRequirementAction(sessionId, action);
      setRequirements(data);
    } catch (err) {
      setError("Could not update requirements. Please try again.");
    }
  };

  const handleSaveEdit = async (bucket, index) => {
    const nextText = editingRequirement?.text?.trim();
    if (!nextText) return;
    await handleAction({ type: "modify", bucket, index, text: nextText });
    setEditingRequirement(null);
  };

  const handleApproveAndGenerate = async () => {
    setLoading(true);
    setError("");
    try {
      await applyRequirementAction(sessionId, { type: "approve" });
      const generated = await generateWorkflow(sessionId);
      setResult(generated);
      navigate("/workflow");
    } catch (err) {
      setError("Pipeline run failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="py-2">
      <div className="hero-panel mb-4">
        <div className="brand-badge">Workflow intelligence</div>
        <h2 className="section-header">Describe your project</h2>
        <p className="section-subtitle mb-0">Turn a project brief into structured requirements, intelligent workflow stages, risk analysis, and delivery planning.</p>

        <div className="metric-strip">
          <div className="metric-pill">
            <strong>4</strong>
            <span>Core outputs</span>
          </div>
          <div className="metric-pill">
            <strong>Risk-aware</strong>
            <span>Decision support</span>
          </div>
          <div className="metric-pill">
            <strong>Fast</strong>
            <span>Workflow generation</span>
          </div>
        </div>
      </div>

      <div className="card p-4 mb-4">
        <Form.Group className="mb-3">
          <Form.Control
            as="textarea"
            rows={5}
            className="modern-input"
            placeholder="e.g. A cloud-based inventory management system with real-time stock tracking and supplier integration"
            value={projectDescription}
            onChange={(e) => setProjectDescription(e.target.value)}
          />
        </Form.Group>

        {error && <div className="alert alert-warning mb-3">{error}</div>}

        <Button className="primary-action" onClick={handleGenerateRequirements} disabled={loading}>
          Generate requirements
        </Button>
      </div>

      {loading && <Loader label="Working..." />}

      {!loading && sessionId && (
        <div className="mt-4">
          <h5>Requirements <span className="badge bg-secondary">{normalizedRequirements.length}</span></h5>
          <ListGroup className="mb-3">
            {normalizedRequirements.length === 0 ? (
              <ListGroup.Item className="text-muted">Generate requirements to begin editing.</ListGroup.Item>
            ) : (
              normalizedRequirements.map((item) => (
                <ListGroup.Item key={`${item.bucket}-${item.index}`} className="d-flex justify-content-between align-items-center gap-3">
                  {editingRequirement?.bucket === item.bucket && editingRequirement.index === item.index ? (
                    <InputGroup>
                      <Form.Control
                        autoFocus
                        value={editingRequirement.text}
                        onChange={(e) => setEditingRequirement({ ...editingRequirement, text: e.target.value })}
                      />
                      <Button variant="outline-primary" onClick={() => handleSaveEdit(item.bucket, item.index)}>
                        Save
                      </Button>
                      <Button variant="outline-secondary" onClick={() => setEditingRequirement(null)}>
                        Cancel
                      </Button>
                    </InputGroup>
                  ) : (
                    <>
                      <span className="me-3 requirement-text">{item.text}</span>
                      <div className="d-flex gap-2">
                        <Button size="sm" variant="outline-secondary" onClick={() => setEditingRequirement({ bucket: item.bucket, index: item.index, text: item.text })}>
                          Edit
                        </Button>
                        <Button size="sm" variant="outline-danger" onClick={() => handleAction({ type: "delete", bucket: item.bucket, index: item.index })}>
                          Delete
                        </Button>
                      </div>
                    </>
                  )}
                </ListGroup.Item>
              ))
            )}
          </ListGroup>

          <Form.Group className="mb-3">
            <InputGroup>
              <Form.Control
                placeholder="Add a new requirement"
                value={newReqText}
                onChange={(e) => setNewReqText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    if (!newReqText.trim()) return;
                    handleAction({ type: "add", bucket: "functional", text: newReqText.trim() });
                    setNewReqText("");
                  }
                }}
              />
              <Button
                variant="outline-primary"
                onClick={() => {
                  if (!newReqText.trim()) return;
                  handleAction({ type: "add", bucket: "functional", text: newReqText.trim() });
                  setNewReqText("");
                }}
              >
                Add
              </Button>
            </InputGroup>
          </Form.Group>

          <Button className="primary-action" onClick={handleApproveAndGenerate}>
            Approve requirements and generate workflow
          </Button>
        </div>
      )}
    </div>
  );
}
