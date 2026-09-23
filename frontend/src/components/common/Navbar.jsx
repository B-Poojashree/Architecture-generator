import React from "react";
import { Navbar as BSNavbar, Nav, Container } from "react-bootstrap";
import { Link, NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <BSNavbar bg="white" expand="lg" className="border-bottom shadow-sm">
      <Container>
        <BSNavbar.Brand as={Link} to="/" className="fw-semibold d-flex align-items-center gap-2">
          <span className="brand-badge">AI</span>
          <span>Workflow Risk Framework</span>
        </BSNavbar.Brand>
        <BSNavbar.Toggle aria-controls="main-nav" />
        <BSNavbar.Collapse id="main-nav">
          <Nav className="ms-auto">
            <Nav.Link as={NavLink} to="/">Home</Nav.Link>
            <Nav.Link as={NavLink} to="/workflow">Workflow</Nav.Link>
            <Nav.Link as={NavLink} to="/risk-analysis">Risk Analysis</Nav.Link>
            <Nav.Link as={NavLink} to="/task-schedule">Task Schedule</Nav.Link>
            <Nav.Link as={NavLink} to="/model-comparison">Model Comparison</Nav.Link>
            <Nav.Link as={NavLink} to="/results">Results</Nav.Link>
          </Nav>
        </BSNavbar.Collapse>
      </Container>
    </BSNavbar>
  );
}
