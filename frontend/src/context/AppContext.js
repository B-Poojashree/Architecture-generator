import React, { createContext, useContext, useState } from "react";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [sessionId, setSessionId] = useState(null);
  const [projectDescription, setProjectDescription] = useState("");
  const [requirements, setRequirements] = useState({ functional_requirements: [], non_functional_requirements: [] });
  const [result, setResult] = useState(null); // full GenerateResponse from /generate
  const [loading, setLoading] = useState(false);

  const value = {
    sessionId,
    setSessionId,
    projectDescription,
    setProjectDescription,
    requirements,
    setRequirements,
    result,
    setResult,
    loading,
    setLoading,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within an AppProvider");
  return ctx;
}
