import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export const getBackendAssetUrl = (assetPath) => {
  if (!assetPath) return "";
  if (/^https?:\/\//i.test(assetPath)) return assetPath;
  return `${API_BASE_URL}${assetPath.startsWith("/") ? assetPath : `/${assetPath}`}`;
};

export const startRequirements = (projectDescription) =>
  api.post("/requirements/start", { project_description: projectDescription }).then((res) => res.data);

export const applyRequirementAction = (sessionId, action) =>
  api
    .post("/requirements/action", {
      session_id: sessionId,
      type: action.type,
      bucket: action.bucket,
      text: action.text,
      index: action.index,
    })
    .then((res) => res.data);

export const generateWorkflow = (sessionId) =>
  api.post(`/generate?session_id=${sessionId}`).then((res) => res.data);

export default api;
