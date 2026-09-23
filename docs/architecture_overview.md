# Architecture overview

Input (project description)
  -> RAG retrieval (Sentence-BERT + FAISS: GitHub + JIRA indexes)
  -> Requirement Agent (LLM + human-in-the-loop approval)
  -> Workflow Agent (LLM -> NetworkX DAG -> Graphviz diagram)
  -> Risk Agent (RAG-grounded risk prediction per workflow stage)
  -> Evaluation Agent (structural + coverage validation, 0-100 score)
  -> Benchmark Agent (fans out across Gemini, DeepSeek, Llama, Qwen, Mistral;
     scores each on requirement coverage, workflow completeness/correctness,
     risk quality, response time; selects the single best output)
  -> FastAPI /generate endpoint
  -> React dashboard (Workflow, Risk Analysis, Model Comparison, Results)

All five agents are wired together as one LangGraph StateGraph
(see backend/app/graph/langgraph_workflow.py), giving the pipeline
shared state, conditional routing, and human-in-the-loop pausing.
