# Multi-Agent AI Software Workflow Architecture Generation and Risk Prediction Framework

A multi-agent system (LangGraph + RAG + FAISS + Sentence-BERT + multiple LLMs)
that turns a plain-English software project description into:
- approved functional/non-functional requirements (human-in-the-loop)
- a dependency-aware software development workflow (DAG)
- a risk analysis grounded in real historical JIRA issues
- an automatic benchmark across 5 LLMs, surfacing only the best result

## Structure
- `backend/` - FastAPI + LangGraph + RAG + FAISS pipeline (Python)
- `frontend/` - React + Bootstrap dashboard (JavaScript)

## Quick start

### Backend
```
cd backend
pip install -r requirements.txt --break-system-packages
cp .env.example .env   # fill in your LLM API keys
uvicorn app.main:app --reload --port 8000
```

### Frontend
```
cd frontend
npm install
npm start
```

## Pipeline order
1. `app/preprocessing/preprocessing.py` - clean GitHub + JIRA datasets
2. `app/rag/embedding_utils.py` - generate Sentence-BERT embeddings
3. `app/rag/vector_store.py` - build FAISS indexes
4. `app/rag/rag_utils.py` - retrieval used by agents
5. `app/agents/*.py` - Requirement, Workflow, Risk, Evaluation, Benchmark agents
6. `app/graph/langgraph_workflow.py` - top-level LangGraph orchestration
7. `app/main.py` - FastAPI server exposing `/requirements/*` and `/generate`
