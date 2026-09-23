#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../backend"
python3 -m app.rag.embedding_utils
python3 -m app.rag.vector_store
