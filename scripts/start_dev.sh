#!/usr/bin/env bash
# Starts backend and frontend dev servers together.
set -e
cd "$(dirname "$0")/../backend"
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

cd "../frontend"
npm start &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
