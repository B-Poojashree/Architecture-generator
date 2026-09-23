#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../backend"
python3 -m app.preprocessing.preprocessing
