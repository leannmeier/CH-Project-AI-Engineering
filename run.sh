#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
docker compose up -d --wait redis
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
