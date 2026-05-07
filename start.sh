#!/bin/bash
# ZeaCares API — safe start script
# Usage: bash start.sh
# Stops any old instance, frees port 8000, then starts fresh via PM2.

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== ZeaCares API Startup ==="
echo "Project root: $PROJECT_ROOT"

# ── 1. Kill anything on port 8000 ────────────────────────────────────────────
PORT=8000
PID=$(lsof -ti tcp:$PORT 2>/dev/null || true)
if [ -n "$PID" ]; then
  echo "[1/5] Killing existing process on port $PORT (PID $PID)..."
  kill -9 $PID 2>/dev/null || true
  sleep 1
else
  echo "[1/5] Port $PORT is free."
fi

# ── 2. Stop any PM2 instance with this name ───────────────────────────────────
echo "[2/5] Stopping old PM2 instances..."
pm2 delete zeacares-api 2>/dev/null || true
sleep 1

# ── 3. Create required directories ───────────────────────────────────────────
echo "[3/5] Ensuring directories exist..."
mkdir -p "$PROJECT_ROOT/results"
mkdir -p "$PROJECT_ROOT/model_cache"
mkdir -p "$PROJECT_ROOT/logs"

# ── 4. Activate venv ──────────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  echo "[4/5] Activating virtualenv..."
  source "$PROJECT_ROOT/venv/bin/activate"
else
  echo "[4/5] WARNING: venv not found at $PROJECT_ROOT/venv — using system Python."
fi

# ── 5. Start via PM2 ─────────────────────────────────────────────────────────
echo "[5/5] Starting ZeaCares API via PM2..."
pm2 start ecosystem.config.js

echo ""
echo "=== ZeaCares API Started ==="
echo "API:      http://0.0.0.0:8000"
echo "Docs:     http://0.0.0.0:8000/docs"
echo "Logs:     pm2 logs zeacares-api"
echo "Status:   pm2 status"
echo "Stop:     pm2 stop zeacares-api"
