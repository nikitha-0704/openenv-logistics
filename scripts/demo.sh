#!/usr/bin/env bash
# Push-button demo for the Round 2 video / mini-blog.
#
# What this does:
#   1. Boots the OpenEnv server with ADVERSARIAL=1 in the background.
#   2. Waits for /docs to be reachable.
#   3. Runs the multi-agent harness end-to-end (Dispatcher + Drivers + bus + re-plan).
#   4. Prints the [PLAN] / [MSG] / [STEP] / [REPLAN] / [END] log to stdout.
#   5. Stops the server cleanly on exit.
#
# Requirements:
#   - OPENAI_API_KEY *or* HF_TOKEN exported (the orchestrator needs an LLM).
#   - Optional: OPENENV_BASE_URL to point at a remote env (e.g. an HF Space).
#     If unset, we boot a local server on PORT (default 7860).
#
# Usage:
#   bash scripts/demo.sh                 # boots local server + runs demo
#   bash scripts/demo.sh --no-server     # assumes server already running

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-7860}"
HOST_URL="${OPENENV_BASE_URL:-http://localhost:${PORT}}"
START_SERVER=1
SERVER_PID=""

for arg in "$@"; do
  case "$arg" in
    --no-server) START_SERVER=0 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "${OPENAI_API_KEY:-}" && -z "${HF_TOKEN:-}" && -z "${API_KEY:-}" ]]; then
  echo "ERROR: set OPENAI_API_KEY or HF_TOKEN (or API_KEY for LiteLLM) before running." >&2
  exit 2
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "[demo] stopping local env server (pid=$SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$START_SERVER" -eq 1 && -z "${OPENENV_BASE_URL:-}" ]]; then
  echo "[demo] starting env server on $HOST_URL with ADVERSARIAL=1"
  ADVERSARIAL=1 PORT="$PORT" python -m server.app >/tmp/openenv-demo.log 2>&1 &
  SERVER_PID=$!

  echo "[demo] waiting for $HOST_URL/docs ..."
  for _ in $(seq 1 40); do
    if curl -sf "$HOST_URL/docs" >/dev/null 2>&1; then
      echo "[demo] env server is up (pid=$SERVER_PID)"
      break
    fi
    sleep 0.5
  done

  if ! curl -sf "$HOST_URL/docs" >/dev/null 2>&1; then
    echo "[demo] env server failed to come up. Last 30 log lines:" >&2
    tail -n 30 /tmp/openenv-demo.log >&2 || true
    exit 1
  fi
fi

export OPENENV_BASE_URL="$HOST_URL"

echo "[demo] running multi-agent harness against $OPENENV_BASE_URL"
echo "------------------------------------------------------------"
python inference_multi.py
echo "------------------------------------------------------------"
echo "[demo] done."
