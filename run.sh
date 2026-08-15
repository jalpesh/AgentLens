#!/usr/bin/env bash
# AgentLens — one-file local setup/run script.
#
# Usage:
#   ./run.sh setup            create .venv, install agentlens-cli + dev deps
#   ./run.sh ingest [args]    parse local agent history into the local db
#   ./run.sh start [--port N] start the dashboard (default port 7878)
#   ./run.sh stop             stop the dashboard if this script started it
#   ./run.sh restart          stop, then start
#   ./run.sh status           is a server running, and on what port/pid
#   ./run.sh test             ruff + pytest
#   ./run.sh                  setup (if needed) + ingest + start
#
# Uses `python3` explicitly throughout — never bare `python`, which doesn't
# exist on a lot of machines (notably plain macOS).
#
# Safe to re-run: `start` checks for an already-running instance (by PID file
# AND by port) before launching a second one, `setup` skips reinstalling into
# an already-good .venv, and every step is a no-op if its result already
# exists and is current.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
DB="agentlens.db"
PORT="${AGENTLENS_PORT:-7878}"
PIDFILE=".agentlens.pid"
LOGFILE=".agentlens.log"

PY="$VENV/bin/python3"  # a plain variable, not a function — nohup below
                        # execs it directly and cannot see shell functions

# ---------------------------------------------------------------------------

setup() {
  if [ -x "$VENV/bin/python3" ] && "$PY" -c "import agentlens" 2>/dev/null; then
    echo "[setup] .venv already has agentlens installed — skipping. (Use 'rm -rf $VENV' to force a clean rebuild.)"
    return 0
  fi
  echo "[setup] creating virtualenv at $VENV ..."
  python3 -m venv "$VENV"
  echo "[setup] installing agentlens-cli (editable) + dev deps ..."
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -e ".[dev]"
  echo "[setup] done. $("$PY" -m agentlens --version 2>/dev/null || echo agentlens) ready."
}

ingest() {
  need_venv
  echo "[ingest] scanning local agent history into $DB ..."
  "$PY" -m agentlens ingest --db "$DB" "$@"
}

running_pid() {
  # Prints "pid port" (space-separated) if this script's own server is still
  # running, else nothing. Cross-checks the pidfile against `ps` so a stale
  # pidfile left over from a killed/crashed process doesn't look like a live
  # server.
  #
  # The pidfile stores BOTH the pid and the port it was launched with — not
  # just the pid. Storing only the pid was a real bug found while testing
  # this script: `start` would report an already-running instance as being
  # on whatever port was *currently requested*, even if that instance was
  # actually launched on a different port. Two fields, always read and
  # written together, make that class of mismatch structurally impossible.
  if [ -f "$PIDFILE" ]; then
    local pid port
    read -r pid port < "$PIDFILE" 2>/dev/null || true
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid $port"
      return 0
    fi
  fi
  return 1
}

port_owner_pid() {
  # Best-effort: is *something* (possibly not us — another AgentLens
  # instance, or an unrelated process) already listening on $PORT?
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null | head -1
  elif command -v fuser >/dev/null 2>&1; then
    fuser "$PORT/tcp" 2>/dev/null | awk '{print $1}'
  fi
}

need_venv() {
  if [ ! -x "$VENV/bin/python3" ]; then
    echo "[error] no .venv found — run './run.sh setup' first (or just './run.sh' to do everything)." >&2
    exit 1
  fi
}

start() {
  need_venv
  if info="$(running_pid)"; then
    local run_pid="${info%% *}" run_port="${info#* }"
    if [ "$run_port" = "$PORT" ]; then
      echo "[start] already running — pid $run_pid on port $run_port (this script started it). Not launching a second instance."
    else
      echo "[start] already running — pid $run_pid, but on port $run_port, not the $PORT you asked for."
      echo "        This script tracks one managed instance at a time. Stop it first if you want to"
      echo "        switch ports: './run.sh stop' then 'AGENTLENS_PORT=$PORT ./run.sh start'."
    fi
    echo "        Open http://127.0.0.1:$run_port — or './run.sh stop' first if you want to restart."
    return 0
  fi
  if owner="$(port_owner_pid)"; then
    echo "[start] port $PORT is already in use by pid $owner (not started by this script)."
    echo "        Either stop that process yourself, or run with a different port:"
    echo "          AGENTLENS_PORT=7879 ./run.sh start"
    return 1
  fi
  echo "[start] launching dashboard on http://127.0.0.1:$PORT ..."
  nohup "$PY" -m agentlens serve --db "$DB" --port "$PORT" > "$LOGFILE" 2>&1 &
  local newpid=$!
  disown "$newpid" 2>/dev/null || true
  echo "$newpid $PORT" > "$PIDFILE"
  sleep 1
  if kill -0 "$newpid" 2>/dev/null; then
    echo "[start] up — pid $newpid. Logs: $LOGFILE. Stop with './run.sh stop'."
  else
    echo "[start] failed to start — check $LOGFILE" >&2
    rm -f "$PIDFILE"
    exit 1
  fi
}

stop() {
  if info="$(running_pid)"; then
    local run_pid="${info%% *}" run_port="${info#* }"
    echo "[stop] stopping pid $run_pid (port $run_port) ..."
    kill "$run_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$run_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$run_pid" 2>/dev/null; then
      echo "[stop] still running after 5s, forcing (kill -9) ..."
      kill -9 "$run_pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    echo "[stop] stopped."
  else
    echo "[stop] nothing running (no valid pidfile). If something is still bound to port $PORT,"
    echo "       it wasn't started by this script — find and stop it manually, e.g.:"
    echo "         lsof -ti tcp:$PORT | xargs kill"
  fi
}

status() {
  if info="$(running_pid)"; then
    local run_pid="${info%% *}" run_port="${info#* }"
    echo "running — pid $run_pid, http://127.0.0.1:$run_port (started by this script)"
  elif owner="$(port_owner_pid)"; then
    echo "port $PORT is in use by pid $owner, but not by a server this script started"
  else
    echo "not running"
  fi
}

test_() {
  need_venv
  "$PY" -m ruff check src tests fixtures
  "$PY" -m pytest -q
}

# ---------------------------------------------------------------------------

cmd="${1:-}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  setup)   setup ;;
  ingest)  ingest "$@" ;;
  start)
    while [ $# -gt 0 ]; do
      case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    start
    ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  test)    test_ ;;
  ""|run)
    setup
    ingest
    start
    ;;
  *)
    echo "unknown command: $cmd" >&2
    echo "usage: ./run.sh {setup|ingest|start|stop|restart|status|test}" >&2
    exit 1
    ;;
esac
