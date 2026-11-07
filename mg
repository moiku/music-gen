#!/usr/bin/env bash
# music-gen server control.
#
#   ./mg start     start the gallery server in the background
#   ./mg stop      stop the server and the ACE worker (frees model memory)
#   ./mg restart
#   ./mg status
#   ./mg log       follow the server log
#
# Host/port come from musicgen.local.env (not in git), e.g.
#   MUSICGEN_HOST=100.64.0.5
#   MUSICGEN_PORT=8890
# Defaults: 127.0.0.1:8890.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f musicgen.local.env ]] && set -a && source musicgen.local.env && set +a
HOST="${MUSICGEN_HOST:-127.0.0.1}"
PORT="${MUSICGEN_PORT:-8890}"
PIDFILE="output/server.pid"
LOG="output/server.log"
URL="http://${HOST}:${PORT}/"
mkdir -p output

server_pid() {
    [[ -f "$PIDFILE" ]] || return 1
    local pid; pid="$(cat "$PIDFILE")"
    kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= | grep -q "musicgen.server" && echo "$pid"
}

worker_pids() { pgrep -f "python -m musicgen.worker" || true; }

start() {
    if pid="$(server_pid)"; then echo "すでに起動しています（PID ${pid}）: $URL"; return; fi
    uv sync -q
    nohup .venv/bin/python -m musicgen.server --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    for _ in $(seq 1 30); do
        if curl -s -o /dev/null "$URL"; then echo "起動しました: $URL"; return; fi
        sleep 0.5
    done
    echo "起動を確認できませんでした。ログ: $LOG" >&2; tail -5 "$LOG" >&2; exit 1
}

stop() {
    local stopped=0
    if pid="$(server_pid)"; then kill "$pid"; stopped=1; fi
    for w in $(worker_pids); do kill "$w"; stopped=1; done
    rm -f "$PIDFILE"
    if [[ $stopped == 1 ]]; then
        sleep 1
        echo "停止しました（AIモデルのメモリも解放）"
    else
        echo "起動していません"
    fi
}

status() {
    if pid="$(server_pid)"; then echo "サーバー: 起動中（PID ${pid}） $URL"; else echo "サーバー: 停止"; fi
    if [[ -n "$(worker_pids)" ]]; then echo "AIモデル: 読み込み済み（作成ジョブ用）"; else echo "AIモデル: 停止"; fi
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    log) tail -f "$LOG" ;;
    *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
