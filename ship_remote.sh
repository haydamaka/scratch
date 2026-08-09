#!/usr/bin/env bash
#
# Remote-side half of ship_to_host.py. Runs ON the target host.
#
# Everything that has to share one shell lives here: the exports, the venv, and
# the pip install and uvicorn launch that depend on both. Splitting these across
# separate ssh invocations silently loses them — `export` and `source` do not
# outlive the shell that ran them.
#
#     ./ship_remote.sh unpack [PATH...]   # extract the bundle; each PATH given
#                                         # is wiped first, so it is replaced
#                                         # rather than merged over
#     ./ship_remote.sh setup              # venv + pip.conf + requirements
#     ./ship_remote.sh start              # nohup uvicorn, detached
#     ./ship_remote.sh stop               # stop it (kills the process group)
#     ./ship_remote.sh restart
#     ./ship_remote.sh status
#     ./ship_remote.sh all                # unpack + setup
#
# Run configuration: module uvicorn, app.main:app, PYTHONUNBUFFERED=1, ENV=uat.
# It binds 0.0.0.0 rather than 127.0.0.1, because a loopback-only bind is
# unreachable from anywhere but the box itself.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — edit here rather than passing flags every time.
# ---------------------------------------------------------------------------
# Site-specific: export these rather than editing them in. Only `setup`
# needs them, so they are validated there, not here — an unset host must
# not stop `start`/`stop`/`status` from working.
ARTIFACTORY_USER="${ARTIFACTORY_USER:-}"
ARTIFACTORY_HOST="${ARTIFACTORY_HOST:-}"
ARTIFACTORY_INDEX="${ARTIFACTORY_INDEX:-/artifactory/api/pypi/pypi-dev/simple}"

# Paste the Artifactory reference token (the `cmVmdGtu…` string from your
# pip.conf) between the quotes to stop being asked for it.
#
# Left empty on purpose: this file sits inside a git working tree, so a token
# written here is one `git add -A` from being committed and pushed. If you do
# fill it in, add ship_remote.sh to .gitignore first. Until then it is read
# from $ARTIFACTORY_TOKEN or prompted for.
ARTIFACTORY_TOKEN="${ARTIFACTORY_TOKEN:-}"

APP_MODULE="${APP_MODULE:-app.main:app}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-}"                 # non-empty to pass --reload
PYTHON="${PYTHON:-python3.11}"

PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BUNDLE="${BUNDLE:-$PROJECT_DIR/_bundle.zip}"

# Derive the runtime dir rather than hardcoding one uid, so this works for
# whoever runs it. XDG_RUNTIME_DIR is usually already set by the login.
: "${HOME:=/tmp/$(id -un)}"
XDG="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
VENV_DIR="${VENV_DIR:-$XDG/pyvenv}"

export HOME XDG_RUNTIME_DIR="$XDG"
export ENV="${ENV:-uat}"

LOG_DIR="$PROJECT_DIR/logs/std"
LOG_FILE="$LOG_DIR/uvicorn.log"
PID_FILE="$LOG_DIR/uvicorn.pid"

log() { printf '[remote] %s\n' "$*" >&2; }
die() { printf '[remote] ERROR: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

# Percent-encode for the userinfo part of a URL. Reference tokens are base64 and
# may contain '/' or '+', either of which silently corrupts the index URL.
urlencode() {
    "$PYTHON" -c 'import sys,urllib.parse; sys.stdout.write(urllib.parse.quote(sys.stdin.read().strip(), safe=""))' 2>/dev/null \
        || python3 -c 'import sys,urllib.parse; sys.stdout.write(urllib.parse.quote(sys.stdin.read().strip(), safe=""))'
}

resolve_token() {
    [ -n "$ARTIFACTORY_TOKEN" ] && return 0

    # Never argv: it is world-readable through `ps`.
    if [ -t 0 ]; then
        read -rsp "Artifactory token for $ARTIFACTORY_USER: " ARTIFACTORY_TOKEN
        printf '\n' >&2
    else
        read -r ARTIFACTORY_TOKEN || true
    fi
    [ -n "$ARTIFACTORY_TOKEN" ] || die "no Artifactory token (set it in the defaults block, \$ARTIFACTORY_TOKEN, or on stdin)"
}

write_pip_conf() {
    local encoded conf="$HOME/.config/pip/pip.conf"
    encoded=$(printf '%s' "$ARTIFACTORY_TOKEN" | urlencode)

    mkdir -p "$(dirname "$conf")"
    # Create at 600 *before* writing, so the token is never briefly world-readable.
    umask 077
    cat > "$conf" <<EOF
[global]
index-url = https://${ARTIFACTORY_USER}:${encoded}@${ARTIFACTORY_HOST}${ARTIFACTORY_INDEX}
trusted-host = ${ARTIFACTORY_HOST}
EOF
    chmod 600 "$conf"
    log "wrote $conf (host=${ARTIFACTORY_HOST}, token redacted)"
}

# --------------------------------------------------------------------------
# unpack
# --------------------------------------------------------------------------

# Refuse anything that would escape PROJECT_DIR or wipe the whole tree. This is
# an `rm -rf` target built from an argument, so it gets checked properly.
assert_safe_target() {
    local target="$1"
    case "$target" in
        /*|.|./|""|*..*) die "refusing to replace unsafe path: '${target}'" ;;
    esac
}

do_unpack() {
    [ -f "$BUNDLE" ] || die "bundle not found: $BUNDLE"
    mkdir -p "$PROJECT_DIR"

    # Each named path is removed first, so shipping a subdirectory *replaces* it
    # — files deleted locally do not linger on the host. Without names (a whole
    # project ship) we extract over the top instead: wiping PROJECT_DIR would
    # take logs/, .env files and any data the host has accumulated with it.
    for target in "$@"; do
        assert_safe_target "$target"
        if [ -e "$PROJECT_DIR/$target" ]; then
            log "replacing $target"
            rm -rf -- "${PROJECT_DIR:?}/$target"
        fi
    done
    [ "$#" -eq 0 ] && log "extracting over the existing tree (no paths named)"

    log "unpacking $(basename "$BUNDLE") into $PROJECT_DIR"
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$BUNDLE" -d "$PROJECT_DIR"
    else
        # unzip is not installed everywhere; the stdlib always is.
        "$PYTHON" -m zipfile -e "$BUNDLE" "$PROJECT_DIR" \
            || python3 -m zipfile -e "$BUNDLE" "$PROJECT_DIR"
    fi
    rm -f "$BUNDLE"
    log "unpacked."
}

# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

do_setup() {
    cd "$PROJECT_DIR" || die "no such directory: $PROJECT_DIR"
    [ -f requirements.txt ] || die "requirements.txt not found in $PROJECT_DIR"

    [ -n "$ARTIFACTORY_USER" ] || die "set \$ARTIFACTORY_USER (package index account)"
    [ -n "$ARTIFACTORY_HOST" ] || die "set \$ARTIFACTORY_HOST (package index host)"
    resolve_token
    write_pip_conf

    if [ -x "$VENV_DIR/bin/python" ]; then
        log "reusing venv at $VENV_DIR"
    else
        log "creating venv at $VENV_DIR with $PYTHON"
        command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not on PATH"
        "$PYTHON" -m venv "$VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    log "python: $(python -V 2>&1) at $(command -v python)"

    log "installing requirements (reads the pip.conf written above)"
    pip install --no-cache-dir -r requirements.txt

    mkdir -p "$LOG_DIR"
    log "done. activate with: source $VENV_DIR/bin/activate"
}

# --------------------------------------------------------------------------
# start / stop / status
# --------------------------------------------------------------------------

running_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    printf '%s' "$pid"
}

do_start() {
    cd "$PROJECT_DIR" || die "no such directory: $PROJECT_DIR"
    [ -x "$VENV_DIR/bin/python" ] || die "no venv at $VENV_DIR — run '$0 setup' first"

    local pid
    if pid=$(running_pid); then
        log "already running (pid $pid) — use restart"
        return 0
    fi

    mkdir -p "$LOG_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    export PYTHONUNBUFFERED=1

    local args=("$APP_MODULE" --host "$BIND_HOST" --port "$PORT")
    [ -n "$RELOAD" ] && args+=(--reload)

    # stdin from /dev/null and both streams to the log: without all three
    # redirected, ssh keeps the channel open and the calling command hangs
    # waiting for a server that never exits.
    log "starting: python -m uvicorn ${args[*]}"
    nohup python -m uvicorn "${args[@]}" </dev/null >>"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    # A server that dies on import would otherwise be reported as started.
    sleep 2
    if ! pid=$(running_pid); then
        log "uvicorn exited during startup — last 30 log lines:"
        tail -n 30 "$LOG_FILE" >&2 || true
        exit 1
    fi
    log "started pid $pid on ${BIND_HOST}:${PORT} (ENV=$ENV), logging to $LOG_FILE"
}

do_stop() {
    local pid
    if ! pid=$(running_pid); then
        log "not running"
        rm -f "$PID_FILE"
        return 0
    fi

    # --reload makes uvicorn fork a worker, so signalling the pid alone leaves
    # the child behind. Signal the whole process group when we can resolve it.
    local pgid
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    if [ -n "$pgid" ]; then
        kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    else
        kill -TERM "$pid" 2>/dev/null || true
    fi

    for _ in $(seq 1 20); do
        running_pid >/dev/null || break
        sleep 0.5
    done
    if running_pid >/dev/null; then
        log "still alive after SIGTERM — sending SIGKILL"
        [ -n "$pgid" ] && kill -KILL "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    log "stopped (was pid $pid)"
}

do_status() {
    local pid
    if pid=$(running_pid); then
        log "running: pid $pid, ${BIND_HOST}:${PORT}"
        ps -o pid,ppid,etime,cmd -p "$pid" >&2 2>/dev/null || true
    else
        log "not running"
    fi
    if [ -f "$LOG_FILE" ]; then
        log "last 10 log lines ($LOG_FILE):"
        tail -n 10 "$LOG_FILE" >&2 || true
    fi
}

command="${1:-all}"
shift || true
case "$command" in
    unpack)  do_unpack "$@" ;;
    setup)   do_setup ;;
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    all)     do_unpack "$@"; do_setup ;;
    *)       die "unknown command '$command' (expected: unpack | setup | all | start | stop | restart | status)" ;;
esac
