#!/usr/bin/env python3
"""Start, stop and watch the server on the remote host.

The counterpart of ``ship_to_host.py``: that one puts code on the host, this
one runs it. Both read the same ``ship_config.py``, and both drive the same
``ship_remote.sh``, so there is one definition of where things live.

The server is detached on the host — ``nohup``, its own process group, pid in
``logs/std/uvicorn.pid`` and output appended to ``logs/std/uvicorn.log``.

A first ``start`` against a host with no virtualenv builds one before
launching, rather than telling you to go and run setup — that needs the
Artifactory token, which is read from ``ship_config.py`` or ``--ask-token``
and sent on stdin, never argv. ``--no-auto-setup`` turns it back into an
error.
Following those logs is a separate, disposable thing: Ctrl-C stops the tail
and returns your prompt, and the server carries on serving. Only ``stop``
stops it.

    python ship_run.py                # start, then follow the log
    python ship_run.py start --no-follow
    python ship_run.py logs           # just follow, start nothing
    python ship_run.py setup          # build the venv, start nothing
    python ship_run.py restart
    python ship_run.py status
    python ship_run.py stop
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

# By path, so the working directory a run configuration happens to use cannot
# hide the module sitting right next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ship_to_host as ship            # noqa: E402  (path set up above)

# Mirrors LOG_DIR/LOG_FILE in ship_remote.sh, relative to the project root on
# the host. Change one and change the other.
REMOTE_LOG = "logs/std/uvicorn.log"


def remote_command(remote_dir: str, verb: str, env_prefix: str = "") -> str:
    """``ship_remote.sh <verb>``, run through bash from the project root.

    bash rather than executing it: the project directory is mounted noexec on
    these hosts, where chmod +x succeeds and the exec still fails with 126.
    """
    script = f"{remote_dir}/{ship.SETUP_SCRIPT}"
    prefix = f"{env_prefix} " if env_prefix else ""
    return (f"cd {shlex.quote(remote_dir)} && {prefix}"
            f"bash {shlex.quote(script)} {verb}")


def follow_command(remote_dir: str, lines: int) -> str:
    """Tail the log, creating it first so a not-yet-started server still works.

    ``-f`` rather than ``-F``: this host's tail is not GNU's, and the file is
    appended to rather than rotated under us, so retry-on-rename buys nothing.
    """
    log_path = f"{remote_dir}/{REMOTE_LOG}"
    return (f"mkdir -p {shlex.quote(f'{remote_dir}/logs/std')} && "
            f"touch {shlex.quote(log_path)} && "
            f"tail -n {lines} -f {shlex.quote(log_path)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start, stop and watch the remote server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("verb", nargs="?", default="start",
                        choices=("start", "stop", "restart", "status", "logs",
                                 "setup"),
                        help="default: start. `start` provisions the venv "
                             "first if the host has none; `setup` does that "
                             "and nothing else")
    parser.add_argument("--no-follow", action="store_true",
                        help="do not tail the log after start/restart")
    parser.add_argument("--lines", type=int, default=200, metavar="N",
                        help="how much existing log to show first (default: 200)")

    parser.add_argument("--host", default=ship.DEFAULT_HOST)
    parser.add_argument("--user", default=ship.DEFAULT_USER)
    parser.add_argument("--remote-dir", default=None,
                        help=f"default: {ship.DEFAULT_REMOTE_DIR}")
    parser.add_argument("--ssh-port", type=int, default=int(ship.DEFAULT_SSH_PORT),
                        metavar="N")
    parser.add_argument("--password-env", default=ship.PASSWORD_ENV, metavar="VAR")
    parser.add_argument("--ask-password", action="store_true",
                        help="prompt for the SSH password")
    parser.add_argument("--ssh-option", action="append", default=[], metavar="OPT",
                        help="passed through as -o OPT; system-ssh fallback only")

    parser.add_argument("--bind-host", default="0.0.0.0", metavar="ADDR",
                        help="address uvicorn binds on the host (default: "
                             "0.0.0.0; a loopback bind would only be reachable "
                             "from the box itself)")
    parser.add_argument("--port", type=int, default=8000,
                        help="port uvicorn listens on (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="pass --reload to uvicorn; off by default because "
                             "the reloader forks a watcher that outlives a "
                             "plain kill")
    parser.add_argument("--token-env", default=ship.TOKEN_ENV, metavar="VAR",
                        help="environment variable holding the Artifactory "
                             "token, used only if the venv has to be built "
                             f"(default: {ship.TOKEN_ENV})")
    parser.add_argument("--ask-token", action="store_true",
                        help="prompt for the Artifactory token rather than "
                             "reading it from the config or environment")
    parser.add_argument("--no-auto-setup", action="store_true",
                        help="fail if the venv is missing instead of "
                             "building it")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run, connect to nothing")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.host:
        ship.die(f"no target host — set HOST in {ship.CONFIG_FILE}, export "
                 f"$SHIP_HOST, or pass --host")

    remote_dir = (args.remote_dir or ship.DEFAULT_REMOTE_DIR).format(
        user=args.user or "user")

    password  = ship.read_password(args.password_env, ask=args.ask_password)
    transport = ship.open_transport(
        args.host, args.user, args.ssh_port, password,
        dry_run=args.dry_run,
        ssh_options=args.ssh_option,
    )

    try:
        if args.verb != "logs":
            settings: "list[str]" = []
            stdin_bytes = None

            if args.verb in ("start", "restart", "setup"):
                # Sent every time, though only a host with no venv will use
                # them. Asking first whether setup is needed would cost a
                # round trip, and a round trip here costs a whole connection.
                for name, value in (("ARTIFACTORY_USER", ship.ARTIFACTORY_USER),
                                    ("ARTIFACTORY_HOST", ship.ARTIFACTORY_HOST)):
                    if value:
                        settings.append(f"{name}={shlex.quote(value)}")
                if args.no_auto_setup:
                    settings.append("AUTO_SETUP=0")
                token = ship.read_token(args.token_env, ask=args.ask_token,
                                        required=args.verb == "setup")
                if token:
                    stdin_bytes = (token + "\n").encode()

            # Only start and restart care about these. Passing them to stop
            # or status would state something about a server this invocation
            # did not launch.
            if args.verb in ("start", "restart"):
                settings += [f"BIND_HOST={shlex.quote(args.bind_host)}",
                             f"PORT={args.port}",
                             f"RELOAD={'1' if args.reload else ''}"]

            transport.exec(remote_command(remote_dir, args.verb,
                                          " ".join(settings)),
                           stdin_bytes=stdin_bytes)

        following = (args.verb in ("start", "restart", "logs")
                     and not args.no_follow)
        if not following:
            return 0

        ship.log(f"following {remote_dir}/{REMOTE_LOG} — Ctrl-C stops watching, "
                 f"not the server")
        try:
            transport.exec(follow_command(remote_dir, args.lines))
        except KeyboardInterrupt:
            # The tail dies with its channel; uvicorn was nohup'd into its own
            # process group by ship_remote.sh and never notices.
            ship.log("stopped following. The server is still running — "
                     f"`{Path(__file__).name} stop` to stop it")
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
