#!/usr/bin/env python3
"""
Ship a project subdirectory to a remote host: zip locally, scp, unpack there.

Written for hosts that cannot reach github.com — cloning is not an option, so
the working tree travels as a bundle instead. Uncommitted work is included on
purpose (that is usually the reason for shipping at all).

This script does local work and transport only. Every remote step lives in
``ship_remote.sh`` and runs in a single shell there, because ``export`` and
``source`` do not survive separate ssh invocations.

Transport is paramiko when it is installed, and the system ``ssh``/``scp``
when it is not. paramiko is what makes a password work everywhere: OpenSSH
reads one only from its own tty, and the usual stand-ins are platform-bound
(``sshpass`` is POSIX-only, PuTTY is Windows-only). ``upload-util.py`` in
this project already takes the paramiko route.

Host, account, pip index and what to send live in ``ship_config.py`` beside
this file — copy ``ship_config.sample.py`` and fill it in. ``REMOTE_DIR`` is
the project root on the host; ``UPLOAD_DIR`` narrows a run to one part of it,
such as ``app/rag``, and empty means the whole project. Nothing site-specific is stored in
this script, so it can be copied to another project as-is; only the config
file has to be recreated. Each entry falls back to an environment variable,
and a command-line flag overrides both.

Authentication is supplied, not inherited from ``~/.ssh/config`` or an agent.
A password comes from ``PASSWORD`` in that config, ``$SHIP_PASSWORD``, or
``--ask-password``, and never touches argv. Leave it empty and keys are used
instead. Host keys are accepted without being checked or recorded — see
ParamikoTransport for why, and for what that costs.

    python ship_to_host.py --subdir app --subdir requirements.txt
    python ship_to_host.py --subdir app --setup      # + venv and pip install
    python ship_to_host.py --subdir app --dry-run

Running the thing once it is there is ship_run.py's job — start, stop,
status and logs — so that shipping code and controlling a server stay
separate commands. Both read the same ship_config.py.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import logging
import os
import posixpath
import shlex
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, NoReturn, Optional, Sequence

try:
    import paramiko
except ImportError:                     # falls back to the system ssh/scp
    paramiko = None
else:
    # paramiko logs a full traceback from its own transport thread for things
    # this script reports properly itself, so a refused connection would print
    # twice — once as noise, once as the actual message. Quiet unless --debug.
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)

def log(message: str) -> None:
    print(f"[ship] {message}", file=sys.stderr, flush=True)


def die(message: str) -> NoReturn:
    print(f"[ship] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


CONFIG_FILE        = "ship_config.py"
SETUP_SCRIPT       = "ship_remote.sh"
LAUNCH_SCRIPT      = "ship_run.py"      # start/stop/logs live there, not here
BUNDLE_NAME        = "_bundle.zip"

PASSWORD_ENV       = "SHIP_PASSWORD"
TOKEN_ENV          = "ARTIFACTORY_TOKEN"


def _load_config():
    """Import ``ship_config.py`` from beside this script, or return None.

    By path rather than by name: run configurations often set a working
    directory that is not the project root, and a plain ``import`` would
    find nothing from there.
    """
    path = Path(__file__).resolve().parent / CONFIG_FILE
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ship_config", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:                      # a typo should not be fatal
        log(f"ignoring {CONFIG_FILE}: {error}")
        return None
    return module


_CONFIG = _load_config()


def configured(name: str, env: str, default: str = "") -> str:
    """One setting, resolved: ``ship_config.py``, then ``$env``, then default.

    A command-line flag beats all three — argparse takes what this returns as
    its default, so passing the flag simply overrides it for that run.
    """
    value = getattr(_CONFIG, name, "") if _CONFIG else ""
    return value or os.environ.get(env, "") or default


def configured_list(name: str, env: str) -> "list[str]":
    """Same resolution as :func:`configured`, for a setting that may repeat.

    Accepts a list in ``ship_config.py`` or a comma-separated string, since
    an environment variable can only ever be the latter.
    """
    value = getattr(_CONFIG, name, None) if _CONFIG else None
    # Empty counts as unset, not as an answer — otherwise the ``UPLOAD_DIR = ""``
    # that ships in the sample would shadow the environment variable, which is
    # not how configured() behaves for every other setting.
    if not value:
        value = os.environ.get(env, "")
    if isinstance(value, str):
        value = value.split(",")
    return [part.strip() for part in value if part and part.strip()]


# ===========================================================================
# CONFIGURATION — the values live in ship_config.py, next to this script.
#
# Nothing site-specific is stored in this file, so it can be copied between
# projects, or published, without carrying credentials along. Copy
# ship_config.sample.py to ship_config.py and fill that in; anything left out
# of it falls back to the environment variable named beside it below.
# ===========================================================================
DEFAULT_HOST       = configured("HOST", "SHIP_HOST")
DEFAULT_USER       = configured("USER", "SHIP_USER")
# Where the project lives on the host — its root there, the counterpart of
# this checkout. Everything uploaded is placed relative to it.
DEFAULT_REMOTE_DIR = configured("REMOTE_DIR", "SHIP_REMOTE_DIR",
                                "/tmp/{user}/project")

# The part of the project to actually send, as paths relative to the root
# above — "app/rag" refreshes just that. Each named directory is wiped on
# the host before the new copy lands, so local deletions propagate. Empty
# means the whole project, extracted over whatever is already there.
DEFAULT_UPLOAD     = configured_list("UPLOAD_DIR", "SHIP_UPLOAD_DIR")

DEFAULT_SSH_PORT   = configured("SSH_PORT", "SHIP_SSH_PORT", "22")

PASSWORD           = configured("PASSWORD", PASSWORD_ENV)
ARTIFACTORY_USER   = configured("ARTIFACTORY_USER", "ARTIFACTORY_USER")
ARTIFACTORY_HOST   = configured("ARTIFACTORY_HOST", "ARTIFACTORY_HOST")
ARTIFACTORY_TOKEN  = configured("ARTIFACTORY_TOKEN", TOKEN_ENV)
# ===========================================================================

# Never shipped, whatever git says, and matched by name at any depth. This
# repo's .gitignore does not cover __pycache__, so `git ls-files --others
# --exclude-standard` happily offers every .pyc in the tree — bytecode for the
# wrong interpreter, and pure weight.
EXCLUDE_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".idea", ".pytest_cache",
    ".mypy_cache", "node_modules",
})
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log")

# Excluded by default but overridable, and matched as project-relative *paths*,
# not names — "data" drops the top-level data/ and leaves app/rag/data alone.
# Extend with --exclude, or drop the lot with --no-default-excludes.
DEFAULT_EXCLUDE_PATHS = ("data",)

# Bundle size past which the heaviest directories are named in the log.
BULKY_MB = 25

# Upload size past which progress is worth printing.
PROGRESS_FROM_BYTES = 4 * 1_048_576


def is_shippable(path: Path, root: Path, exclude_paths: Sequence[str] = ()) -> bool:
    relative = path.relative_to(root)
    # Never the credentials, whatever git thinks. .gitignore covers it in
    # this repo, but this script is meant to be copied into others, and a
    # missing ignore rule there must not put a password on a server.
    if relative.name == CONFIG_FILE:
        return False
    if EXCLUDE_DIRS.intersection(relative.parts):
        return False
    if relative.name.endswith(EXCLUDE_SUFFIXES):
        return False
    for excluded in exclude_paths:
        parts = Path(excluded).parts
        if relative.parts[:len(parts)] == parts:
            return False
    return True


def run(
    cmd: Sequence[str],
    *,
    dry_run: bool = False,
    stdin: Optional[bytes] = None,
) -> None:
    """Run a command, streaming its output; raise on a non-zero exit.

    ``stdin`` is written to the child rather than passed as an argument — it
    carries the Artifactory token, and argv is world-readable through ``ps``.
    """
    printable = " ".join(shlex.quote(part) for part in cmd)
    log(printable)
    result = subprocess.run(cmd, input=stdin, check=False)
    if result.returncode != 0:
        die(f"command failed with exit {result.returncode}: {printable}")


class Transport:
    """Runs remote commands and uploads files over a single SSH connection.

    Two implementations, picked at run time by :func:`open_transport`.
    paramiko is preferred and is what makes a password work at all on
    Windows: OpenSSH takes one only from its own tty, and the usual stand-ins
    are platform-bound (``sshpass`` is POSIX-only, PuTTY is Windows-only).
    Being pure Python it sidesteps that entirely, and ``upload-util.py`` in
    this project already ships this way.

    The ``ssh``/``scp`` implementation is the fallback for when paramiko is
    not installed. It handles key auth perfectly well; it just cannot be
    handed a password.
    """

    def exec(self, command: str, *, stdin_bytes: Optional[bytes] = None) -> None:
        """Run ``command`` on the host; raise unless it exits zero."""
        raise NotImplementedError

    def put(self, paths: Sequence[Path], remote_dir: str) -> None:
        """Upload ``paths`` into ``remote_dir``, keeping their basenames."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class ParamikoTransport(Transport):
    """One connection, reconnected whenever the host runs out of channels."""

    def __init__(self, host: str, user: str, port: int, password: str) -> None:
        self._host, self._port = host, port
        self._user = user
        self._connect_kwargs: "dict[str, object]" = {
            "hostname": host, "port": port, "timeout": 30,
        }
        if user:
            self._connect_kwargs["username"] = user
        if password:
            # Offering keys first would let a host that accepts one succeed
            # while ignoring the password — a pass for the wrong reason.
            self._connect_kwargs.update(password=password, look_for_keys=False,
                                        allow_agent=False)

        self._client = self._connect()
        log(f"connected to {host}:{port} as {user or '(default user)'}")

    def _connect(self):
        client = paramiko.SSHClient()
        # Any host key is accepted and none is remembered. Consulting
        # known_hosts only produced misleading refusals here — these hosts are
        # reachable solely from the corporate network, and their keys change
        # often enough that a stale entry is the usual outcome. The trade is
        # the real one: no protection against a spoofed host.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(**self._connect_kwargs)
        except paramiko.AuthenticationException:
            die(f"authentication failed for "
                f"{self._user or '(default user)'}@{self._host}")
        except paramiko.SSHException as error:
            die(f"could not connect to {self._host}:{self._port}: {error}")
        except OSError as error:
            die(f"could not reach {self._host}:{self._port}: {error}")

        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)  # pip install can outlast an idle timeout
        return client

    def _session(self):
        """A channel, reconnecting once if the host will not grant another.

        Some hosts allow exactly one session channel per connection and do not
        free the slot when it closes, so the second ``open_session`` fails with
        ChannelException(2, 'Connect failed') however carefully the first was
        cleaned up. ``upload-util.py`` sidesteps this by doing everything in a
        single channel; reconnecting instead costs one handshake per step and
        keeps the steps independent.
        """
        failure: "Optional[BaseException]" = None
        for attempt in (1, 2):
            transport = self._client.get_transport()
            if transport is not None and transport.is_active():
                try:
                    return transport.open_session()
                except paramiko.SSHException as error:
                    failure = error
            if attempt == 1:
                log("host granted no further channels — reconnecting")
                self._client.close()
                self._client = self._connect()
        die(f"could not open a channel even on a fresh connection: {failure}")

    def exec(self, command: str, *, stdin_bytes: Optional[bytes] = None) -> None:
        log(f"$ {command}" + ("  <<< (secret on stdin)" if stdin_bytes else ""))
        channel = self._session()
        try:
            # stderr folded into stdout: ship_remote.sh logs there, and a
            # `setup` that installs for minutes should not look like a hang.
            channel.exec_command(f"{command} 2>&1")
            if stdin_bytes is not None:
                channel.sendall(stdin_bytes)
            channel.shutdown_write()

            with channel.makefile("r") as output:
                for line in output:
                    print(line.rstrip("\n"), file=sys.stderr, flush=True)
            status = channel.recv_exit_status()
        finally:
            # Closed explicitly, not left to the GC: a host with a low
            # MaxSessions refuses the *next* channel while this one lingers,
            # and the failure surfaces somewhere unrelated.
            channel.close()

        if status != 0:
            die(f"remote command failed with exit {status}: {command}")

    def put(self, paths: Sequence[Path], remote_dir: str) -> None:
        """Stream each file through ``cat``, one channel at a time.

        Not SFTP: that needs the sftp subsystem, which these hosts do not
        offer — ``open_sftp`` fails with ChannelException(2, 'Connect
        failed'). ``upload-util.py`` hit the same wall, which is why its
        default method is tar-over-one-channel rather than sftp. ``cat``
        needs nothing the exec channel above does not already prove works.
        """
        for path in paths:
            destination = posixpath.join(remote_dir, path.name)
            total = path.stat().st_size
            log(f"upload {path.name} ({total / 1_048_576:.1f} MB) -> {destination}")

            channel = self._session()
            try:
                channel.exec_command(
                    f"mkdir -p {shlex.quote(remote_dir)} && "
                    f"cat > {shlex.quote(destination)}")
                sent = milestone = 0
                # A 17 MB upload over a slow link is a long silence; a 10 KB
                # one is over before a percentage means anything.
                step = total // 4 if total >= PROGRESS_FROM_BYTES else 0
                with path.open("rb") as source:
                    while chunk := source.read(65536):
                        channel.sendall(chunk)
                        sent += len(chunk)
                        if step and sent - milestone >= step:
                            milestone = sent
                            log(f"  {sent * 100 // total}%")
                channel.shutdown_write()

                errors = channel.makefile_stderr("r").read().decode(
                    errors="replace").strip()
                status = channel.recv_exit_status()
            finally:
                channel.close()

            if status != 0:
                die(f"upload of {path.name} failed with exit {status}"
                    + (f": {errors}" if errors else ""))

    def close(self) -> None:
        self._client.close()


class OpenSshTransport(Transport):
    """The system ``ssh``/``scp``. Key auth only — see Transport."""

    def __init__(self, host: str, user: str, port: int,
                 ssh_options: Sequence[str] = ()) -> None:
        self._target = f"{user}@{host}" if user else host
        # Same bargain as the paramiko path above: accept the key, keep no
        # record of it. Listed first so --ssh-option can still override.
        defaults = ("StrictHostKeyChecking=no", "UserKnownHostsFile=/dev/null",
                    "LogLevel=ERROR")
        self._options = [part for option in (*defaults, *ssh_options)
                         for part in ("-o", option)]
        self._port = port

    def exec(self, command: str, *, stdin_bytes: Optional[bytes] = None) -> None:
        run(["ssh", "-p", str(self._port), *self._options, self._target, command],
            stdin=stdin_bytes)

    def put(self, paths: Sequence[Path], remote_dir: str) -> None:
        run(["scp", "-P", str(self._port), *self._options,
             *[str(p) for p in paths], f"{self._target}:{remote_dir}/"])


class DryRunTransport(Transport):
    """Prints what the real thing would do, and never opens a connection."""

    def __init__(self, host: str, user: str, port: int) -> None:
        self._where = f"{user}@{host}:{port}" if user else f"{host}:{port}"

    def exec(self, command: str, *, stdin_bytes: Optional[bytes] = None) -> None:
        log(f"DRY-RUN {self._where} $ {command}"
            + ("  <<< (secret on stdin)" if stdin_bytes else ""))

    def put(self, paths: Sequence[Path], remote_dir: str) -> None:
        for path in paths:
            log(f"DRY-RUN upload {path} -> {self._where}:{remote_dir}/{path.name}")


def open_transport(host: str, user: str, port: int, password: str, *,
                   dry_run: bool, ssh_options: Sequence[str]) -> Transport:
    if dry_run:
        return DryRunTransport(host, user, port)
    if paramiko is not None:
        return ParamikoTransport(host, user, port, password)
    if password:
        die("a password needs paramiko, which is not installed — ssh accepts "
            "one only from its own tty.\n"
            "    pip install paramiko\n"
            "Or leave PASSWORD empty and use key authentication, which the "
            "system ssh handles on its own.")
    log("paramiko is not installed — falling back to the system ssh/scp")
    return OpenSshTransport(host, user, port, ssh_options)


def git_files(root: Path, targets: Sequence[str]) -> Optional[list[Path]]:
    """Paths git would ship for ``targets``: tracked plus untracked-not-ignored.

    Returns ``None`` when git cannot answer (not a repo, git missing), so the
    caller can fall back to a plain walk.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z",
             "--cached", "--others", "--exclude-standard", "--", *targets],
            capture_output=True, check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None

    names = [n for n in result.stdout.decode().split("\0") if n]
    # ls-files lists index entries, which may name files deleted from the tree.
    return [root / n for n in names if (root / n).is_file()]


def walked_files(root: Path, targets: Sequence[str]) -> "list[Path]":
    """Fallback file list — used only when git cannot answer."""
    found: "list[Path]" = []
    for target in targets:
        path = root / target
        if path.is_file():
            found.append(path)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            found += [Path(dirpath) / name for name in filenames]
    return found


def build_bundle(
    root: Path,
    targets: Sequence[str],
    destination: Path,
    exclude_paths: Sequence[str] = (),
) -> Path:
    """Zip ``targets`` (paths relative to ``root``) into ``destination``."""
    for target in targets:
        if not (root / target).exists():
            die(f"no such path in the project: {target}")

    files = git_files(root, targets)
    if files is None:
        log("git could not list files — falling back to a filesystem walk")
        files = walked_files(root, targets)

    artefacts = [f for f in files if not is_shippable(f, root)]
    kept      = [f for f in files if is_shippable(f, root, exclude_paths)]
    if artefacts:
        log(f"skipped {len(artefacts)} build artefact(s) — "
            f"{', '.join(sorted(EXCLUDE_DIRS))}")
    if excluded := len(files) - len(artefacts) - len(kept):
        log(f"excluded {excluded} file(s) under: {', '.join(exclude_paths)}")
    if not kept:
        die(f"nothing to ship for {list(targets)} (everything excluded?)")
    files = kept

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            bundle.write(path, path.relative_to(root).as_posix())

    size_mb = destination.stat().st_size / 1_048_576
    log(f"bundled {len(files)} file(s), {size_mb:.1f} MB → {destination.name}")
    if size_mb > BULKY_MB:
        warn_if_bulky(root, files)
    return destination


def warn_if_bulky(root: Path, files: Iterable[Path]) -> None:
    """Name the directories responsible for a large bundle.

    Shipping the whole project is the default, and this tree carries data and
    screenshot directories that dwarf the source. Worth one line of output
    before a slow upload, rather than excluding them on the user's behalf.
    """
    weight: "dict[str, int]" = {}
    for path in files:
        top = path.relative_to(root).parts[0]
        weight[top] = weight.get(top, 0) + path.stat().st_size

    heaviest = sorted(weight.items(), key=lambda kv: -kv[1])[:3]
    summary = ", ".join(f"{name} {size / 1_048_576:.0f} MB" for name, size in heaviest)
    log(f"  bulk is {summary} — narrow it with --subdir if that is not wanted")


def read_token(explicit_env: str, *, ask: bool = False,
               required: bool = True) -> str:
    """Resolve the Artifactory token: ``ship_config.py``, then the environment.

    Prompts as a last resort when there is a tty, so --setup does not fail at
    the very end of a slow upload over something the caller has to hand.

    ``required=False`` returns "" instead of exiting, for callers that send
    the token speculatively — ship_run.py cannot know whether the host still
    needs provisioning without spending a round trip to ask.
    """
    token = ARTIFACTORY_TOKEN or os.environ.get(explicit_env, "")
    if token:
        return token
    if not required and not ask:
        return ""

    if ask or sys.stdin.isatty():
        try:
            token = getpass.getpass("Artifactory token: ")
        except (EOFError, KeyboardInterrupt):
            token = ""
    if not token and not required:
        return ""
    if not token:
        die(
            f"--setup needs the Artifactory token. Get it from Artifactory's "
            f'"Set Me Up" dialog for the pypi repo, then either set '
            f"ARTIFACTORY_TOKEN in {CONFIG_FILE}, or export it (keeping it "
            f"out of shell history):\n"
            f"    read -rs {explicit_env} && export {explicit_env}"
        )
    return token


def read_password(env_var: str, *, ask: bool) -> str:
    """Resolve the SSH password.

    Empty is the normal answer and means "authenticate however ssh otherwise
    would" — this only takes over when a password is actually supplied.
    """
    if ask:
        try:
            password = getpass.getpass("SSH password: ")
        except (EOFError, KeyboardInterrupt):
            # No tty to prompt on, or the caller gave up. Either way this is a
            # normal way to fail, not a traceback.
            die(f"could not read a password — set ${env_var} instead")
        if not password:
            die("--ask-password given but nothing was entered")
        return password
    return PASSWORD or os.environ.get(env_var, "")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zip a project subdirectory, scp it to a host, unpack it there.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--subdir", action="append", metavar="PATH", default=None,
                        help="project-relative path to ship; repeatable. "
                             "Overrides UPLOAD_DIR in the config, which is the "
                             "same setting. With neither, the whole project "
                             "goes. A named directory REPLACES its remote copy "
                             "(wiped first, so local deletions propagate); the "
                             "whole-project default extracts over the existing "
                             "tree instead")
    parser.add_argument("--exclude", action="append", metavar="PATH", default=[],
                        help="project-relative path to leave out of the zip; "
                             "repeatable. Added to the defaults "
                             f"({', '.join(DEFAULT_EXCLUDE_PATHS)})")
    parser.add_argument("--no-default-excludes", action="store_true",
                        help=f"do not exclude {', '.join(DEFAULT_EXCLUDE_PATHS)} "
                             "(build artefacts are still skipped)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER,
                        help="remote user")
    parser.add_argument("--password-env", default=PASSWORD_ENV, metavar="VAR",
                        help="environment variable holding the SSH password "
                             f"(default: {PASSWORD_ENV}). Empty means ssh "
                             "authenticates as it normally would")
    parser.add_argument("--ask-password", action="store_true",
                        help="prompt for the SSH password instead of reading "
                             "it from the environment")
    parser.add_argument("--remote-dir", default=None,
                        help=f"default: {DEFAULT_REMOTE_DIR.format(user='<user>')}")
    parser.add_argument("--project-root", default=None, type=Path,
                        help="default: the directory holding this script")
    parser.add_argument("--setup", action="store_true",
                        help="also run the prerequisite setup (venv, pip.conf, "
                             "requirements) on the host")
    parser.add_argument("--token-env", default=TOKEN_ENV, metavar="VAR",
                        help="environment variable holding the Artifactory token, "
                             f"read only with --setup (default: {TOKEN_ENV})")
    parser.add_argument("--artifactory-user", default=None, metavar="NAME",
                        help="pip index account; default: the CONFIGURATION "
                             "block, then $ARTIFACTORY_USER")
    parser.add_argument("--artifactory-host", default=None, metavar="HOST",
                        help="pip index host; default: the CONFIGURATION "
                             "block, then $ARTIFACTORY_HOST")
    parser.add_argument("--ssh-port", type=int, default=int(DEFAULT_SSH_PORT),
                        metavar="N", help="SSH port (default: 22)")
    parser.add_argument("--ssh-option", action="append", default=[], metavar="OPT",
                        help="passed through as -o OPT; repeatable. Only used "
                             "by the system-ssh fallback, not by paramiko")
    parser.add_argument("--keep-bundle", action="store_true",
                        help="do not delete the local zip afterwards")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run; still builds the zip")
    parser.add_argument("--debug", action="store_true",
                        help="unmute paramiko's own logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.debug and paramiko is not None:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("paramiko").setLevel(logging.DEBUG)

    if not args.host:
        die(f"no target host — set HOST in {CONFIG_FILE} (copy "
            f"ship_config.sample.py if it is missing), export $SHIP_HOST, "
            f"or pass --host")

    root = (args.project_root or Path(__file__).resolve().parent).resolve()
    if not (root / SETUP_SCRIPT).is_file():
        die(f"{SETUP_SCRIPT} not found next to the project root ({root})")

    remote_dir = args.remote_dir or DEFAULT_REMOTE_DIR.format(
        user=args.user or os.environ.get("USER", "user")
    )
    remote_setup = f"{remote_dir}/{SETUP_SCRIPT}"

    # Resolved before anything is built or uploaded, so a prompt appears while
    # the caller is still watching and a bad password fails in a second.
    password  = read_password(args.password_env, ask=args.ask_password)
    transport = open_transport(
        args.host, args.user, args.ssh_port, password,
        dry_run=args.dry_run,
        ssh_options=args.ssh_option,
    )

    def remote(command: str, *, stdin: Optional[bytes] = None,
               env_prefix: str = "") -> None:
        """One call = one remote shell, so exports and the venv stay in scope.

        Run as ``bash <script>`` rather than executed directly: /tmp on these
        hosts is mounted noexec, where chmod +x succeeds and the exec still
        fails with 126. Handing the file to an interpreter only needs read
        permission. It also pins bash — the login shell here is ksh, which
        does not take the script's arrays and += syntax.

        ``env_prefix`` goes before the script name, so it applies to that
        command only. Nothing passed this way is secret — the token travels
        on stdin.
        """
        prefix = f"{env_prefix} " if env_prefix else ""
        transport.exec(f"cd {shlex.quote(remote_dir)} && {prefix}"
                       f"bash {shlex.quote(remote_setup)} {command}",
                       stdin_bytes=stdin)

    # No --subdir means the whole project. Named paths are replaced wholesale on
    # the host; the whole-project default extracts over the top, because wiping
    # the remote root would take logs/, .env files and accumulated data with it.
    # --subdir beats UPLOAD_DIR. Resolved here rather than as the argparse
    # default, because action="append" would extend that default instead of
    # replacing it, and the flag would silently mean "as well as".
    chosen  = args.subdir or DEFAULT_UPLOAD
    targets = chosen or ["."]
    replace = [t for t in chosen if (root / t).is_dir()]
    if chosen:
        source = "--subdir" if args.subdir else "UPLOAD_DIR"
        log(f"shipping {', '.join(chosen)} ({source}); "
            f"the rest of the project is left alone on the host")
    else:
        log("shipping the whole project (no --subdir, no UPLOAD_DIR)")

    excludes = list(args.exclude)
    if not args.no_default_excludes:
        excludes += list(DEFAULT_EXCLUDE_PATHS)
    # Asking for a path outright beats excluding it by default; silently
    # shipping nothing for `--subdir data` would be the wrong reading.
    conflicting = {e for e in excludes for t in chosen
                   if Path(t).parts[:len(Path(e).parts)] == Path(e).parts}
    if conflicting:
        log(f"--subdir names {', '.join(sorted(conflicting))} explicitly — "
            f"not applying that exclusion")
        excludes = [e for e in excludes if e not in conflicting]

    # Flag beats the CONFIGURATION block beats the environment.
    artifactory_user = (args.artifactory_user if args.artifactory_user is not None
                        else ARTIFACTORY_USER or os.environ.get("ARTIFACTORY_USER", ""))
    artifactory_host = (args.artifactory_host if args.artifactory_host is not None
                        else ARTIFACTORY_HOST or os.environ.get("ARTIFACTORY_HOST", ""))

    # Checked before any transport, so a missing token fails in a second rather
    # than after a multi-megabyte upload.
    token = read_token(args.token_env) if args.setup else None

    workdir = Path(tempfile.mkdtemp(prefix="ship-"))
    bundle  = build_bundle(root, targets, workdir / BUNDLE_NAME, excludes)

    try:
        transport.put([bundle, root / SETUP_SCRIPT], remote_dir)

        if replace:
            log(f"replacing on the host: {', '.join(replace)}")
        remote("unpack " + " ".join(shlex.quote(p) for p in replace))

        if args.setup:
            index = " ".join(
                f"{name}={shlex.quote(value)}"
                for name, value in (
                    ("ARTIFACTORY_USER", artifactory_user),
                    ("ARTIFACTORY_HOST", artifactory_host),
                )
                if value
            )
            log("running remote setup (token piped on stdin, never in argv)")
            remote("setup", stdin=((token or "") + "\n").encode(),
                   env_prefix=index)

    finally:
        if args.keep_bundle:
            log(f"bundle kept at {bundle}")
        else:
            bundle.unlink(missing_ok=True)
            workdir.rmdir()
        transport.close()

    where = f"{args.user}@{args.host}" if args.user else args.host
    log(f"shipped to {where}:{remote_dir}")
    # Named as this script's own flags: the transport may be paramiko, in which
    # case there is no ssh command line to copy.
    if not args.setup:
        log(f"run setup later with: {Path(__file__).name} --setup")
    log(f"start it with:        {LAUNCH_SCRIPT} start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
