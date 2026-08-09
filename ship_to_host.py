#!/usr/bin/env python3
"""
Ship a project subdirectory to a remote host: zip locally, scp, unpack there.

Written for hosts that cannot reach github.com — cloning is not an option, so
the working tree travels as a bundle instead. Uncommitted work is included on
purpose (that is usually the reason for shipping at all).

This script does local work and transport only. Every remote step lives in
``ship_remote.sh`` and runs in a single shell there, because ``export`` and
``source`` do not survive separate ssh invocations.

Transport is the system ``ssh``/``scp``, not a library: they need no package
installed from an index this host may not be able to reach.

Host, account and pip index live in ``ship_config.py`` beside this file — copy
``ship_config.sample.py`` and fill it in. Nothing site-specific is stored in
this script, so it can be copied to another project as-is; only the config
file has to be recreated. Each entry falls back to an environment variable,
and a command-line flag overrides both.

Authentication is supplied, not inherited from ``~/.ssh/config`` or an agent.
A password can come from ``PASSWORD`` in that config, ``$SHIP_PASSWORD``, or
``--ask-password``. Getting it to ssh needs a helper, because OpenSSH accepts
a password only from its own tty — ``sshpass`` on Linux and macOS, PuTTY's
``plink``/``pscp`` on Windows. Either way it travels by environment or file,
never argv. Leave it empty and ssh authenticates however it otherwise would,
which needs no helper at all.

    python ship_to_host.py --subdir app --subdir requirements.txt
    python ship_to_host.py --subdir app --setup      # + venv and pip install
    python ship_to_host.py --subdir app --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import getpass
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, NoReturn, Optional, Sequence

def log(message: str) -> None:
    print(f"[ship] {message}", file=sys.stderr, flush=True)


def die(message: str) -> NoReturn:
    print(f"[ship] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


CONFIG_FILE        = "ship_config.py"
SETUP_SCRIPT       = "ship_remote.sh"
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
DEFAULT_REMOTE_DIR = configured("REMOTE_DIR", "SHIP_REMOTE_DIR",
                                "/tmp/{user}/project")

PASSWORD           = configured("PASSWORD", PASSWORD_ENV)
ARTIFACTORY_USER   = configured("ARTIFACTORY_USER", "ARTIFACTORY_USER")
ARTIFACTORY_HOST   = configured("ARTIFACTORY_HOST", "ARTIFACTORY_HOST")
ARTIFACTORY_TOKEN  = configured("ARTIFACTORY_TOKEN", TOKEN_ENV)
# ===========================================================================

# Applied only when a password is in play. Without them ssh still offers every
# key in the agent first, and a host that accepts one would quietly ignore the
# password it was handed — succeeding for a reason the caller did not ask for.
PASSWORD_SSH_OPTIONS = (
    "PubkeyAuthentication=no",
    "PreferredAuthentications=password,keyboard-interactive",
)

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


def is_shippable(path: Path, root: Path, exclude_paths: Sequence[str] = ()) -> bool:
    relative = path.relative_to(root)
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
    stdin: Optional[str] = None,
    env_extra: Optional[dict] = None,
) -> None:
    """Run a command, streaming its output; raise on a non-zero exit.

    ``stdin`` is written to the child rather than passed as an argument — it
    carries the Artifactory token, and argv is world-readable through ``ps``.
    ``env_extra`` carries a secret for the same reason: see Transport.
    """
    environ = {**os.environ, **env_extra} if env_extra else None

    printable = " ".join(shlex.quote(part) for part in cmd)
    if dry_run:
        log(f"DRY-RUN {printable}" + ("  <<< (secret on stdin)" if stdin else ""))
        return

    log(printable)
    result = subprocess.run(
        cmd,
        input=stdin.encode() if stdin is not None else None,
        env=environ,
        check=False,
    )
    if result.returncode != 0:
        die(f"command failed with exit {result.returncode}: {printable}")


class Transport:
    """``ssh``/``scp``, plus whatever must stand in front to supply a password.

    OpenSSH reads a password only from its own tty, so automating one means a
    helper program — and which helper depends on the platform. Neither ships
    with the OS:

    ============  =========================================================
    Linux, macOS  ``sshpass -e``, password in ``$SSHPASS``
    Windows       PuTTY's ``plink``/``pscp``, password in a ``-pwfile``
    ============  =========================================================

    With no password these are plain ``ssh`` and ``scp`` and none of the rest
    applies — which is why key auth stays the path of least resistance.
    """

    def __init__(self, password: str = "", ssh_options: Sequence[str] = ()) -> None:
        self.password = password
        self.env: "dict[str, str]" = {}
        self._pwfile: Optional[Path] = None
        extra = list(ssh_options)

        if not password:
            self.ssh, self.scp = ["ssh"], ["scp"]

        elif shutil.which("sshpass"):
            self.ssh = ["sshpass", "-e", "ssh"]
            self.scp = ["sshpass", "-e", "scp"]
            self.env = {"SSHPASS": password}
            extra = list(PASSWORD_SSH_OPTIONS) + extra

        elif shutil.which("plink") and shutil.which("pscp"):
            flag = self._putty_password_flag()
            self.ssh = ["plink", "-batch", *flag]
            self.scp = ["pscp", "-batch", *flag]
            log("using PuTTY for password auth. -batch means an unfamiliar "
                "host key aborts instead of prompting — run plink against the "
                "host once by hand first to cache it")
            if extra:
                # -o is OpenSSH syntax; plink rejects it outright.
                log(f"plink takes no -o options — ignoring: {', '.join(extra)}")
            extra = []

        else:
            die("a password was given but nothing here can hand it to ssh, "
                "which accepts one only from its own tty. Install a helper:\n"
                "    Linux, macOS   sshpass          (apt install sshpass)\n"
                "    Windows        PuTTY's plink and pscp, both on PATH\n"
                "Or leave PASSWORD empty and use key authentication, which "
                "needs no helper on either platform.")

        self.options = [part for option in extra for part in ("-o", option)]

    @staticmethod
    def _putty_supports_pwfile() -> bool:
        """``-pwfile`` arrived in PuTTY 0.77; before that there is only -pw."""
        try:
            result = subprocess.run(["plink", "-V"], capture_output=True,
                                    text=True, check=False)
        except OSError:
            return False
        release = re.search(r"Release (\d+)\.(\d+)",
                            (result.stdout or "") + (result.stderr or ""))
        if not release:
            return False
        return (int(release.group(1)), int(release.group(2))) >= (0, 77)

    def _putty_password_flag(self) -> "list[str]":
        """Prefer a file over ``-pw``, which puts the password in argv."""
        if not self._putty_supports_pwfile():
            log("this PuTTY predates 0.77 and has no -pwfile, so -pw is all "
                "that is left — the password is visible in the process list "
                "for as long as each command runs. Upgrading PuTTY fixes it")
            return ["-pw", self.password]

        handle, name = tempfile.mkstemp(prefix="ship-pw-")
        pwpath = Path(name)
        with os.fdopen(handle, "w") as pwfile:
            pwfile.write(self.password + "\n")
        pwpath.chmod(0o600)
        self._pwfile = pwpath
        # Registered rather than left to a finally: die() raises SystemExit
        # from arbitrary depth, and this file must not outlive the process.
        atexit.register(self.cleanup)
        return ["-pwfile", str(pwpath)]

    def cleanup(self) -> None:
        if self._pwfile is not None:
            self._pwfile.unlink(missing_ok=True)
            self._pwfile = None


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


def ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}" if user else host


def read_token(explicit_env: str, *, ask: bool = False) -> str:
    """Resolve the Artifactory token: CONFIGURATION block, then the environment.

    Prompts as a last resort when there is a tty, so --setup does not fail at
    the very end of a slow upload over something the caller has to hand.
    """
    token = ARTIFACTORY_TOKEN or os.environ.get(explicit_env, "")
    if token:
        return token

    if ask or sys.stdin.isatty():
        try:
            token = getpass.getpass("Artifactory token: ")
        except (EOFError, KeyboardInterrupt):
            token = ""
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
                        help="project-relative path to ship; repeatable. Omit to "
                             "ship the whole project. A named directory REPLACES "
                             "its remote copy (wiped first, so local deletions "
                             "propagate); the whole-project default extracts over "
                             "the existing tree instead")
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
    parser.add_argument("--start", action="store_true",
                        help="start uvicorn under nohup after shipping")
    parser.add_argument("--restart", action="store_true",
                        help="stop uvicorn if running, then start it")
    parser.add_argument("--stop", action="store_true",
                        help="stop uvicorn and do nothing else")
    parser.add_argument("--status", action="store_true",
                        help="report whether uvicorn is running, and do nothing else")
    parser.add_argument("--bind-host", default="0.0.0.0", metavar="ADDR",
                        help="address uvicorn binds on the host (default: 0.0.0.0; "
                             "a loopback bind would only be reachable from the "
                             "box itself)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="pass --reload to uvicorn; off by default because the "
                             "reloader forks a watcher that outlives a plain kill")
    parser.add_argument("--token-env", default=TOKEN_ENV, metavar="VAR",
                        help="environment variable holding the Artifactory token, "
                             f"read only with --setup (default: {TOKEN_ENV})")
    parser.add_argument("--artifactory-user", default=None, metavar="NAME",
                        help="pip index account; default: the CONFIGURATION "
                             "block, then $ARTIFACTORY_USER")
    parser.add_argument("--artifactory-host", default=None, metavar="HOST",
                        help="pip index host; default: the CONFIGURATION "
                             "block, then $ARTIFACTORY_HOST")
    parser.add_argument("--ssh-option", action="append", default=[], metavar="OPT",
                        help="passed through to ssh/scp as -o OPT; repeatable")
    parser.add_argument("--keep-bundle", action="store_true",
                        help="do not delete the local zip afterwards")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run; still builds the zip")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

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
    target       = ssh_target(args.user, args.host)
    remote_setup = f"{remote_dir}/{SETUP_SCRIPT}"

    # Resolved before anything is built or uploaded, so a prompt appears while
    # the caller is still watching and a missing helper fails in a second.
    password  = read_password(args.password_env, ask=args.ask_password)
    transport = Transport(password, args.ssh_option)
    options   = transport.options

    def remote(command: str, *, stdin: Optional[str] = None, env_prefix: str = "") -> None:
        """One ssh call = one shell, so exports and the venv stay in scope.

        ``env_prefix`` goes before the script name, so it applies to that
        command only. Nothing passed this way is secret — the token travels
        on stdin.
        """
        prefix = f"{env_prefix} " if env_prefix else ""
        run([*transport.ssh, *options, target,
             f"cd {shlex.quote(remote_dir)} && {prefix}"
             f"{shlex.quote(remote_setup)} {command}"],
            dry_run=args.dry_run, stdin=stdin, env_extra=transport.env)

    # Pure control verbs: nothing to build or upload.
    if args.stop or args.status:
        remote("stop" if args.stop else "status")
        return 0

    # No --subdir means the whole project. Named paths are replaced wholesale on
    # the host; the whole-project default extracts over the top, because wiping
    # the remote root would take logs/, .env files and accumulated data with it.
    targets = args.subdir or ["."]
    replace = [t for t in (args.subdir or []) if (root / t).is_dir()]

    excludes = list(args.exclude)
    if not args.no_default_excludes:
        excludes += list(DEFAULT_EXCLUDE_PATHS)
    # Asking for a path outright beats excluding it by default; silently
    # shipping nothing for `--subdir data` would be the wrong reading.
    conflicting = {e for e in excludes for t in (args.subdir or [])
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
        run([*transport.ssh, *options, target,
             f"mkdir -p {shlex.quote(remote_dir)}"],
            dry_run=args.dry_run, env_extra=transport.env)
        run([*transport.scp, *options, str(bundle), str(root / SETUP_SCRIPT),
             f"{target}:{remote_dir}/"],
            dry_run=args.dry_run, env_extra=transport.env)
        run([*transport.ssh, *options, target,
             f"chmod +x {shlex.quote(remote_setup)}"],
            dry_run=args.dry_run, env_extra=transport.env)

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
            remote("setup", stdin=(token or "") + "\n", env_prefix=index)

        if args.start or args.restart:
            env = (
                f"BIND_HOST={shlex.quote(args.bind_host)} PORT={args.port} "
                f"RELOAD={'1' if args.reload else ''}"
            )
            # The env goes before the script name so it applies to that command
            # only; nothing here is secret, unlike the token above.
            run([*transport.ssh, *options, target,
                 f"cd {shlex.quote(remote_dir)} && {env} "
                 f"{shlex.quote(remote_setup)} {'restart' if args.restart else 'start'}"],
                dry_run=args.dry_run, env_extra=transport.env)
    finally:
        if args.keep_bundle:
            log(f"bundle kept at {bundle}")
        else:
            bundle.unlink(missing_ok=True)
            workdir.rmdir()

    log(f"shipped to {target}:{remote_dir}")
    if not args.setup:
        log(f"run setup later with: ssh {target} '{remote_setup} setup'")
    if not (args.start or args.restart):
        log(f"start it with:        ssh {target} '{remote_setup} start'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
