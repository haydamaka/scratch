#!/usr/bin/env python3
"""
Ship a project subdirectory to a remote host: zip locally, scp, unpack there.

Written for hosts that cannot reach github.com — cloning is not an option, so
the working tree travels as a bundle instead. Uncommitted work is included on
purpose (that is usually the reason for shipping at all).

This script does local work and transport only. Every remote step lives in
``ship_remote.sh`` and runs in a single shell there, because ``export`` and
``source`` do not survive separate ssh invocations.

Transport is the system ``ssh``/``scp``, not a library: they already honour
``~/.ssh/config``, agent auth, ProxyJump and known_hosts, and need no package
installed from an index this host may not be able to reach.

    export SHIP_HOST=build01.example.net SHIP_USER=alice
    python ship_to_host.py --subdir app --subdir requirements.txt
    python ship_to_host.py --subdir app --setup      # + venv and pip install
    python ship_to_host.py --subdir app --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Site-specific; set them in the environment rather than editing this file.
DEFAULT_HOST       = os.environ.get("SHIP_HOST", "")
DEFAULT_USER       = os.environ.get("SHIP_USER", "")
DEFAULT_REMOTE_DIR = os.environ.get("SHIP_REMOTE_DIR", "/tmp/{user}/project")
SETUP_SCRIPT       = "ship_remote.sh"
BUNDLE_NAME        = "_bundle.zip"

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


def log(message: str) -> None:
    print(f"[ship] {message}", file=sys.stderr, flush=True)


def die(message: str) -> "None":
    print(f"[ship] ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def run(cmd: Sequence[str], *, dry_run: bool = False, stdin: Optional[str] = None) -> None:
    """Run a command, streaming its output; raise on a non-zero exit.

    ``stdin`` is written to the child rather than passed as an argument — it
    carries the Artifactory token, and argv is world-readable through ``ps``.
    """
    printable = " ".join(shlex.quote(part) for part in cmd)
    if dry_run:
        log(f"DRY-RUN {printable}" + ("  <<< (secret on stdin)" if stdin else ""))
        return

    log(printable)
    result = subprocess.run(
        cmd,
        input=stdin.encode() if stdin is not None else None,
        check=False,
    )
    if result.returncode != 0:
        die(f"command failed with exit {result.returncode}: {printable}")


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


def read_token(explicit_env: str) -> str:
    token = os.environ.get(explicit_env, "")
    if not token:
        die(
            f"--setup needs the Artifactory token in ${explicit_env}. "
            f"Export it first (and keep it out of shell history):\n"
            f"    read -rs {explicit_env} && export {explicit_env}"
        )
    return token


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
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="target host; default: $SHIP_HOST")
    parser.add_argument("--user", default=DEFAULT_USER,
                        help="remote user; pass '' to rely on ~/.ssh/config")
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
    parser.add_argument("--token-env", default="ARTIFACTORY_TOKEN", metavar="VAR",
                        help="environment variable holding the Artifactory token, "
                             "read only with --setup (default: ARTIFACTORY_TOKEN)")
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
        die("no target host — pass --host or set $SHIP_HOST")

    root = (args.project_root or Path(__file__).resolve().parent).resolve()
    if not (root / SETUP_SCRIPT).is_file():
        die(f"{SETUP_SCRIPT} not found next to the project root ({root})")

    remote_dir = args.remote_dir or DEFAULT_REMOTE_DIR.format(
        user=args.user or os.environ.get("USER", "user")
    )
    target       = ssh_target(args.user, args.host)
    remote_setup = f"{remote_dir}/{SETUP_SCRIPT}"

    options: "list[str]" = []
    for option in args.ssh_option:
        options += ["-o", option]

    def remote(command: str, *, stdin: Optional[str] = None) -> None:
        """One ssh call = one shell, so exports and the venv stay in scope."""
        run(["ssh", *options, target,
             f"cd {shlex.quote(remote_dir)} && {shlex.quote(remote_setup)} {command}"],
            dry_run=args.dry_run, stdin=stdin)

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

    # Checked before any transport, so a missing token fails in a second rather
    # than after a multi-megabyte upload.
    token = read_token(args.token_env) if args.setup else None

    workdir = Path(tempfile.mkdtemp(prefix="ship-"))
    bundle  = build_bundle(root, targets, workdir / BUNDLE_NAME, excludes)

    try:
        run(["ssh", *options, target, f"mkdir -p {shlex.quote(remote_dir)}"],
            dry_run=args.dry_run)
        run(["scp", *options, str(bundle), str(root / SETUP_SCRIPT),
             f"{target}:{remote_dir}/"],
            dry_run=args.dry_run)
        run(["ssh", *options, target, f"chmod +x {shlex.quote(remote_setup)}"],
            dry_run=args.dry_run)

        if replace:
            log(f"replacing on the host: {', '.join(replace)}")
        remote("unpack " + " ".join(shlex.quote(p) for p in replace))

        if args.setup:
            log("running remote setup (token piped on stdin, never in argv)")
            remote("setup", stdin=(token or "") + "\n")

        if args.start or args.restart:
            env = (
                f"BIND_HOST={shlex.quote(args.bind_host)} PORT={args.port} "
                f"RELOAD={'1' if args.reload else ''}"
            )
            # The env goes before the script name so it applies to that command
            # only; nothing here is secret, unlike the token above.
            run(["ssh", *options, target,
                 f"cd {shlex.quote(remote_dir)} && {env} "
                 f"{shlex.quote(remote_setup)} {'restart' if args.restart else 'start'}"],
                dry_run=args.dry_run)
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
