"""Running a coding agent over a compile workspace.

The compiler is an agent, not a model call: it needs to read the target's source,
write a package, and run what it wrote until the package works. This module owns
that invocation — the command, the sandbox around it, and a record of what
happened — so the research question stays about prompts and artifacts rather than
about process plumbing.

Target knowledge lives elsewhere. A caller supplies the workspace, the prompt, and
the model; nothing here knows what a task is.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROMPT = Path(__file__).with_name("prompt.md")

# rootfs the agent may read. `/etc` carries TLS roots; `/bin`, `/lib` and `/lib64`
# are symlinks into `/usr` on Arch and the launchers are reached through them, so
# all of them have to be mounted or `execvp` fails obscurely. `/run` is there for
# one reason: `/etc/resolv.conf` is a systemd symlink into it, and without it DNS
# inside the sandbox fails with "Temporary failure in name resolution".
BASE_READS = (
    Path("/usr"),
    Path("/etc"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/run"),
)


@dataclass(frozen=True)
class Attempt:
    """One agent run, and where its evidence is."""

    exit_code: int
    seconds: float
    run_dir: Path
    session_dir: Path
    stdout_path: Path
    stderr_path: Path

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def sandbox_command(
    command: list[str],
    *,
    writable: tuple[tuple[Path, Path], ...],
    readable: tuple[tuple[Path, Path], ...],
    env: dict[str, str],
    chdir: Path,
) -> list[str]:
    """Wrap a command so it sees only these paths, each at its given location.

    Reads are mounted read-only, writes read-write, and everything else simply
    does not exist inside the namespace. Adding a path here is a decision about
    what the compiler may read, so each entry deserves a reason.
    """
    sources = [source for source, _ in (*readable, *writable)]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot sandbox: no such path to bind: {', '.join(missing)}")
    argv = ["bwrap", "--dev", "/dev", "--proc", "/proc", "--dir", "/tmp"]
    for source, target in readable:
        argv += ["--ro-bind", str(source), str(target)]
    for source, target in writable:
        argv += ["--bind", str(source), str(target)]
    argv += ["--chdir", str(chdir)]
    for name, value in env.items():
        argv += ["--setenv", name, value]
    return [*argv, "--", *command]


def read_profile() -> dict[str, str]:
    """Machine paths the agent needs to run, derived rather than hardcoded."""
    pi_launcher = shutil.which("pi")
    node = shutil.which("node")
    if pi_launcher is None or node is None:
        raise RuntimeError("cannot sandbox: `pi` and `node` must be on PATH")
    return {
        # The agent's own installation, read-only: its launcher, its JavaScript,
        # and the interpreter that runs it.
        "pi_bin": str(Path(pi_launcher).parent),
        "pi_install": str(Path(pi_launcher).parent.parent / "install"),
        "node": str(Path(node).resolve().parents[1]),
        # What the compiled package runs on: the interpreter and the environment
        # holding its dependencies. The interpreter is reached through a version
        # symlink, so its parent directory is what has to be mounted, not the
        # resolved version it points at.
        "python": str(Path(sys.base_prefix).parent),
        "venv": str(sys.prefix),
    }


def run_agent(
    workspace: Path,
    *,
    model: str,
    run_dir: Path,
    env: dict[str, str] | None = None,
    prompt: str | None = None,
    thinking: str = "high",
    timeout: float = 3600.0,
    sandbox: bool = True,
) -> Attempt:
    """Run the compiler agent once, over `workspace`, recording everything.

    The workspace is the agent's working directory and keeps its host path inside
    the sandbox, so `env` — the target's own variables, such as where its sources
    and data live — means the same thing whether or not the run is sandboxed. The
    run directory holds what the agent needs to configure itself: a model entry,
    an agent directory, a writable home, and everything it produced.
    """
    prompt = PROMPT.read_text() if prompt is None else prompt
    # The caller may already have created the run directory to build the
    # workspace in; `logs` is what makes a fresh directory required.
    run_dir.mkdir(parents=True, exist_ok=True)
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=False)
    session = run_dir / "session"
    agent_dir = run_dir / "agent"
    home = run_dir / "home"
    agent_dir.mkdir()
    home.mkdir()

    key = os.environ.get("COMPILER_API_KEY")
    base_url = os.environ.get("COMPILER_BASE_URL")
    if not key or not base_url:
        raise RuntimeError("COMPILER_API_KEY and COMPILER_BASE_URL must be set")
    (agent_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "deepseek": {
                        "baseUrl": base_url,
                        "api": "openai-completions",
                        "apiKey": "$COMPILER_API_KEY",
                        "models": [{"id": model}],
                    }
                }
            },
            indent=2,
        )
    )

    command = [
        "pi", "--print",
        "--model", f"deepseek/{model}",
        "--thinking", thinking,
        "--tools", "read,bash,edit,write",
        "--session-dir", str(session),
        prompt,
    ]
    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "COMPILER_API_KEY": key,
        "LANG": "C.UTF-8",
        **(env or {}),
    }
    if sandbox:
        profile = read_profile()
        # `python` must be the venv's, or the compiled package cannot import its
        # dependencies; the venv is bound alongside the interpreter it points at.
        child_env["PATH"] = ":".join(
            [
                str(Path(profile["node"]) / "bin"),
                profile["pi_bin"],
                str(Path(profile["venv"]) / "bin"),
                "/usr/bin",
                "/bin",
            ]
        )
        command = sandbox_command(
            command,
            writable=((workspace, workspace), (run_dir, run_dir)),
            readable=(
                *((path, path) for path in BASE_READS),
                (Path(profile["pi_bin"]), Path(profile["pi_bin"])),
                (Path(profile["pi_install"]), Path(profile["pi_install"])),
                (Path(profile["node"]), Path(profile["node"])),
                (Path(profile["python"]), Path(profile["python"])),
                (Path(profile["venv"]), Path(profile["venv"])),
            ),
            env=child_env,
            chdir=workspace,
        )

    stdout_path = logs / "stdout.txt"
    stderr_path = logs / "stderr.txt"
    started = time.monotonic()
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        completed = subprocess.run(
            command,
            cwd=None if sandbox else workspace,
            env=None if sandbox else child_env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            timeout=timeout,
            check=False,
        )
    seconds = time.monotonic() - started

    (run_dir / "prompt.txt").write_text(prompt)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "model": model,
                "thinking": thinking,
                "sandboxed": sandbox,
                "command": [part for part in command if part != key],
                "workspace": str(workspace),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "manifest_sha256": _manifest_hash(workspace),
                "seconds": round(seconds, 3),
                "exit_code": completed.returncode,
                "pi_version": _pi_version(),
            },
            indent=2,
        )
    )
    return Attempt(
        exit_code=completed.returncode,
        seconds=seconds,
        run_dir=run_dir,
        session_dir=session,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _pi_version() -> str | None:
    try:
        completed = subprocess.run(
            ["pi", "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _manifest_hash(workspace: Path) -> str | None:
    """The workspace builder's own hash of the material it shipped.

    What belongs in a workspace is the target's business, so the builder writes
    the manifest and this only records which one was used.
    """
    manifest = workspace / "MANIFEST.json"
    return sha256(manifest) if manifest.is_file() else None
