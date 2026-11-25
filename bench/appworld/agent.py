"""The trace agent: one pi run over one task.

The agent is a general coding agent with a single tool, `run`, that executes
python in the task's apps, and the benchmark's own instructions ported to that
channel. Its prompt is a system prompt replaced wholesale, so pi's coding
guidance is not in the agent's context; the tool's schema reaches the model in
the request, which is why the agent can call `run` without being told how.

What it leaves behind is pi's own session log. That log is the trace: thinking,
each call with its code, each output, and the token counts per turn.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "bench/appworld/tools/run.ts"
PROMPTS = ROOT / "bench/appworld/prompts"
MODEL = "opencode-go/deepseek-v4.1-flash"
TIMEOUT = 1800.0


def render(task: dict, directory: Path) -> tuple[Path, Path]:
    """Fill the two prompt files with this task's supervisor and instruction."""
    supervisor = task["supervisor"]
    values = {
        "{{SUPERVISOR_NAME}}": f"{supervisor['first_name']} {supervisor['last_name']}",
        "{{SUPERVISOR_EMAIL}}": supervisor["email"],
        "{{SUPERVISOR_PHONE}}": supervisor["phone_number"],
        "{{APP_DESCRIPTIONS}}": task["app_descriptions"],
        "{{INSTRUCTION}}": task["instruction"],
    }
    written = []
    for name in ("trace", "task"):
        text = (PROMPTS / f"{name}.md").read_text()
        for placeholder, value in values.items():
            text = text.replace(placeholder, value)
        path = directory / f"{name}.txt"
        path.write_text(text)
        written.append(path)
    return written[0], written[1]


def summarize(session: Path) -> dict:
    """Token counts and tool calls, summed from the session's own records."""
    turns = calls = 0
    usage = {"input": 0, "cacheRead": 0, "cacheWrite": 0, "output": 0, "reasoning": 0}
    for line in session.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        message = record.get("message") if record.get("type") == "message" else None
        if not message or message.get("role") != "assistant":
            continue
        turns += 1
        for key in usage:
            usage[key] += message.get("usage", {}).get(key, 0)
        calls += sum(1 for block in message.get("content", []) if block.get("type") == "toolCall")
    return {"turns": turns, "calls": calls, **usage}


def run(
    task_id: str,
    instruction: str,
    supervisor: dict,
    directory: Path,
    *,
    env_url: str,
    model: str = MODEL,
    thinking: str = "high",
    app_descriptions: str = "",
) -> dict:
    """Run the agent once, in `directory`, and return what it recorded."""
    directory.mkdir(parents=True, exist_ok=True)
    system_prompt, task_line = render(
        {
            "supervisor": supervisor,
            "instruction": instruction,
            "app_descriptions": app_descriptions,
        },
        directory,
    )
    session_directory = directory / "session"
    session_directory.mkdir(exist_ok=True)
    command = [
        "pi", "--print",
        "--model", model,
        "--thinking", thinking,
        "--system-prompt", system_prompt.read_text(),
        "--tools", "run",
        "--extension", str(TOOL),
        "--session-dir", str(session_directory),
        "--name", task_id,
        "--no-extensions", "--no-context-files", "--no-skills",
        "--no-prompt-templates", "--no-mcp",
        task_line.read_text(),
    ]
    env = {
        **os.environ,
        "APPWORLD_ENV_URL": env_url,
        "APPWORLD_TASK_ID": task_id,
    }
    started = time.monotonic()
    timed_out = False
    with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
        try:
            completed = subprocess.run(
                command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, timeout=TIMEOUT, check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
    seconds = round(time.monotonic() - started, 1)

    sessions = sorted(session_directory.glob("*.jsonl"))
    result = {
        "task_id": task_id,
        "instruction": instruction,
        "model": model,
        "thinking": thinking,
        "prompt_sha256": hashlib.sha256(system_prompt.read_bytes()).hexdigest(),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "seconds": seconds,
    }
    if sessions:
        result["session"] = str(sessions[-1].relative_to(directory))
        result.update(summarize(sessions[-1]))
    return result
