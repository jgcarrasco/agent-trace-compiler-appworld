"""The all-generate baseline: one model call with the family's traces in context.

Each task gets one prompt. It holds the system prompt the tracing agent received,
every trace the compiler was given for that family rendered as the conversation
it was, and the new task. The model answers with python, which is executed once,
block by block, against the same live world the other arms run against; then the
task is graded. There is no tool, no repair round and no second call, so what
comes back the first time is what the arm is worth.

The prompt is ordered family-constant first: the instructions, the example and
the traces are the same text for every instance of a family, and only the
request at the end carries the task's identity and its instruction. That has no
methodological content, since a model reads the whole prompt either way, but it
lets the provider's prefix cache answer most of the repeated context, which
makes this the cheapest version of the arm and the comparison against the
compiled workflow conservative.

The traces are in context verbatim and untruncated. If they did not fit the
model's window the right answer would be to say so, not to shorten them, so
nothing here trims: `prompt_chars` is recorded, and a prompt too long for the
provider fails at the provider instead of being quietly made smaller.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from traceir.engine.backend import _post, Usage
from traceir.engine.cli import load_env_file

from bench.appworld.serve import World

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "plans"
TRACES = ROOT / "runs/appworld/traces"
PROMPTS = ROOT / "bench/appworld/prompts"
STORE = ROOT / "runs/appworld/baseline/all-generate"
CONFIG = "pi-dsv41flash-high-v1"
MAX_TOKENS = 16384


def system_prompt(supervisor: dict, app_descriptions: str) -> str:
    """The tracing agent's system prompt, with its example filled in once.

    The identity here is the example conversation's, not the solved task's: the
    prompt already says the example's credentials are stale, and holding it
    constant is what lets a family's ten prompts share a prefix. The task's own
    identity travels with the request.
    """
    values = {
        "{{SUPERVISOR_NAME}}": f"{supervisor['first_name']} {supervisor['last_name']}",
        "{{SUPERVISOR_EMAIL}}": supervisor["email"],
        "{{SUPERVISOR_PHONE}}": supervisor["phone_number"],
        "{{APP_DESCRIPTIONS}}": app_descriptions,
    }
    text = (PROMPTS / "trace.md").read_text()
    for placeholder, value in values.items():
        text = text.replace(placeholder, value)
    return text


def example_identity(world: World, family: str) -> dict:
    """The identity the example conversation is filled with, constant per family.

    trace.md is written around one example task, so the identity it is rendered
    with belongs to that example. Taking it from the family's first traced
    instance keeps it a fact on disk rather than an invention, and keeps it the
    same for every run of the family.
    """
    first = sorted(path.name for path in (TRACES / CONFIG / family).iterdir() if path.is_dir())[0]
    return world.initialize(first)["supervisor"]


def transcript(session: Path) -> str:
    """One trace as readable text: the task, the reasoning, each call, each result.

    The session's own records are read in order and nothing is rewritten. The
    system message is skipped because its text is empty: the prompt that became
    it is the one this call already carries.
    """
    parts: list[str] = []
    for line in session.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "message":
            continue
        message = record["message"]
        role = message["role"]
        if role == "system":
            continue
        if role == "user":
            parts.append("".join(block.get("text", "") for block in message["content"]).strip())
        elif role == "assistant":
            for block in message["content"]:
                if block.get("type") == "thinking" and block["thinking"].strip():
                    parts.append("ASSISTANT (thinking):\n" + block["thinking"].strip())
                elif block.get("type") == "text" and block["text"].strip():
                    parts.append("ASSISTANT:\n" + block["text"].strip())
                elif block.get("type") == "toolCall":
                    parts.append(f"run(code={json.dumps(block['arguments']['code'])})")
        elif role == "toolResult":
            output = "".join(block.get("text", "") for block in message["content"])
            parts.append("Result:\n```\n" + output.strip() + "\n```")
    return "\n\n".join(parts)


def traces_of(family: str) -> tuple[str, int]:
    """Every traced instance of this family, in task order, and how many there are."""
    tasks = sorted(path for path in (TRACES / CONFIG / family).iterdir() if path.is_dir())
    rendered = [transcript(next((task / "session").glob("*.jsonl"))) for task in tasks]
    return "\n\n".join(rendered), len(tasks)


def request(task: dict) -> str:
    """What to ask once the demonstrations are in context."""
    supervisor = task["supervisor"]
    return (
        "You have no tools in this conversation. Write the python that performs this task: "
        "it will be executed once, exactly as written, in the environment the traces above "
        "ran in, and its output is the only feedback there will be. Variables you define "
        "stay available between blocks. Reply with the python in fenced blocks in the order "
        "they must run, and nothing else.\n\n"
        f"My name is: {supervisor['first_name']} {supervisor['last_name']}. "
        f"My personal email is {supervisor['email']} and phone number is {supervisor['phone_number']}.\n"
        f"Task: {task['instruction']}"
    )


def python_blocks(reply: str) -> list[str]:
    """Every fenced python block in the reply, in the order it appears."""
    return re.findall(r"```(?:python|py)\s*\n(.*?)```", reply, re.DOTALL)


def ask(prompt: str, usage: Usage, client: httpx.Client) -> tuple[str, str | None]:
    """Make the arm's one call and return the reply and how it ended."""
    payload = _post(
        client,
        os.environ["GENERATE_URL"],
        os.environ["GENERATE_API_KEY"],
        {
            "model": os.environ["GENERATE_MODEL"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
        },
        os.environ["GENERATE_MODEL"],
    )
    usage.record("generate", "one_shot", os.environ["GENERATE_MODEL"], payload.get("usage"))
    choice = payload["choices"][0]
    # A reply that ran into the token cap is still the arm's one answer: it is
    # executed as far as it goes and recorded with how it ended.
    return choice["message"]["content"], choice.get("finish_reason")


def run_task(
    family: str, task: dict, world: World, identity: dict, app_descriptions: str,
    traces: tuple[str, int], client: httpx.Client,
) -> tuple[dict, str, str, list[dict], dict]:
    """Solve one task with one call and return what the arm recorded."""
    task_id = task["task_id"]
    prompt = (system_prompt(identity, app_descriptions) + "\n\n" + traces[0]
              + "\n\n" + request(task))
    usage = Usage()
    started = time.monotonic()
    reply, finish_reason = ask(prompt, usage, client)
    seconds = round(time.monotonic() - started, 1)
    blocks = [{"code": code, "output": world.execute(task_id, code)} for code in python_blocks(reply)]
    evaluation = world.evaluate(task_id)
    reported = usage.calls[0]["usage"] or {}
    record = {
        "task": task_id,
        "family": family,
        "model": os.environ["GENERATE_MODEL"],
        "traces": traces[1],
        "prompt_chars": len(prompt),
        "reply_chars": len(reply),
        "finish_reason": finish_reason,
        "blocks": blocks,
        # What the provider says it answered from its prefix cache, so the paper
        # can state the caching effect instead of assuming it.
        "cache": {
            "hit_tokens": reported.get("prompt_cache_hit_tokens", 0),
            "miss_tokens": reported.get("prompt_cache_miss_tokens", 0),
        },
        # The benchmark's completion flag: whether the run told the system it was
        # done. There is no other sense of "completed" here, no process to exit.
        "completed": world.task_completed(task_id),
        "success": evaluation["success"],
        "tests_passed": evaluation["num_tests"] - len(evaluation["failures"]),
        "tests_total": evaluation["num_tests"],
        "seconds": seconds,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return record, prompt, reply, usage.calls, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default="p1")
    parser.add_argument("--families", default=None, help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=None, help="stop after N instances per family")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    arguments = parser.parse_args()
    load_env_file(arguments.env_file)

    plan = json.loads((PLANS / f"{arguments.plan}.json").read_text())
    families = plan["families"]
    if arguments.families:
        wanted = arguments.families.split(",")
        families = {name: value for name, value in families.items() if name in wanted}

    client = httpx.Client(timeout=1800.0)
    with World(port=arguments.port) as world:
        app_descriptions = ""
        for family, entry in families.items():
            pool = entry["heldout"][: arguments.limit]
            traces = traces_of(family)
            identity = example_identity(world, family)
            print(f"{family}: {len(pool)} held out, {traces[1]} traces in context, "
                  f"{len(traces[0])} chars, example identity "
                  f"{identity['first_name']} {identity['last_name']}")
            for index, task_id in enumerate(pool, start=1):
                directory = STORE / family / task_id
                if (directory / "record.json").is_file():
                    print(f"  [{index}/{len(pool)}] {task_id}: already run")
                    continue
                task = world.initialize(task_id)
                task["task_id"] = task_id
                if not app_descriptions:
                    # The same app set for every task, fetched once, as tracing did.
                    app_descriptions = world.execute(
                        task_id, "print(apis.api_docs.show_app_descriptions())"
                    )
                record, prompt, reply, calls, evaluation = run_task(
                    family, task, world, identity, app_descriptions, traces, client
                )
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "prompt.txt").write_text(prompt)
                (directory / "reply.txt").write_text(reply)
                (directory / "record.json").write_text(json.dumps(record, indent=2, default=str))
                (directory / "usage.json").write_text(json.dumps(calls, indent=2, default=str))
                (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
                verdict = "solved" if record["success"] else "failed"
                print(f"  [{index}/{len(pool)}] {task_id}: {verdict}, "
                      f"{record['tests_passed']}/{record['tests_total']} tests, "
                      f"{len(record['blocks'])} blocks, "
                      f"{calls[0]['usage']['prompt_tokens']} prompt tokens "
                      f"({record['cache']['hit_tokens']} cached), "
                      f"{record['seconds']}s")


if __name__ == "__main__":
    main()
