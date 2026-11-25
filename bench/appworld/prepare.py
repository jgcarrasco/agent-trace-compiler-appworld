"""Generate the instance set for one experiment and split it.

AppWorld gives three things that shape this: a generator yields about six tasks
per process and then exhausts its search space, task ids are positional so a
second run overwrites the first, and more seeds buy new worlds at the same
instructions. So a family is built from several seeded runs, each in its own
root, merged here under fresh ids.

The split keeps no instruction out of the traces: every instruction in the
held-out set appears in at least one trace. What the held-out set withholds is
the world — databases, users, identifiers, dates. Compiling a branch the traces
never exercised is a different question, measured elsewhere.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPWORLD = ROOT / "appworld"
APPWORLD_PYTHON = ROOT / "bench/appworld/.venv/bin/python"
DATA = ROOT / "appworld_gen"
SHARED = ROOT / "appworld_data/data"
TASKS_PER_RUN = 6
MAX_SEEDS = 6
TARGET_INSTANCES = 20
HELDOUT = 10


def run_generation(generator: str, seed: int, staging: Path) -> None:
    """One generation process, into its own root, so ids cannot overwrite."""
    (staging / "data/tasks").mkdir(parents=True, exist_ok=True)
    for name in ("base_dbs", "api_docs", "datasets", "version.txt"):
        link = staging / "data" / name
        if not link.exists():
            link.symlink_to(SHARED / name)
    env = {
        **os.environ,
        "APPWORLD_ROOT": str(staging),
        "APPWORLD_CACHE": str(staging / ".cache"),
        "PYTHONHASHSEED": "0",
        # The generator shells out to ruff for its static checks.
        "PATH": f"{APPWORLD_PYTHON.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    completed = subprocess.run(
        [
            str(APPWORLD_PYTHON), "-m", "generate.tasks.generate_and_validate_tasks",
            generator,
            "--generator_num_tasks", str(TASKS_PER_RUN),
            "--random_seed", str(seed),
        ],
        cwd=APPWORLD, env=env, capture_output=True, text=True, timeout=3600,
    )
    count = len(list((staging / "data/tasks").iterdir()))
    status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
    print(f"  seed {seed}: {count} tasks ({status})")


def signature(task: Path) -> str:
    """What makes two instances the same task: the world and the request."""
    specs = json.loads((task / "specs.json").read_text())
    parts = [specs["instruction"], json.dumps(specs["supervisor"], sort_keys=True), specs["datetime"]]
    for path in sorted((task / "dbs").glob("*.jsonl")):
        parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def collect(generator: str, seeds: int, target: int) -> list[dict]:
    """Generate until the family is big enough, keeping every distinct world."""
    staging_root = ROOT / "runs/appworld/gen" / generator
    instances: list[dict] = []
    seen: set[str] = set()
    for seed in range(1, seeds + 1):
        staging = staging_root / f"seed{seed}"
        tasks = staging / "data/tasks"
        if not tasks.is_dir() or not any(tasks.iterdir()):
            run_generation(generator, seed, staging)
        for task in sorted(tasks.iterdir()):
            digest = signature(task)
            if digest in seen:
                continue
            seen.add(digest)
            specs = json.loads((task / "specs.json").read_text())
            instances.append(
                {"path": task, "seed": seed, "signature": digest, "instruction": specs["instruction"]}
            )
        print(f"  after seed {seed}: {len(instances)} distinct instances")
        if len(instances) >= target:
            break
    return instances


def split(instances: list[dict], heldout_size: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """Choose the held-out set, keeping every instruction covered by the traces.

    One instance of every instruction goes to the dev pool first, so a held-out
    instruction can never be one the compiler has not seen. The rest are shuffled
    and the first ten become held out.
    """
    rng = random.Random(seed)
    by_instruction: dict[str, list[dict]] = {}
    for instance in instances:
        by_instruction.setdefault(instance["instruction"], []).append(instance)
    dev, rest = [], []
    for group in by_instruction.values():
        rng.shuffle(group)
        dev.append(group[0])
        rest.extend(group[1:])
    rng.shuffle(rest)
    heldout = rest[: min(heldout_size, len(rest))]
    dev.extend(rest[len(heldout):])
    return [i["signature"] for i in dev], [i["signature"] for i in heldout]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", default="07b42fd,302c169,ce359b5")
    parser.add_argument("--experiment", default="p1")
    parser.add_argument("--seeds", type=int, default=MAX_SEEDS)
    parser.add_argument("--target", type=int, default=TARGET_INSTANCES,
                        help="instances to gather per family before stopping")
    parser.add_argument("--heldout", type=int, default=HELDOUT)
    parser.add_argument("--force", action="store_true", help="regenerate the staging roots")
    arguments = parser.parse_args()
    if arguments.force:
        shutil.rmtree(ROOT / "runs/appworld/gen", ignore_errors=True)
        shutil.rmtree(DATA / "data/tasks", ignore_errors=True)
    (DATA / "data/tasks").mkdir(parents=True, exist_ok=True)

    plan = {"experiment": arguments.experiment, "families": {}}
    for generator in arguments.families.split(","):
        print(f"{generator}:")
        instances = collect(generator, arguments.seeds, arguments.target)
        dev, heldout = split(instances, arguments.heldout)
        # Fresh ids: the generator's own are positional and collide across seeds.
        named = []
        for index, instance in enumerate(instances, start=1):
            task_id = f"{generator}_{index:02d}"
            destination = DATA / "data/tasks" / task_id
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(instance["path"], destination, ignore=shutil.ignore_patterns("__pycache__"))
            named.append(
                {
                    "task_id": task_id,
                    "signature": instance["signature"],
                    "instruction": instance["instruction"],
                    "seed": instance["seed"],
                }
            )
        renamed = {i["signature"]: n["task_id"] for i, n in zip(instances, named)}
        plan["families"][generator] = {
            "instances": named,
            "dev_pool": [renamed[s] for s in dev],
            "heldout": [renamed[s] for s in heldout],
            "instruction_count": len({i["instruction"] for i in instances}),
        }
        print(f"  {len(instances)} instances, {plan['families'][generator]['instruction_count']} instructions, "
              f"{len(dev)} dev pool, {len(heldout)} held out")
    path = ROOT / "plans" / f"{arguments.experiment}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
