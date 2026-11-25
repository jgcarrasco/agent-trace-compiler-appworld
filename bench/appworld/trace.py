"""Gather the traces: run the agent on every dev instance of every family.

One world serves a family; each dev instance is initialized, run once, and
evaluated. A run is kept whatever its outcome — a failure is evidence about the
task, and the trace store is where "how often does this agent solve this family"
becomes a number rather than an impression. Nothing is retried.

Held-out instances are never touched here: they exist for the comparison in the
run stage, and the agent's traces must not contain them.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.appworld import agent
from bench.appworld.serve import World

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "plans"
TRACES = ROOT / "runs/appworld/traces"


def traced(store: Path, config: str, family: str, task_id: str) -> bool:
    return (store / config / family / task_id / "meta.json").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", default="p1")
    parser.add_argument("--config", default="pi-dsv41flash-high-v1")
    parser.add_argument("--families", default=None, help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=None, help="stop after N instances per family")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--force", action="store_true", help="re-trace instances that already have a meta.json")
    parser.add_argument("--pool", choices=("dev", "heldout"), default="dev",
                        help="which side of the split to run (the held-out side is the baseline arm)")
    parser.add_argument("--store", type=Path, default=None,
                        help="where runs are kept (default: traces/ for dev, baseline/ for heldout)")
    arguments = parser.parse_args()

    store = arguments.store or (TRACES if arguments.pool == "dev" else ROOT / "runs/appworld/baseline")
    plan = json.loads((PLANS / f"{arguments.plan}.json").read_text())
    families = plan["families"]
    if arguments.families:
        wanted = arguments.families.split(",")
        families = {name: value for name, value in families.items() if name in wanted}

    started = time.monotonic()
    with World(port=arguments.port) as world:
        app_descriptions = ""
        for family, entry in families.items():
            key = "dev_pool" if arguments.pool == "dev" else "heldout"
            pool = entry[key][: arguments.limit]
            print(f"{family}: {len(pool)} {arguments.pool} instances")
            for index, task_id in enumerate(pool, start=1):
                directory = store / arguments.config / family / task_id
                if traced(store, arguments.config, family, task_id) and not arguments.force:
                    print(f"  [{index}/{len(pool)}] {task_id}: already traced")
                    continue
                task = world.initialize(task_id)
                if not app_descriptions:
                    # The same app set for every task; fetched once, as the
                    # benchmark's own prompt shows it in an example output.
                    app_descriptions = world.execute(
                        task_id, "print(apis.api_docs.show_app_descriptions())"
                    )
                record = agent.run(
                    task_id,
                    task["instruction"],
                    task["supervisor"],
                    directory,
                    env_url=world.url,
                    app_descriptions=app_descriptions,
                )
                completed_flag = world.task_completed(task_id)
                evaluation = world.evaluate(task_id)
                (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
                meta = {
                    **record,
                    "family": family,
                    "config": arguments.config,
                    "pool": arguments.pool,
                    "success": evaluation["success"],
                    "tests_passed": evaluation["num_tests"] - len(evaluation["failures"]),
                    "tests_total": evaluation["num_tests"],
                    "task_completed": completed_flag,
                    "datetime": task["datetime"],
                    "traced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                (directory / "meta.json").write_text(json.dumps(meta, indent=2))
                verdict = "solved" if meta["success"] else "failed"
                print(f"  [{index}/{len(pool)}] {task_id}: {verdict}, {meta['tests_passed']}/{meta['tests_total']} "
                      f"tests, {meta.get('calls', 0)} calls, {meta['seconds']}s")
    print(f"\ndone in {(time.monotonic() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
