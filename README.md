# agent-trace-compiler-appworld

Code for the paper *Compiling Agent Traces into Typed Workflows for Cheaper and
Faster Execution of Recurring Tasks* (Knowledge-Based Systems, under review).

The repository has two halves. `traceir/` is the intermediate representation and
its engine: a workflow is a directed graph of EXECUTE (computation and tool
calls), DECIDE (one bounded choice over declared candidates) and GENERATE
(open-ended inference), and the engine runs it, calling a model only at the
nodes that declare one. `bench/appworld/` is the experiment: it traces an agent
on AppWorld, compiles the traces into such a graph, and measures the graph
against the agent on held-out instances.

## Reproducing the paper's numbers

The experiment records are archived separately on Zenodo and published with the
paper. Download the archive and extract it at the repository root, so that
`runs/appworld/` sits next to `bench/`, then:

    uv sync
    uv run python -m bench.appworld.audit

The audit recomputes every number in the paper from the records, in the order
the paper uses them, and fails when its output differs from
`bench/appworld/audit.txt`. It needs no API key, no model and no AppWorld
installation.

## Layout

    traceir/            the IR specification, validation, engine, compiler runner
      compile/          the compiler's instructions
    bench/appworld/     the experiment: one module per stage, driven by -m
    bench/pricing.py    the price table every cost in the paper is computed with
    plans/              the instance split: which instances were traced, which held out
    tests/              the library's tests
    workflows/          a worked example the tests and docs/ir-v0.1.md use
    docs/ir-v0.1.md     the specification the compiler agent reads, not prose about it

## Running the experiment from scratch

Prerequisites: Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), the
AppWorld environment (a pinned clone and its 189 MB dataset, about 1 GB in
total). The tracing and compiling stages also need pi, node on `PATH`, and, for
compiling only, `bubblewrap` (the compiler runs the agent in a sandbox). The
figure stage needs graphviz's `dot`. The benchmark pins
fastapi 0.150.0, which builds its models with pydantic 2.12.5, while the library
tracks a later pydantic; the two do not share an environment, so the benchmark
gets its own:

    bash bench/appworld/setup.sh

That script clones AppWorld at the commit the paper used (`42b5bcf`), downloads
the `0.2.0` data, unpacks the task generators and creates
`bench/appworld/.venv`. Re-running it is cheap.

Then the stages, in order:

    uv run python -m bench.appworld.prepare      # plans/p1.json: 76 instances, 30 held out
    uv run python -m bench.appworld.trace        # the agent on every dev instance
    uv run python -m bench.appworld.compile --family 07b42fd
    uv run python -m bench.appworld.run --package <out> --task <task_id> --out <dir>
    uv run python -m bench.appworld.trace --pool heldout   # the agent on the held-out set
    uv run python -m bench.appworld.baseline_generate      # the all-generate arm
    uv run python -m bench.appworld.run --decide-backend chat ...  # the IR arm with a chat model
    uv run python -m bench.appworld.coverage     # held-out success against the number of traces
    uv run python -m bench.appworld.figures      # the paper's figures, written to figures/

`prepare` seeds a world per process and merges the generated tasks under fresh
ids; it rewrites `plans/p1.json`, which is why the committed plan is the recorded
split and not a scratch output; `trace` runs one pi session per dev instance and
keeps the session log as the trace; `compile` gives those traces to the compiler agent, which writes a
package and exercises it against a live world that holds only the traced
instances. Held-out instances are never reachable from the compiler.

`run.py` executes one package on one task and records the result, the node
trace, the state and the usage of every model call; `--package` is the `out/`
directory of a compile. The held-out sweep of the packages is that command in a
loop over the plan's held-out ids, writing one `--out` directory per instance; the agent arm
is `trace --pool heldout`, which stores its runs under `runs/appworld/baseline`,
and the all-generate arm is `baseline_generate`, one call per task with the
family's traces in the prompt.

`coverage.py --ks 1,3,5,10,all --repeats 2` reproduces both the coverage curve
and the stability recompiles.

## What the re-run needs

Tracing and compiling drive the [pi](https://github.com/earendil-works/pi) coding
agent. The benchmark talks to two provider APIs: an OpenAI-compatible chat
endpoint for the generated nodes, and the typed Decisions endpoint for the
bounded ones. A `.env` at the repository root supplies them (see `.env.example`):

    GENERATE_URL, GENERATE_MODEL, GENERATE_API_KEY     the open-ended nodes
    DECIDE_URL, DECIDE_MODEL, DECIDE_API_KEY          the bounded decisions
    COMPILER_BASE_URL, COMPILER_MODEL, COMPILER_API_KEY   the compiler agent

The trace model is fixed in `bench/appworld/agent.py`; the compiler model comes
from the environment. The recorded compiles used `deepseek-flash` for both the
compiler and the generated nodes, and `jev-preview` for the decisions. The price
table in `bench/pricing.py` records the prices the paper used and the date they
were read, so a later price change shows up as a difference rather than as a
silent one.

## Tests

    uv run --with pytest python -m pytest tests

The suite covers the IR, the engine, the backend and the package format. One
test asserts that `traceir/` imports no benchmark, which is the boundary the
library is built on.

## Third-party work

AppWorld is not distributed here. Clone it from
[StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld) at the pinned
commit, as `bench/appworld/setup.sh` does, and cite its paper if you use it.

`bench/appworld/prompts/trace.md` adapts AppWorld's
`experiments/prompts/full_code_agent/full_code_instructions.txt` (Apache-2.0),
rewriting the tool channel and the completion rule for this harness. See
`bench/appworld/prompts/README.md`.

## Licence

Apache-2.0 for the code. The experiment records are CC BY 4.0.
