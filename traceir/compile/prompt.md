# Compiling traces into a workflow

You are given a workspace containing successful agent traces for one task. Your
job is to produce an IR v0.1 workflow that performs the same task and reaches
the same outcome, and to check that it does.

Read `ir/docs.md` for the artifact contract and `ir/schema.json` for the exact
field types. The traces are in `traces/`.

## What to produce

Write the package into `out/`:

- `out/graph.json` — the workflow, validated against `ir/schema.json`;
- `out/operations.py` — the code the graph's EXECUTE nodes call, defining
  `WORKFLOW_OPERATIONS` as a `dict[str, Callable]`, plus optional
  `setup(initial_input)` and `teardown()`;
- `out/input.example.json` — the input that makes one run work;
- `out/REPORT.md` — what you built, how you checked it, and what you know is
  missing.

`setup` prepares anything the operations need to reach, and receives the initial
input. `teardown` takes no arguments and runs however the run ended, so it is
where a record of the run belongs. Neither is required if the operations need
neither.

## How to compile

A workflow encodes the *procedure* the traces followed, not one instance of it:

- every value that came from the task — user, order, item, product, variant,
  payment method — is a binding or a node output, never a constant in the code;
- filtering, arithmetic, comparisons, and choosing according to a fixed rule are
  EXECUTE, not a model call;
- matching natural language against explicit candidates is a DECIDE question
  whose criteria are those candidates;
- proposing an action and receiving the answer to the proposal are separate
  steps, and the answer's branch is what authorizes the mutation;
- a GENERATE node writes text and must not reach anything that changes state;
- a branch that the traces never exercised is a safeguard: keep it, and mark it
  as unobserved.

Keep the graph honest about what it is:
 
- no EXECUTE operation may call a model;
- no single node may implement the whole workflow — the graph must show the
  steps;
- no operation may take over routing, looping, or state the engine already
  handles.

## How to check it

`ir/docs.md` describes how to run a package: `python -m traceir.engine
--workflow out/ --input out/input.example.json`. Run it against the system the
traces reached; the deployment says how to reach it. A package whose operations
answer from data they carry reproduces the trace by construction and has
checked nothing.

Check, at least:

- the graph validates;
- a run reaches its terminal node;
- the state changes it makes are the ones the traces made, with the same
  identifiers and amounts;
- the recorded run in `out/run.json` matches what actually happened.

Iterate until that holds, then write `out/REPORT.md` and stop. If something
cannot be made to work, say so in the report rather than leaving it out — a
known gap is more useful than a silent omission.
