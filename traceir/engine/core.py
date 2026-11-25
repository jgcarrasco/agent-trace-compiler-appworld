"""Straight-through execution of IR v0.1 graphs.

The engine is the only graph-aware component: it resolves bindings, invokes
operations and model backends, validates results, stores state, and follows
transitions. It contains no host behavior, makes no model calls, and prints
nothing; node implementations never see the graph.

An operation is an ordinary function of its resolved inputs, so it cannot pause
the run and come back later: an operation that asks a person for something gets
the answer inside the call, and returns it. A host whose protocol cannot block
has a problem the graph does not: the engine never re-enters a node, because a
node whose effect was already delivered must never run twice.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from traceir.ir.models import (
    DecideNode,
    ExecuteNode,
    FlowNode,
    GenerateNode,
    Reference,
    Workflow,
    answer_schema,
)

# A loop is allowed, so a run needs a bound or a workflow whose exit condition
# never fires runs forever -- burning model calls while it does. This is that
# bound, and nothing more: it sits far outside any legitimate number of steps,
# because steps scale with the data a workflow happens to meet.
DEFAULT_MAX_STEPS = 1000


@dataclass(frozen=True)
class Completed:
    """The workflow terminated, with the state it accumulated and the nodes it ran."""

    output: Any
    state: dict[str, Any]
    trace: list[tuple[str, Any]]


@dataclass(frozen=True)
class Failed:
    """The workflow stopped, with as much state and trace as it had established."""

    detail: str
    state: dict[str, Any]
    trace: list[tuple[str, Any]]


Result = Completed | Failed

Generator = Callable[[GenerateNode, dict[str, Any]], Any]
Decider = Callable[[DecideNode, dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class _Context:
    """What a node can reach: the input, the state so far, and the providers."""

    initial_input: dict[str, Any]
    state: dict[str, Any]
    operations: Mapping[str, Callable[..., Any]]
    generate: Generator | None
    decide: Decider | None


def run(
    workflow: Workflow,
    initial_input: dict[str, Any],
    *,
    operations: Mapping[str, Callable[..., Any]],
    generate: Generator | None = None,
    decide: Decider | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    setup: Callable[[dict[str, Any]], None] | None = None,
    teardown: Callable[[], None] | None = None,
) -> Result:
    """Execute the graph from its entrypoint, one node at a time.

    Every EXECUTE operation must be in `operations`; a missing one fails here,
    before anything runs, rather than after the workflow has done half its work.
    Initial input that does not match `input_schema` raises, since that is the
    caller's mistake. Anything that goes wrong after that comes back as a `Failed`
    carrying the state and trace reached, because how far a run got is what you
    need when it breaks.

    `setup` prepares whatever the operations need to reach — an environment, a
    session — and runs after the input is known to be valid. Its failure is a
    `Failed` result rather than a raise, because a missing resource is not the
    caller's mistake. `teardown` runs either way and takes no arguments: it lets a
    package write down what happened without learning about this result type.

    The trace lists the nodes that completed, in order, with their values; a node
    the engine rejected is absent, so a route or output that failed its check did
    not happen as far as this run is concerned.
    """
    unbound = sorted(
        node.operation
        for node in workflow.nodes
        if isinstance(node, ExecuteNode) and node.operation not in operations
    )
    if unbound:
        raise ValueError(f"unbound operations: {', '.join(unbound)}")
    _check(workflow.input_schema, initial_input, "initial input")

    context = _Context(initial_input, {}, operations, generate, decide)
    nodes = {node.id: node for node in workflow.nodes}
    current: str | None = workflow.entrypoint
    output: Any = None
    trace: list[tuple[str, Any]] = []
    try:
        try:
            if setup is not None:
                setup(initial_input)
            while current is not None:
                if len(trace) >= max_steps:
                    raise ValueError(f"step limit of {max_steps} reached")
                node = nodes[current]
                value = _run_node(node, context)
                output = value
                current = _complete(workflow, node, value, context.state)
                trace.append((node.id, value))
        except Exception as error:
            return Failed(f"{type(error).__name__}: {error}", context.state, trace)
        return Completed(output=output, state=context.state, trace=trace)
    finally:
        # A broken teardown means the run's own record is unfinished, which is
        # worth hearing about even when the run itself succeeded.
        if teardown is not None:
            teardown()


def _run_node(node: ExecuteNode | GenerateNode | DecideNode, context: _Context) -> Any:
    """Get this node's value from the provider its type names."""
    generate, decide = context.generate, context.decide
    inputs = {
        name: _resolve(binding, context) for name, binding in node.inputs.items()
    }
    try:
        if isinstance(node, ExecuteNode):
            return context.operations[node.operation](**inputs)
        if isinstance(node, GenerateNode):
            if generate is None:
                raise ValueError("no GENERATE backend is configured")
            value = generate(node, inputs)
            _check(node.output_schema, value, f"node {node.id!r} value")
            return value
        if decide is None:
            raise ValueError("no DECIDE backend is configured")
        question = _question(node, context)
        answer = decide(node, inputs, question)
        _check(
            answer_schema(question, require_confidence=node.on == "confidence"),
            answer,
            f"node {node.id!r} answer",
        )
        return answer
    except Exception as error:
        raise ValueError(f"node {node.id!r}: {type(error).__name__}: {error}") from error


def _complete(workflow: Workflow, node: FlowNode, value: Any, state: dict) -> str | None:
    """Put the value where it belongs and say which node comes next."""
    if isinstance(node, DecideNode):
        # The answer is data for later nodes whether or not it also chose a branch.
        if node.write_to is not None:
            _store(state, node.write_to, value)
        target = _decide_target(node, value)
    elif "routes" in node.model_fields_set:
        if not isinstance(value, str) or value not in node.routes:
            raise ValueError(
                f"node {node.id!r} returned {value!r}, "
                f"which is not one of its declared routes"
            )
        target = node.routes[value]
    else:
        if node.write_to is not None:
            _store(state, node.write_to, value)
        target = node.next
    if target is None:
        _check(workflow.output_schema, value, "workflow output")
    return target


def _decide_target(node: DecideNode, answer: dict) -> str | None:
    """Follow the node's own branch. The model never chooses a successor."""
    key = node.question.type  # the answer's value field is named after it
    if "routes" in node.model_fields_set:
        label = answer[key]
        if label not in node.routes:
            # Reachable when the criteria, and so the labels, come from state.
            raise ValueError(f"node {node.id!r} answered {label!r}, which is not routed")
        return node.routes[label]
    if "bands" in node.model_fields_set:
        # The schema already required certainty when the node branches on it.
        value = answer["confidence" if node.on == "confidence" else key]
        for band in node.bands:
            if band.max is None or value <= band.max:
                return band.next
        raise ValueError(f"node {node.id!r} value {value!r} matched no band")
    return node.next


def _question(node: DecideNode, context: _Context) -> dict:
    """Resolve any criteria reference so every backend sees the same data."""
    data = node.question.model_dump(exclude_unset=True)
    criteria = node.question.criteria
    if isinstance(criteria, Reference):
        data["criteria"] = _resolve(criteria, context)
    return data


def _resolve(binding: Reference | Any, context: _Context) -> Any:
    if isinstance(binding, Reference):
        return _deref(binding.ref, context)
    return binding.literal


def _deref(pointer: str, context: _Context) -> Any:
    value: Any = {"input": context.initial_input, "state": context.state}
    for token in pointer[1:].split("/"):
        token = _decode(token)
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError(f"unresolved reference {pointer!r}")
    return value


def _check(schema: Any, value: Any, label: str) -> None:
    """Fail with the first schema violation, named by what was being checked."""
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error is not None:
        location = "/".join(str(part) for part in error.path)
        detail = f"{error.message} (at {location})" if location else error.message
        raise ValueError(f"{label} does not match its schema: {detail}")


def _decode(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _store(state: dict, pointer: str, value: Any) -> None:
    if not pointer.startswith("/state/"):
        raise ValueError(f"invalid state destination {pointer!r}")
    tokens = [_decode(token) for token in pointer[len("/state/") :].split("/")]
    target = state
    for token in tokens[:-1]:
        target = target.setdefault(token, {})
    target[tokens[-1]] = value
