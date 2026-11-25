"""Validate IR artifacts without importing operations or contacting services."""

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path
from urllib.parse import unquote

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError

from traceir.ir.models import (
    DecideNode,
    ExecuteNode,
    GenerateNode,
    Workflow,
    graph_schema,
)

DRAFT = "https://json-schema.org/draft/2020-12/schema"


class IRValidationError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("\n".join(issues))


def _location(parts) -> str:
    tokens = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(tokens)


def _local_reference(root, reference: str):
    """Return the local target and its document path, or None."""
    fragment = unquote(reference[1:])
    if fragment == "":
        return root, ()
    if not fragment.startswith("/") or re.search(r"~(?![01])", fragment):
        return None
    value = root
    path = []
    for token in fragment[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
            path.append(token)
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", token):
            index = int(token)
            if index >= len(value):
                return None
            value = value[index]
            path.append(index)
        else:
            return None
    return (value, tuple(path)) if isinstance(value, (dict, bool)) else None


def _schema_nodes(schema, path=(), *, root=None, seen=None) -> Iterator[tuple[dict, tuple]]:
    """Visit schema positions and referenced targets, not unused literal data."""
    if not isinstance(schema, dict):
        return
    if root is None:
        root = schema
    if seen is None:
        seen = set()
    if id(schema) in seen:
        return
    seen.add(id(schema))
    yield schema, path
    # A local pointer can target a schema under an arbitrary keyword, not just
    # $defs. Inspect it too, before any instance validation can follow its refs.
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#"):
        target = _local_reference(root, reference)
        if target is not None:
            yield from _schema_nodes(target[0], target[1], root=root, seen=seen)
    for keyword in ("$defs", "properties", "patternProperties", "dependentSchemas"):
        for name, child in schema.get(keyword, {}).items():
            yield from _schema_nodes(child, (*path, keyword, name), root=root, seen=seen)
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for index, child in enumerate(schema.get(keyword, [])):
            yield from _schema_nodes(child, (*path, keyword, index), root=root, seen=seen)
    for keyword in (
        "additionalProperties", "unevaluatedProperties", "propertyNames", "contains",
        "items", "not", "if", "then", "else", "unevaluatedItems", "contentSchema",
    ):
        if keyword in schema:
            yield from _schema_nodes(schema[keyword], (*path, keyword), root=root, seen=seen)


def _embedded_schema_issues(schema, location: str) -> list[str]:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        return [f"{location}{_location(error.path)}: invalid JSON Schema: {error.message}"]

    issues = []
    for node, path in _schema_nodes(schema):
        prefix = location + (_location(path) if path else "")
        try:
            # Referenced targets outside standard schema positions were not
            # necessarily checked as schemas by the root metaschema validation.
            Draft202012Validator.check_schema(node)
        except SchemaError as error:
            return issues + [f"{prefix}: invalid referenced/nested JSON Schema: {error.message}"]
        if "$schema" in node and node["$schema"] != DRAFT:
            issues.append(f"{prefix}: only JSON Schema draft 2020-12 is supported")
        if "$id" in node or "$dynamicRef" in node:
            issues.append(f"{prefix}: schema IDs and dynamic references are not supported in v0.1")
        if "$ref" in node:
            reference = node["$ref"]
            if not reference.startswith("#"):
                issues.append(f"{prefix}/$ref: external schema references are not allowed")
            elif _local_reference(schema, reference) is None:
                issues.append(f"{prefix}/$ref: unresolved local JSON Pointer reference {reference!r}")
    return issues


def _successors(node) -> list:
    """Every node this node can hand over to, including band targets."""
    if "routes" in node.model_fields_set:
        return list(node.routes.values())
    if isinstance(node, DecideNode) and "bands" in node.model_fields_set:
        return [band.next for band in node.bands]
    return [node.next]


def _decide_issues(node: DecideNode, prefix: str) -> list[str]:
    """Check that a DECIDE branch can be followed with the answer it can get.

    The model decides the answer; the graph must already say where each answer
    goes, otherwise the run stops on a node with nowhere to continue.
    """
    issues: list[str] = []
    kind = node.question.type
    routed = "routes" in node.model_fields_set
    banded = "bands" in node.model_fields_set

    if routed and (kind != "choice" or node.on != "answer"):
        issues.append(f"{prefix}: 'routes' need a 'choice' question with on='answer'")
    if banded and kind == "choice" and node.on == "answer":
        issues.append(f"{prefix}: branch on a 'choice' answer with 'routes', not 'bands'")
    if "next" in node.model_fields_set and node.on == "confidence":
        issues.append(f"{prefix}: 'on': 'confidence' has no effect without 'routes' or 'bands'")

    if routed and kind == "choice" and node.on == "answer":
        criteria = node.question.criteria
        if isinstance(criteria, dict):
            missing = sorted(set(criteria) - set(node.routes))
            extra = sorted(set(node.routes) - set(criteria))
            if missing:
                issues.append(f"{prefix}/routes: criteria without a route: {', '.join(missing)}")
            if extra:
                issues.append(
                    f"{prefix}/routes: routes that are not criteria: {', '.join(extra)}"
                )
        # Criteria read from state name their options at run time and cannot be
        # compared here; the engine rejects an unrouted answer instead.

    if banded:
        maxima = [band.max for band in node.bands]
        if maxima[-1] is not None:
            issues.append(f"{prefix}/bands: the last band must catch everything (max: null)")
        for index, maximum in enumerate(maxima[:-1]):
            if maximum is None:
                issues.append(
                    f"{prefix}/bands: band {index} catches everything, "
                    f"so later bands never match"
                )
        limits = [maximum for maximum in maxima if maximum is not None]
        if any(lower >= upper for lower, upper in zip(limits, limits[1:])):
            issues.append(f"{prefix}/bands: band maxima must increase from top to bottom")
        span = _band_span(node, kind)
        if span is not None and any(not span[0] <= maximum <= span[1] for maximum in limits):
            issues.append(
                f"{prefix}/bands: a maximum lies outside "
                f"the 0..{span[1]:g} range this answer can take"
            )
    return issues


def _band_span(node: DecideNode, kind: str) -> tuple[float, float] | None:
    """The range of the number the bands compare, or None if it is not fixed."""
    if node.on == "confidence" or kind == "noul":
        return (0.0, 1.0)
    criteria = node.question.criteria
    if kind == "score" and isinstance(criteria, list):
        return (0.0, float(len(criteria) - 1))
    return None


def validate_graph(document, *, operations: Collection[str] | None = None) -> Workflow:
    """Return a validated graph, or raise IRValidationError with located issues.

    Operation availability is checked only when the caller supplies names.
    No functions are imported, and runtime data/candidate membership is not tested.
    """
    try:
        # Reject non-finite numbers even in arbitrary literal/schema JSON values.
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise IRValidationError([f"/: not finite JSON data: {error}"]) from error

    issues = [
        f"{_location(error.path)}: {error.message}"
        for error in Draft202012Validator(graph_schema()).iter_errors(document)
    ]
    try:
        workflow: Workflow | None = Workflow.model_validate(document)
    except ValidationError as error:
        # The generated schema reports a branch failure as "not valid under any of
        # the given schemas"; the model says which continuation was misdeclared.
        workflow = None
        issues.extend(
            f"{_location(item['loc'])}: {item['msg']}"
            for item in error.errors(include_url=False, include_input=False)
        )
    if issues or workflow is None:
        raise IRValidationError(issues)

    counts = Counter(node.id for node in workflow.nodes)
    for name, count in counts.items():
        if count > 1:
            issues.append(f"/nodes: duplicate node ID {name!r}")
    if workflow.entrypoint not in counts:
        issues.append(f"/entrypoint: unknown node ID {workflow.entrypoint!r}")

    available = None if operations is None else set(operations)
    for index, node in enumerate(workflow.nodes):
        prefix = f"/nodes/{index}"
        targets = _successors(node)
        for target in targets:
            if target is not None and target not in counts:
                issues.append(f"{prefix}: unknown successor {target!r}")
        if isinstance(node, DecideNode):
            issues.extend(_decide_issues(node, prefix))
        if isinstance(node, ExecuteNode) and available is not None:
            if node.operation not in available:
                issues.append(f"{prefix}/operation: unavailable operation {node.operation!r}")

    schemas = [
        ("/input_schema", workflow.input_schema),
        ("/output_schema", workflow.output_schema),
    ]
    schemas.extend(
        (f"/nodes/{index}/output_schema", node.output_schema)
        for index, node in enumerate(workflow.nodes)
        if isinstance(node, GenerateNode)
    )
    for location, schema in schemas:
        issues.extend(_embedded_schema_issues(schema, location))

    if issues:
        raise IRValidationError(issues)
    return workflow


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def load_graph(path: str | Path):
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, prog="python -m traceir.ir.validate"
    )
    parser.add_argument("graph", type=Path)
    parser.add_argument(
        "--operation", action="append", default=None,
        help="Available operation name; repeat to check availability (no names are built in)",
    )
    arguments = parser.parse_args(argv)
    try:
        workflow = validate_graph(load_graph(arguments.graph), operations=arguments.operation)
    except (OSError, ValueError) as error:
        print(f"Invalid IR: {error}", file=sys.stderr)
        return 1
    print(f"Valid IR v{workflow.schema_version}: {len(workflow.nodes)} nodes")
    if arguments.operation is None:
        print("Operation availability was not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
