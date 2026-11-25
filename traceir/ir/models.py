"""Typed IR v0.1 definitions and their generated JSON Schema."""

import json
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

NonemptyString = Annotated[str, Field(min_length=1)]
# JSON Pointer tokens permit literal characters other than '/' and '~', or ~0/~1.
ReferencePath = Annotated[
    str, Field(pattern=r"^/(?:input|state)(?:/(?:[^~/]|~[01])*)*$")
]
StateDestination = Annotated[
    str, Field(pattern=r"^/state/(?:[^~/]|~[01])*(?:/(?:[^~/]|~[01])*)*$")
]
JSONSchema = dict[str, JsonValue] | bool


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Reference(ClosedModel):
    ref: ReferencePath          # -> {"ref": "..."}


class LiteralValue(ClosedModel):
    literal: JsonValue          # -> {"literal": ...}


Binding = Reference | LiteralValue


class ChoiceQuestion(ClosedModel):
    type: Literal["choice"]
    instructions: NonemptyString
    criteria: Annotated[dict[NonemptyString, NonemptyString], Field(min_length=2)] | Reference


class ScoreQuestion(ClosedModel):
    type: Literal["score"]
    instructions: NonemptyString
    criteria: Annotated[list[NonemptyString], Field(min_length=2)] | Reference


class NoulCriteria(ClosedModel):
    true: NonemptyString | None = None
    false: NonemptyString | None = None


class NoulQuestion(ClosedModel):
    type: Literal["noul"]
    instructions: NonemptyString
    criteria: NoulCriteria | Reference | None = None


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")
]


def answer_schema(question: dict, *, require_confidence: bool = False) -> dict:
    """What the answer of a DECIDE node must look like.

    The graph does not store this. The answer depends on the node's question, and
    a question may only know its options while the workflow runs, because they are
    read from state. So the shape is worked out here, each time, from the question
    the node really has.

    The result is used twice: the engine checks the answer against it, and the
    model backend asks the model for exactly this shape. So neither side can
    invent its own format.
    """
    kind = question["type"]
    # The value field is named after the question type: {"type": "choice", "choice": ...}
    key = kind
    if kind == "choice":
        value: dict[str, Any] = {"type": "string", "enum": list(question["criteria"])}
    elif kind == "score":
        value = {"type": "number", "minimum": 0, "maximum": len(question["criteria"]) - 1}
    elif kind == "noul":
        value = {"type": "number", "minimum": 0, "maximum": 1}
    else:
        # A new question type needs a range here, not a guess at one.
        raise ValueError(f"unsupported question type {kind!r}")
    required = ["type", key]
    if require_confidence:
        # The node branches on certainty, so the answer has no meaning without it.
        required.append("confidence")
    return {
        "type": "object",
        "properties": {
            "type": {"const": kind},
            key: value,
            # Model-reported certainty. Optional unless the node routes on it.
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "probabilities": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "required": required,
        "additionalProperties": False,
    }


class Node(ClosedModel):
    id: NonemptyString
    inputs: dict[NonemptyString, Binding]
    write_to: StateDestination | None = None


class Band(ClosedModel):
    """One band of a DECIDE branch: the first whose maximum covers the value."""

    max: float | None
    next: NonemptyString | None


class FlowNode(Node):
    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {"required": ["next"], "not": {"required": ["routes"]}},
                {"required": ["routes"], "not": {"required": ["next"]}},
            ]
        },
    )

    # A subclass with more ways to continue overrides this and its own oneOf.
    continuation_fields: ClassVar[tuple[str, ...]] = ("next", "routes")

    next: NonemptyString | None = None
    routes: Annotated[dict[NonemptyString, NonemptyString | None], Field(min_length=1)] = (
        Field(default_factory=dict)
    )

    @model_validator(mode="after")
    def require_one_continuation(self) -> Self:
        declared = [name for name in self.continuation_fields if name in self.model_fields_set]
        if len(declared) != 1:
            names = "' or '".join(self.continuation_fields)
            raise ValueError(f"declare exactly one of '{names}'")
        return self


class ExecuteNode(FlowNode):
    type: Literal["EXECUTE"]
    operation: NonemptyString


class DecideNode(FlowNode):
    """One bounded question, one typed answer, and the branch that follows it.

    The node routes on its own answer and nothing else: the engine, not the model,
    picks the successor. Policy that compares the answer with other data, or with
    another answer, stays in an EXECUTE router that follows a plain `next`.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {
                    "required": ["next"],
                    "not": {"anyOf": [{"required": ["routes"]}, {"required": ["bands"]}]},
                },
                {
                    "required": ["routes"],
                    "not": {"anyOf": [{"required": ["next"]}, {"required": ["bands"]}]},
                },
                {
                    "required": ["bands"],
                    "not": {"anyOf": [{"required": ["next"]}, {"required": ["routes"]}]},
                },
            ]
        },
    )

    type: Literal["DECIDE"]
    # The question is decided at compile time; only its criteria may be resolved
    # from state, so that a decision can enumerate runtime candidates.
    question: Question
    # What the branch reads: the answer itself, or how certain the answer was.
    on: Literal["answer", "confidence"] = "answer"
    bands: Annotated[list[Band], Field(min_length=1)] = Field(default_factory=list)

    continuation_fields: ClassVar[tuple[str, ...]] = ("next", "routes", "bands")


class GenerateNode(FlowNode):
    type: Literal["GENERATE"]
    template: NonemptyString
    output_schema: JSONSchema


WorkflowNode = Annotated[
    ExecuteNode | DecideNode | GenerateNode, Field(discriminator="type")
]


class Workflow(ClosedModel):
    schema_version: Literal["0.1"]
    entrypoint: NonemptyString
    input_schema: JSONSchema
    output_schema: JSONSchema
    nodes: Annotated[list[WorkflowNode], Field(min_length=1)]


def graph_schema() -> dict:
    """Generate the schema; do not maintain a second hand-written definition."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **Workflow.model_json_schema(),
    }


if __name__ == "__main__":
    print(json.dumps(graph_schema(), indent=2, allow_nan=False))
