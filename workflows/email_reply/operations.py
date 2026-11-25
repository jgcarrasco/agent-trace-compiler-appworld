"""Reviewed EXECUTE operations for the email-reply workflow.

Deterministic only: no model calls, no network, and no dependency on the
workflow library. `graph.json` binds these by operation name. A node that
declares `routes` must return one of those route names as its value; routing
over several answers, or over an answer compared with data, belongs here and not
in a DECIDE node, which routes on its own answer alone.

This example has no mail service to talk to, so `send_reply` records messages in
`OUTBOX` instead. That scaffolding lives here rather than in the engine, which
never learns that an outbox exists.

The workflow asks for approval by talking to a terminal, which makes this
package code for that target: `request_input` prints and reads a line. Anyone
running it may replace any operation by name, and a host that cannot block on a
human would supply its own `request_input` instead. The graph stays silent about
all of this; it names an operation and moves on.
"""

from collections.abc import Callable
from typing import Any

# Messages "sent" by this example, in order, for its own tests to inspect.
OUTBOX: list[dict[str, str]] = []


def prepare_review(*, recipient: str, body: str) -> str:
    """Format the review message shown to the user before anything is sent.

    The draft and recipient are reproduced exactly as received, so the message
    is the object the user approves.
    """
    for name, text in (("recipient", recipient), ("body", body)):
        if not isinstance(text, str) or not text:
            raise ValueError(f"{name} must be a non-empty string")
    return (
        f"To: {recipient}\n\n"
        f"Draft reply:\n\n{body}\n\n"
        "Send this reply? Answer yes to send it, or no to skip it."
    )


def request_input(*, message: str) -> str:
    """Put a question to whoever is running this, and wait for their answer.

    The message is shown as it arrives, so the text the user reads is the text
    `prepare_review` built for them to approve.
    """
    print(f"\n{message}")
    return input("> ")


def report_not_sent() -> dict[str, Any]:
    """Construct the result of a declined reply, without contacting a mail tool."""
    return {"status": "not_sent", "message_id": None}


def send_reply(*, in_reply_to: str, recipient: str, body: str) -> dict[str, Any]:
    """Stand in for a mail tool: record the message exactly as received.

    Nothing is reproduced or reformatted here, so whatever the user approved is
    whatever is recorded.
    """
    OUTBOX.append({"in_reply_to": in_reply_to, "recipient": recipient, "body": body})
    return {"status": "submitted", "message_id": f"message-{len(OUTBOX)}"}


WORKFLOW_OPERATIONS: dict[str, Callable[..., object]] = {
    "prepare_review": prepare_review,
    "report_not_sent": report_not_sent,
    "request_input": request_input,
    "send_reply": send_reply,
}
