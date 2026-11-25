# Prompts

`trace.md` is the system prompt of the tracing agent. It adapts AppWorld's own
full-code-agent instructions
(`experiments/prompts/full_code_agent/full_code_instructions.txt`, Apache-2.0):
the channel is a single `run` tool instead of AppWorld's code execution loop,
and the completion rule is rewritten for this harness. `task.md` renders the
task's supervisor and instruction.

They are copied into the workspace the agent runs in. `trace.md` describes the
AppWorld apps and APIs, so it is specific to this benchmark, and it is adapted
from AppWorld's own instructions as the notice says.
