/**
 * The `run` tool: execute python against the task's apps.
 *
 * This is the trace agent's only app interface, and the equivalent of the python
 * REPL the benchmark's own agent writes code into: one chunk of code in, its
 * printed output back, variables and logins persisting between calls because the
 * world keeps a single shell per task.
 *
 * It knows nothing about apps, tasks or the process of solving anything: the
 * address comes from the environment, and the code is forwarded unchanged.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const PARAMETERS = Type.Object({
	code: Type.String({ description: "Python code to execute in the task environment." }),
});

export default function runTool(pi: ExtensionAPI) {
	pi.registerTool({
		name: "run",
		label: "Run code",
		description:
			"Execute python code against the user's apps and return what it printed. " +
			"The code runs in a persistent shell with an `apis` object holding the app APIs, " +
			"so variables defined in one call are available in the next.",
		parameters: PARAMETERS,
		async execute(_toolCallId, params) {
			const url = process.env.APPWORLD_ENV_URL;
			const taskId = process.env.APPWORLD_TASK_ID;
			if (!url || !taskId) {
				throw new Error("APPWORLD_ENV_URL and APPWORLD_TASK_ID must be set");
			}
			const response = await fetch(`${url}/execute`, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ task_id: taskId, code: params.code }),
			});
			if (!response.ok) {
				throw new Error(`environment returned HTTP ${response.status}: ${await response.text()}`);
			}
			const body = (await response.json()) as { output?: string };
			return {
				content: [{ type: "text", text: body.output ?? "" }],
				details: {},
			};
		},
	});
}
