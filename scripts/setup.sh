#!/usr/bin/env bash
# Sync the library environment.
#
# AppWorld runs in its own environment: it pins fastapi 0.150.0, which builds
# its models with pydantic 2.12.5, while traceir tracks a later pydantic, so the
# benchmark cannot share a process with the library. That environment is created
# by bench/appworld/setup.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync
