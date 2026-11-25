#!/usr/bin/env bash
# Create the AppWorld environment: the clone, its data, and a venv that can
# import it.
#
# AppWorld cannot share a process with traceir: it pins fastapi 0.150.0, which
# needs pydantic 2.12.5, while the library tracks a later pydantic. This script
# therefore builds a second environment under bench/appworld/.venv and leaves the
# root .venv alone.
#
#   appworld/          the upstream clone, gitignored, read-only for us
#   appworld_data/     the dataset (189 MB), gitignored
#   appworld_gen/      generated tasks, written by prepare.py, gitignored
#
# Re-running is cheap: every step checks whether its output already exists.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
APPWORLD="$REPO/appworld"
DATA="$REPO/appworld_data"
VENV="$REPO/bench/appworld/.venv"

# The commit the paper's numbers were produced with. Moving it means re-running
# the experiment: task ids are positional, so a different commit generates
# different instances under the same names.
COMMIT=42b5bcf3cd334fee33f0c37c02070a9f5807add5
DATA_VERSION=0.2.0

# Two pins the experiment depends on, both because of fastapi.
#   fastapi 0.150.0 is the last release where `Path(..., embed=True)` still
#     accepts **extra; the environment's /execute endpoint needs it.
#   pydantic 2.12.5 is the last version fastapi 0.150 can build models with.
PIN_FASTAPI=0.150.0
PIN_PYDANTIC=2.12.5

step() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- clone
step "clone at ${COMMIT:0:7}"
if [ ! -d "$APPWORLD/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/StonyBrookNLP/appworld.git "$APPWORLD"
fi
git -C "$APPWORLD" fetch --quiet origin "$COMMIT" 2>/dev/null || true
if [ "$(git -C "$APPWORLD" rev-parse HEAD)" != "$COMMIT" ]; then
  echo "  checked out $(git -C "$APPWORLD" rev-parse --short HEAD), moving to the pin"
  git -C "$APPWORLD" checkout --quiet "$COMMIT"
fi

# ---------------------------------------------------------------- LFS bundles
# Four files are stored in git-lfs and the clone left them as pointer files.
# git-lfs is not installed and not needed: the batch endpoint serves public
# repositories over plain HTTPS, so we read the pointer, ask where the object
# lives, and download it.
step "fetch LFS bundles over HTTPS"
for path in src/appworld/.source/apps.bundle \
            src/appworld/.source/tests.bundle \
            generate/.source/data.bundle \
            generate/.source/tasks.bundle; do
  if [ "$(head -c 7 "$APPWORLD/$path")" != "version" ]; then
    echo "  $path already fetched"
    continue
  fi
  pointer=$(git -C "$APPWORLD" show "HEAD:$path")
  oid=$(awk '$1 == "oid" {sub("sha256:", "", $2); print $2}' <<<"$pointer")
  size=$(awk '$1 == "size" {print $2}' <<<"$pointer")
  href=$(curl -sS -X POST \
    -H "Accept: application/vnd.git-lfs+json" \
    -H "Content-Type: application/vnd.git-lfs+json" \
    -d "{\"operation\":\"download\",\"transfers\":[\"basic\"],\
\"objects\":[{\"oid\":\"$oid\",\"size\":$size}]}" \
    "https://github.com/StonyBrookNLP/appworld.git/info/lfs/objects/batch" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["objects"][0]["actions"]["download"]["href"])')
  curl -sSL "$href" -o "$APPWORLD/$path"
  echo "  $path  $(stat -c%s "$APPWORLD/$path") bytes"
done

# ---------------------------------------------------------------- venv
# The [ci] extra carries ruff, which the task generators shell out to.
step "virtual environment"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
fi
uv pip install --quiet --python "$VENV/bin/python" -e "$APPWORLD[ci]"
uv pip install --quiet --python "$VENV/bin/python" \
  "fastapi==$PIN_FASTAPI" "pydantic==$PIN_PYDANTIC"

# ---------------------------------------------------------------- unpack code
# Unpacks the encrypted portion of the package into the repository itself, which
# is where prepare.py looks for the task generators.
step "appworld install --repo"
(cd "$APPWORLD" && "$VENV/bin/appworld" install --repo)

# ---------------------------------------------------------------- data
step "data $DATA_VERSION"
if [ ! -f "$DATA/data/version.txt" ]; then
  mkdir -p "$DATA"
  "$VENV/bin/appworld" download data --version "$DATA_VERSION" --root "$DATA"
fi
echo "  $DATA/data/version.txt: $(cat "$DATA/data/version.txt")"

# ---------------------------------------------------------------- check
# `appworld verify` expects the tests, the data and the package under one root,
# which is not how this repository lays them out (the clone is read-only and the
# dataset is 189 MB), so it is not the check we can run. Check the artifacts the
# experiment actually needs instead.
step "check"
[ -d "$APPWORLD/generate/tasks/task_generators" ] || {
  echo "the task generators were not unpacked" >&2
  exit 1
}
"$VENV/bin/python" - <<'PY'
import appworld, fastapi, pydantic

assert pydantic.VERSION.startswith("2.12."), pydantic.VERSION
assert fastapi.__version__ == "0.150.0", fastapi.__version__
print(f"  appworld {appworld.__version__}, pydantic {pydantic.VERSION}, "
      f"fastapi {fastapi.__version__}")
PY

cat <<EOF

Ready. The benchmark owns its environment under bench/appworld/.venv, which git
ignores. The next step generates the instances the experiment uses:

  uv run python -m bench.appworld.prepare        # writes plans/p1.json

Then the trace, compile, run and analysis stages, in that order; see README.md.
EOF
