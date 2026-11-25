"""The AppWorld environment server, for the duration of a tracing session.

One server serves every task in a family: `/initialize` replaces the world, so
tasks run one at a time against the same process. The server lives in its own
venv (fastapi 0.150 pins pydantic 2.12.5), so it is started as a subprocess and spoken to
over HTTP rather than imported.
"""

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPWORLD = ROOT / "bench/appworld/.venv/bin/appworld"
DATA_ROOT = ROOT / "appworld_gen"


class World:
    """An environment server holding one task at a time."""

    def __init__(self, port: int = 9100, root: Path = DATA_ROOT):
        self.port = port
        self.root = root
        self.url = f"http://127.0.0.1:{port}"
        self.process: subprocess.Popen | None = None

    def start(self, timeout: float = 120.0) -> "World":
        self.process = subprocess.Popen(
            [str(APPWORLD), "serve", "environment", "--port", str(self.port), "--no-show-usage"],
            cwd=self.root,
            env={
                "APPWORLD_ROOT": str(self.root),
                "APPWORLD_CACHE": str(self.root / ".cache"),
                "PATH": "/usr/bin:/bin",
                "HOME": str(self.root),
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.get("/")
                return self
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        self.stop()
        raise RuntimeError(f"the environment server did not come up on port {self.port}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __enter__(self) -> "World":
        return self.start()

    def __exit__(self, *exception) -> None:
        self.stop()

    def _request(self, method: str, path: str, payload: dict | None, timeout: float) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.url + path, data=data, headers={"content-type": "application/json"}, method=method
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def get(self, path: str, timeout: float = 10) -> dict:
        return self._request("GET", path, None, timeout)

    def post(self, path: str, payload: dict, timeout: float = 300) -> dict:
        return self._request("POST", path, payload, timeout)

    def initialize(self, task_id: str) -> dict:
        return self.post("/initialize", {"task_id": task_id})["output"]

    def evaluate(self, task_id: str) -> dict:
        return self.post("/evaluate", {"task_id": task_id})["output"]

    def task_completed(self, task_id: str) -> bool:
        return self.post("/task_completed", {"task_id": task_id})["output"]

    def execute(self, task_id: str, code: str) -> str:
        return self.post("/execute", {"task_id": task_id, "code": code})["output"]
