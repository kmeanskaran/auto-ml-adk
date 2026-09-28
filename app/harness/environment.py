"""Where agent-written code runs.

ProjectEnvironment implements ADK's BaseEnvironment with one difference from
ADK's LocalEnvironment that matters: the child process gets a clean environment
instead of a copy of the server's, so agent-written code cannot read API keys.
The source screen is a tripwire for obvious mistakes, not a sandbox; isolation
comes from the deploy target's container.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from google.adk.environment import BaseEnvironment, ExecutionResult

from app.harness.project import LAKE

BLOCKED = {
    "network access": r"\b(import|from)\s+(socket|requests|urllib|http|httpx|aiohttp|ftplib|smtplib)\b",
    "starting processes": r"\b(import|from)\s+(subprocess|multiprocessing)\b|os\.(system|popen|exec|spawn|fork)",
    "dynamic code": r"\b(eval|exec|compile|__import__)\s*\(",
    "deleting folders": r"shutil\.rmtree",
}


def screen(source: str) -> list[str]:
    """Names of blocked behaviours found in a script's source."""
    return [name for name, pattern in BLOCKED.items() if re.search(pattern, source)]


class ProjectEnvironment(BaseEnvironment):
    """Runs commands in one folder with a minimal, secret-free environment."""

    def __init__(self, working_dir: Path, extra_env: dict[str, str] | None = None):
        super().__init__()
        self._working_dir = working_dir
        self._extra_env = extra_env or {}

    @property
    def working_dir(self) -> Path:
        return self._working_dir

    async def initialize(self) -> None:
        (self._working_dir / ".home").mkdir(parents=True, exist_ok=True)
        self.is_initialized = True

    async def execute(
        self, command: str, *, timeout: float | None = None
    ) -> ExecutionResult:
        if not self.is_initialized:
            await self.initialize()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self._working_dir,
            env=self._child_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            timed_out = False
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            timed_out = True
        return ExecutionResult(
            exit_code=-1
            if timed_out or process.returncode is None
            else process.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            timed_out=timed_out,
        )

    async def read_file(self, path: Path) -> bytes:
        return (self._working_dir / path).read_bytes()

    async def write_file(self, path: Path, content: str | bytes) -> None:
        target = self._working_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)

    def _child_env(self) -> dict[str, str]:
        return {
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "HOME": str(self._working_dir / ".home"),
            "DATA_DIR": str(LAKE),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "4",
            "LANG": "C.UTF-8",
            **self._extra_env,
        }
