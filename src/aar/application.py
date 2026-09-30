"""The application service: one place that loads, plans and runs a pipeline.

The CLI and the Workbench both need the same sequence - load the file, plan
it, apply policy, execute, and hand back what happened. Written twice, the
two copies drift: the Workbench's called ``Executor().run(...)``, a method
that does not exist, so ``/api/run`` failed for *every* valid pipeline while
the CLI kept working. The tests missed it because they only asserted that a
missing file produced an error - a test that passes while the success path is
broken.

Both front ends now call :class:`PipelineService`, so a change to how a
pipeline is loaded, planned or governed reaches the CLI and the browser at
the same time. That also removes the ``cli -> workbench.server -> cli``
cycle: the Workbench no longer imports the CLI.

Note what this service deliberately does *not* do: it does not swallow
failures. A front end decides whether an exception becomes an error page or
an exit code, because only that front end knows what its caller expects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

__all__ = ["PipelineService", "RunReport"]


@dataclass(slots=True)
class RunReport:
    """The outcome of one load-plan-execute cycle."""

    plan: Any
    result: Any
    root: Any


class PipelineService:
    """Load, plan, and execute pipelines on behalf of a front end.

    ``policy_path`` is a path rather than a parsed policy so that loading is
    the same call in both front ends, and so that a policy file which fails
    to parse stays a hard error. Absent rules deny, so a typo in a policy
    path must never quietly produce an unprotected run.
    """

    def __init__(self, planner: Any | None = None,
                 history: Any | None = None) -> None:
        self._planner = planner
        self._history = history

    def _make_planner(self) -> Any:
        if self._planner is not None:
            return self._planner
        from .planner import AdaptivePlanner

        self._planner = AdaptivePlanner()
        return self._planner

    @staticmethod
    def _load_policy(path: str | None) -> Any:
        if not path:
            return None
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such policy file: {path}")
        from .governance import load_policy

        return load_policy(path)

    @staticmethod
    def _subject(role: str | None, actor: str | None) -> Any:
        if role is None and actor is None:
            return None
        from .governance import Subject

        return Subject(name=actor or "cli",
                       roles=frozenset({role}) if role else frozenset())

    # ---------------------------------------------------------------- public
    def load(self, path: str) -> Any:
        """Load a pipeline file into a root node, or raise."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such pipeline: {path}")
        from .sdk import load_pipeline

        return load_pipeline(path)

    def plan(self, root: Any) -> Any:
        """Produce a physical plan for an already-loaded root."""
        return self._make_planner().plan(root)

    def explain(self, path: str) -> Any:
        """Load and plan a file, returning the plan for display."""
        return self.plan(self.load(path))

    def run(self, path: str, role: str | None = None,
            policy_path: str | None = None,
            actor: str | None = None) -> RunReport:
        """Load, plan and execute one pipeline end to end.

        Planning and execution stay inside one call on purpose. A caller that
        plans separately from running can pair one pipeline with another's
        plan, and the resulting error names neither.
        """
        from .runtime import Executor

        root = self.load(path)
        plan = self.plan(root)
        policy = self._load_policy(policy_path)
        subject = self._subject(role, actor)
        with Executor(policy=policy, subject=subject,
                      history=self._history) as executor:
            result = executor.execute(plan)
        return RunReport(plan=plan, result=result, root=root)