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
from dataclasses import dataclass, field
from typing import Any

from .cost import EstimationLog
from .governance.disclosure import DisclosureGuard, apply_disclosure_control
from .governance.rewrite import apply_row_security, assert_barriers_intact

__all__ = ["PipelineService", "RunReport"]


@dataclass(slots=True)
class RunReport:
    """The outcome of one load-plan-execute cycle."""

    plan: Any
    result: Any
    root: Any
    #: What the profiler measured before planning, keyed by node type.
    profiles: dict = field(default_factory=dict)
    #: Row-level-security barriers injected into the logical plan, in the order
    #: they were applied. Present so a front end can say what was protected
    #: without re-deriving it, and so a caller can prove a plan was secured.
    barriers: list = field(default_factory=list)


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
        #: Prediction-vs-actual errors from the most recent profiled run, or
        #: ``None`` when nothing was profiled. Public because a caller that
        #: wants to know whether the numbers were any good has to be able to
        #: ask, and a private attribute would just move the guessing.
        self.estimation_log: Any | None = None

    def _make_planner(self) -> Any:
        if self._planner is not None:
            return self._planner
        from .planner import AdaptivePlanner

        self._planner = AdaptivePlanner()
        return self._planner

    @staticmethod
    def _apply_disclosure(root: Any, policy: Any, subject: Any
                          ) -> list[DisclosureGuard]:
        """Attach small-cell suppression to every aggregate in ``root``.

        Called after row security so the contributor counts are measured over
        the rows the subject may actually see. A no-op when the policy declares
        no ``disclosure`` rule, which is the default: a control nobody asked
        for must not silently change anybody's numbers.

        Returns the guards for the run report, so a plan can say which
        aggregates were subject to suppression rather than only which ones
        produced a suppressed group.
        """
        rules = list(getattr(policy.policy, "disclosure", ()) or ())
        if not rules:
            return []
        return apply_disclosure_control(root, rules, subject)

    @staticmethod
    def _load_policy(path: str | None) -> Any:
        """Build the policy engine for a run.

        ``None`` means *no policy file was supplied*, and it builds the safe
        baseline :class:`PolicyEngine` rather than disabling governance. The
        old behaviour - ``return None``, and have the executor skip every
        check - meant the default-deny egress rule only existed for whoever
        remembered to instantiate a policy, which is the opposite of what a
        default is for.

        The baseline permits local processing and local output and denies
        network egress, which keeps ordinary local use working while making
        "no policy file" a *restrictive* state rather than a permissive one. A
        genuine bypass is explicit and separate: ``--unsafe-disable-policy``.
        """
        from .governance import Policy, PolicyEngine

        if not path:
            return PolicyEngine(Policy())
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no such policy file: {path}")
        from .governance import load_policy

        return PolicyEngine(load_policy(path))

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

    def prepare(self, path: str, profile: bool = True,
               role: str | None = None, policy_path: str | None = None,
               actor: str | None = None) -> tuple:
        """Load, secure, profile, then plan - returning ``(root, plan)``.

        Exists because a front end that wants to *show* a plan before
        running it used to plan twice, and the first plan was made before
        the sources were measured. That ordering is not a cosmetic problem:
        a source declared as 100 MB and actually 8 GB gets planned from
        100 MB, and the profiler then measures 8 GB and does nothing with
        it. The measurement exists to change the plan, so it has to happen
        first.

        This is the same sequence :meth:`run` uses, which is the point: one
        order, one behaviour, whichever front end asks. That includes the
        security rewrite - ``aar explain`` must show the secured plan, or an
        analyst is shown row counts they are not entitled to see.
        """
        from .governance import Subject

        policy = self._load_policy(policy_path)
        subject = (self._subject(role, actor)
                   or Subject(name="explain"))
        root = self.load(path)
        barriers = apply_row_security(root, policy.policy, subject)
        self._apply_disclosure(root, policy, subject)
        self.profile_sources(root) if profile else {}
        plan = self.plan(root)
        assert_barriers_intact(root, barriers)
        return root, plan

    def explain(self, path: str, role: str | None = None,
                policy_path: str | None = None,
                actor: str | None = None) -> Any:
        """Load and plan a file, returning the plan for display.

        Profiles first, like :meth:`prepare`, so ``aar explain`` and
        ``aar run`` cannot disagree about the same pipeline.
        """
        return self.prepare(path, role=role, policy_path=policy_path,
                            actor=actor)[1]

    def run(self, path: str, role: str | None = None,
            policy_path: str | None = None,
            actor: str | None = None,
            profile: bool = True) -> RunReport:
        """Load, profile, plan and execute one pipeline end to end.

        Planning and execution stay inside one call on purpose. A caller that
        plans separately from running can pair one pipeline with another's
        plan, and the resulting error names neither.

        ``profile`` measures each source before planning. That is the whole
        point of the profiler: the cost model otherwise sizes segments from
        numbers the pipeline declared about itself. Profiling is bounded -
        a Parquet footer or a database's own statistics, else a capped
        sample - and a source that cannot be measured is left alone, with
        the plan falling back to the declared estimate. It never fails a run
        because a file was unreadable.

        Afterwards the predicted row count for each profiled source is
        compared with the rows the scan actually produced, and the gap is
        recorded. That comparison is the only thing that can catch a
        profiler which is confidently wrong.
        """
        from .runtime import Executor

        # Load and validate the policy *before* planning. A policy path that
        # does not exist is a more fundamental error than a plan that does
        # not fit, and planning first reported the memory failure instead -
        # sending the user to resize their machine when the real problem was
        # a typo in a path. Absent rules deny, so this must fail loudly
        # before anything is measured or scheduled.
        policy = self._load_policy(policy_path)
        subject = self._subject(role, actor)

        root = self.load(path)

        # Row-level security is injected into the *logical* plan, before
        # profiling and before planning. Two reasons, and the second is the
        # one that is easy to miss:
        #
        # 1. RLS decides which rows are legally input to a computation. Once an
        #    aggregate has run, "region = 'EU'" cannot be applied to its output
        #    - the filter belongs below the group-by, not above it.
        # 2. Profiling an unsecured source measures what the analyst is *not*
        #    allowed to see. If the table holds 1e9 rows and the subject may
        #    read 2e6, planning from 1e9 is both a worse plan and a disclosure
        #    of the restricted size.
        #
        # This is where an ambiguity in the policy becomes an error, before
        # anything has been read, measured or scheduled.
        barriers = apply_row_security(root, policy.policy, subject)
        # Small-cell control, attached after RLS so the contributor counts are
        # measured over the rows the subject may actually see. Attaching it
        # first would count rows a barrier is about to remove.
        self._apply_disclosure(root, policy, subject)

        predicted = self.profile_sources(root) if profile else {}
        plan = self.plan(root)
        # Re-check the obligations *after* planning: a barrier that was removed,
        # weakened or hoisted above an aggregation between here and execution
        # must fail loudly rather than leak.
        assert_barriers_intact(root, barriers)
        with Executor(policy=policy, subject=subject,
                      history=self._history) as executor:
            result = executor.execute(plan)
        report = RunReport(plan=plan, result=result, root=root,
                           profiles=predicted, barriers=barriers)
        self.score_estimates(root, result)
        return report

    def score_estimates(self, root: Any, result: Any) -> list:
        """Compare each profiled source with what the run actually read.

        The executor records ``rows_out`` per node and the profiler recorded
        what it expected, so the two can be compared directly. Errors are
        labelled by node id, which is unique within a run; a stable
        identifier across runs would be needed to accumulate history, and
        inventing one that could silently collide would be worse than
        keeping this per-run.

        Takes ``root`` and ``result`` rather than a :class:`RunReport` so a
        caller that already planned and executed separately - the CLI, which
        prints the plan before running it - can still score the estimates.
        """
        log = EstimationLog()
        for node in root.walk():
            profile = getattr(node, "aar_profile", None)
            if profile is None:
                continue
            outcome = next((o for o in result.outcomes
                            if o.node_id == node.id), None)
            if outcome is None:
                continue
            log.add(label=f"{node.type.value}:{node.id}",
                    predicted=profile.rows, actual=outcome.rows_out,
                    source=profile.source)
        if log.errors():
            self.estimation_log = log
        return log.errors()

    def profile_sources(self, root: Any) -> dict:
        """Measure every source a pipeline reads, and attach the results.

        Returns the profiles that were taken, so a caller can show them.
        A source that cannot be measured is skipped rather than reported as
        zero rows - a zero would plan a trivially cheap pipeline, which is
        the one answer a missing measurement must never produce.
        """
        from .ir import NodeType
        from .stats import DataProfiler

        # SCAN_ARROW is an in-memory handle and SCAN_CONST has no source at
        # all; neither has anything to read from disk.
        skip = (NodeType.SCAN_ARROW, NodeType.SCAN_CONST)
        profiler = DataProfiler()
        taken: dict = {}
        for node in root.walk():
            if node.type in skip or node.aar_profile is not None:
                continue
            profile = profiler.profile_node(node)
            if profile is None:
                continue
            node.aar_profile = profile
            taken[str(node.type.value)] = profile
        return taken