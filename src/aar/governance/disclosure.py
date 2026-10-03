"""Small-cell suppression: RLS's answer is not the whole answer.

Row-level security removes rows a subject may not see. That is necessary and it
is not sufficient, because an aggregate can disclose what RLS correctly
withheld::

    SELECT department, AVG(salary) FROM employees GROUP BY department

RLS has already done its job - the analyst sees only their own rows - and the
result still discloses a colleague's salary whenever a department contains one
person. Nothing was leaked that RLS was responsible for; the number is simply
an inference from a legitimate answer. Suppressing that is a *different
control* with a different input, and conflating the two is how a system ends up
claiming privacy it does not have.

**Contributor cardinality, not source cardinality.** The input to this decision
is how many rows contributed to *this group*, not how many rows the source held
before and after an RLS barrier. Whole-table ``rows_before``/``rows_after`` are
the wrong measurement twice over: they are measured above the join rather than
below the group-by, and they say nothing about whether one group has three
contributors and forty others have three thousand. A single global count cannot
express a per-group rule, so a policy written against one would protect the
average department and expose the smallest one - which is the opposite of the
intent.

So the count is computed *by the aggregation itself*, as a hidden
``__aar_group_count`` column, and the guard reads it. Four properties follow
from putting it here rather than in a post-hoc filter:

* **It counts post-RLS, pre-aggregation.** Above the barrier, so it counts the
  rows the subject may actually see; inside the group-by, so it is the group's
  own contributor count. Round 26 established why this ordering is not
  negotiable: ``AVG`` cannot be un-averaged, so a count taken after the
  aggregate is not a count of anything.
* **Fail closed.** If the count cannot be established - an aggregate over a
  window, a group-by whose engine dropped the hidden column, a policy that
  names a group-by AAR cannot see - the run is refused rather than allowed
  through on an assumption. A disclosure control that silently passes when it
  cannot measure is not a control.
* **The count is not a column.** It is internal control metadata: stripped
  before exposure, never projected, never written. An analyst who could see
  ``__aar_group_count`` would learn exactly the group sizes the rule exists to
  protect, which is why the guard drops it as part of deciding, not as a
  separate step somebody can forget.
* **Honest scope.** This is minimum-group-size protection. It is not
  differential privacy, and it is not protection against differencing: an
  analyst who runs the same query twice with a differing filter, or who already
  knows one value, can still subtract to recover a small cell. Closing that is
  a different mechanism - query-memory accounting - and it is recorded as a
  known limitation rather than implied by this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..failures import PolicyDenied

__all__ = [
    "GROUP_COUNT_COLUMN", "DisclosureRule", "DisclosureGuard", "Suppression",
    "apply_disclosure_control", "disclosure_guards", "strip_group_counts",
]


#: The hidden contributor-count column. Named with a leading dunder so it is
#: conspicuous in a plan and obvious to strip, and namespaced so it cannot
#: collide with a user column.
GROUP_COUNT_COLUMN = "__aar_group_count"


@dataclass(frozen=True, slots=True)
class DisclosureRule:
    """A minimum-group-size rule: no aggregate may describe fewer than ``k``.

    ``k`` is the number of *contributing rows*, not the number of output rows.
    A rule of ``k=5`` on ``AVG(salary)`` permits a department of six and
    suppresses one of four, and the group is suppressed rather than rounded -
    rounding an average over four people is still an average over four people.
    """

    min_group_size: int = 5
    #: Free-text provenance, so a denial can say which rule fired.
    rule: str = "disclosure.min_group_size"

    def __post_init__(self) -> None:
        if self.min_group_size < 1:
            # A k of zero or one means "suppress nothing", which would be a rule
            # that appears configured and protects nothing - the exact failure
            # mode this codebase refuses elsewhere.
            raise ValueError(
                f"min_group_size must be at least 1, got "
                f"{self.min_group_size}: a rule that suppresses nothing is "
                f"not a disclosure control")


@dataclass(frozen=True, slots=True)
class Suppression:
    """What one guarded aggregate decided, and why.

    ``suppressed`` is the count of *groups removed*, and ``smallest`` is the
    smallest contributing group seen. Both are evidence, and both belong in the
    run's record: a guard that fired on every group is a misconfigured policy,
    and nothing else would say so.
    """

    node_id: str
    rule: str
    suppressed_groups: int
    smallest_group: int | None
    reason: str

    def describe(self) -> str:
        if self.suppressed_groups == 0:
            return (f"[{self.rule}] every group had at least enough "
                    f"contributors")
        return (f"[{self.rule}] suppressed {self.suppressed_groups} group(s) "
                f"below the minimum; smallest was {self.smallest_group}")


@dataclass(frozen=True, slots=True)
class DisclosureGuard:
    """A group-by that must be checked before its results are exposed.

    Holds the *node* rather than a computed answer: like a
    :class:`~aar.governance.rewrite.SecurityBarrier`, this is a standing
    obligation re-checked after planning, not a one-time verdict.
    """

    node: Any
    rule: DisclosureRule
    subject: str = ""
    reason: str = ""

    @property
    def count_column(self) -> str:
        return GROUP_COUNT_COLUMN

    def describe(self) -> str:
        return (f"[{self.rule.rule}] groups smaller than "
                f"{self.rule.min_group_size} contributors are suppressed "
                f"before {self.node.type.value} results are exposed")

    def __repr__(self) -> str:  # pragma: no cover - display
        return f"<DisclosureGuard k={self.rule.min_group_size}>"


def _aggregates(root: Any) -> list[Any]:
    """Every node whose output is a statistic over contributor rows.

    ``AGGREGATE`` (a whole-table statistic) is included as well as ``GROUPBY``:
    ``AVG(salary)`` with no grouping discloses an individual's salary just as
    readily, and a rule that only guarded grouped output would miss the more
    dangerous half.
    """
    from ..ir import NodeType, topological_order

    return [n for n in topological_order(root)
            if n.type in (NodeType.GROUPBY, NodeType.AGGREGATE)]


def _requires_cardinality(node: Any) -> bool:
    """Whether a node's result is an inference about contributors.

    Only computed aggregates can disclose by inference. A scan, a filter, a
    join or a projection returns the rows themselves, so RLS and CLS are
    already sufficient for them and no count is needed. Guarding them would be
    a rule that fires on a filter.
    """
    return bool(node.agg_functions)


def apply_disclosure_control(root: Any, rules: list[DisclosureRule],
                             subject: Any) -> list[DisclosureGuard]:
    """Attach a contributor-count obligation to every disclosing aggregate.

    The obligation is recorded on the node itself (``disclosure_rules``), which
    is what makes it survive planning: a rewrite that rebuilt the tree, or an
    optimiser that moved the aggregate, would drop a rule held only in this
    function's return value. The guard returned here is for explanation, not
    for enforcement - enforcement reads the node.

    Fails closed. A node that cannot carry a count is a
    :class:`~aar.failures.PolicyDenied`, because "I could not measure the group
    size" must never become "so I allowed the group".
    """
    if not rules:
        return []

    guards: list[DisclosureGuard] = []
    for node in _aggregates(root):
        if not _requires_cardinality(node):
            continue
        attached = list(getattr(node, "disclosure_rules", ()))
        for rule in rules:
            if rule.rule not in {r.rule for r in attached}:
                attached.append(rule)
        strictest = min(attached, key=lambda r: r.min_group_size)
        try:
            node.disclosure_rules = tuple(attached)
        except AttributeError:
            raise PolicyDenied(
                f"this node cannot carry a disclosure rule "
                f"({type(node).__name__} has no disclosure_rules field), so "
                f"[{strictest.rule}] cannot be enforced on "
                f"{node.type.value}. Nothing is allowed through, because a "
                f"disclosure control that cannot be attached must not be "
                f"assumed to be satisfied.",
                rule=strictest.rule) from None
        guards.append(DisclosureGuard(
            node=node, rule=strictest,
            subject=getattr(subject, "name", ""),
            reason=f"{len(attached)} disclosure rule(s) apply to this "
                   f"aggregate"))
    return guards


def disclosure_guards(root: Any) -> list[DisclosureGuard]:
    """Every guarded aggregate in ``root``, in topological order."""
    guards: list[DisclosureGuard] = []
    for node in _aggregates(root):
        rule = enforced_rule(node)
        if rule is not None:
            guards.append(DisclosureGuard(node=node, rule=rule, subject=""))
    return guards


def enforced_rule(node: Any) -> DisclosureRule | None:
    """The strictest rule attached to ``node``, or ``None``.

    Strictest rather than first: two rules with different ``k`` on one
    aggregate is a policy that wants both, and satisfying the looser would
    satisfy neither the author nor the analyst's expectation.
    """
    rules = tuple(getattr(node, "disclosure_rules", ()))
    if not rules:
        return None
    return min(rules, key=lambda r: r.min_group_size)


def group_count_aggregate() -> Any:
    """The ``COUNT(*)`` AAR injects to learn a group's contributor size.

    Built here, once, so the executor and any engine-side pushdown agree on
    what the hidden column contains. A ``COUNT(DISTINCT ...)`` would be a
    different and stronger measure; contributor rows is what the policy
    language promises, and promising a stronger guarantee than is delivered
    would be the more dangerous mistake.
    """
    from ..ir import Agg

    return Agg(func="COUNT", arg=None, distinct=False)


def strip_group_counts(table: Any) -> Any:
    """Remove the hidden count from a table about to be exposed or written.

    Called as part of the guard's decision, not as an optional tidy-up: if the
    column can outlive the check, an analyst can read ``__aar_group_count`` and
    learn precisely the group sizes the rule protects. Returns a new table with
    the column gone and the original untouched, so the check stays re-runnable.
    """
    if table is None:
        return table
    names = list(getattr(table, "column_names", ()) or ())
    if GROUP_COUNT_COLUMN not in names:
        return table
    keep = [n for n in names if n != GROUP_COUNT_COLUMN]
    try:
        return table.project(keep)
    except Exception as exc:  # noqa: BLE001 - becomes a refusal
        raise PolicyDenied(
            f"the disclosure guard produced a {GROUP_COUNT_COLUMN} column it "
            f"cannot remove before exposure ({type(exc).__name__}: {exc}). The "
            f"run is refused rather than handing the analyst the group sizes "
            f"the rule exists to protect.") from exc