"""Row-level security as a *logical* barrier, not a write-time patch.

The defect this module exists to fix: RLS was applied by
:meth:`~aar.governance.policy.PolicyEngine.enforce_write`, at the very end of a
run. By then every row has already passed through every join, group-by,
window and UDF. For the analyst's own question - "what is the average salary
in my region?" - that is not a slightly-wrong filter, it is the wrong
computation: ``AVG`` over all rows cannot be un-averaged afterwards, and
filtering the single output row of an aggregate either drops the whole result
or does nothing at all.

So RLS is injected **into the logical plan, directly above each secured
source**, before profiling, planning or execution. The shape is::

    ScanSQL  --  SECURITY_FILTER(tenant = 42)  --  FILTER(amount > 100)
                        ^ dominates everything downstream

and for a join, each input is secured independently *before* the join, which is
why rules are source-scoped::

    Orders -- SECURITY_FILTER(region = 'EU') --\\
                                                  JOIN
    Users -- SECURITY_FILTER(tenant_id = 42) --/

Three properties this module is built around:

* **Fail closed.** Every ambiguity - a rule that names no source in a
  multi-source pipeline, a rule naming a source that is not present, a
  predicate that cannot be parsed exactly, a referenced column that does not
  exist - raises. None is resolved by guessing. A security layer that picks a
  plausible reading of an ambiguous rule is worse than one that refuses,
  because the refusal is visible and the guess is not.
* **The barrier is non-elidable.** A ``SECURITY_FILTER`` may be pushed *down*
  into a connector, which is faster and safer, but never removed, hoisted
  above an aggregation or a join, weakened, or approximated.
  :func:`assert_barriers_intact` re-checks that after planning, so a future
  optimiser which "helpfully" drops a redundant-looking filter fails loudly
  instead of quietly leaking rows.
* **RLS and CLS stay separate.** RLS decides which rows are legally *input*.
  Column masking decides what may be *seen*, and belongs at the exposure
  boundary - ``AVG(salary)`` may be permitted even when ``salary`` is not. The
  two are not synonyms and this module does not conflate them.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..failures import PolicyDenied
from ..ir import BinOp, Col, Expr, Lit, Node, NodeType

__all__ = [
    "SecurityBarrier", "apply_row_security", "security_barriers",
    "assert_barriers_intact", "parse_row_predicate",
]


class SecurityBarrier:
    """One injected barrier: which rule, which source, which node."""

    __slots__ = ("rule", "source", "node", "subject", "predicate")

    def __init__(self, rule: str, source: str, node: Node,
                 subject: str) -> None:
        self.rule = rule
        self.source = source
        self.node = node
        self.subject = subject
        #: The predicate as policy wrote it, snapshotted at injection.
        #:
        #: Snapshot rather than re-read, because ``node`` and ``barrier.node``
        #: are the *same object*. A check of the form
        #: ``node.predicate != barrier.node.predicate`` compares a value with
        #: itself and is therefore always false - it would have reported every
        #: barrier as intact, including one whose predicate had been rewritten.
        self.predicate = node.predicate

    def describe(self) -> str:
        return (f"[{self.rule}] {self.subject} sees only {self.source} rows "
                f"matching {self.node.predicate}")

    def __repr__(self) -> str:  # pragma: no cover - display
        return f"<SecurityBarrier {self.rule} on {self.source}>"


def _walk(root: Node) -> list[Node]:
    from ..ir import topological_order

    return topological_order(root)


def security_barriers(root: Node) -> list[SecurityBarrier]:
    """Every barrier in ``root``, in topological order."""
    return [SecurityBarrier(rule=n.security_rule or "rls",
                            source=n.source_scope, node=n, subject="")
            for n in _walk(root) if n.type is NodeType.SECURITY_FILTER]


def _sources(root: Node) -> list[tuple[Node, str]]:
    """Every scan node and the name a rule must use to address it.

    Ordered, and duplicates preserved: two scans of the same source are two
    separate places the obligation has to be discharged.
    """
    return [(n, n.scan.source_scope()) for n in _walk(root)
            if n.is_source and n.scan is not None]


def _applicable_rules(subject: Any, policy: Any) -> list[tuple[str, Any]]:
    """``(rule_name, RLSRule)`` for every role this subject holds."""
    rules = getattr(policy, "rls_rules", None)
    if not rules:
        return []
    return [(f"rls.{role}", rule) for role in sorted(rules)
            if subject.has_role(role) for rule in rules[role]]


def apply_row_security(root: Node, policy: Any, subject: Any
                       ) -> list[SecurityBarrier]:
    """Insert a barrier above every source this subject is restricted on.

    Returns the barriers injected. Raises :class:`~aar.failures.PolicyDenied`
    rather than guessing when a rule cannot be attached to exactly one source.
    """
    rules = _applicable_rules(subject, policy)
    if not rules:
        return []
    scans = _sources(root)
    if not scans:
        # No source to secure, so nothing to refuse: a pipeline built entirely
        # from constants has no rows for a role to be denied.
        return []

    scopes = {scope for _node, scope in scans if scope}
    barriers: list[SecurityBarrier] = []
    substitutions: dict[int, Node] = {}
    for rule_name, rule in rules:
        for target in _targets_for(rule, rule_name, scans, scopes):
            barrier = _build_barrier(target, rule, rule_name, subject)
            substitutions[id(target)] = barrier.node
            barriers.append(SecurityBarrier(
                rule=rule_name, source=barrier.node.source_scope,
                node=barrier.node, subject=subject.name))

    if substitutions:
        _rewire(root, substitutions)
    return barriers


def _targets_for(rule: Any, rule_name: str, scans: list[tuple[Node, str]],
                 scopes: set[str]) -> list[Node]:
    """Which scan nodes one rule addresses, or raise."""
    declared = getattr(rule, "source", "") or ""
    if declared:
        matched = [n for n, scope in scans if scope == declared]
        if not matched:
            # A rule aimed at a source this pipeline does not read is almost
            # always a typo, and it means the author believed a restriction
            # was in force that is not.
            raise PolicyDenied(
                f"{rule_name} restricts source {declared!r}, which this "
                f"pipeline does not read. The sources available are: "
                f"{', '.join(sorted(scopes)) or '<none>'}. Nothing is applied, "
                f"because a rule that cannot be placed must not be assumed to "
                f"be satisfied.",
                rule=rule_name, source=declared)
        return matched

    # Unscoped: legal only when exactly one source could mean it.
    if len(scopes) != 1:
        raise PolicyDenied(
            f"{rule_name} is an unscoped row-level rule ({rule.predicate!r}) "
            f"but this pipeline reads {len(scopes)} sources "
            f"({', '.join(sorted(scopes)) or 'none named'}). An unscoped "
            f"predicate cannot say which input it belongs to, and guessing "
            f"decides whose rows the subject may see. Give the rule a 'source' "
            f"key, or declare source_name= on the scan.",
            rule=rule_name, sources=sorted(scopes))
    only = next(iter(scopes))
    return [node for node, scope in scans if scope == only]


def _build_barrier(scan: Node, rule: Any, rule_name: str, subject: Any
                   ) -> SecurityBarrier:
    """Build the barrier, parsing the predicate against the scan's schema."""
    scope = scan.scan.source_scope() if scan.scan is not None else ""
    predicate = parse_row_predicate(rule.predicate,
                                    columns=_declared_columns(scan),
                                    rule=rule_name, source=scope)
    node = Node(NodeType.SECURITY_FILTER, inputs=[scan], predicate=predicate,
                security_rule=rule_name, source_scope=scope)
    node.estimated_bytes = scan.estimated_bytes
    node.output_schema = scan.output_schema
    node.input_schemas = scan.input_schemas
    node.privacy = scan.privacy
    return SecurityBarrier(rule=rule_name, source=scope, node=node,
                           subject=subject.name)


def _declared_columns(scan: Node) -> tuple[str, ...]:
    """Column names the pipeline *declares* for this source, if any.

    Deliberately the declared schema only. Opening the real source to validate
    a security predicate would mean reading data before the policy has been
    applied to it, which is the ordering this module exists to establish. A
    pipeline that declares no schema gets its predicate parsed without column
    checking here; the engines still refuse a missing column when they
    evaluate it.
    """
    fields = getattr(scan.output_schema, "fields", None)
    if not fields:
        return ()
    return tuple(f.name for f in fields if getattr(f, "name", None))


def _rewire(root: Node, substitutions: Mapping[int, Node]) -> None:
    """Point every consumer of a substituted scan at its barrier instead."""
    for node in _walk(root):
        for i, child in enumerate(node.inputs):
            replacement = substitutions.get(id(child))
            if replacement is not None:
                node.inputs[i] = replacement


def assert_barriers_intact(root: Node, barriers: list[SecurityBarrier]) -> None:
    """Verify every barrier survived planning with its obligation intact.

    Called after planning. It checks three things a future optimiser could
    plausibly break:

    1. the node is still reachable from the root,
    2. its predicate is still the one policy wrote,
    3. it still sits *below* any aggregation, join or window on its path, so
       the restriction dominates the computation rather than following it.

    A check that cannot be performed is reported as a failure, not skipped.
    """
    if not barriers:
        return
    present = {id(n) for n in _walk(root)}
    for barrier in barriers:
        node = barrier.node
        if id(node) not in present:
            raise PolicyDenied(
                f"{barrier.rule} was applied to {barrier.source} but the "
                f"barrier is no longer part of the plan. A row-level "
                f"restriction removed before execution protects nothing.",
                rule=barrier.rule, source=barrier.source)
        if node.predicate != barrier.predicate:
            raise PolicyDenied(
                f"{barrier.rule} on {barrier.source} was rewritten to a "
                f"different predicate ({node.predicate} instead of "
                f"{barrier.predicate}). A weakened restriction is not a "
                f"faster one, it is a different rule.",
                rule=barrier.rule, source=barrier.source)
        _assert_dominates(node, root, barrier)


def _assert_dominates(node: Node, root: Node, barrier: SecurityBarrier) -> None:
    """No combining operation may sit *between* the source and the barrier.

    Walks the barrier's own **ancestors**. If one of them is a join, group-by,
    aggregate, window, UDF, sort or dedup, then the barrier is applied *after*
    that operation - the rows reaching it have already been combined, reduced
    or reordered, which is exactly the case RLS exists to prevent.

    The direction is easy to get backwards, and it was: an earlier version
    walked *downstream* instead, which flags every healthy plan (a barrier
    above a source is always above a group-by) and therefore reports the
    correct arrangement as a violation. ``Node.walk`` visits ancestors only, so
    checking "what did this filter come after" is the walk it already provides;
    it is checking "what does this filter feed" that needs the reverse index.
    """
    combining = {NodeType.JOIN, NodeType.GROUPBY, NodeType.AGGREGATE,
                 NodeType.WINDOW, NodeType.PYTHON_UDF, NodeType.SORT,
                 NodeType.DEDUPLICATE}
    for ancestor in _walk(node):
        if ancestor.type in combining:
            raise PolicyDenied(
                f"{barrier.rule} on {barrier.source} would be applied after a "
                f"{ancestor.type.value}. Row-level security has to dominate "
                f"every operation that can combine, reduce or reorder rows; "
                f"applied after one, it filters the output of a computation "
                f"that already saw the restricted rows.",
                rule=barrier.rule, source=barrier.source)


# ------------------------------------------------------------------ parsing
def parse_row_predicate(expression: str, columns: tuple[str, ...] = (),
                        rule: str = "rls", source: str = "") -> Expr:
    """Parse a policy predicate into IR, exactly or not at all.

    The grammar is a documented, closed subset - conjunctions of
    ``column = literal`` - parsed into the same :class:`~aar.ir.BinOp` the
    planner uses, so a policy cannot smuggle SQL in through a string.

    Literals are typed rather than left as text: ``tenant_id = 42`` against an
    integer column must become ``42`` and not the string ``"42"``, which
    compares unequal to every integer and silently filters the whole relation.

    Raises on anything it cannot represent. There is no partial parse.
    """
    text = (expression or "").strip()
    if not text:
        raise PolicyDenied(
            f"{rule} on {source or 'its source'} has an empty predicate",
            rule=rule, source=source)

    known = {name.lower(): name for name in columns}
    parts = [_parse_term(term, known, rule, source)
             for term in text.split(" and ")]
    predicate = parts[0]
    for extra in parts[1:]:
        predicate = BinOp(predicate, "AND", extra)
    return predicate


def _parse_term(term: str, known: Mapping[str, str], rule: str,
                source: str) -> Expr:
    where = source or "its source"
    raw = term.strip()
    name, sep, value = raw.partition("=")
    if not sep:
        raise PolicyDenied(
            f"{rule} on {where} uses {raw!r}, which AAR cannot parse. "
            f"Row-level rules are conjunctions of 'column = literal' and "
            f"nothing else - AAR will not run 'what it thinks' a security "
            f"rule.", rule=rule, source=source, term=raw)
    column_name = name.strip()
    if not column_name:
        raise PolicyDenied(
            f"{rule} on {where} has a term with no column ({raw!r})",
            rule=rule, source=source, term=raw)
    if not _is_identifier(column_name):
        # Without this, "1=1; DROP TABLE users --" parses happily as
        # ``column "1" = "1; DROP TABLE users --"``. Nothing is executed - the
        # engine quotes identifiers - so it is not injection, but it *is* a
        # security rule silently meaning something nobody wrote, filtering on a
        # column that does not exist. A rule AAR cannot state plainly is a rule
        # it should refuse.
        raise PolicyDenied(
            f"{rule} on {where} uses {column_name!r}, which is not a column "
            f"name AAR can represent. Row-level rules are conjunctions of "
            f"'column = literal' with a plain column name.",
            rule=rule, source=source, term=raw)
    if known:
        resolved = known.get(column_name.lower())
        if resolved is None:
            raise PolicyDenied(
                f"{rule} on {where} references column {column_name!r}, which "
                f"is not among this source's declared columns "
                f"({', '.join(known.values())}). Nothing is applied: a rule "
                f"naming a column that does not exist protects nothing and "
                f"looks identical to one that works.",
                rule=rule, source=source, column=column_name)
        column_name = resolved
    return BinOp(Col(column_name), "=", Lit(_literal_value(value.strip())))


def _is_identifier(name: str) -> bool:
    """Whether ``name`` is a plain column name AAR is willing to use.

    Letters, digits, underscore, space, dot and brackets - which together cover
    the quoted identifiers, dotted paths and bracketed column names that appear
    in real schemas, including Excel headers with spaces.

    A leading character must be alphabetic or an underscore. That single rule is
    what refuses ``1=1; DROP TABLE users --``: without it, ``1`` is "alphanumeric"
    and the rest parses as a column named ``1`` compared against a string
    containing a DROP statement. Requiring the shape an identifier actually has
    catches that without needing a blocklist of attack strings, which would only
    ever catch the ones already thought of.
    """
    if not name or len(name) > 256:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch in "_ .[]" for ch in name)


def _literal_value(raw: str) -> Any:
    """Turn a policy literal into a Python value.

    Quoted text stays text. Anything else is read as a number when it parses as
    one, so ``tenant_id = 42`` does not become the string ``"42"`` and compare
    unequal to every integer in the column.
    """
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() == "null":
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw