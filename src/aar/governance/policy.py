"""Policy: the difference between *carrying* a privacy tag and *honouring* it.

Every field in AAR can carry a classification. Until this module, that tag
was documentation — it travelled faithfully and changed nothing. The
specification asks for architectural privacy: "a CONFIDENTIAL tag in
PostgreSQL is still there in the Excel output." This module is what makes
that sentence true, and it is deliberately small enough to audit by reading.

Four obligations, matching §13:

* **Egress** — data may not leave the machine to a network sink unless a rule
  permits it. The default is deny, so forgetting to write a policy is safe.
* **Classification** — data at or above a given sensitivity may not reach a
  given sink.
* **Row-level security** — a role's rows are removed by injecting a
  predicate, not by hoping the analyst remembers.
* **Column-level security** — a role sees masked or dropped columns rather
  than raw values.

Two properties matter more than the feature list:

* **Every decision names a reason.** A denied plan that does not say which
  rule refused it is unusable, so :class:`Decision` always carries one.
* **Nothing happens implicitly.** A policy with no matching rule denies.
  Default-allow would make the policy a suggestion.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..failures import PolicyDenied, PrivacyViolation
from ..types import Sensitivity, sensitivity_of

__all__ = [
    "Sensitivity", "sensitivity_of", "Action", "Decision", "Obligation",
    "Subject", "Sink", "Policy", "PolicyEngine", "mask_value",
]


class Action(str, enum.Enum):
    """What the engine did about a rule."""

    ALLOW = "allow"
    DENY = "deny"
    MASK = "mask"
    DROP_COLUMN = "drop_column"
    FILTER_ROWS = "filter_rows"


@dataclass(frozen=True, slots=True)
class Obligation:
    """One policy obligation the executor must apply.

    Obligations are data, not code paths: the policy engine decides *what*
    must happen and the executor decides how. That separation is what lets a
    policy be audited, printed and diffed without reading the runtime.
    """

    action: Action
    column: str | None = None
    expression: str | None = None
    rule: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Decision:
    """A verdict, with the reason attached. Never returned without one."""

    allowed: bool
    reason: str
    action: Action = Action.ALLOW
    obligations: tuple[Obligation, ...] = ()
    rule: str = ""

    def __bool__(self) -> bool:
        return self.allowed

    def render(self) -> str:
        verb = "ALLOW" if self.allowed else "DENY"
        parts = [f"{verb}: {self.reason}"]
        if self.rule:
            parts.append(f"[{self.rule}]")
        if self.obligations:
            parts.append(f"{len(self.obligations)} obligation(s)")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Subject:
    """Who is asking: a person, a service, a pipeline.

    ``roles`` drive RLS and CLS; ``attributes`` drive ABAC rules. Both are
    plain data so a policy can be expressed without writing Python.
    """

    name: str = "anonymous"
    roles: frozenset[str] = frozenset()
    attributes: Mapping[str, str] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        return role in self.roles or "*" in self.roles

    def attribute(self, key: str, default: str = "") -> str:
        return self.attributes.get(key, default)



# ------------------------------------------------------------------- sinks
#: Anything that leaves the process is egress. AAR has no network client of
#: its own; these are the sinks a *connector* would use, and a rule that does
#: not mention one is not a rule about it.
NETWORK_SINKS: frozenset[str] = frozenset({
    "postgres", "mysql", "sqlserver", "oracle", "sqlite_network",
    "mongodb", "redis", "kafka", "s3", "gcs", "azure_blob", "http", "webhook",
    "email", "lakesnow", "bigquery", "databricks", "snowflake",
})

#: The sinks AAR knows write to the local filesystem. Membership here is what
#: earns a name the benefit of the doubt, which is why the list is explicit
#: and short rather than a default.
LOCAL_SINKS: frozenset[str] = frozenset({
    "excel", "xlsx", "parquet", "csv", "json", "jsonl", "arrow", "feather",
    "sqlite", "memory", "in_memory", "local", "local_file", "none", "stdout",
})


@dataclass(frozen=True, slots=True)
class Sink:
    """A destination for data, described well enough to reason about.

    ``kind`` matches the connector's system name (``excel``, ``parquet``,
    ``postgres``...). ``network`` says whether bytes leave the machine, which
    is the single fact the egress rule turns on.
    """

    kind: str
    network: bool = False
    name: str = ""

    @classmethod
    def of(cls, kind: str) -> "Sink":
        """Infer the sink kind from a connector name.

        Inference is not a shortcut past the policy - it is the default
        description, and an explicit :class:`Sink` always overrides it.

        The inference **fails closed**: only a name in :data:`LOCAL_SINKS`
        is believed to be local, and everything else - an unknown connector,
        a typo, an empty string - is treated as network egress. The
        asymmetry is the point. Believing a remote sink is local is a data
        leak; believing a local sink is remote is an inconvenience the
        analyst can fix by naming it.

        The earlier version of this inferred locality by *absence* from
        ``NETWORK_SINKS``, which inverted that: ``ftp``, ``smb`` and any
        name AAR had not heard of were all classified local and therefore
        permitted, under a default-deny policy that documented the opposite.
        A test pins the difference.
        """
        k = (kind or "").strip().lower()
        return cls(kind=k, network=k not in LOCAL_SINKS)

    def describe(self) -> str:
        where = "network" if self.network else "local"
        return f"{self.kind or '<unnamed>'} ({where})"

    def __str__(self) -> str:
        return self.describe()



# ------------------------------------------------------------------ policy
@dataclass(frozen=True, slots=True)
class RLSRule:
    """One row-level restriction, addressed to a named source.

    ``source`` is the logical name a :class:`~aar.ir.ScanSpec` declares (or the
    one derived from its table, collection or file stem). The empty string
    means "unscoped", which is accepted only when exactly one source could
    mean it - see :func:`aar.governance.rewrite.apply_row_security`.

    Scoping exists because a bare predicate is ambiguous the moment a pipeline
    reads more than one source::

        Orders --\\
                   JOIN  --  region = 'EU' belongs to which side?
        Users  --/

    Answering "the first one" or "both" would be a guess, and a guess in this
    position decides whose rows an analyst may see.
    """

    predicate: str
    source: str = ""

    def describe(self) -> str:
        where = self.source or "<any single source>"
        return f"{where}: WHERE {self.predicate}"


@dataclass(frozen=True, slots=True)
class Policy:
    """A set of rules. Absent rules deny; that is the whole design.

    Attributes
    ----------
    allow_network:
        Whether data may leave the machine at all. ``False`` by default.
    allow_network_kinds:
        The specific network sinks permitted when ``allow_network`` is
        false. An empty set with ``allow_network=False`` means none.
    max_sensitivity:
        The highest sensitivity permitted to reach a network sink.
    rls:
        role -> SQL predicate. A role with no entry sees every row,
        because row filtering is a grant, not a default; a *deny*
        needs a rule.

        Accepts either a bare predicate string (unscoped - legal only when the
        pipeline has exactly one source that could mean it) or a list of
        :class:`RLSRule`-shaped mappings with ``source`` and ``predicate``
        keys. Both forms normalise to :attr:`rls_rules`, which is what the
        rewrite pass reads.
    cls_mask:
        role -> list of (column, mask_kind). Applied on the way out.
    cls_drop:
        role -> columns removed entirely. Dropping beats masking when the
        value is not needed at all, and it is auditable.
    mask_default:
        The mask applied to a column that matches no explicit rule but is
        above ``mask_threshold``.
    mask_threshold:
        Sensitivity at or above which masking applies.
    disclosure:
        Optional ``min_group_size`` rules. RLS and CLS both constrain *access*
        - which rows, which columns. Neither constrains *inference*: a subject
        who may read every row in their own department can still learn an
        individual's salary from ``AVG`` over a group of one. This is that
        separate control, and it is absent unless a policy asks for it.
    """

    allow_network: bool = False
    allow_network_kinds: frozenset[str] = frozenset()
    max_sensitivity: Sensitivity = Sensitivity.INTERNAL
    rls: Mapping[str, Any] = field(default_factory=dict)
    #: Minimum-group-size rules enforced on aggregate results. Empty by
    #: default: a disclosure control nobody configured must not silently
    #: change results, and one that is configured must be enforced rather than
    #: advisory.
    disclosure: tuple[Any, ...] = ()
    #: ``role -> (RLSRule, ...)``, derived from :attr:`rls` in
    #: ``__post_init__``. Declared as a field rather than computed on demand so
    #: that a frozen dataclass still exposes it as data.
    rls_rules: Mapping[str, tuple[RLSRule, ...]] = field(default_factory=dict)
    cls_mask: Mapping[str, Sequence[tuple[str, str]]] = field(
        default_factory=dict)
    cls_drop: Mapping[str, Sequence[str]] = field(default_factory=dict)
    mask_default: str = "redact"
    mask_threshold: Sensitivity = Sensitivity.CONFIDENTIAL
    name: str = "default"

    def __post_init__(self) -> None:
        """Normalise ``rls`` into :attr:`rls_rules` once, at construction.

        Storing both shapes and interpreting at each use site would mean two
        readers that could disagree. Normalising in one place means
        :attr:`rls_rules` is the only thing the rewrite pass has to understand,
        and a caller who wrote the legacy bare-string form gets identical
        behaviour to one who was explicit.
        """
        normalised: dict[str, tuple[RLSRule, ...]] = {}
        for role, value in dict(self.rls).items():
            normalised[role] = _coerce_rls_rules(role, value)
        object.__setattr__(self, "rls_rules", normalised)

    def rules_for_role(self, role: str) -> tuple[RLSRule, ...]:
        """Every row rule that applies to ``role``."""
        return tuple(self.rls_rules.get(role, ()))

    def permits_network_kind(self, kind: str) -> bool:
        if self.allow_network:
            return True
        return (kind or "").strip().lower() in self.allow_network_kinds

    def render(self) -> str:
        lines = [f"Policy {self.name!r}"]
        lines.append(
            f"  egress: {'allow' if self.allow_network else 'deny'}"
            + (f" except {sorted(self.allow_network_kinds)}"
               if self.allow_network_kinds and not self.allow_network
               else ""))
        lines.append(f"  max sensitivity at a network sink: "
                     f"{self.max_sensitivity.name}")
        for role in sorted(self.rls_rules):
            for rule in self.rls_rules[role]:
                lines.append(f"  RLS {role}: {rule.describe()}")
        for role in sorted(self.cls_drop):
            lines.append(f"  CLS {role}: drop {list(self.cls_drop[role])}")
        for role in sorted(self.cls_mask):
            pairs = ", ".join(f"{c}->{m}" for c, m in self.cls_mask[role])
            lines.append(f"  CLS {role}: mask {pairs}")
        return "\n".join(lines)



# ------------------------------------------------------------------ masks
def _coerce_rls_rules(role: str, value: Any) -> tuple[RLSRule, ...]:
    """Accept both RLS spellings, reject the rest.

    Two forms are supported, because the second is a strict superset of the
    first and existing policies must keep working::

        "rls": {"analyst": "region = 'EU'}

        "rls": {"analyst": [
            {"source": "orders", "predicate": "region = 'EU'"}]}

    Anything else raises. A policy file is not a place to be forgiving: a rule
    that is silently misread is a rule that silently fails to protect.
    """
    if isinstance(value, str):
        return (RLSRule(predicate=value),) if value.strip() else ()
    if isinstance(value, RLSRule):
        return (value,)
    if isinstance(value, Mapping):
        value = [value]
    if isinstance(value, Sequence):
        rules: list[RLSRule] = []
        for item in value:
            if isinstance(item, str):
                rules.append(RLSRule(predicate=item))
                continue
            if not isinstance(item, Mapping):
                raise PolicyDenied(
                    f"RLS rule for role {role!r} must be a string or a mapping "
                    f"with 'source' and 'predicate'; got "
                    f"{type(item).__name__}", rule=f"rls.{role}")
            unknown = sorted(set(item) - {"source", "predicate"})
            if unknown:
                raise PolicyDenied(
                    f"RLS rule for role {role!r} has unknown key(s) "
                    f"{', '.join(unknown)}; expected 'source' and 'predicate'",
                    rule=f"rls.{role}")
            predicate = str(item.get("predicate", "") or "").strip()
            if not predicate:
                raise PolicyDenied(
                    f"RLS rule for role {role!r} has an empty predicate; a "
                    f"rule that matches everything protects nothing",
                    rule=f"rls.{role}")
            rules.append(RLSRule(predicate=predicate,
                                 source=str(item.get("source", "") or "")))
        return tuple(rules)
    raise PolicyDenied(
        f"RLS rules for role {role!r} must be a string or a list of rules; "
        f"got {type(value).__name__}", rule=f"rls.{role}")


_MASKERS: dict[str, Any] = {}


def mask_value(value: Any, kind: str, column: str = "") -> Any:
    """Replace a value according to a masking rule.

    The kinds are chosen so that a masked column is still *usable*: a masked
    number stays numeric, so an average of a masked column does not become a
    type error. A redaction that produced ``"***"`` in a numeric column would
    break every downstream aggregate while appearing to protect the data.

    ``full`` nulls the value; ``hash`` replaces it with a stable digest, which
    keeps join keys workable; ``partial`` keeps the last four characters,
    which is the usual "I need to recognise this customer" case.
    """
    global _MASKERS
    if not _MASKERS:
        _MASKERS = _build_maskers()
    fn = _MASKERS.get((kind or "").strip().lower())
    if fn is None:
        raise PolicyDenied(
            f"unknown mask {kind!r} for column {column!r}; known masks are "
            f"{', '.join(sorted(_MASKERS))}", mask=kind, column=column)
    return fn(value)


def _build_maskers() -> dict[str, Any]:
    import hashlib

    def full(_v: Any) -> Any:
        return None

    def hash_(v: Any) -> Any:
        digest = hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:12]
        return digest

    def partial(v: Any) -> Any:
        text = str(v)
        if len(text) <= 4:
            return "*" * len(text)
        return "*" * (len(text) - 4) + text[-4:]

    def redact(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # Keep the type. A numeric column masked to text breaks the next
            # aggregate, and a broken pipeline gets the mask removed.
            return 0
        return "[redacted]"

    def email(v: Any) -> Any:
        text = str(v or "")
        if "@" not in text:
            return partial(text)
        local, _, domain = text.partition("@")
        head = local[:1] if local else ""
        return f"{head}***@{domain}"

    return {"full": full, "hash": hash_, "partial": partial,
            "redact": redact, "email": email}



def _retype_literals(predicate: Any, table: Any) -> None:
    """Re-type a parsed predicate's literals against the real schema.

    The parser in :mod:`aar.governance.rewrite` types literals from the text,
    because at rewrite time it has only a *declared* schema and often none at
    all. By the time a rule reaches the write boundary the real Arrow types
    are known, and that is strictly better information.

    It matters most for numerics. A float64 column compared against the string
    ``"3.0"`` is unequal to every value, so the rule silently filters the whole
    relation - and the analyst sees an empty result, indistinguishable from
    "no EU rows today". This walks the parsed predicate and rebuilds each
    literal as the type its column actually holds.
    """
    from ..ir import BinOp, Col, Lit

    if isinstance(predicate, BinOp):
        left, right = predicate.left, predicate.right
        if (isinstance(left, Col) and isinstance(right, Lit)
                and predicate.op == "="):
            column = left.name
            try:
                field = table.schema.get(column)
            except Exception:  # noqa: BLE001 - unknown column, leave as parsed
                return
            typed = _typed_literal(right.value, field)
            if typed != right.value:
                object.__setattr__(predicate, "right", Lit(typed))
            return
        _retype_literals(left, table)
        _retype_literals(right, table)


def _typed_literal(value: Any, field: Any) -> Any:
    """The value ``field``'s type implies, when that is unambiguous."""
    if not isinstance(value, str):
        return value
    kind = str(getattr(getattr(field, "type", None), "kind", "")).upper()
    if "INT" in kind:
        try:
            return int(float(value))
        except ValueError:
            return value
    if "FLOAT" in kind or "DOUBLE" in kind or "DECIMAL" in kind:
        try:
            return float(value)
        except ValueError:
            return value
    return value


# ------------------------------------------------------------------ engine
class PolicyEngine:
    """Applies a :class:`Policy` to data and to plans.

    The engine produces :class:`Decision` and :class:`Obligation` objects. It
    does not mutate data and it executes nothing: the executor asks what it
    must do and then does it. Keeping the two apart is what makes a policy
    printable, diffable and auditable without reading the runtime.
    """

    __slots__ = ("policy", "decisions")

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or Policy()
        #: Every decision this engine has made, in order. A run's policy
        #: audit is exactly this list.
        self.decisions: list[Decision] = []

    # ------------------------------------------------------------ egress
    def check_egress(self, sink: Sink | str, subject: Subject,
                     schema: Any = None) -> Decision:
        """May this data reach this sink?

        Two independent questions, because they fail differently: is the
        sink a network sink at all (egress), and is the data sensitive enough
        for it (classification). Reporting only one would leave the operator
        guessing which half to fix.
        """
        s = sink if isinstance(sink, Sink) else Sink.of(sink)
        if not s.network:
            d = Decision(True, f"{s.describe()} is local; no egress",
                         rule="egress.local")
            self.decisions.append(d)
            return d
        if not self.policy.permits_network_kind(s.kind):
            d = Decision(
                False,
                f"policy {self.policy.name!r} denies egress to {s.kind!r}; "
                f"data would leave this machine",
                action=Action.DENY, rule="egress.network")
            self.decisions.append(d)
            return d
        level = _schema_sensitivity(schema)
        if level > self.policy.max_sensitivity:
            d = Decision(
                False,
                f"data is {level.name} but policy {self.policy.name!r} "
                f"permits at most {self.policy.max_sensitivity.name} at a "
                f"network sink",
                action=Action.DENY, rule="egress.sensitivity")
            self.decisions.append(d)
            return d
        d = Decision(True, f"{s.kind} is permitted and data is {level.name}",
                     rule="egress.ok")
        self.decisions.append(d)
        return d

    # -------------------------------------------------------------- rls
    def row_predicate(self, subject: Subject) -> Decision:
        """The write-time row predicate, for a plan that carries no barrier.

        The *primary* application of RLS is
        :func:`aar.governance.rewrite.apply_row_security`, which puts the
        restriction into the logical plan before anything is computed. This
        method is the fallback for a plan that was never rewritten - a bare
        :class:`~aar.runtime.Executor` used directly, or a run with policy
        enforcement explicitly disabled - and it can only express the simple
        case, because a single output table has no notion of which input a
        scoped rule belonged to.

        So it applies the rules for the *first* matching role, and refuses
        outright if that role's rules are source-scoped. Applying one rule of a
        multi-source policy and ignoring the rest would be a partial grant
        presented as a complete one.
        """
        for role in sorted(self.policy.rls_rules):
            if not subject.has_role(role):
                continue
            rules = self.policy.rls_rules[role]
            if not rules:
                continue
            scoped = [r for r in rules if r.source]
            if scoped:
                raise PolicyDenied(
                    f"role {role!r} has source-scoped row rules "
                    f"({', '.join(r.source for r in scoped)}) but this plan "
                    f"carries no security barriers, so there is no way to "
                    f"apply them to the right input. Run the pipeline through "
                    f"aar's policy rewrite, which injects them per source.",
                    rule=f"rls.{role}")
            expr = rules[0].predicate
            d = Decision(True,
                         f"role {role!r} restricts rows to WHERE {expr}",
                         action=Action.FILTER_ROWS,
                         obligations=(
                             Obligation(Action.FILTER_ROWS,
                                        expression=expr,
                                        rule=f"rls.{role}",
                                        reason="row-level security"),),
                         rule=f"rls.{role}")
            self.decisions.append(d)
            return d
        d = Decision(True, "no row-level rule applies to this subject",
                     rule="rls.none")
        self.decisions.append(d)
        return d

    # -------------------------------------------------------------- cls
    def column_actions(self, subject: Subject, schema: Any) -> list[Obligation]:
        """What must happen to each column before this subject sees it.

        Explicit rules win over the threshold default. Dropping beats masking,
        because a dropped column cannot be recovered by anything downstream;
        masking at least keeps the row count and the aggregates honest.
        """
        obligations: list[Obligation] = []
        fields = getattr(schema, "fields", ()) or ()
        drops: set[str] = set()
        masks: set[str] = set()

        for role in sorted(self.policy.cls_drop):
            if subject.has_role(role):
                for column in self.policy.cls_drop[role]:
                    drops.add(column)
                    obligations.append(Obligation(
                        Action.DROP_COLUMN, column=column,
                        rule=f"cls.{role}",
                        reason=f"role {role!r} may not see {column!r}"))
        for role in sorted(self.policy.cls_mask):
            if subject.has_role(role):
                for column, _kind in self.policy.cls_mask[role]:
                    masks.add(column)
                    obligations.append(Obligation(
                        Action.MASK, column=column, rule=f"cls.{role}",
                        reason=f"role {role!r} sees a masked {column!r}"))

        for schema_field in fields:
            name = getattr(schema_field, "name", None)
            if not name or name in drops or name in masks:
                continue
            level = sensitivity_of(getattr(schema_field, "classification", ()))
            if level >= self.policy.mask_threshold:
                obligations.append(Obligation(
                    Action.MASK, column=name, rule="cls.default",
                    reason=f"column is {level.name}, at or above the "
                           f"{self.policy.mask_threshold.name} threshold"))
        return obligations


    # --------------------------------------------------------- enforcement
    def enforce_write(self, table: Any, sink: Sink | str,
                      subject: Subject, rls_applied: bool = False) -> Any:
        """Apply every obligation to ``table`` and return the result.

        Raises on egress or sensitivity denial; applies RLS and CLS otherwise.
        The returned table keeps its classification tags, so a value that
        left masked is still *known* to have been masked by policy rather
        than by someone remembering to.

        ``rls_applied`` says the plan already carried row-level security
        barriers, injected by :func:`aar.governance.rewrite.apply_row_security`.
        It must be true whenever the pipeline was secured, and it changes what
        happens here in a way that is not cosmetic:

        * the rows were already restricted *before* every join, group-by and
          window, so filtering again would be redundant work; and
        * more importantly, applying the predicate here could **refuse** the
          write. ``region = 'EU'`` against the output of
          ``GROUP BY region`` matches; the same rule against the output of
          ``AVG(salary)`` names a column that does not exist, and a strict
          implementation raises - so a correctly-secured analytic query would
          fail at the write, having computed the right answer.

        So: with barriers in the plan, RLS is the plan's job and only CLS and
        egress remain. Without them - a bare :class:`Executor` used directly,
        or an explicitly disabled policy - this method still applies RLS
        itself, which is the only safe reading of a plan nobody secured.
        """
        schema = getattr(table, "schema", None)
        decision = self.check_egress(sink, subject, schema)
        if not decision.allowed:
            raise PrivacyViolation(
                f"policy {self.policy.name!r} denied this write: "
                f"{decision.reason}", rule=decision.rule,
                sink=str(sink), subject=subject.name)

        result = table
        if not rls_applied:
            predicate = self.row_predicate(subject)
            if predicate.obligations:
                result = self._apply_rls(result, predicate.obligations[0])

        obligations = self.column_actions(subject, getattr(result, "schema",
                                                            None))
        if obligations:
            result = self._apply_cls(result, obligations)
        return result

    def _mask_kind_for(self, column: str) -> str:
        """The mask a column gets, preferring an explicit rule."""
        for pairs in self.policy.cls_mask.values():
            for name, kind in pairs:
                if name == column:
                    return kind
        return self.policy.mask_default

    def _apply_rls(self, table: Any, obligation: Obligation) -> Any:
        """Remove rows the subject may not see.

        The predicate is a restricted, documented subset - conjunctions of
        ``column = literal`` - parsed into the same IR the planner uses, so a
        policy cannot smuggle arbitrary SQL in through a string. A rule
        naming a column that is not in the result raises rather than
        silently filtering nothing.
        """
        from ..engines.arrow_engine import ArrowEngine
        from .rewrite import parse_row_predicate

        expression = (obligation.expression or "").strip()
        if not expression:
            return table
        # The same parser the logical rewrite uses, so a rule cannot behave one
        # way when injected into the plan and another when applied at the
        # write. Literals are then re-typed from the *actual* schema, which is
        # the one advantage this path has: "3" against a float column has to
        # become 3.0, because comparing as text matches nothing and a rule
        # that silently filters every row looks exactly like one that works.
        predicate = parse_row_predicate(
            expression, columns=tuple(table.column_names),
            rule=obligation.rule, source="<result>")
        _retype_literals(predicate, table)
        return ArrowEngine().filter(table, predicate)



    def _apply_cls(self, table: Any,
                   obligations: Sequence[Obligation]) -> Any:
        """Drop and mask columns, preserving classification tags.

        Tags survive because "this column was masked" is provenance worth
        keeping: a downstream reader that sees an unmasked-looking
        identifier should still be able to ask what happened to it.
        """
        import pyarrow as pa

        from ..interchange import Table

        drops = {o.column for o in obligations
                 if o.action is Action.DROP_COLUMN and o.column}
        masks = {o.column: self._mask_kind_for(o.column)
                 for o in obligations
                 if o.action is Action.MASK and o.column}

        keep = [n for n in table.column_names if n not in drops]
        if drops and not keep:
            raise PrivacyViolation(
                "the policy removes every column from this result; there is "
                "nothing left to write", columns=sorted(drops))
        result = table.select(keep) if drops else table
        if not masks or result.num_rows == 0:
            return result

        rows = result.arrow.to_pylist()
        for row in rows:
            for column, kind in masks.items():
                if column in row:
                    row[column] = mask_value(row[column], kind, column)
        rebuilt = pa.Table.from_pylist(rows, schema=result.arrow.schema)
        return Table(rebuilt, result.schema)

    def render(self) -> str:
        """The policy plus every decision made, as one auditable block."""
        lines = [self.policy.render(), ""]
        if not self.decisions:
            lines.append("No policy decisions were made.")
        else:
            lines.append(f"Policy decisions ({len(self.decisions)}):")
            for d in self.decisions:
                lines.append("  " + d.render())
        return "\n".join(lines)


def _schema_sensitivity(schema: Any) -> Sensitivity:
    """The highest sensitivity anywhere in a schema.

    An absent schema is ``PUBLIC``, not ``RESTRICTED``: a write that has
    declared no columns has declared nothing sensitive, and refusing every
    such write would make the engine unusable for the many pipelines that do
    not yet annotate their sources. Annotation is opt-in here precisely
    because it is opt-in everywhere else too.
    """
    if schema is None:
        return Sensitivity.PUBLIC
    fields = getattr(schema, "fields", None)
    if fields is None:
        return Sensitivity.PUBLIC
    level = Sensitivity.PUBLIC
    for item in fields:
        level = max(level, sensitivity_of(
            getattr(item, "classification", ())))
    return level


# ------------------------------------------------------------------ files
#: A worked example, written by ``aar policy check --write-example``. It is
#: deliberately strict, and every field is commented, because a policy nobody
#: can read is a policy nobody will maintain.
EXAMPLE_POLICY: dict[str, Any] = {
    "name": "example-strict",
    "allow_network": False,
    "allow_network_kinds": ["postgres"],
    "max_sensitivity": "INTERNAL",
    "mask_threshold": "CONFIDENTIAL",
    "mask_default": "redact",
    # Row rules are *source-scoped*. The bare-string form ("emea": "region = EU")
    # is still accepted, but it is only legal when the pipeline reads exactly
    # one source that could mean it - in a join, "region = EU" does not say
    # whether it constrains Orders or Users, and guessing that decides whose
    # rows an analyst may see.
    "rls": {
        "emea": [{"source": "orders", "predicate": "region = EU"}],
    },
    "cls_drop": {"junior": ["ssn", "national_id"]},
    "cls_mask": {
        "analyst": [["email", "email"]],
        "junior": [["email", "email"], ["amount", "redact"]],
    },
}


def load_policy(path: str) -> Policy:
    """Load a :class:`Policy` from JSON.

    Unknown keys are rejected rather than ignored. A misspelled
    ``mask_threshold`` that is silently dropped would leave a policy that
    looks configured and masks nothing - the most dangerous possible failure
    for a file whose whole job is protection.
    """
    import json

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return policy_from_dict(raw)


def policy_from_dict(raw: Any) -> Policy:
    """Build a :class:`Policy` from a plain mapping."""
    if not isinstance(raw, dict):
        raise TypeError(
            f"a policy must be a JSON object, got {type(raw).__name__}")
    known = {"name", "allow_network", "allow_network_kinds",
             "max_sensitivity", "rls", "cls_drop", "cls_mask",
             "mask_default", "mask_threshold", "disclosure"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"unknown policy key(s): {', '.join(unknown)}; "
            f"valid keys are {', '.join(sorted(known))}")
    sensitivity = raw.get("max_sensitivity")
    threshold = raw.get("mask_threshold")
    return Policy(
        name=raw.get("name", "default"),
        allow_network=bool(raw.get("allow_network", False)),
        allow_network_kinds=frozenset(raw.get("allow_network_kinds", ())),
        max_sensitivity=(Sensitivity.parse(sensitivity) if sensitivity
                         else Sensitivity.INTERNAL),
        mask_threshold=(Sensitivity.parse(threshold) if threshold
                        else Sensitivity.CONFIDENTIAL),
        mask_default=raw.get("mask_default", "redact"),
        rls=dict(raw.get("rls", {})),
        cls_drop={k: list(v) for k, v in raw.get("cls_drop", {}).items()},
        cls_mask={k: [tuple(pair) for pair in v]
                  for k, v in raw.get("cls_mask", {}).items()},
        disclosure=_disclosure_rules(raw.get("disclosure")),
    )


def _disclosure_rules(raw: Any) -> tuple[Any, ...]:
    """Normalise the ``disclosure`` key into :class:`DisclosureRule` objects.

    Accepts an integer (the common case - one minimum group size), a list of
    integers, or a list of mappings with ``min_group_size`` and an optional
    ``rule`` name. An unknown key here is rejected for the same reason
    ``mask_threshold`` is: a misspelled disclosure key would leave a policy that
    looks like it protects small cells and does not.
    """
    from .disclosure import DisclosureRule

    if raw is None:
        return ()
    if isinstance(raw, (int, str)) and not isinstance(raw, bool):
        raw = [raw]
    rules: list[DisclosureRule] = []
    for item in raw:
        if isinstance(item, bool):
            raise ValueError(
                f"disclosure must be a minimum group size, got {item!r}")
        if isinstance(item, int):
            rules.append(DisclosureRule(min_group_size=item))
            continue
        if isinstance(item, str):
            # A bare string is not accepted: "5" and "five" are both plausible
            # and neither says what the number counts.
            raise ValueError(
                f"disclosure rule {item!r} must be a number or a mapping with "
                f"'min_group_size'; a string does not say what it counts")
        if isinstance(item, dict):
            unknown = sorted(set(item) - {"min_group_size", "rule"})
            if unknown:
                raise ValueError(
                    f"unknown disclosure key(s): {', '.join(unknown)}; valid "
                    f"keys are min_group_size, rule")
            rules.append(DisclosureRule(
                min_group_size=int(item.get("min_group_size", 5)),
                rule=str(item.get("rule", "disclosure.min_group_size"))))
            continue
        raise ValueError(
            f"a disclosure rule must be a number or a mapping, got "
            f"{type(item).__name__}")
    if not rules:
        raise ValueError(
            "disclosure is present but empty; omit the key entirely to have no "
            "small-cell control, because a rule that is written but empty looks "
            "configured and protects nothing")
    return tuple(rules)
