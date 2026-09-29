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
        role -> SQL predicate. A role with no entry sees every row, because
        row filtering is a grant, not a default; a *deny* needs a rule.
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
    """

    allow_network: bool = False
    allow_network_kinds: frozenset[str] = frozenset()
    max_sensitivity: Sensitivity = Sensitivity.INTERNAL
    rls: Mapping[str, str] = field(default_factory=dict)
    cls_mask: Mapping[str, Sequence[tuple[str, str]]] = field(
        default_factory=dict)
    cls_drop: Mapping[str, Sequence[str]] = field(default_factory=dict)
    mask_default: str = "redact"
    mask_threshold: Sensitivity = Sensitivity.CONFIDENTIAL
    name: str = "default"

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
        for role in sorted(self.rls):
            lines.append(f"  RLS {role}: WHERE {self.rls[role]}")
        for role in sorted(self.cls_drop):
            lines.append(f"  CLS {role}: drop {list(self.cls_drop[role])}")
        for role in sorted(self.cls_mask):
            pairs = ", ".join(f"{c}->{m}" for c, m in self.cls_mask[role])
            lines.append(f"  CLS {role}: mask {pairs}")
        return "\n".join(lines)



# ------------------------------------------------------------------ masks
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
        """The predicate this subject's rows must satisfy, if any."""
        for role in sorted(self.policy.rls):
            if subject.has_role(role):
                expr = self.policy.rls[role]
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
                      subject: Subject) -> Any:
        """Apply every obligation to ``table`` and return the result.

        Raises on egress or sensitivity denial; applies RLS and CLS otherwise.
        The returned table keeps its classification tags, so a value that
        left masked is still *known* to have been masked by policy rather
        than by someone remembering to.
        """
        schema = getattr(table, "schema", None)
        decision = self.check_egress(sink, subject, schema)
        if not decision.allowed:
            raise PrivacyViolation(
                f"policy {self.policy.name!r} denied this write: "
                f"{decision.reason}", rule=decision.rule,
                sink=str(sink), subject=subject.name)

        result = table
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
        from ..ir import BinOp, Col, Lit

        expression = (obligation.expression or "").strip()
        if not expression:
            return table
        lower = {name.lower(): name for name in table.column_names}
        parts: list[Any] = []
        for term in expression.split(" and "):
            name, eq, raw = term.strip().partition("=")
            if not eq:
                raise PolicyDenied(
                    f"row-level rule {expression!r} uses an unsupported term "
                    f"{term.strip()!r}; AAR understands only conjunctions "
                    f"of 'column = literal'",
                    rule=obligation.rule, term=term.strip())
            column = lower.get(name.strip().lower())
            if column is None:
                raise PolicyDenied(
                    f"row-level rule references column {name.strip()!r}, "
                    f"which is not in the result "
                    f"(have: {', '.join(table.column_names)})",
                    rule=obligation.rule, column=name.strip())
            # The literal is typed from the column, not from the string.
            # Comparing a float column to "3.0" as text matches nothing, and
            # a rule that silently filters every row looks identical to one
            # that works until someone checks the output.
            raw_value = raw.strip()
            if raw_value[:1] in ("'", '"') and raw_value[-1:] == raw_value[:1]:
                literal: Any = raw_value[1:-1]
            else:
                field_type = table.schema.get(column).type
                kind = getattr(field_type, "kind", None)
                if kind is not None and "INT" in str(kind).upper():
                    literal = int(float(raw_value))
                elif kind is not None and "FLOAT" in str(kind).upper():
                    literal = float(raw_value)
                else:
                    literal = raw_value
            parts.append(BinOp(Col(column), "=", Lit(literal)))

        predicate = parts[0]
        for extra in parts[1:]:
            predicate = BinOp(predicate, "AND", extra)
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
    "rls": {"emea": "region = EU"},
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
             "mask_default", "mask_threshold"}
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
    )
