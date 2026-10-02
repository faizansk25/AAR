"""Evidence: what happened in a governed execution, and proof it has not changed.

AAR's buyer is the Head of Data Governance / Data Platform / Security and
Compliance. The analyst is the daily user; governance is what creates the
budget. So the runtime does not sell speed - a governance/compliance buyer is
asked to trust statements like *"RLS ran before aggregation"* and *"no
confidential data left this machine"*, and those statements are only worth
anything if the code that produces them is inspectable.

Which is why this package, and the runtime, the IR, the policy semantics and
the local evidence store beneath it, stay Apache-2.0. The mechanism that
produces the evidence is the part that must not be hidden. The commercial
boundary, when it comes, is organization-scale operation - centralised
evidence, policy distribution, retention and legal hold, SSO/SCIM, signed
attestations, SIEM/GRC integration - not the local mechanism.

Three modules, in dependency order:

* :mod:`aar.audit.canonical` - one reproducible byte encoding. Everything
  hashes against it, so it is the foundation and changes least often.
* :mod:`aar.audit.contract` - the event: immutable, versioned, hashable.
* :mod:`aar.audit.recorder` - one ordered sink with one hash chain, replacing
  four independent ledgers that could not reconstruct a run.

What is deliberately **not** here yet: the SQLite store, the query API, and any
authentication. AAR records identity *asserted by an external system* and names
the provider that vouched for it; it does not invent a second identity database
that an enterprise would have to operate alongside the one it already has.
"""

from .canonical import (  # noqa: F401
    GENESIS_HASH, CanonicalisationError, canonical_bytes, canonical_text,
    chain_digest, digest, indexed_columns,
)
from .contract import (  # noqa: F401
    EVIDENCE_SCHEMA_VERSION, EventType, EvidenceEvent, SubjectIdentity,
    new_event_id, new_run_id,
)

#: Shorthand for the subject type an audit record carries. Named to read well
#: next to :class:`EventType` and :class:`EvidenceEvent` at a call site, and
#: distinct from :mod:`aar.governance`'s ``Subject`` so the two are never
#: confused for one another - that one carries a live identity, this one a
#: record of what an external system asserted.
Subject = SubjectIdentity
from .recorder import (  # noqa: F401
    EvidenceRecorder, RunContext, VerificationResult, event_fields,
    rows_removed, stamp, verify_chain,
)

from .view import (  # noqa: F401
    Capability, EvidenceAuthorizer, EvidenceView, FieldState, FieldValue,
    RESTRICTED_COUNT_FIELDS, project, trust_boundary_notice,
)

__all__ = [
    "CanonicalisationError", "Capability", "EVIDENCE_SCHEMA_VERSION",
    "EvidenceAuthorizer", "EvidenceEvent", "EvidenceRecorder", "EvidenceView",
    "EventType", "FieldState", "FieldValue", "GENESIS_HASH",
    "RESTRICTED_COUNT_FIELDS", "RunContext", "Subject", "SubjectIdentity",
    "VerificationResult", "canonical_bytes", "canonical_text", "chain_digest",
    "digest", "event_fields", "indexed_columns", "new_event_id", "new_run_id",
    "project", "rows_removed", "stamp", "trust_boundary_notice",
    "verify_chain",
]