"""Canonical encoding: the one way an evidence event becomes bytes.

A hash is only worth something if the bytes it covers are reproducible. Two
processes on two machines, the same version of AAR, must produce identical
canonical bytes for identical events - otherwise the hash chain cannot be
verified and every "this evidence has not been modified" claim is decorative.

Getting this wrong is silent. A canonical form that ignores key order breaks
reproducibility; one that renders ``True`` and ``1`` identically lets a
tampered event hash to the same value as the original; one that concatenates
strings without a length prefix lets a crafted string impersonate a
different structure. So the rules are explicit, typed, and enforced rather
than left to ``json.dumps``.

The rules:

* **Every value carries its type tag.** ``true`` and ``1`` are different
  values and must not encode to the same bytes - in Python they compare
  equal, which is exactly why they need distinguishing.
* **Strings are length-prefixed.** No delimiter can be smuggled inside one.
* **Floats are rejected.** A float's shortest round-trip representation is a
  property of the language, not of the event, and it has changed between
  versions. Measured durations are recorded as integer microseconds instead,
  which is also the honest precision for an audit trail - sub-microsecond
  timing noise on a wall clock is not a fact about the data.
* **The schema version is hashed, not merely stored.** Evidence written by one
  version cannot be replayed as another.

Nothing here is clever and nothing here is negotiable. It is the layer that
makes the rest of the audit subsystem worth anything.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Mapping, Sequence

__all__ = [
    "GENESIS_HASH", "CanonicalisationError", "canonical_bytes",
    "canonical_text", "chain_digest", "digest",
]

#: Domain separator. Mixing this into every digest means a digest produced for
#: one purpose can never be replayed as a digest for another - the same
#: technique the semantic-identity work already uses.
_EVIDENCE_DOMAIN = b"aar-evidence-v1\x00"

#: Used for the first event of a run, before any hash exists.
GENESIS_HASH = "0" * 64


class CanonicalisationError(ValueError):
    """A value cannot be canonically encoded, and must not be guessed at."""


def canonical_bytes(value: Any) -> bytes:
    """Encode ``value`` to its canonical byte form.

    Supported: ``None``, ``bool``, ``int``, ``str``, and arbitrarily nested
    lists/tuples and string-keyed mappings. Everything else raises, because a
    silent coercion here produces a hash that verifies against a value the
    caller did not write.
    """
    out: list[bytes] = []
    _encode(value, out)
    return b"".join(out)


def canonical_text(value: Any) -> str:
    """Canonical bytes, decoded. For display and debugging only.

    Hashing goes through :func:`canonical_bytes`; this exists so a developer
    can *see* what was hashed, which is the only practical way to debug a
    mismatched chain.
    """
    return canonical_bytes(value).decode("utf-8")


def _encode(value: Any, out: list[bytes]) -> None:
    if value is None:
        out.append(b"n;")
        return
    # bool before int: in Python ``True == 1``, so testing int first would let a
    # boolean encode as a number and collide with the integer one.
    if isinstance(value, bool):
        out.append(b"b:1;" if value else b"b:0;")
        return
    if isinstance(value, int):
        if not -(2 ** 63) <= value < 2 ** 63:
            raise CanonicalisationError(
                f"integer {value} does not fit in 64 bits; evidence amounts "
                f"should be bounded, and an unbounded counter is usually a sign "
                f"that the wrong quantity is being recorded")
        out.append(f"i:{value};".encode("ascii"))
        return
    if isinstance(value, float):
        raise CanonicalisationError(
            "floats are not canonicalisable: their shortest representation "
            "is a property of the Python version, not of the event. Record a "
            "duration as integer microseconds (`duration_us`) instead.")
    if isinstance(value, str):
        # NFC first, so "é" written two ways is one value. Length-prefixed so
        # no content can terminate the field early or imitate structure.
        encoded = unicodedata.normalize("NFC", value).encode("utf-8")
        out.append(f"s:{len(encoded)}:".encode("ascii"))
        out.append(encoded)
        out.append(b";")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise CanonicalisationError(
            "raw bytes are not canonicalisable; base64 or hex them into a "
            "string so the encoding choice is explicit")
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalisationError(
                    f"mapping key {key!r} is not a string; a non-string key "
                    f"has no stable canonical order")
            items.append((key, item))
        items.sort(key=lambda kv: kv[0])
        out.append(f"m:{len(items)}:".encode("ascii"))
        for key, item in items:
            _encode(key, out)
            _encode(item, out)
        out.append(b";")
        return
    if isinstance(value, Sequence):
        out.append(f"l:{len(value)}:".encode("ascii"))
        for item in value:
            _encode(item, out)
        out.append(b";")
        return
    raise CanonicalisationError(
        f"{type(value).__name__} cannot be canonicalised; evidence records "
        f"must be built from None, bool, int, str, list and string-keyed dict")


#: The subset of an event worth putting in an indexed column. Kept beside the
#: encoder rather than in a storage layer so that "what is queryable" is a
#: property of the contract, not of whichever backend happens to be in use.
INDEXED_COLUMNS: tuple[str, ...] = (
    "run_id", "sequence", "event_type", "subject_id", "rule_id", "node_id",
    "recorded_at", "event_hash", "previous_hash",
)


def indexed_columns(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The queryable subset of an event payload."""
    return {name: payload[name] for name in INDEXED_COLUMNS
            if payload.get(name) is not None}
def digest(value: Any) -> str:
    """The SHA-256 of a canonical encoding, as lowercase hex."""
    hasher = hashlib.sha256()
    hasher.update(_EVIDENCE_DOMAIN)
    hasher.update(canonical_bytes(value))
    return hasher.hexdigest()


def chain_digest(previous_hash: str, payload: Any) -> str:
    """One link of the tamper-evident chain.

    ``H(domain || previous || canonical(payload))``. Binding the previous hash
    in is what makes the sequence a *chain*: altering, deleting or reordering
    any event invalidates every link after it, which is the property the
    compliance claim actually rests on.
    """
    hasher = hashlib.sha256()
    hasher.update(_EVIDENCE_DOMAIN)
    hasher.update(b"chain\x00")
    hasher.update(previous_hash.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(canonical_bytes(payload))
    return hasher.hexdigest()