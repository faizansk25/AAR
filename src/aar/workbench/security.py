"""The Workbench trust boundary.

The Workbench executes a caller-supplied file with the operator's privileges.
Everything else it serves is inert; that one route is a policy engine and an
arbitrary-code loader, so it is the only part of this package that has to be
right about who is asking.

It was not. ``do_POST`` dispatched on the request path alone, so any process
that could open a socket to the port could name any pipeline file on the disk
and have it executed under the user's account, with any policy or role the
request body asked for. That combination is the worst available: the caller
chooses both the code *and* the authority it runs with.

Five checks, each closing a hole the others do not:

1. **A session token.** Generated at startup from ``secrets``, never derived
   from anything guessable. The UI reads it from the page it was served.
2. **Host validation.** A browser can be pointed at ``http://127.0.0.1:8765``
   by any site the user visits; a *DNS-rebinding* attacker resolves their own
   domain to loopback first, so the request arrives with a ``Host`` that is not
   ours. Refusing a foreign ``Host`` is what makes the token check meaningful.
3. **Origin validation.** A cross-origin page cannot read our response, but it
   can *send* one. ``Origin`` is checked on every mutating request.
4. **A custom request header.** ``Content-Type: application/json`` is a
   CORS-*simple* header, so a form post carries it without a preflight. A
   custom header forces a preflight, and since this server sends no CORS
   headers the preflight fails and the request is never sent.
5. **A resolved pipeline root.** The path arrives as text, so it may be
   absolute, relative, or ``../../etc/passwd``. It is resolved *first*, then
   required to sit inside the root, then checked for symlink escape.

None substitutes for another: the token stops a local process, ``Host`` stops
DNS rebinding, ``Origin`` stops a cross-origin page, and the custom header
stops the simple-request shape that ``Origin`` alone does not cover. Binding is
restricted to loopback for the same reason - see :func:`is_loopback`.
"""

from __future__ import annotations

import ipaddress
import os
import secrets
from typing import Any

__all__ = [
    "MAX_BODY_BYTES", "MUTATING_HEADER", "TOKEN_HEADER", "Refused",
    "is_loopback", "new_token", "resolve_pipeline", "check_host",
    "check_origin", "check_mutating_request", "check_body_length",
]

#: A run request is a path and some options. Anything larger is not one, and
#: reading it into memory first would be the wrong way to find that out.
MAX_BODY_BYTES = 64 * 1024

#: Forces a CORS preflight, which this server will never satisfy. A request
#: carrying it from another origin therefore never leaves the browser.
MUTATING_HEADER = "X-AAR-Workbench"

#: The session token, as a header rather than a body field so that it is not
#: echoed back in any response or logged by anything reading the request.
TOKEN_HEADER = "X-AAR-Token"


class Refused(Exception):
    """A request the boundary declined, carrying the reason.

    Refusing *and saying why* is the whole design. A workbench that answers a
    blocked request with "forbidden" teaches an operator nothing; one that says
    "pipeline is outside the allowed root" tells them which of five things to
    fix. This is not an exception for control flow at the boundary - it is the
    boundary's return value, typed.
    """

    def __init__(self, reason: str, status: int = 403) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def new_token() -> str:
    """A session token from the OS CSPRNG.

    ``secrets`` rather than ``random`` or a hash of something: this value is
    the only thing standing between a local socket and arbitrary code
    execution, so it must not be guessable from any observable state.
    """
    return secrets.token_urlsafe(32)


def is_loopback(host: str) -> bool:
    """Whether ``host`` names the local machine and nothing else.

    Resolved rather than string-matched, because ``localhost``, ``127.0.0.1``,
    ``127.0.0.2`` and ``::1`` are all loopback while ``0.0.0.0`` is every
    interface and an empty string is ambiguous. A prefix check would accept
    ``127.0.0.1.evil.com``.
    """
    if not host:
        return False
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True
    # Strip an IPv6 zone/bracket form before parsing.
    candidate = host.strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def pipeline_root_help() -> str:
    """The explanation shown when a path is refused, naming the real root.

    Naming the resolved root is deliberate. "Access denied" from a tool the
    operator just started is indistinguishable from a bug; the resolved
    absolute path in the message is the one fact that makes it fixable.
    """
    root = os.path.realpath(os.getcwd())
    return (f"pipelines must live under {root}; start the workbench from the "
            f"directory that holds them, or pass --pipeline-root")


def resolve_pipeline(path: str, root: str) -> str:
    """Resolve ``path`` and require it to sit inside ``root``.

    Three separate escapes are closed here, and they are closed in order,
    because each defeats the previous one:

    * ``..`` traversal - handled by ``realpath``, not by string rejection.
    * a symlink pointing outside the root - ``realpath`` resolves the *final*
      target, so a link inside the root that points out is still caught. This
      is the one a prefix check on the input path misses.
    * a prefix that is a prefix but not a parent - a sibling directory named
      ``pipelines-evil`` shares a string prefix with ``pipelines``. The
      comparison below is on path *components*, so it cannot be fooled.

    The resolved real path is returned, not the input: every later use should
    be working from the checked value, and returning the unchecked one would
    invite a caller to re-derive it wrongly.
    """
    if not path or not path.strip():
        raise Refused("no pipeline path was supplied")
    real_root = os.path.realpath(root)
    # A relative path is resolved against the root, not the process CWD: the
    # operator names a root, so paths in the UI mean "relative to the root".
    candidate = path if os.path.isabs(path) else os.path.join(real_root, path)
    resolved = os.path.realpath(candidate)
    try:
        common = os.path.commonpath([real_root, resolved])
    except ValueError:
        # Different drives on Windows, or a mix of absolute and relative.
        raise Refused(pipeline_root_help()) from None
    if common != real_root:
        raise Refused(pipeline_root_help())
    if resolved == real_root:
        raise Refused("that is the pipeline root itself, not a pipeline: "
                      f"{pipeline_root_help()}")
    return resolved


def _split_host_port(value: str) -> tuple[str, str]:
    """Split a ``Host`` header into ``(host, port)``, tolerating IPv6."""
    value = value.strip()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return value, ""
        host = value[1:end]
        rest = value[end + 1:]
        return host, rest[1:] if rest.startswith(":") else ""
    if value.count(":") == 1:
        host, _, port = value.partition(":")
        return host, port
    # A bare IPv6 address, or a host with no port.
    return value, ""


def check_host(host_header: str, allowed_hosts: tuple[str, ...]) -> None:
    """Reject a ``Host`` that is not one this server was told to answer for.

    This is the DNS-rebinding defence. The attacker's page lives at some
    domain they control; the browser is induced to send the request to
    loopback while ``Host`` still names their domain. Every other check then
    passes, and the session token - which their origin cannot read - is the
    only thing left. Refusing the ``Host`` closes it before the token is
    consulted.
    """
    host, _port = _split_host_port(host_header or "")
    if not host:
        raise Refused("a Host header is required", status=400)
    if host.lower() in {h.lower() for h in allowed_hosts}:
        return
    # Ports are not compared: rebinding controls the name, not the port, and
    # the name is what identifies the origin.
    raise Refused(f"refusing a request addressed to {host!r}; this workbench "
                  f"answers only for {', '.join(sorted(allowed_hosts))}")


def check_origin(origin: str, allowed_origins: tuple[str, ...]) -> None:
    """Validate ``Origin`` on a mutating request.

    A cross-origin page cannot *read* a response, but it can cause one. This is
    what stops a page the analyst happens to be visiting from running a
    pipeline as them.
    """
    if not origin:
        # No Origin at all: a same-origin fetch in some browsers, or a
        # non-browser client such as curl. Neither is a CSRF vector, and
        # requiring it would break the documented command-line use. The custom
        # header requirement covers the browser case this would otherwise miss.
        return
    if origin not in allowed_origins:
        raise Refused(f"refusing a cross-origin request from {origin!r}")


def check_body_length(length: int) -> None:
    """Refuse an oversized or malformed body before reading it."""
    if length < 0:
        raise Refused("malformed Content-Length", status=400)
    if length > MAX_BODY_BYTES:
        raise Refused(f"request body exceeds {MAX_BODY_BYTES} bytes",
                      status=413)


def check_mutating_request(headers: Any, token: str,
                           allowed_hosts: tuple[str, ...],
                           allowed_origins: tuple[str, ...]) -> None:
    """Run every check that a state-changing request must pass.

    Order matters and is deliberate: the cheap, spoofable checks come first so
    that an obviously foreign request is rejected before any secret is
    compared, and the token - the check with real work behind it - comes last,
    once the request has already proved it is aimed at us.
    """
    check_host(headers.get("Host", ""), allowed_hosts)
    check_origin(headers.get("Origin", ""), allowed_origins)

    if MUTATING_HEADER not in headers:
        # The absence of a custom header means this was a simple cross-origin
        # request: a form post, or a fetch whose preflight was never answered.
        # Both are attacks; neither is the UI, which always sends it.
        raise Refused(f"mutating requests must carry the {MUTATING_HEADER} "
                      f"header")
    supplied = headers.get(TOKEN_HEADER, "") or ""
    if not secrets.compare_digest(supplied, token or ""):
        raise Refused("missing or invalid session token", status=403)