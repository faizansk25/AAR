"""The Analyst Workbench: a local, dependency-free UI over the real system.

Deliberately built on the standard library. The specification requires the
core to work air-gapped, and a UI that needed a package install would be a
UI that cannot start on the machines that matter most.

Everything served here is *real*: the engine catalogue comes from a live
probe, the plan comes from the real planner, and running a pipeline runs
it. There is no mock data, because a workbench that shows sample numbers
is worse than no workbench - it is a workbench that lies.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .. import __version__
from .i18n import LANGUAGES, STRINGS
from .security import (
    Refused, check_body_length,
    check_mutating_request, is_loopback, new_token, resolve_pipeline,
)

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

__all__ = ["WorkbenchServer", "serve", "api_state", "api_explain",
           "api_run", "api_rows", "api_i18n", "main"]


def api_state() -> dict:
    """What this machine is, and what AAR can actually do on it.

    Everything here is measured, not declared: ``probe()`` imports each
    engine's module or does not, and a failure carries the reason. The UI
    shows that reason verbatim, because an analyst who sees "duckdb is not
    installed" can act on it and one who sees a greyed-out box cannot.
    """
    from ..capability import default_registry
    from ..hardware import HardwareProfile

    profile = HardwareProfile()
    registry = default_registry()
    caps = registry.probe()
    return {
        "version": __version__,
        "hardware": profile.to_dict(),
        "engines": {eid: {"available": cap.available, "reason": cap.reason,
                          "version": cap.version,
                          "blocked_by": cap.blocked_by}
                    for eid, cap in caps.items()},
    }


def api_rows(token: str, offset: int = 0, limit: int = 100,
             sort: str = "", descending: bool = False) -> dict:
    """Page and sort a previous run's rows.

    Sorting happens in the engine, not in the browser. Shipping 3M rows to
    a tab for JavaScript to reorder is how a workbench becomes the slowest
    part of a fast pipeline, and it is the difference between a grid that
    works on real data and one that only demos well.
    """
    cached = _RESULTS.get(token)
    if cached is None:
        return {"ok": False, "error": "that result is no longer held; re-run"}
    order = cached
    if sort and order.schema.has(sort):
        # Through a real engine, not a direct Arrow call: the point is that
        # the grid agrees with whatever the engine would have done, and
        # `Table` deliberately has no sort of its own.
        from ..engines.factory import create_engine

        order = create_engine("arrow").sort(order, [(sort, not descending)])
    window = order.slice(int(offset), int(limit))
    columns = [{"name": f.name,
                "classification": sorted(f.classification)}
               for f in cached.schema.fields
               if f.name in window.column_names]
    return {"ok": True, "total": cached.num_rows, "offset": int(offset),
            "rows": window.arrow.to_pylist(), "columns": columns}


#: Recent results, so the grid can page without re-running the pipeline.
#: Bounded, because a workbench left open all day should not become a
#: memory leak with a friendly icon.
_RESULTS: dict = {}
_ORDER: list = []
_SEQ = itertools.count(1)
MAX_CACHED = 8


def _remember(table: Any) -> str:
    """Cache a result and return the token that addresses it.

    The token comes from a monotonic counter, *not* from the length of
    ``_ORDER``. Deriving it from the length reuses names once the cache is
    full - the ninth and tenth inserts both get "r9" - which silently
    evicts the wrong entries and leaves far fewer distinct results
    resident than MAX_CACHED promises.
    """
    token = f"r{next(_SEQ)}"
    _RESULTS[token] = table
    _ORDER.append(token)
    while len(_ORDER) > MAX_CACHED:
        _RESULTS.pop(_ORDER.pop(0), None)
    return token


def _failure(exc: BaseException) -> dict:
    """Turn any failure into a JSON body, including a CLI's SystemExit.

    `_plan_for` and `load_pipeline` exit by raising `SystemExit`, which is a
    `BaseException` - so a plain ``except Exception`` misses it and a typo
    in a filename takes down the request handler instead of returning a
    message. An API that dies on bad input is worse than one that
    complains.
    """
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def api_explain(path: str) -> dict:
    """Plan a pipeline and return the decision trace."""
    from ..application import PipelineService

    try:
        plan = PipelineService().explain(path)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        return _failure(exc)
    return {"ok": True, "plan": plan.render()}


def api_run(path: str, role: str | None = None,
            policy_path: str | None = None, root: str | None = None) -> dict:
    """Plan and execute a pipeline, returning what actually happened.

    This used to call ``Executor().run(node, policy=..., role=...)``. The
    executor has no ``run`` method - it has ``execute(plan)`` - and it takes
    a plan rather than a root node, so the endpoint raised ``AttributeError``
    for every valid pipeline. The tests passed because they only checked the
    missing-file path, which fails earlier and for a different reason.

    ``root`` confines both paths. The HTTP handler always passes it; the
    in-process callers do not, because a Python caller already has the
    operator's privileges and gains nothing from a boundary it could bypass.
    """
    from ..application import PipelineService

    if root:
        path = resolve_pipeline(path, root)
        if policy_path:
            policy_path = resolve_pipeline(policy_path, root)
    try:
        report = PipelineService().run(path, role=role,
                                       policy_path=policy_path)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        return _failure(exc)
    result = report.result
    table = getattr(result, "table", None)
    fields = table.schema.fields if table is not None else ()
    token = _remember(table) if table is not None else ""
    return {
        "ok": True,
        "token": token,
        "rows": table.num_rows if table is not None else 0,
        "columns": list(table.column_names) if table is not None else [],
        # ``DataType`` has no ``render()``; it renders through ``__str__``.
        # Calling ``.render()`` here raised AttributeError for every field, so
        # this line only ever worked for a result with no schema at all.
        "schema": [{"name": f.name, "type": str(f.type),
                    "classification": sorted(f.classification)}
                   for f in fields],
        "degradations": [getattr(d, "reason", str(d)) for d in
                         getattr(result, "degradations", [])],
    }



def api_i18n(language: str = "en") -> dict:
    """Interface strings for one language, falling back to English.

    A missing translation must degrade to a working English interface, not
    to a blank button with no label - an unlabelled control is unusable
    with a screen reader, so the fallback is part of accessibility, not
    just tidiness.
    """
    table = STRINGS.get(language) or STRINGS["en"]
    merged = dict(STRINGS["en"])
    merged.update(table)
    return {"language": language if language in STRINGS else "en",
            "strings": merged, "available": sorted(LANGUAGES)}


def _ctype(name: str) -> str:
    for suffix, kind in ((".html", "text/html"),
                         (".js", "application/javascript"),
                         (".css", "text/css"),
                         (".json", "application/json"),
                         (".svg", "image/svg+xml")):
        if name.endswith(suffix):
            return kind
    return "text/plain"



class _Handler(BaseHTTPRequestHandler):
    server_version = f"AAR-Workbench/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: Any,
                    close: bool = False) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if close:
            # The request body may not have been read, so the connection is no
            # longer framed correctly and must not be reused.
            self.send_header("Connection", "close")
            self.close_connection = True
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        # The workbench executes local files, so it fetches nothing remote.
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'")
        self.send_header("X-Content-Type-Options", "nosniff")

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            return self._serve_file("index.html")
        if route.startswith("/static/"):
            return self._serve_file(route[len("/static/"):])
        if route == "/api/state":
            return self._send_json(200, api_state())
        if route == "/api/i18n":
            language = ""
            if "lang=" in self.path:
                language = self.path.split("lang=")[-1].split("&")[0]
            return self._send_json(200, api_i18n(language))
        if route == "/api/health":
            return self._send_json(200, {"ok": True, "version": __version__})
        self._send_json(404, {"error": f"no such route: {route}"})

    def do_POST(self) -> None:  # noqa: N802
        """Every state-changing route passes the boundary first.

        The dispatch below is only reachable once the caller has proven it is
        this session. Before that, three things are refused outright: an
        oversized body (before it is read), a request that is not ours, and a
        pipeline path outside the configured root.
        """
        try:
            check_mutating_request(
                self.headers,
                self.server.session_token,
                self.server.allowed_hosts,
                self.server.allowed_origins,
            )
            length = int(self.headers.get("Content-Length") or 0)
            check_body_length(length)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send_json(400,
                                       {"error": "request body is not JSON"})
            route = self.path.split("?")[0]
            if route == "/api/explain":
                return self._send_json(
                    200, api_explain(self._pipeline(payload.get("path", "")),
                                     self.server.pipeline_root))
            if route == "/api/run":
                return self._send_json(
                    200, api_run(self._pipeline(payload.get("path", "")),
                                 payload.get("role"),
                                 self._policy(payload.get("policy")),
                                 self.server.pipeline_root))
            if route == "/api/rows":
                return self._send_json(200, api_rows(
                    payload.get("token", ""),
                    int(payload.get("offset") or 0),
                    int(payload.get("limit") or 100),
                    payload.get("sort", ""),
                    bool(payload.get("descending"))))
            self._send_json(404, {"error": f"no such route: {route}"})
        except Refused as refusal:
            # Reply *without* draining the body, and say so.
            #
            # Answering while a body is still in flight makes the OS abort the
            # connection (WinError 10053): the caller sees a socket error
            # instead of the 403 and its reason, so the refusal is reported as
            # a flaky connection rather than as a policy decision. Draining
            # first looks like the fix and is worse - a request that declares
            # more bytes than it sends leaves the server blocked in ``read``
            # until the client times out, which is the same symptom plus a
            # stuck thread. ``Connection: close`` is the correct signal: the
            # response is already complete, and the client is told the message
            # ends here rather than that more is expected.
            self._send_json(refusal.status, {"error": refusal.reason,
                                             "refused": True},
                            close=True)

    def _pipeline(self, path: str) -> str:
        """Resolve a caller-supplied pipeline path against the allowed root."""
        return resolve_pipeline(path, self.server.pipeline_root)

    def _policy(self, path: str) -> str:
        """A policy path is confined to the same root as the pipeline.

        A policy file is executable content too - it is parsed and compiled -
        so an unchecked policy path is the same hole as an unchecked pipeline
        path, and one hop from ``../``.
        """
        if not path:
            return ""
        return resolve_pipeline(path, self.server.pipeline_root)

    def _serve_file(self, name: str) -> None:
        # Resolve inside STATIC and refuse anything that escapes it. The
        # comparison is on resolved path *components*, not a string prefix:
        # "/static-evil" starts with "/static" and is not inside it.
        full = os.path.realpath(os.path.join(STATIC, name))
        root = os.path.realpath(STATIC)
        if os.path.commonpath([root, full]) != root or \
                not os.path.isfile(full):
            return self._send_json(404, {"error": f"no such file: {name}"})
        with open(full, "rb") as handle:
            body = handle.read()
        kind = _ctype(full)
        if name.endswith(".html"):
            # The UI is same-origin and needs the session token to make any
            # mutating request at all. It is served to the page rather than
            # taken from a URL, so it does not land in shell history, a
            # Referer header, or a browser bookmark. It is still scoped to this
            # one process and this one origin.
            body = self._inject_token(body)
        self.send_response(200)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # No CORS headers at all. That is deliberate and load-bearing: the
        # absence of `Access-Control-Allow-Origin` is what makes the custom
        # header requirement in `check_mutating_request` work.
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _inject_token(self, body: bytes) -> bytes:
        """Publish this session's token to the page it is served with."""
        meta = (f'<meta name="aar-session-token" '
                f'content="{self.server.session_token}">').encode()
        for anchor in (b"</head>", b"</body>"):
            if anchor in body:
                return body.replace(anchor, meta + anchor, 1)
        return meta + body


class WorkbenchServer:
    """A local workbench. Refuses to bind anywhere but loopback by default.

    Loopback is the default for a reason: the workbench executes pipeline
    files, and the specification's first principle is that outbound is denied
    by default. Binding to 0.0.0.0 is possible, but it should be a decision
    the user makes out loud - so it takes ``allow_remote=True``, and even then
    the boundary still applies. Exposure is a separate axis from
    authentication, and turning one on does not turn the other off.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 open_browser: bool = False, verbose: bool = False,
                 pipeline_root: str | None = None,
                 allow_remote: bool = False) -> None:
        if not is_loopback(host) and not allow_remote:
            # Refused at construction, not at bind: an exception after the
            # socket is open would leave a listener nobody intended.
            raise Refused(
                f"refusing to bind {host!r}: the workbench executes local "
                f"files, so it binds to loopback unless you pass "
                f"allow_remote=True to say that you mean it",
                status=400)
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.verbose = verbose
        #: A fresh token per process. Not persisted, not derived: restarting
        #: the workbench invalidates every session that was talking to it,
        #: which is the correct behaviour for a credential of this kind.
        self.session_token = new_token()
        #: Pipelines must resolve inside this directory. Defaults to the
        #: working directory, so an operator who starts the workbench beside
        #: their pipelines has to configure nothing.
        self.pipeline_root = os.path.realpath(
            pipeline_root or os.getcwd())

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        """Names this server answers for, for ``Host`` validation.

        ``localhost`` is included even when bound to ``127.0.0.1``, because
        a user who types ``localhost:8765`` and a user who types
        ``127.0.0.1:8765`` are the same person and neither should be told to
        fix their own address bar.
        """
        return tuple(dict.fromkeys((self.host, "localhost")))

    @property
    def allowed_origins(self) -> tuple[str, ...]:
        """Same-origin spellings permitted to send a mutating request."""
        hosts = self.allowed_hosts
        return tuple(f"http://{h}:{self.port}" for h in hosts)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._httpd.verbose = self.verbose
        # The handler reads its policy off the server object rather than a
        # module global: two servers in one process - as the tests do - must
        # not be able to see each other's token or root.
        self._httpd.session_token = self.session_token
        self._httpd.allowed_hosts = self.allowed_hosts
        self._httpd.allowed_origins = self.allowed_origins
        self._httpd.pipeline_root = self.pipeline_root
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        if self.open_browser:
            webbrowser.open(self.url)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def serve_forever(self) -> None:
        if self._httpd is None:
            self.start()
        print(f"AAR Workbench {__version__} on {self.url}")
        print("  loopback only; Ctrl-C to stop.")
        try:
            while self._thread.is_alive():
                self._thread.join(0.5)
        except KeyboardInterrupt:
            self.stop()


def serve(port: int = 8765, open_browser: bool = False) -> None:
    WorkbenchServer(port=port, open_browser=open_browser).serve_forever()


def main(argv: list | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="aar workbench")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (loopback by default)")
    parser.add_argument("--open", action="store_true",
                        help="open a browser window")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pipeline-root", default=None,
                        help="only serve pipelines under this directory "
                             "(defaults to the working directory)")
    parser.add_argument("--allow-remote", action="store_true",
                        help="permit a non-loopback bind; this exposes the "
                             "workbench to your network and is not "
                             "recommended")
    args = parser.parse_args(argv)
    try:
        server = WorkbenchServer(host=args.host, port=args.port,
                                 open_browser=args.open,
                                 verbose=args.verbose,
                                 pipeline_root=args.pipeline_root,
                                 allow_remote=args.allow_remote)
    except Refused as refusal:
        print(f"aar workbench: {refusal.reason}")
        return 2
    server.serve_forever()
    return 0
