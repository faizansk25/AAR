"""The Analyst Workbench: API, server, and the accessibility claims.

The UI's own docstrings make promises - translated strings never render
blank, right-to-left is set rather than guessed, every engine's reason is
visible, and nothing is served from a CDN. Each of those is a test here,
because an accessibility promise that is not checked is an aspiration.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request

import pytest

from aar.workbench import (  # noqa: E402
    STRINGS, WorkbenchServer, api_explain, api_i18n, api_run, api_state,
    text_direction,
)
from aar.workbench.security import (  # noqa: E402
    Refused, is_loopback, new_token, resolve_pipeline,
)
from aar.workbench.server import STATIC  # noqa: E402


class TestState:
    def test_reports_a_version_and_hardware(self):
        state = api_state()
        assert state["version"]
        assert "cpu" in state["hardware"] or "os" in state["hardware"]

    def test_every_engine_carries_a_reason(self):
        """A greyed-out box with no explanation is worse than useless.

        This is the specification's ninth principle: a degradation the
        analyst cannot see is one they will not trust.
        """
        for engine_id, cap in api_state()["engines"].items():
            assert "available" in cap, engine_id
            if not cap["available"]:
                assert cap["reason"], (
                    f"{engine_id} is unavailable with no reason given")

    def test_it_probes_rather_than_hardcodes(self):
        """`arrow` needs no third-party import, so it must be present."""
        assert api_state()["engines"]["arrow"]["available"] is True


class TestInternationalisation:
    def test_every_language_translates_the_essentials(self):
        essential = ("app.title", "action.run", "action.explain",
                     "label.engine", "panel.explain", "status.error")
        for code in STRINGS:
            for key in essential:
                assert STRINGS[code].get(key), f"{code} lacks {key}"

    def test_a_partial_translation_still_works(self):
        """Never render a blank control.

        An unlabelled button is unusable with a screen reader, so a missing
        string must degrade to working English, not to nothing.
        """
        strings = api_i18n("fr")["strings"]
        for key in STRINGS["en"]:
            assert key in strings, f"fr dropped {key}"
            assert strings[key], f"fr has a blank {key}"

    def test_an_unknown_language_is_english_not_an_error(self):
        data = api_i18n("kl")
        assert data["language"] == "en"
        assert data["strings"]["action.run"] == STRINGS["en"]["action.run"]

    def test_right_to_left_is_declared_not_guessed(self):
        assert text_direction("ar") == "rtl"
        assert text_direction("he") == "rtl"
        assert text_direction("en") == "ltr"
        assert text_direction("klingon") == "ltr"

    def test_the_language_list_is_not_thin(self):
        assert len(api_i18n()["available"]) >= 5


def _write_pipeline(directory, csv_text: str = "region,amount\nnorth,10\nsouth,20\n"):
    """A real, runnable pipeline file - not a mock of one."""
    src = directory / "orders.csv"
    src.write_text(csv_text, encoding="utf-8")
    pipeline = directory / "pipeline.py"
    pipeline.write_text(
        "from aar.sdk import csv, write_csv\n"
        "\n"
        "def build():\n"
        "    return write_csv(csv(r%r), r%r)\n"
        % (str(src), str(directory / "out.csv")),
        encoding="utf-8")
    return pipeline


class TestPipelineApi:
    def test_a_missing_pipeline_reports_an_error_not_a_crash(self):
        result = api_explain("definitely_not_a_real_file.py")
        assert result["ok"] is False
        assert result["error"]

    def test_run_executes_a_real_pipeline_and_returns_rows(self, tmp_path):
        """The success path, which no test covered until it was broken.

        ``api_run`` called ``Executor().run(...)`` - a method that does not
        exist - so every valid pipeline failed with ``AttributeError``. The
        existing tests all passed because they only asserted that a *missing*
        file reports an error, and that assertion was satisfied by a code
        path that never reached the broken line. A test suite can be green
        while the product's main function is dead.
        """
        pipeline = _write_pipeline(tmp_path)
        result = api_run(str(pipeline))
        assert result["ok"] is True, result.get("error")
        assert result["rows"] == 2
        assert result["columns"]
        assert result["token"]

    def test_the_result_grid_can_read_back_the_rows(self, tmp_path):
        """A run that returns a token the grid cannot read is still broken."""
        from aar.workbench import api_rows

        pipeline = _write_pipeline(tmp_path)
        run = api_run(str(pipeline))
        assert run["ok"] is True, run.get("error")
        page = api_rows(run["token"])
        assert page["ok"] is True
        assert page["total"] == 2
        assert {r["region"] for r in page["rows"]} == {"north", "south"}

    def test_explain_returns_a_real_plan_for_a_real_pipeline(self, tmp_path):
        """``api_explain`` had the same CLI-import coupling as ``api_run``."""
        pipeline = _write_pipeline(tmp_path)
        result = api_explain(str(pipeline))
        assert result["ok"] is True, result.get("error")
        assert "segment" in result["plan"].lower()

    def test_the_cli_and_the_workbench_agree_on_the_result(self, tmp_path):
        """Both front ends now share one service, so both must see one answer.

        This is the equivalence the duplicated orchestration made impossible
        to state: same pipeline, same rows, through two different entry
        points.
        """
        from aar.application import PipelineService

        pipeline = _write_pipeline(tmp_path)
        report = PipelineService().run(str(pipeline))
        via_api = api_run(str(pipeline))
        assert via_api["ok"] is True, via_api.get("error")
        assert report.result.table.num_rows == via_api["rows"]
        assert sorted(report.result.table.column_names) == \
            sorted(via_api["columns"])

    def test_run_reports_an_error_for_a_missing_pipeline(self):
        result = api_run("definitely_not_a_real_file.py")
        assert result["ok"] is False
        assert result["error"]

    def test_a_systemexit_does_not_escape_the_api(self):
        """`_plan_for` exits by raising SystemExit, a BaseException.

        A plain `except Exception` does not catch it, so a typo in the
        filename killed the request handler and the browser saw a dropped
        connection instead of a sentence saying what was wrong. Caught by
        a test because the symptom - a dead handler - is invisible in
        normal use.
        """
        for call in (lambda: api_explain("nope.py"),
                     lambda: api_run("nope.py")):
            try:
                result = call()
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (AssertionError, KeyboardInterrupt)):
                    raise
                pytest.fail(f"SystemExit escaped the API: {exc!r}")
            assert result["ok"] is False
            assert "nope.py" in result["error"] or result["error"]


@pytest.fixture(scope="module")
def server():
    instance = WorkbenchServer(port=8791)
    instance.start()
    yield instance
    instance.stop()


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read(), dict(response.headers)


def _auth_headers(server) -> dict:
    """Headers a legitimate mutating request carries.

    Kept next to the other HTTP helpers rather than inlined at each call site:
    a test that hand-rolls these is a test that can silently stop testing the
    thing it means to.
    """
    from aar.workbench.security import MUTATING_HEADER, TOKEN_HEADER

    return {"Content-Type": "application/json",
            MUTATING_HEADER: "1",
            TOKEN_HEADER: server.session_token}


def post(server, route: str, body: dict, headers: dict | None = None):
    """POST to a running workbench, returning ``(status, decoded)``.

    A 4xx from the boundary is a legitimate outcome here, so the HTTP error is
    caught rather than raised: a test asserting "this is refused" would
    otherwise fail with an exception instead of with its own assertion.
    """
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        server.url + route, data=data,
        headers=headers if headers is not None else _auth_headers(server))
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


class TestTheTrustBoundary:
    """The Workbench executes a caller-supplied file. These are its defences.

    Every test here is an *attack that must fail*. The shape matters: a suite
    of "the happy path still works" would pass just as happily against the
    old unauthenticated handler, because the happy path was never the problem.
    """

    @pytest.fixture(scope="class")
    def secured(self, tmp_path_factory):
        """A server whose root is known and which refuses outside it."""
        directory = tmp_path_factory.mktemp("secroot")
        _write_pipeline(directory)
        instance = WorkbenchServer(port=8793, pipeline_root=str(directory))
        instance.start()
        yield instance
        instance.stop()

    def test_a_run_with_no_token_is_refused(self, secured):
        """The original hole, reproduced: any socket, any pipeline."""
        status, body = post(secured, "api/run", {"path": "pipeline.py"},
                            headers={"Content-Type": "application/json"})
        assert status == 403, body
        assert body.get("refused") is True
        assert body["error"]

    def test_a_run_with_a_wrong_token_is_refused(self, secured):
        headers = _auth_headers(secured)
        headers["X-AAR-Token"] = "not-the-token"
        status, body = post(secured, "api/run", {"path": "pipeline.py"},
                            headers=headers)
        assert status == 403
        assert "token" in body["error"].lower()

    def test_a_token_is_required_on_every_mutating_route(self, secured):
        for route, payload in (("api/run", {"path": "pipeline.py"}),
                               ("api/explain", {"path": "pipeline.py"}),
                               ("api/rows", {"token": "x"})):
            status, _ = post(secured, route, payload,
                             headers={"Content-Type": "application/json"})
            assert status == 403, f"{route} accepted an unauthenticated POST"

    def test_a_custom_header_is_required(self, secured):
        """The preflight forcing function, asserted directly.

        `Content-Type: application/json` is CORS-simple, so a cross-origin
        form post carries it without a preflight. Without this header a
        cross-origin page can reach a mutating route in one shot.
        """
        headers = _auth_headers(secured)
        del headers["X-AAR-Workbench"]
        status, body = post(secured, "api/run", {"path": "pipeline.py"},
                            headers=headers)
        assert status == 403
        assert "X-AAR-Workbench" in body["error"]

    def test_a_foreign_host_is_refused(self, secured):
        """DNS rebinding: same socket, a Host the server never claimed."""
        headers = _auth_headers(secured)
        request = urllib.request.Request(
            secured.url + "api/run",
            data=json.dumps({"path": "pipeline.py"}).encode(),
            headers=headers)
        request.add_header("Host", "evil.example")
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("a foreign Host was accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 403
            assert b"evil.example" in error.read()

    def test_a_cross_origin_post_is_refused(self, secured):
        headers = _auth_headers(secured)
        headers["Origin"] = "http://evil.example"
        status, body = post(secured, "api/run", {"path": "pipeline.py"},
                            headers=headers)
        assert status == 403
        assert "cross-origin" in body["error"]

    def test_same_origin_is_accepted(self, secured):
        headers = _auth_headers(secured)
        headers["Origin"] = f"http://{secured.host}:{secured.port}"
        status, body = post(secured, "api/run", {"path": "pipeline.py"},
                            headers=headers)
        assert status == 200, body
        assert body["ok"] is True

    def test_no_cors_headers_are_sent(self, secured):
        """The absence is load-bearing, not an omission.

        The custom-header check only works because the server never answers a
        preflight. Adding a permissive `Access-Control-Allow-Origin` would
        silently disarm it.
        """
        _, _, headers = fetch(secured.url + "api/health")
        assert "Access-Control-Allow-Origin" not in headers

    def test_a_pipeline_outside_the_root_is_refused(self, secured, tmp_path):
        """`..` and absolute paths both resolve before the check."""
        outside = _write_pipeline(tmp_path)
        for path in (str(outside), "../../../etc/passwd"):
            status, body = post(secured, "api/run", {"path": path})
            assert status == 403, f"{path} was allowed"
            assert body["error"]


    def test_a_symlink_out_of_the_root_is_refused(self, secured, tmp_path):
        """A link inside the root pointing out must not be followed.

        This is the escape a prefix check on the *supplied* path cannot see:
        the string is inside the root, and the file it names is not.
        """
        outside = _write_pipeline(tmp_path)
        link = os.path.join(secured.pipeline_root, "innocent.py")
        try:
            os.symlink(str(outside), link)
        except (OSError, NotImplementedError, AttributeError):
            pytest.skip("symlinks unavailable to this user")
        try:
            status, _ = post(secured, "api/run", {"path": link})
            assert status == 403
        finally:
            os.unlink(link)

    def test_a_policy_path_is_confined_too(self, secured, tmp_path):
        """A policy is executable content, so `../` on it is the same hole."""
        policy = tmp_path / "policy.yaml"
        policy.write_text("version: 1\nrules: []\n", encoding="utf-8")
        status, body = post(secured, "api/run",
                            {"path": "pipeline.py", "policy": str(policy)})
        assert status == 403, body

    def test_an_oversized_body_is_refused(self, secured):
        """Refused on the declared length, before the body is read."""
        from aar.workbench.security import MAX_BODY_BYTES

        request = urllib.request.Request(
            secured.url + "api/run", data=b"{}",
            headers=_auth_headers(secured))
        request.add_header("Content-Length", str(MAX_BODY_BYTES + 1))
        try:
            urllib.request.urlopen(request, timeout=10)
            raise AssertionError("an oversized body was accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 413

    def test_a_legitimate_request_still_works(self, secured):
        """The boundary must not be a wall that breaks the product."""
        status, body = post(secured, "api/run", {"path": "pipeline.py"})
        assert status == 200, body
        assert body["ok"] is True
        assert body["rows"] == 2

    def test_the_ui_is_served_the_token(self, secured):
        """The page must be able to authenticate without a URL parameter."""
        _, body, _ = fetch(secured.url)
        assert secured.session_token.encode() in body
        # And not in the URL, where it would leak via Referer or history.
        assert "?" not in secured.url

    def test_a_remote_bind_is_refused_by_default(self):
        for host in ("0.0.0.0", "10.0.0.5", "192.168.1.4"):
            with pytest.raises(Refused):
                WorkbenchServer(host=host)

    def test_a_remote_bind_is_possible_but_says_so(self):
        server = WorkbenchServer(host="0.0.0.0", allow_remote=True)
        assert server.host == "0.0.0.0"

    def test_the_token_is_fresh_per_server(self):
        assert WorkbenchServer().session_token != WorkbenchServer().session_token


class TestTheBoundaryUnits:
    """The pieces, tested directly so a failure names the actual check."""

    @pytest.mark.parametrize("host,expected", [
        ("127.0.0.1", True), ("localhost", True), ("::1", True),
        ("127.0.0.2", True),
        ("0.0.0.0", False), ("10.0.0.5", False), ("", False),
        # The case a string prefix check would get wrong.
        ("127.0.0.1.evil.com", False),
    ])
    def test_loopback_is_resolved_not_prefixed(self, host, expected):
        assert is_loopback(host) is expected

    def test_each_server_generates_a_distinct_token(self):
        tokens = {new_token() for _ in range(50)}
        assert len(tokens) == 50
        assert all(len(t) >= 32 for t in tokens)

    def test_a_sibling_directory_sharing_a_prefix_is_refused(self, tmp_path):
        root = tmp_path / "pipelines"
        root.mkdir()
        evil = tmp_path / "pipelines-evil"
        evil.mkdir()
        target = evil / "x.py"
        target.write_text("# escape\n", encoding="utf-8")
        with pytest.raises(Refused):
            resolve_pipeline(str(target), str(root))

    def test_the_root_itself_is_not_a_pipeline(self, tmp_path):
        with pytest.raises(Refused):
            resolve_pipeline(str(tmp_path), str(tmp_path))

    def test_a_refusal_explains_itself(self, tmp_path):
        """A refusal with no reason is indistinguishable from a bug."""
        with pytest.raises(Refused) as caught:
            resolve_pipeline(str(tmp_path / "nope.py"), str(tmp_path / "sub"))
        assert "pipelines must live under" in caught.value.reason
        assert caught.value.status == 403


class TestServer:
    def test_it_serves_the_ui(self, server):
        status, body, _ = fetch(server.url)
        assert status == 200
        assert b"<html" in body.lower()

    def test_health_is_reported(self, server):
        status, body, _ = fetch(server.url + "api/health")
        assert status == 200
        assert json.loads(body)["ok"] is True

    def test_state_is_served_as_json(self, server):
        status, body, headers = fetch(server.url + "api/state")
        assert status == 200
        assert "application/json" in headers["Content-Type"]
        assert json.loads(body)["version"]

    def test_i18n_is_served_per_language(self, server):
        _status, body, _ = fetch(server.url + "api/i18n?lang=hi")
        assert json.loads(body)["language"] == "hi"

    def test_an_unknown_route_is_a_clean_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as caught:
            fetch(server.url + "api/nonsense")
        assert caught.value.code == 404

    def test_it_refuses_files_outside_the_static_root(self, server):
        """Path traversal is the obvious attack on a file-serving UI."""
        with pytest.raises(urllib.error.HTTPError) as caught:
            fetch(server.url + "static/../../../secrets.txt")
        assert caught.value.code == 404


class TestAccessibilityClaims:
    """The UI promises things; this checks the promises are in the markup."""

    @pytest.fixture(scope="class")
    def html(self):
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            return fh.read()

    def test_the_page_declares_a_language_and_direction(self, html):
        assert 'lang="en"' in html
        assert 'dir="ltr"' in html

    def test_every_interactive_control_is_labelled(self, html):
        """A control with no accessible name is unusable with a reader."""
        for match in re.finditer(r"<(input|select)\b[^>]*>", html):
            tag = match.group(0)
            assert "aria-label" in tag or 'id="' in tag, \
                f"unlabelled control: {tag}"

    def test_there_is_a_skip_link(self, html):
        assert 'class="skip"' in html

    def test_tabs_declare_their_role(self, html):
        assert 'role="tablist"' in html
        assert 'role="tab"' in html

    def test_live_regions_announce_results(self, html):
        assert "aria-live" in html

    def test_no_information_is_hover_only(self, html):
        """Reasons are rendered as text, not as tooltips.

        The specification calls the explainability panel a requirement for
        adoption; a tooltip nobody can screenshot is not evidence.
        """
        assert "title=" not in html


    def test_the_ui_loads_nothing_from_a_cdn(self, server):
        """The specification denies outbound by default.

        A UI that pulled a font or a framework from the internet would
        break on exactly the air-gapped machines the design is for.
        """
        _status, body, _ = fetch(server.url)
        html = body.decode("utf-8", "replace").lower()
        assert "http://" not in html
        assert "https://" not in html

    def test_it_binds_loopback_by_default(self):
        assert WorkbenchServer().host == "127.0.0.1"

    def test_the_static_assets_are_all_present(self):
        for name in ("index.html", "app.js", "app.css"):
            assert os.path.isfile(os.path.join(STATIC, name)), name

    @staticmethod
    def _client_source() -> str:
        """The Workbench client script, read once for the checks below."""
        with open(os.path.join(STATIC, "app.js"), encoding="utf-8") as fh:
            return fh.read()

    def test_the_client_script_is_syntactically_complete(self):
        """A truncated statement makes the whole UI inert.

        ``setPanel`` ended mid-expression at ``CACHE[name] ||`` with nothing
        after it. That is a *syntax* error, so the browser discarded the
        entire script: ``wire()`` never ran, no listener was ever bound, and
        the symptom was "nothing in the UI responds to a click" rather than
        anything a reader could act on.

        Every existing Workbench test passed while this was true, because
        they all exercise the HTTP layer and never the client's ability to
        parse.

        **These are heuristics, not a parse.** They catch the shapes seen so
        far - a truncated expression, a function left open, an orphaned
        fragment. They cannot catch ``const x = ;``, which no amount of
        counting would notice. :meth:`test_node_parses_the_client` is the real
        gate; this one stays because it runs without Node and its failure
        messages are far more specific than a line number.
        """
        source = self._client_source()
        lines = source.splitlines()

        import re

        unterminated = []
        for index, line in enumerate(lines):
            if not re.match(r"^(async )?function \w+\(", line):
                continue
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].startswith("}"):
                cursor += 1
            if cursor >= len(lines):
                unterminated.append(line.strip())
        assert not unterminated, (
            f"these functions never close: {unterminated}")

        dangling = [
            (index + 1, line.strip())
            for index, line in enumerate(lines)
            if re.search(r"(\|\||&&|\+)\s*$", line.rstrip())
            and index + 1 < len(lines)
            and (not lines[index + 1].strip()
                 or re.match(r"^(function|async|const|let|for|if)\b",
                             lines[index + 1].strip()))
        ]
        assert not dangling, (
            f"these lines end in an operator with nothing to continue them: "
            f"{dangling}")

        assert source.count("(") == source.count(")"), "unbalanced parentheses"

    def test_node_parses_the_client(self):
        """The acceptance test for "the browser can run this", literally.

        Every Python check in this file inspects *text*. None of them asks
        whether the text is valid JavaScript, which is the property that
        actually matters: a syntax error anywhere makes the browser discard
        the whole script, so the Workbench is not degraded, it is dead, and
        no HTTP or API assertion notices.

        Two rounds of structural guessing were not enough. Round 20 added a
        dangling-operator check that passed on a file the browser still could
        not parse. So the gate is now the actual parser.

        Skipped when Node is absent, since AAR's runtime dependencies are
        Python-only and a contributor without Node should still be able to
        run the suite. CI runs this on all six jobs, where Node is always
        present, so it is enforced rather than merely available.
        """
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed; structural checks still apply")

        result = subprocess.run(
            [node, "--check", os.path.join(STATIC, "app.js")],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            "app.js is not valid JavaScript, so the browser discards it and "
            f"the Workbench is inert:\n{result.stdout}{result.stderr}")

    def test_node_rejects_the_shape_that_motivated_this(self):
        """Prove the gate above can actually fail.

        A test that calls `node --check` on a file we know is fine only shows
        that Node runs. The question worth asking is whether it would notice
        a broken file, so this feeds Node a script containing precisely the
        errors the structural tests miss.
        """
        node = shutil.which("node")
        if not node:
            pytest.skip("node is not installed")

        for broken in ("const x = ;", "function f() {", "if (a) { } }"):
            result = subprocess.run(
                [node, "--check", "-"], input=broken,
                capture_output=True, text=True, timeout=60,
            )
            assert result.returncode != 0, (
                f"node --check accepted invalid JavaScript: {broken!r}")

    def test_the_client_brace_balance_ignores_template_literals(self):
        """A stray ``}`` after the last declaration breaks the whole file.

        Repairing the truncation above left the original expression's tail at
        the end of ``app.js`` - a template literal and a closing brace with
        nothing to attach to. The editor reported "Declaration or statement
        expected" and the script stopped parsing, so the UI was inert again.

        Braces cannot simply be counted: ``${...}`` inside a template literal
        contains them legitimately, which is why a naive count reads -1 on a
        perfectly good file. So template literals are blanked first, and what
        remains is real code whose braces must balance.
        """
        import re

        source = self._client_source()
        # Remove backtick-delimited literals, including `${...}` expressions.
        # ``re.DOTALL`` must be compiled into the pattern; passing ``flags``
        # to ``Pattern.sub`` is a TypeError.
        literal = re.compile(r"`(?:[^`\\]|\\.)*`", re.DOTALL)
        without_literals = literal.sub('""', source)
        opens = without_literals.count("{")
        closes = without_literals.count("}")
        assert opens == closes, (
            f"braces are unbalanced outside template literals ({opens} open, "
            f"{closes} close) - the file will not parse")

    def test_the_click_handlers_the_ui_promises_are_wired(self):
        """The wiring must exist, since a parse failure removes it all."""
        source = self._client_source()
        for needed in (
            'addEventListener("click"',
            "function wire(",
            "wire();",
            'addEventListener("keydown"',
            "data-sort",
        ):
            assert needed in source, f"the UI lost {needed!r}"

    def test_a_rows_request_for_nothing_is_a_clean_error(self, server):
        """A stale token must say so, not raise.

        The request now has to authenticate first: `/api/rows` is a mutating
        route, and a test that reached the handler without a token would be
        asserting the behaviour of a path the boundary no longer permits.
        """
        payload = json.dumps({"token": "r-nope", "offset": 0,
                              "limit": 10}).encode()
        request = urllib.request.Request(
            server.url + "api/rows", data=payload,
            headers=_auth_headers(server))
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read())
        assert body["ok"] is False
        assert body["error"]


class TestRowsEndpoint:
    """A grid that only shows a row count is not a grid.

    Paging and sorting are done in the engine, not in the browser: shipping
    3M rows to a tab for JavaScript to reorder makes the workbench the
    slowest part of a fast pipeline.
    """

    @pytest.fixture(scope="class")
    def table(self):
        import pyarrow as pa

        from aar.interchange import Table
        from aar.types import Field, INT64, Schema, UTF8

        schema = Schema((Field("region", UTF8), Field("n", INT64)))
        return Table(pa.table({"region": [f"r{i % 4}" for i in range(200)],
                               "n": list(range(200))}), schema)

    def test_an_unknown_token_is_refused(self):
        from aar.workbench import api_rows
        result = api_rows("nope")
        assert result["ok"] is False
        assert "re-run" in result["error"]

    def test_it_pages(self, table):
        from aar.workbench import api_rows
        from aar.workbench.server import _remember

        token = _remember(table)
        page = api_rows(token, offset=0, limit=10)
        assert page["ok"] and page["total"] == 200
        assert len(page["rows"]) == 10
        second = api_rows(token, offset=10, limit=10)
        assert second["rows"][0]["n"] != page["rows"][0]["n"]

    def test_it_sorts_in_the_engine(self, table):
        from aar.workbench import api_rows
        from aar.workbench.server import _remember

        token = _remember(table)
        ascending = api_rows(token, 0, 5, sort="n")["rows"]
        descending = api_rows(token, 0, 5, sort="n", descending=True)["rows"]
        assert [r["n"] for r in ascending] == [0, 1, 2, 3, 4]
        assert [r["n"] for r in descending] == [199, 198, 197, 196, 195]

    def test_the_cache_is_bounded(self, table):
        """A workbench left open all day must not become a memory leak."""
        from aar.workbench.server import (
            _ORDER, _RESULTS, MAX_CACHED, _remember,
        )

        tokens = [_remember(table) for _ in range(MAX_CACHED + 4)]
        assert len(_ORDER) <= MAX_CACHED
        assert len(_RESULTS) == len(_ORDER)
        # Distinct tokens, or eviction pops names it has already dropped and
        # the cache quietly holds fewer results than MAX_CACHED promises.
        assert len(set(tokens)) == len(tokens)
        assert len(_RESULTS) == MAX_CACHED

