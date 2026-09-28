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
import urllib.error
import urllib.request

import pytest

from aar.workbench import (  # noqa: E402
    STRINGS, WorkbenchServer, api_explain, api_i18n, api_run, api_state,
    text_direction,
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


class TestPipelineApi:
    def test_a_missing_pipeline_reports_an_error_not_a_crash(self):
        result = api_explain("definitely_not_a_real_file.py")
        assert result["ok"] is False
        assert result["error"]

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

    def test_a_rows_request_for_nothing_is_a_clean_error(self, server):
        """A stale token must say so, not raise."""
        payload = json.dumps({"token": "r-nope", "offset": 0,
                              "limit": 10}).encode()
        request = urllib.request.Request(
            server.url + "api/rows", data=payload,
            headers={"Content-Type": "application/json"})
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

