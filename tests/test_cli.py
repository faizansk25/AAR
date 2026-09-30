"""CLI contract.

These tests exist because the console script once shipped pointing at a
module that did not exist: ``pip install`` succeeded, the ``aar`` command was
created, and running it raised ``ModuleNotFoundError``. A declared entry point
must be tested, not assumed.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import sys

import pytest

from aar import __version__
from aar.cli import build_parser, main


class TestEntryPointIsReal:
    def test_console_script_target_imports(self):
        """The function declared in [project.scripts] must exist."""
        import importlib

        module = importlib.import_module("aar.cli")
        assert callable(getattr(module, "main", None))

    def test_dunder_main_module_imports(self):
        import importlib

        assert importlib.import_module("aar.__main__") is not None

    def test_console_script_actually_runs(self):
        """Run it as a real subprocess.

        This is the only test that would have caught the original defect: the
        module existed and imported fine, but the *generated* entry point did
        not.
        """
        result = subprocess.run(
            [sys.executable, "-m", "aar", "version"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert __version__ in result.stdout

    def test_module_and_library_entry_points_agree(self):
        assert main(["version"]) == 0


class TestParser:
    def test_bare_invocation_prints_help(self, capsys):
        assert main([]) == 2
        assert "usage" in capsys.readouterr().out.lower()

    def test_no_docstring_points_at_a_section_that_does_not_exist(self):
        """A dangling cross-reference is a claim with nothing behind it.

        `cudf_engine.py` pointed readers at "report.md section 8.2" for its
        GPU evidence. No such section exists, and no test noticed, because
        prose is not code. The evidence actually lives in
        `data/gpu/README.md`; this is the check that would have said so.
        """
        import re

        root = Path(__file__).resolve().parents[1]
        report = (root / "report.md").read_text(encoding="utf-8")
        sections = set(re.findall(r"^#{2,3}\s+(\d+(?:\.\d+)?)[\s.]", report,
                                  re.MULTILINE))
        assert sections, "report.md has no numbered sections to check against"

        offenders = []
        for path in (root / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"report\.md`?\s+section\s+(\d+(?:\.\d+)?)",
                                 text, re.IGNORECASE):
                if m.group(1) not in sections:
                    offenders.append(
                        f"{path.relative_to(root)} -> report.md section "
                        f"{m.group(1)} (which does not exist)")
        assert not offenders, (
            "docstrings reference report.md sections that do not exist:\n  "
            + "\n  ".join(offenders))

    def test_the_readme_test_count_is_the_real_one(self, capsys):
        """A wrong number in the README is how an external reviewer goes wrong.

        The README said "540 tests" for a long time after the suite passed
        600+. Nobody noticed, and the number is the first thing a reader
        checks - it is the cheapest possible credibility test to run. So it
        is asserted rather than trusted.
        """
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
            encoding="utf-8")
        import os
        import re
        import subprocess
        import sys

        claimed = re.search(r"pytest -q\s+#\s*([\d,]+) tests", readme)
        assert claimed, "README no longer states a test count"
        total = int(claimed.group(1).replace(",", ""))

        # `sys.executable -m pytest`, not a bare "pytest": on Windows the
        # console script is not on PATH for a subprocess launched this way,
        # and a FileNotFoundError here would be a confusing failure rather
        # than the honest "the count is wrong".
        # Count the collected node ids rather than parsing a summary line.
        # pyproject's addopts already passes `-q`; adding another makes it
        # `-qq`, which suppresses "N tests collected" entirely - so a test
        # that parses the summary silently stops guarding anything. Node ids
        # are printed at every verbosity, so this cannot go quiet.
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only",
             "-p", "no:cacheprovider"],
            capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTEST_ADDOPTS": ""},
            cwd=str(Path(__file__).resolve().parents[1]))
        # Count collected node ids. pytest also prints class-level lines for
        # ``unittest.TestCase`` classes, which carry no ``::method`` suffix,
        # so the pattern is anchored on the ``tests/`` prefix rather than
        # requiring a method name - the stricter pattern silently omitted
        # 31 Workbench tests, which is the kind of undercount that makes a
        # correct README look wrong.
        node = re.compile(r"^tests[/\\].*::", re.MULTILINE)
        collected = len(node.findall(proc.stdout))
        if not collected:
            pytest.skip(f"could not count collected tests: {proc.stdout[-300:]}")
        assert total == collected, (
            f"README claims {total} tests; pytest collected {collected}. "
            "Fix the README - a stale count is the cheapest way to lose a "
            "reader's trust.\n"
            "If this is wrong after installing or removing a dependency, the "
            "count moved with the environment: `pip install -e \".[dev]\"` "
            "should give the same set everywhere, and `tests/test_cli.py::"
            "TestNoSilentLint` fails if `dev` is incomplete.")

    def test_the_readme_lists_every_subcommand(self):
        """A command that exists but is undocumented is a command nobody finds."""
        from aar.cli import _COMMANDS

        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
            encoding="utf-8")
        for name in _COMMANDS:
            assert f"aar {name}" in readme, (
                f"`aar {name}` exists but the README never mentions it")

    def test_explain_accepts_the_specifications_spelling(self, tmp_path):
        """`aar explain plan pipeline.py` is how the specification writes it.

        The CLI only ever accepted `aar explain pipeline.py`, so the command
        in the specification failed with a confusing "no such file: plan" -
        a documented command that does not run is worse than an undocumented
        one. Two `nargs="?"` positionals do not fix it either: argparse fills
        them left to right, so `plan` becomes the path and the real path
        lands in a slot that rejects it. Both spellings must work.
        """
        target = tmp_path / "hello.py"
        assert main(["examples", "--write", "hello", str(target)]) == 0
        assert main(["explain", "plan", str(target)]) == 0, (
            "aar explain plan <file> is the documented spelling")
        assert main(["explain", str(target)]) == 0, (
            "aar explain <file> is the spelling the printed hint uses")

    def test_explain_with_no_file_says_both_spellings(self, capsys):
        assert main(["explain"]) != 0
        err = capsys.readouterr().err
        assert "explain plan" in err, err
        assert "explain pipeline.py" in err, err

    def test_every_subcommand_has_a_real_handler(self):
        """No command may exist without something real behind it."""
        from aar.cli import _COMMANDS

        parser = build_parser()
        actions = [a for a in parser._actions
                   if isinstance(getattr(a, "choices", None), dict)]
        declared = set(actions[0].choices)
        assert declared == set(_COMMANDS)
        for name, fn in _COMMANDS.items():
            assert callable(fn), name

    def test_unknown_command_is_rejected(self):
        with pytest.raises(SystemExit):
            main(["not-a-command"])


class TestVersion:
    def test_version_reports_the_package_version(self, capsys):
        assert main(["version"]) == 0
        assert __version__ in capsys.readouterr().out

    def test_version_lists_installed_extras(self, capsys):
        main(["version"])
        out = capsys.readouterr().out
        assert "pyarrow" in out
        assert "python" in out.lower()


class TestDoctor:
    def test_doctor_runs_and_reports_hardware(self, capsys):
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        for label in ("os", "cpu", "memory", "gpu", "storage", "network"):
            assert label in out

    def test_doctor_json_is_valid_json(self, capsys):
        assert main(["doctor", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "fingerprint" in payload
        assert "gpu" in payload


class TestEngines:
    def test_engines_lists_every_engine(self, capsys):
        assert main(["engines"]) == 0
        out = capsys.readouterr().out
        assert "ENGINE CAPABILITY" in out
        for engine in ("duckdb", "polars_cpu", "polars_gpu", "python_worker"):
            assert engine in out

    def test_engines_json_explains_every_absence(self, capsys):
        assert main(["engines", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload
        for entry in payload.values():
            assert entry["reason"], "an engine reported no reason"


class TestCalibrate:
    def test_calibrate_writes_a_real_profile(self, capsys, tmp_path):
        out = tmp_path / "profile.json"
        rc = main(["calibrate", "--quick", "--no-gpu", "--no-disk",
                   "--out", str(out)])
        assert rc == 0
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["curves"], "calibration produced no curves"
        assert "groupby" in capsys.readouterr().out

    @pytest.mark.slow
    def test_calibrate_reports_failure_rather_than_claiming_success(
            self, capsys, tmp_path, monkeypatch):
        """A calibration that measures nothing must exit non-zero."""
        import aar.cli as cli_mod
        import aar.hardware.calibrate as cal

        def empty(*a, **kw):
            store = cal.CalibrationStore()
            store.fingerprint = "test"
            return store

        monkeypatch.setattr(cal, "calibrate", empty)
        rc = main(["calibrate", "--quick", "--out",
                   str(tmp_path / "p.json")])
        assert rc == 2
        assert "no curves" in capsys.readouterr().err
        assert cli_mod is not None


class TestNoSilentLint:
    """The codebase is lint-clean on the rules that find real defects.

    `ruff` and `vulture` are dev-only tools; this test shells out to whichever
    is present and skips cleanly when neither is, so a developer without them
    is not blocked from running the suite.

    The selected rules are the ones that found actual bugs here, not a style
    wish-list. F821 in particular caught `Schema` and `Field` being used in
    `arrow_engine._arrow_group_by` without ever being imported - a NameError
    that only fires on an Arrow build where the native kernel works, which is
    why the whole test suite was green while the fast path was broken.
    """

    RULES = "F401,F811,F821,F841,F632"

    def test_no_undefined_names_or_unused_code(self):
        import shutil
        import subprocess

        ruff = shutil.which("ruff") or str(
            Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "ruff.exe")
        if not Path(ruff).exists():
            pytest.skip("ruff not installed (pip install ruff)")

        root = Path(__file__).resolve().parents[1]
        # No --select: the rule set lives in pyproject.toml under
        # [tool.ruff.lint], with a written justification for each rule, so
        # this test and a developer running ruff by hand cannot disagree.
        proc = subprocess.run(
            [ruff, "check", "--no-cache", "--statistics", "src", "tests"],
            cwd=root, capture_output=True, text=True, timeout=300)
        # ruff exits 1 when it finds anything; 0 when clean. Anything else is
        # a tool failure (a malformed pyproject.toml, say) which must not be
        # mistaken for a clean run - that mistake is what let a real
        # TypeError ship to five failing tests once already.
        assert proc.returncode in (0, 1), (
            f"ruff could not run (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        assert proc.returncode == 0, (
            "lint findings - run `ruff check --fix src tests`:\n"
            f"{proc.stdout}")

    def test_no_unreachable_code(self):
        import shutil
        import subprocess

        vulture = shutil.which("vulture") or str(
            Path(__file__).resolve().parents[1] / ".venv" / "Scripts"
            / "vulture.exe")
        if not Path(vulture).exists():
            pytest.skip("vulture not installed (pip install vulture)")

        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [vulture, os.path.join("src", "aar"), "--min-confidence", "100"],
            cwd=root, capture_output=True, text=True, timeout=300)
        assert not proc.stdout.strip(), (
            "unreachable code or unused variables:\n" + proc.stdout)


class TestExamples:
    """The examples are code, not prose, so they are tested as code.

    An example that does not compile is a bug in the documentation, and it
    only surfaces when a new user copies it - the worst possible moment to
    find out. These tests are cheap: parsing a string is instant, and the
    shortest one is planned for real.
    """

    def test_every_example_parses(self):
        import ast

        from aar import examples

        for name in examples.names():
            try:
                ast.parse(examples.source(name), filename=f"<{name}>")
            except SyntaxError as exc:
                pytest.fail(f"example {name!r} does not parse: {exc}")

    def test_every_example_imports_only_names_the_sdk_exports(self):
        """A typo in an import is a crash on the user's very first run.

        Checked against the live ``__all__`` rather than a copy of it, so
        renaming an SDK function fails here instead of in a terminal.
        """
        import ast
        import importlib

        from aar import examples

        sdk = importlib.import_module("aar.sdk")
        for name in examples.names():
            tree = ast.parse(examples.source(name), filename=f"<{name}>")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "aar.sdk":
                    missing = [a.name for a in node.names
                               if a.name not in sdk.__all__]
                    assert not missing, (
                        f"example {name!r} imports {missing} from aar.sdk, "
                        f"which does not export them")
                assert not any(
                    isinstance(n, ast.ImportFrom)
                    and n.module is not None
                    and not n.module.startswith("aar")
                    for n in ast.walk(tree)
                ), f"example {name!r} imports a third-party module"

    def test_the_shortest_example_really_plans(self, tmp_path):
        """`aar explain` on a written-out example must succeed.

        This is the path a new user takes first, and planning touches the
        whole chain: load the file, build the IR, plan it, choose an
        engine. A break anywhere in that shows up here rather than in
        their face.
        """
        pytest.importorskip("pyarrow")
        target = tmp_path / "hello.py"
        assert main(["examples", "--write", "hello", str(target)]) == 0
        assert target.exists()
        assert main(["explain", str(target)]) == 0

    def test_writing_twice_refuses_to_clobber_your_own_work(self, tmp_path):
        target = tmp_path / "hello.py"
        assert main(["examples", "--write", "hello", str(target)]) == 0
        target.write_text("# my own work\n", encoding="utf-8")
        assert main(["examples", "--write", "hello", str(target)]) != 0
        assert target.read_text(encoding="utf-8") == "# my own work\n"

    def test_an_unknown_name_lists_the_real_ones(self, capsys):
        assert main(["examples", "--show", "nope"]) != 0
        err = capsys.readouterr().err
        assert "nope" in err
        assert "hello" in err

    def test_bare_listing_names_every_example(self, capsys):
        from aar import examples

        assert main(["examples"]) == 0
        out = capsys.readouterr().out
        for name in examples.names():
            assert name in out


class TestErrorHandling:
    def test_exceptions_become_a_message_not_a_traceback(self, capsys,
                                                         monkeypatch):
        import aar.hardware as hw

        def boom(self):
            raise RuntimeError("boom")

        monkeypatch.setattr(hw.HardwareProfile, "render", boom)
        rc = main(["doctor"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "RuntimeError" in err
        assert "boom" in err
        assert "Traceback" not in err

    def test_traceback_available_on_request(self, capsys, monkeypatch):
        import aar.hardware as hw

        def boom(self):
            raise RuntimeError("boom")

        monkeypatch.setenv("AAR_TRACEBACK", "1")
        monkeypatch.setattr(hw.HardwareProfile, "render", boom)
        with pytest.raises(RuntimeError):
            main(["doctor"])

    def test_keyboard_interrupt_is_handled(self, capsys, monkeypatch):
        import aar.hardware as hw

        def interrupt(self):
            raise KeyboardInterrupt

        monkeypatch.setattr(hw.HardwareProfile, "render", interrupt)
        assert main(["doctor"]) == 130


# ------------------------------------------------------------------- policy
class TestPolicyCommand:
    def _write(self, tmp_path, payload):
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_write_example_then_check_it(self, tmp_path, capsys):
        from aar.governance import EXAMPLE_POLICY

        path = str(tmp_path / "p.json")
        assert main(["policy", "check", "--write-example", path]) == 0
        assert os.path.exists(path)
        assert main(["policy", "check", path]) == 0
        assert "valid policy" in capsys.readouterr().out
        with open(path, encoding="utf-8") as fh:
            assert json.load(fh) == EXAMPLE_POLICY

    def test_show_prints_the_rules(self, tmp_path, capsys):
        path = self._write(tmp_path, {
            "name": "corp", "rls": {"emea": "region = EU"},
            "cls_drop": {"junior": ["ssn"]}})
        assert main(["policy", "show", path]) == 0
        out = capsys.readouterr().out
        assert "corp" in out
        assert "region = EU" in out
        assert "ssn" in out

    def test_a_typo_in_a_key_is_rejected(self, tmp_path, capsys):
        """A misspelled key must not silently disable the rule it was for.

        This is the failure that matters most for this command: a policy
        that loads, looks configured, and enforces nothing.
        """
        path = self._write(tmp_path, {"name": "x", "mask_threshhold": 2})
        assert main(["policy", "check", path]) == 2
        assert "unknown policy key" in capsys.readouterr().err

    def test_a_missing_policy_file_is_an_error(self, capsys):
        assert main(["policy", "show", "no/such/policy.json"]) == 2
        assert "no such policy file" in capsys.readouterr().err

    def test_policy_without_a_path_is_an_error(self, capsys):
        assert main(["policy", "show"]) == 2
        assert "needs a path" in capsys.readouterr().err

    def test_run_rejects_a_missing_policy_file(self, tmp_path, capsys):
        # `_load_policy` raises SystemExit because a missing policy must
        # not degrade to "no policy": that would turn a typo into a
        # silently unprotected run.
        with pytest.raises(SystemExit) as exc:
            main(["run", "pipelines/example_orders.py",
                  "--policy", str(tmp_path / "absent.json")])
        assert exc.value.code == 2
        assert "no such policy file" in capsys.readouterr().err

    def test_run_with_a_valid_policy_still_executes(self, tmp_path):
        """Enforcement must not break an ordinary, unclassified pipeline."""
        import pyarrow as _pa
        import pyarrow.parquet as pq

        from aar.governance import EXAMPLE_POLICY

        src = str(tmp_path / "plain.parquet")
        out = str(tmp_path / "o.csv")
        pq.write_table(_pa.table({
            "amount": _pa.array([1.0, 2.0], type=_pa.float64())}), src)
        pipeline = tmp_path / "p.py"
        pipeline.write_text(
            "from aar.sdk import parquet, write_csv\n\n\n"
            "def build():\n"
            f"    return write_csv(parquet(r'{src}'), r'{out}')\n",
            encoding="utf-8")

        path = self._write(tmp_path, EXAMPLE_POLICY)
        assert main(["run", str(pipeline), "--policy", path,
                     "--role", "analyst", "--as", "tester"]) == 0
        assert os.path.exists(out)


