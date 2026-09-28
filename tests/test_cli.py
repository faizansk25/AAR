"""CLI contract.

These tests exist because the console script once shipped pointing at a
module that did not exist: ``pip install`` succeeded, the ``aar`` command was
created, and running it raised ``ModuleNotFoundError``. A declared entry point
must be tested, not assumed.
"""

from __future__ import annotations

import json
import subprocess
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

