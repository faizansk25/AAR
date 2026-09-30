# Releasing AAR to PyPI

Distribution name: **`aar-analytics`**
Import name: **`aar`**  ·  Console script: **`aar`**

Work through these in order. Skipping step 5 is how people discover, after
uploading, that the wheel was missing a data file.

---

## Do not publish yet

**This guide is written down and deliberately not followed.** AAR is at
`0.0.1` and stays there until the system is finished, and "finished" is
decided by `report.md`, not by whether a build happens to pass.

A passing build is necessary and not sufficient. The current gaps that
block publication:

- **GPU execution is unverified from the repository.** `data/gpu/` has the
  CPU baseline and run 1; the run that is cited for cuDF end-to-end has no
  committed artifact.
- **The PostgreSQL, MySQL and MongoDB connectors have never spoken to a
  live server.** Dialects and pushdown are written and unit-tested; the
  integration is unproven.
- **There is no data profiler.** Cardinality is declared, not measured, so
  the cost model is choosing engines on estimates it has not checked.
- **The scheduler does no real resource management** — no thread counts, no
  memory ceilings, no worker concurrency.
- **Nine of nineteen specified layers are built.**

Publishing `aar-analytics` is a one-way door: once someone depends on
`0.0.1`, the API they built against is a promise. The gates below are for
*when that is intended*, so they are ready rather than improvised under
pressure. Do not start step 1 until the list above is empty.

---

## 0. The name is not `aar`, and that is deliberate

Three names, and only one of them has to be unique:

| Role | Name | Has to be free on PyPI? |
|---|---|---|
| Distribution (`pip install X`) | `aar-analytics` | **Yes** |
| Import (`import aar`) | `aar` | No — local namespace only |
| Console script (`aar --version`) | `aar` | No — local command only |

**`aar` is already taken on PyPI** by an unrelated project ("aar is a
collection of libraries for building AI applications", first uploaded
November 2024). Anyone who types `pip install aar` gets *that* project, not
this one. So every install instruction in this repository spells the
distribution name out in full. Do not "simplify" them.

`aar-analytics` was confirmed free (HTTP 404 on the PyPI JSON
API) at the time of writing. Re-check immediately before publishing:
names are permanent, and a squatter could take it in between.

A related consequence: searching PyPI for "AAR" returns dozens of unrelated
projects, because the acronym is short and heavily used. The project
description and keywords carry the full expansion for that reason.

---

## 0.1 The one fact that causes the most pain

**You cannot re-upload the same filename. Ever.**

If `twine upload` fails halfway — network drop, 2FA timeout, a typo in the
token — that exact `aar_analytics-0.1.0.tar.gz` is taken
forever, for everyone, not just you. The version is burned. You must bump
`version` in `pyproject.toml` to `0.1.1` and start again.

This is why the whole guide is rehearsal-first. TestPyPI is a separate
index, so a mistake there costs you nothing.

---

## 1. One-time account setup

1. Register at <https://pypi.org/account/register/>. Enable **two-factor
   authentication** — PyPI effectively requires it to upload.
2. Register a **separate** account at <https://test.pypi.org/account/register/>.
   Do not reuse the same credentials.
3. On each account go to **Account settings → API tokens → Add API token**.
   - Scope: **`pypi-…`** for production, **`test-pypi-…`** for testing.
     A token scoped to the wrong index silently fails.
   - Name it `aar-release`.
   - **Copy it once.** PyPI will not show it again.

Store it in a `.pypirc` rather than typing it into shell history:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc<...the rest of the production token...>

[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc<...the TestPyPI token...>
```

Save as `C:\Users\<you>\.pypirc` on Windows, or `~/.pypirc` elsewhere.
**Never commit this file.**

---

## 2. Confirm the name is actually free

Names on PyPI are permanent. Check *before* you build:

- <https://pypi.org/project/aar-analytics/> — a 404 means
  the name is available.
- Also check `aar`, in case you prefer the shorter name.

```powershell
.\.venv\Scripts\python.exe -c "import urllib.request as u; u.urlopen('https://pypi.org/pypi/aar-analytics/json')"
```

A `HTTPError: 404` means the name is free.

---

## 3. Bump the version

Edit `pyproject.toml`:

```toml
[project]
version = "0.1.0"   # -> 0.1.1, 0.2.0, ...
```

`src/aar/__init__.py` carries `__version__` too. Keep the two in step — a
mismatch means `pip show aar` and `aar --version` disagree, which becomes a
support question later.

```powershell
git add pyproject.toml src/aar/__init__.py
git commit -m "Release 0.1.0"
git tag -a v0.1.0 -m "AAR 0.1.0"
git push origin main --tags
```

---

## 4. Build

```powershell
# Start clean. A stale dist/ is how you upload yesterday's wheel.
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force src\aar_analytics.egg-info -ErrorAction SilentlyContinue

.\.venv\Scripts\python.exe -m pip install --upgrade build twine
.\.venv\Scripts\python.exe -m build
```

You should get exactly two files in `dist\`:

```
aar_analytics-0.1.0-py3-none-any.whl
aar_analytics-0.1.0.tar.gz
```

---

## 5. Inspect what you are about to publish

Do not skip this. It is the only chance to notice a missing file.

```powershell
.\.venv\Scripts\python.exe -m twine check --strict dist\*
```
---

## 6. Install-test the wheel in a clean environment

Your working venv has AAR installed in editable mode, so it hides packaging
bugs. A fresh venv does not.

```powershell
# somewhere OUTSIDE the source tree, so aar/ is not found by accident
cd $env:TEMP
Remove-Item -Recurse -Force aar-verify -ErrorAction SilentlyContinue
python -m venv aar-verify
.\aar-verify\Scripts\python.exe -m pip install --upgrade pip
.\aar-verify\Scripts\python.exe -m pip install d:\AAR\dist\aar_analytics-0.1.0-py3-none-any.whl

# The console script must exist and run.
.\aar-verify\Scripts\aar.exe --version

# The library must import with ZERO dependencies - that is the design.
.\aar-verify\Scripts\python.exe -c "import aar; print(aar.__version__)"

# The Workbench must find its static files.
.\aar-verify\Scripts\python.exe -c "from aar.workbench.server import STATIC; import os; print(sorted(os.listdir(STATIC)))"
```

If the last command shows an empty list or `FileNotFoundError`, the
`package-data` entry in `pyproject.toml` is wrong. Fix it, go back to
step 4.

Optionally test the extras too:

```powershell
.\aar-verify\Scripts\python.exe -m pip install "d:\AAR\dist\aar_analytics-0.1.0-py3-none-any.whl[duckdb,polars]"
```

---

## 7. Upload to TestPyPI

```powershell
cd d:\AAR
.\.venv\Scripts\python.exe -m twine upload --repository testpypi dist\*
```

Username `__token__`, password the **TestPyPI** token from step 1.

PyPI caches the rendered page for a few minutes. If it looks wrong, that
is caching, not a mistake.

## 8. Install-test from TestPyPI

```powershell
cd $env:TEMP
Remove-Item -Recurse -Force aar-testpypi -ErrorAction SilentlyContinue
python -m venv aar-testpypi
.\aar-testpypi\Scripts\python.exe -m pip install --upgrade pip
.\aar-testpypi\Scripts\python.exe -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aar-analytics
.\aar-testpypi\Scripts\aar.exe --version
```

`--extra-index-url https://pypi.org/simple/` lets AAR's optional
dependencies come from real PyPI while AAR itself comes from TestPyPI.

Do this on a machine *without* the source tree, or the local `aar/` package
shadows the installed one and you have tested nothing.

---

## 9. Upload to real PyPI

Only once step 8 has passed.

```powershell
cd d:\AAR
.\.venv\Scripts\python.exe -m twine upload dist\*
```

Username `__token__`, password the **production** token.

The upload takes 30–60 seconds to appear. Verify:

- <https://pypi.org/project/aar-analytics/>
- From a clean venv: `pip install aar-analytics`, then
  `aar --version`.

---

## 10. Announcing — and what to say about status

AAR is **alpha**. Say so in the release notes, and separate what is
verified from what is not.

**Genuinely true, safe to claim:**

- No *required* dependencies — the wheel installs and imports on a bare
  interpreter. **Executing a pipeline does need an engine, and pyarrow is
  the minimum.** That distinction is stated everywhere the claim appears,
  because "zero dependencies" unqualified is the sentence that makes an
  air-gapped deployment fail at 2am. `tools/verify_release.py` verifies
  both halves: a bare install refuses to execute *cleanly and with a named
  reason*, and a bare-plus-pyarrow install runs for real.
- Cross-engine agreement on identical data, checked against independent
  Python ground truth.
- Privacy classification and policy enforcement are fail-closed, covered
  by adversarial tests.
- The Workbench runs with no CDN and no frontend build chain.

**Claim carefully:**

- **GPU.** There is no cudf engine implementation. Detection, the cost
  model and degradation are written and tested; cudf execution is not
  implemented. Do not imply otherwise.
- **PostgreSQL / MySQL / MongoDB.** Generation is tested; there are no
  live-server integration tests.
- **Distributed engines** (Ray, Dask, Spark) are declared, not implemented.

---

## 11. After publishing

1. Add a GitHub release with the changelog (the tag is already pushed).
2. Bump `main` to the next development version so the released tag stays
   clean.
3. Record the published version and its SHA-256 in `report.md`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `400 Invalid or non-existent authentication information` | wrong token, or token scoped to the other index | check scope: `pypi-…` vs `test-pypi-…` |
| `403 Forbidden` | 2FA not enabled | enable it, then mint a new token |
| `File already exists` | that filename is already uploaded | **bump the version.** No workaround. |
| `twine check` passes but the wheel lacks a file | missing `package-data` | add it, rebuild, re-inspect step 5 |
| `ModuleNotFoundError: aar` after install | the wheel has no `aar` package | check `packages.find.where` |
| Workbench 404s on CSS/JS | static files not packaged | `[tool.setuptools.package-data]` |
| `aar` command not found | console script not installed | `[project.scripts]`, then reinstall |


Then confirm the Workbench assets are actually inside the wheel — this is
the exact bug caught during development, and it is invisible in
`pip install` output:

```powershell
.\.venv\Scripts\python.exe -c "import zipfile;print('\n'.join(n for n in zipfile.ZipFile('dist/aar_analytics-0.1.0-py3-none-any.whl').namelist() if 'static' in n or 'LICENSE' in n))"
```

You must see:

```
aar/workbench/static/app.css
aar/workbench/static/app.js
aar/workbench/static/index.html
aar_analytics-0.1.0.dist-info/licenses/LICENSE
```
