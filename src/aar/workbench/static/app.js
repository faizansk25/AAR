/* AAR Workbench client.

   No framework, no build step, no CDN. The specification's first principle
   is that outbound is denied by default, and a UI that pulled assets from
   elsewhere would break that on exactly the machines that need it most. */

"use strict";

let STRINGS = {};

const $ = (id) => document.getElementById(id);
/* Never render a blank label. A missing translation falls back to English,
   and if even that is missing the key itself is shown so a developer can
   find it. An unlabelled control is unusable with a screen reader, so this
   is an accessibility requirement, not a nicety. */
const t = (key) => STRINGS[key] || STRINGS["en"]?.[key] || key;

function applyStrings() {
  document.title = t("app.title");
  for (const el of document.querySelectorAll("[data-i18n]")) {
    el.textContent = t(el.dataset.i18n);
  }
}

function applyTheme() {
  const choice = $("theme").value;
  const dark = choice === "dark" ||
    (choice === "system" &&
     matchMedia("(prefers-color-scheme: dark)").matches);
  const root = document.documentElement;
  root.dataset.theme = dark ? "dark" : "light";
  root.dataset.density = $("density").value;
  localStorage.setItem("aar.theme", choice);
  localStorage.setItem("aar.density", $("density").value);
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} -> ${response.status}`);
  return response.json();
}

async function postJSON(url, body) {
  // The workbench executes a caller-supplied file, so a mutating request must
  // carry the session token and a custom header. The custom header is what
  // makes this a preflighted request: the server sends no CORS headers, so a
  // cross-origin page cannot get one of these past the browser at all.
  const token =
    document.querySelector('meta[name="aar-session-token"]')?.content || "";
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-AAR-Workbench": "1",
      "X-AAR-Token": token,
    },
    body: JSON.stringify(body),
  });
  return response.json();
}

function esc(value) {
  return String(value).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function renderEngines(state) {
  const list = $("engine-list");
  list.textContent = "";
  const entries = Object.entries(state.engines || {});
  $("engine-empty").hidden = entries.length > 0;
  for (const [id, cap] of entries) {
    const li = document.createElement("li");
    li.className = "engine";
    const dot = document.createElement("span");
    dot.className = "dot" + (cap.available ? "" : " off");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = id;
    const reason = document.createElement("span");
    reason.className = "reason";
    /* Rendered as text, never hidden behind a tooltip: an analyst who can
       see "duckdb is not installed" can act on it. */
    reason.textContent = cap.available
      ? (cap.version || t("label.available"))
      : (cap.reason || t("label.unavailable"));
    li.append(dot, name, reason);
    list.append(li);
  }
  renderMeters(state.hardware || {});
}

function renderMeters(hw) {
  const box = $("meters");
  box.textContent = "";
  const cpu = hw.cpu || {};
  const mem = hw.memory || {};
  const rows = [];
  if (cpu.cores) rows.push(["CPU", `${cpu.cores} cores`]);
  if (mem.total_gb) rows.push(["Mem", `${mem.total_gb} GB`]);
  if (hw.gpu && hw.gpu.present) {
    rows.push(["GPU", hw.gpu.vram_gb ? `${hw.gpu.vram_gb} GB` : "present"]);
  }
  for (const [label, value] of rows) {
    const wrap = document.createElement("div");
    wrap.className = "meter";
    const name = document.createElement("span");
    name.textContent = label;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = "100%";
    bar.append(fill);
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = value;
    wrap.append(name, bar, chip);
    box.append(wrap);
  }
}

function tableHTML(headers, rows) {
  if (!rows.length) return "";
  const head = headers.map((h) => `<th scope="col">${esc(h)}</th>`).join("");
  const body = rows.map((cells) => "<tr>" + cells.map((c) => {
    /* A confidential column is visibly marked. The analyst should never
       have to open a panel to discover their output is being masked - or
       to discover that it is not. */
    const text = String(c);
    const hot = /confidential/i.test(text);
    return `<td>${hot ? `<span class="chip hot">${esc(text)}</span>`
                      : esc(text)}</td>`;
  }).join("") + "</tr>").join("");
  return `<table><thead><tr>${head}</tr></thead>` +
         `<tbody>${body}</tbody></table>`;
}

const CACHE = {};
/* Grid state. Paging and sorting are *requests to the engine*, not browser
   reordering, so a 3M-row result pages as cheaply as a 10-row one. */
const GRID = { token: "", offset: 0, limit: 100, sort: "", desc: false,
               total: 0 };

const RUNS = [];

function noteRun(ok, text) {
  RUNS.unshift({ ok, text, when: new Date().toLocaleTimeString() });
  while (RUNS.length > 12) RUNS.pop();
  const list = $("history");
  list.textContent = "";
  for (const r of RUNS) {
    const li = document.createElement("li");
    const mark = document.createElement("span");
    mark.className = r.ok ? "ok" : "err";
    mark.textContent = r.ok ? "ok" : "error";
    const when = document.createElement("span");
    when.className = "chip";
    when.textContent = r.when;
    const what = document.createElement("span");
    what.textContent = r.text;
    li.append(mark, when, what);
    list.append(li);
  }
}

async function loadRows() {
  if (!GRID.token) return;
  const res = await postJSON("/api/rows", {
    token: GRID.token, offset: GRID.offset, limit: GRID.limit,
    sort: GRID.sort, descending: GRID.desc,
  });
  if (!res.ok) {
    CACHE.preview = `<p class="empty">${esc(res.error || t("status.error"))}</p>`;
    setPanel("preview");
    return;
  }
  GRID.total = res.total;
  $("grid-bar").hidden = false;
  $("pager").textContent =
    `${res.total.toLocaleString()} rows · ${GRID.offset + 1}–` +
    `${Math.min(GRID.offset + GRID.limit, res.total)}`;
  $("prev").disabled = GRID.offset <= 0;
  $("next").disabled = GRID.offset + GRID.limit >= res.total;
  CACHE.preview = gridHTML(res);
  setPanel("preview");
}

function gridHTML(res) {
  if (!res.rows.length) {
    return `<p class="empty">${esc(t("empty.result"))}</p>`;
  }
  const hot = new Set();
  for (const c of res.columns) {
    if ((c.classification || []).some((k) => /confidential|restricted/i
        .test(k))) hot.add(c.name);
  }
  const head = res.columns.map((c) => {
    const arrow = GRID.sort === c.name ? (GRID.desc ? " ▾" : " ▴") : "";
    return `<th scope="col"><button type="button" data-sort="${esc(c.name)}"` +
      ` aria-label="${esc(c.name)}">${esc(c.name)}${arrow}</button></th>`;
  }).join("");
  const body = res.rows.map((row) => "<tr>" + res.columns.map((c) => {
    const value = row[c.name];
    const text = value === null || value === undefined ? "" : String(value);
    return `<td>${hot.has(c.name)
      ? `<span class="chip hot">${esc(text)}</span>` : esc(text)}</td>`;
  }).join("") + "</tr>").join("");
  return `<table><thead><tr>${head}</tr></thead>` +
         `<tbody>${body}</tbody></table>`;
}

function setPanel(name) {
  for (const b of $("bottom-tabs").children) {
    b.setAttribute("aria-current", b.dataset.panel === name ? "page" : "false");
  }
  const body = $("panel-body");
  if (name === "help") {
    body.innerHTML = `<p class="empty">${esc(t("help.shortcuts"))}</p>`;
    return;
  }
  /* A panel with nothing cached yet still needs to say so. This line was
     truncated mid-expression, which is a *syntax* error rather than a
     rendering one - the whole script failed to parse, so no handler was
     ever bound and nothing in the UI responded to a click. A placeholder
     is also what an analyst needs: "nothing cached" and "panel broken"
     must not look the same. */
  body.innerHTML = CACHE[name] ||
    `<p class="empty">${esc(t("empty.result"))}</p>`;
}

async function refresh() {
  try {
    renderEngines(await getJSON("/api/state"));
  } catch (err) {
    $("engine-list").textContent = `${t("status.error")}: ${err.message}`;
  }
}

async function doExplain() {
  const pre = $("plan");
  pre.textContent = t("status.running");
  pre.hidden = false;
  $("plan-empty").hidden = true;
  const res = await postJSON("/api/explain", { path: $("path").value.trim() });
  if (!res.ok) {
    pre.textContent = `${t("status.error")}: ${res.error}`;
    return;
  }
  pre.textContent = res.plan;
  /* Cached so the "Why" panel shows the same text that is on screen - one
     explanation, not two that can drift apart. */
  CACHE.explain = `<pre>${esc(res.plan)}</pre>`;
  setPanel("explain");
}

async function doRun() {
  const res = await postJSON("/api/run", {
    path: $("path").value.trim(),
    role: $("role").value.trim() || null,
  });
  if (!res.ok) {
    noteRun(false, res.error || "run failed");
    CACHE.preview = `<p class="empty">${esc(t("status.error"))}: ` +
                    `${esc(res.error)}</p>`;
    setPanel("preview");
    return;
  }
  noteRun(true, `${res.rows} rows, ${(res.columns || []).length} columns`);
  const cols = res.columns || [];
  const schema = res.schema || [];
  GRID.token = res.token || "";
  GRID.offset = 0;
  GRID.sort = "";
  GRID.desc = false;
  CACHE.schema = tableHTML(
    ["Column", "Type", "Classification"],
    schema.map((f) => [f.name, f.type,
                       (f.classification || []).join(", ") || t("tag.public")]));
  CACHE.lineage = tableHTML(
    ["Column", "Inherited from"],
    schema.map((f) => [f.name,
                       (f.classification || []).join(", ") || t("tag.public")]));
  CACHE.log = `<pre>${esc(JSON.stringify(
    { rows: res.rows, columns: cols,
      degradations: res.degradations || [] }, null, 2))}</pre>`;
  if (GRID.token) await loadRows();
  else {
    $("grid-bar").hidden = true;
    CACHE.preview = cols.length
      ? tableHTML(cols, [[`${res.rows} rows`]])
      : `<p class="empty">${esc(t("empty.result"))}</p>`;
    setPanel("preview");
  }
}

function wire() {
  $("btn-explain").addEventListener("click", doExplain);
  $("btn-run").addEventListener("click", doRun);
  $("theme").addEventListener("change", applyTheme);
  $("density").addEventListener("change", applyTheme);
  $("lang").addEventListener("change", applyLanguage);
  matchMedia("(prefers-color-scheme: dark)").addEventListener(
    "change", applyTheme);
  for (const b of $("bottom-tabs").children) {
    b.addEventListener("click", () => setPanel(b.dataset.panel));
  }
  $("prev").addEventListener("click", () => {
    GRID.offset = Math.max(0, GRID.offset - GRID.limit);
    loadRows();
  });
  $("next").addEventListener("click", () => {
    GRID.offset += GRID.limit;
    loadRows();
  });
  $("pagesize").addEventListener("change", (ev) => {
    GRID.limit = Number(ev.target.value) || 100;
    GRID.offset = 0;
    loadRows();
  });
  /* Sort by header click, delegated: the grid is re-rendered on every page
     so a per-header listener would be lost each time. */
  $("panel-body").addEventListener("click", (ev) => {
    const button = ev.target.closest("[data-sort]");
    if (!button) return;
    const name = button.dataset.sort;
    GRID.desc = GRID.sort === name ? !GRID.desc : false;
    GRID.sort = name;
    GRID.offset = 0;
    loadRows();
  });
  document.addEventListener("keydown", (ev) => {
    if (!(ev.ctrlKey || ev.metaKey)) {
      if (ev.key === "?") setPanel("help");
      return;
    }
    const key = ev.key.toLowerCase();
    if (key === "e") { ev.preventDefault(); doExplain(); }
    else if (key === "r") { ev.preventDefault(); doRun(); }
    else if (key >= "1" && key <= "5") {
      ev.preventDefault();
      setPanel($("bottom-tabs").children[Number(key) - 1].dataset.panel);
    }
  });
}

const RTL = ["ar", "he", "fa", "ur"];

function setLanguage(code, data) {
  STRINGS = data.strings;
  const root = document.documentElement;
  root.lang = data.language;
  /* Direction is set explicitly rather than inferred by CSS: a mirrored
     layout nobody checked is worse than a left-to-right one. */
  root.dir = RTL.includes(data.language) ? "rtl" : "ltr";
  applyStrings();
  localStorage.setItem("aar.lang", code);
}

async function applyLanguage() {
  const code = $("lang").value;
  setLanguage(code, await getJSON(
    "/api/i18n?lang=" + encodeURIComponent(code)));
}

async function boot() {
  const data = await getJSON("/api/i18n");
  const select = $("lang");
  for (const [code, name] of Object.entries(data.available)) {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = name;
    select.append(opt);
  }
  const stored = localStorage.getItem("aar.lang");
  select.value = (stored && data.available.includes(stored)) ? stored : "en";
  setLanguage(select.value, data);
  $("theme").value = localStorage.getItem("aar.theme") || "system";
  $("density").value = localStorage.getItem("aar.density") || "comfortable";
  applyTheme();
  wire();
  await refresh();
  setPanel("preview");
}

boot();
