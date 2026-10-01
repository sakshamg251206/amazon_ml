// Entity Resolver web client: plain ES module, no build step.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 1) => (v == null ? "–" : (100 * v).toFixed(d) + "%");
const f4 = (v) => (v == null ? "–" : v.toFixed(4));
const fmtInt = (v) => (v == null ? "–" : Number(v).toLocaleString());

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
  return body;
}

// ------------------------------------------------------------------ theme + tabs
const theme = {
  init() {
    let saved = null;
    try { saved = localStorage.getItem("theme"); } catch (_) {}
    if (saved) document.documentElement.dataset.theme = saved;
    $("#theme").addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme
        ? document.documentElement.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      const next = dark ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (_) {}
      if (state.report) renderModel(); // charts read CSS colours at draw time
    });
  },
};

const loaded = {};
function showTab(name) {
  $$("nav.tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$("main > section").forEach((s) => s.classList.toggle("hidden", s.id !== "tab-" + name));
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (!loaded[name]) { loaded[name] = true; ({ explore: loadExplore, model: loadModel })[name]?.(); }
}

const state = { meta: null, report: null, ex: { offset: 0, limit: 20, country: "", outcome: "", q: "" } };

// ------------------------------------------------------------------ tooltip
const tip = $("#tip");
function showTip(e, html) { tip.innerHTML = html; tip.classList.remove("hidden"); moveTip(e); }
function moveTip(e) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.classList.add("hidden"); }
function bindTips(root) {
  $$("[data-tip]", root).forEach((el) => {
    el.addEventListener("mouseenter", (e) => showTip(e, el.dataset.tip));
    el.addEventListener("mousemove", moveTip);
    el.addEventListener("mouseleave", hideTip);
  });
}

// ------------------------------------------------------------------ explanations
function contribBlock(ex) {
  if (!ex) return "";
  const max = Math.max(1, ...ex.groups.map((g) => Math.abs(g.value)));
  const bars = ex.groups.map((g) => {
    const w = (Math.abs(g.value) / max) * 50;
    const left = g.value >= 0 ? 50 : 50 - w;
    const color = g.value >= 0 ? "var(--pos)" : "var(--neg)";
    return `<span>${esc(g.label)}</span>
      <div class="bar" data-tip="<b>${esc(g.label)}</b><br>${g.value >= 0 ? "raises" : "lowers"} the match log-odds by ${Math.abs(g.value).toFixed(2)}"><span style="left:${left}%;width:${Math.max(w, 0.8)}%;background:${color}"></span></div>
      <span class="v">${g.value >= 0 ? "+" : "−"}${Math.abs(g.value).toFixed(2)}</span>`;
  }).join("");
  const reasons = ex.reasons.map((r) => `<li><span>${esc(r.label)} <span class="muted mono">${esc(r.value)}</span></span>
      <span class="${r.impact >= 0 ? "up" : "down"}">${r.impact >= 0 ? "▲ +" : "▼ −"}${Math.abs(r.impact).toFixed(2)}</span></li>`).join("");
  return `<details class="why"><summary>Why this score</summary>
    <div class="contrib">${bars}</div>
    <p class="muted small" style="margin:8px 0 0">Strongest individual signals (stage-1 SHAP, log-odds):</p>
    <ul class="reasons">${reasons}</ul></details>`;
}

function probCell(p, p1) {
  const v = p ?? p1 ?? 0;
  const label = p != null ? pct(p, 1) : p1 != null && p1 < 0.01 ? "< 1%" : pct(p1, 1);
  return `<div class="prob"><div class="num">${label}</div><div class="pbar"><span style="width:${(100 * v).toFixed(1)}%"></span></div>
    <div class="muted small">${p != null ? "match probability" : "pruned at stage 1"}</div></div>`;
}

// ------------------------------------------------------------------ MATCH
const ICON_OK = `<svg viewBox="0 0 24 24" fill="none" stroke="var(--good)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/></svg>`;
const ICON_NO = `<svg viewBox="0 0 24 24" fill="none" stroke="var(--muted)" stroke-width="2.2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M8 12h8"/></svg>`;

async function runMatch(ev) {
  ev?.preventDefault();
  const body = { name: $("#q-name").value, address: $("#q-addr").value, country: $("#q-country").value, top: 5 };
  if (!body.name.trim() && !body.address.trim()) return;
  const btn = $("#match-btn");
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Resolving`;
  try {
    const r = await api("/api/match", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    renderMatch(r);
    $("#match-time").textContent = `${r.elapsed_ms} ms`;
  } catch (e) {
    $("#match-out").innerHTML = `<div class="card pad"><span class="badge bad">Error</span> ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Resolve record";
  }
}

function renderMatch(r) {
  const d = r.decision || {};
  const top = r.candidates.find((c) => c.s1 === d.s1);
  const head = d.matched
    ? `<div class="decision yes">${ICON_OK}<div><div class="big">Matched to ${esc(top?.reference?.business_name)}</div>
        <div class="small">${esc(top?.reference?.business_address)} · <span class="mono">${esc(d.s1)}</span></div>
        <div class="small muted" style="margin-top:4px">${esc(d.reason)}. The reference already has ${top?.cluster_size ?? 0} matched record(s).</div></div></div>`
    : `<div class="decision no">${ICON_NO}<div><div class="big">No confident match</div>
        <div class="small muted">${esc(d.reason || r.error || "")}. Under F0.5 a false merge costs twice as much as a miss, so the resolver abstains.</div></div></div>`;
  const q = r.query || {};
  const norm = `<div class="card pad"><h3>How the record was read</h3><dl class="norm">
      <dt>core name</dt><dd>${esc(q.name_tok) || "–"}</dd><dt>address</dt><dd>${esc(q.addr) || "–"}</dd>
      <dt>house no.</dt><dd>${esc(q.hn) || "–"}</dd><dt>region</dt><dd>${esc(q.state || "not recognised")}</dd>
      <dt>searched</dt><dd>${esc(q.searched || "–")}</dd>${q.nonlatin ? "<dt>script</dt><dd>non-Latin, transliterated</dd>" : ""}</dl></div>`;
  const cands = r.candidates.map((c, i) => `<div class="card cand ${c.selected ? "selected" : ""}">
      <div class="cand-head"><div style="min-width:0">
        <div class="row" style="gap:8px"><span class="muted small">#${i + 1}</span><span class="cand-name">${esc(c.reference?.business_name)}</span>
        ${c.selected ? '<span class="badge good">selected</span>' : ""}</div>
        <div class="cand-addr">${esc(c.reference?.business_address)}</div>
        <div class="muted small mono">${esc(c.s1)} · cluster of ${c.cluster_size}</div></div>
        ${probCell(c.p, c.p1)}</div>${contribBlock(c.explanation)}</div>`).join("");
  $("#match-out").innerHTML = `<div class="stack">${head}${norm}
     <h3 style="margin:18px 0 0">Top candidates</h3>${cands || '<div class="card empty">Blocking retrieved no candidates.</div>'}</div>`;
  bindTips($("#match-out"));
  const first = $("#match-out details.why");
  if (first) first.open = true;
}

// ------------------------------------------------------------------ EXPLORE
const OUTCOME = {
  exact: ["good", "exact"], correct_singleton: ["good", "correct singleton"], false_merge: ["bad", "false merge"],
  has_false_match: ["bad", "wrong record added"], missed_match: ["warn", "missed record"],
};
const outcomeBadge = (o) => (o ? `<span class="badge ${OUTCOME[o][0]}">${OUTCOME[o][1]}</span>` : "");

async function loadExplore() {
  const seg = $("#ex-country");
  seg.innerHTML = `<button data-v="" aria-pressed="true">All</button>` + state.meta.countries.map((c) => `<button data-v="${esc(c)}">${esc(c)}</button>`).join("");
  for (const [sel, key] of [["#ex-country", "country"], ["#ex-outcome", "outcome"]]) {
    $(sel).addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("button", $(sel)).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      state.ex[key] = b.dataset.v; state.ex.offset = 0; fetchEntities();
    });
  }
  let t;
  $("#ex-q").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { state.ex.q = e.target.value; state.ex.offset = 0; fetchEntities(); }, 250); });
  $("#ex-prev").addEventListener("click", () => { state.ex.offset = Math.max(0, state.ex.offset - state.ex.limit); fetchEntities(); });
  $("#ex-next").addEventListener("click", () => { state.ex.offset += state.ex.limit; fetchEntities(); });
  fetchEntities();
}

async function fetchEntities() {
  const p = new URLSearchParams(Object.entries(state.ex).filter(([, v]) => v !== "" && v != null));
  const r = await api("/api/entities?" + p);
  const rows = r.items.map((e) => `<tr class="click" data-id="${esc(e.s1)}">
      <td><div style="font-weight:560">${esc(e.business_name)}</div><div class="muted small">${esc(e.business_address)}</div></td>
      <td>${esc(e.country)}</td><td class="num">${e.n_matched}</td><td class="num">${e.n_true ?? "–"}</td>
      <td class="num">${e.f05 == null ? "–" : e.f05.toFixed(2)}</td><td>${outcomeBadge(e.outcome)}</td></tr>`).join("");
  $("#ex-table").innerHTML = `<thead><tr><th>Reference business</th><th>Country</th><th class="num">Matched</th><th class="num">True</th><th class="num">F0.5</th><th>Outcome</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="6" class="empty">No entities match these filters.</td></tr>'}</tbody>`;
  $$("#ex-table tr.click").forEach((tr) => tr.addEventListener("click", () => openEntity(tr.dataset.id)));
  const end = Math.min(state.ex.offset + state.ex.limit, r.total);
  $("#ex-count").textContent = r.total ? `${fmtInt(state.ex.offset + 1)}–${fmtInt(end)} of ${fmtInt(r.total)}` : "0 results";
  $("#ex-prev").disabled = state.ex.offset === 0;
  $("#ex-next").disabled = end >= r.total;
}

async function openEntity(id) {
  const d = await api("/api/entities/" + encodeURIComponent(id));
  const e = d.entity;
  const recs = d.candidates.map((c) => {
    const cls = c.truth == null ? "" : c.matched && c.truth ? "tp" : c.matched ? "fp" : c.truth ? "fn" : "";
    const tag = c.truth == null ? (c.matched ? '<span class="badge info">matched</span>' : "")
      : c.matched && c.truth ? '<span class="badge good">✓ correct match</span>'
      : c.matched ? '<span class="badge bad">✗ false match</span>'
      : c.truth ? '<span class="badge warn">missed</span>' : '<span class="badge plain">rejected</span>';
    const rec = c.record || {};
    return `<div class="card rec ${cls}"><div class="cand-head"><div style="min-width:0">
        <div class="row" style="gap:8px">${tag}<span class="muted small mono">${esc(c.rec)} · source ${esc(rec.source ?? "?")}</span></div>
        <div class="cand-name" style="margin-top:4px">${esc(rec.business_name)}</div><div class="cand-addr">${esc(rec.business_address) || '<i class="muted">empty address</i>'}</div>
        ${c.note ? `<div class="small" style="color:var(--warn);margin-top:2px">${esc(c.note)}</div>` : ""}</div>
        ${c.p != null || c.p1 != null ? probCell(c.p, c.p1) : ""}</div>${contribBlock(c.explanation)}</div>`;
  }).join("");
  const bg = document.createElement("div"); bg.className = "drawer-bg";
  const dr = document.createElement("aside"); dr.className = "drawer";
  dr.innerHTML = `<div class="drawer-head"><div><div class="muted small mono">${esc(e.s1)} · ${esc(e.country)}</div>
      <h2 style="font-size:20px">${esc(e.business_name)}</h2><div class="cand-addr">${esc(e.business_address)}</div>
      <div class="row" style="margin-top:8px">${outcomeBadge(e.outcome)}<span class="badge plain">${e.n_matched} matched</span>
      ${e.n_true != null ? `<span class="badge plain">${e.n_true} true</span>` : ""}<span class="badge plain">${e.n_candidates} candidates</span></div></div>
      <button class="icon-btn" aria-label="Close">✕</button></div>${recs || '<div class="card empty">No candidates were retrieved for this reference.</div>'}`;
  const close = () => { bg.remove(); dr.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (k) => k.key === "Escape" && close();
  bg.addEventListener("click", close); $("button", dr).addEventListener("click", close);
  document.addEventListener("keydown", onKey);
  document.body.append(bg, dr);
  bindTips(dr);
}

// ------------------------------------------------------------------ BATCH
const files = {};
function initBatch() {
  $$(".drop").forEach((zone) => {
    const input = $("input", zone), key = zone.dataset.key;
    const set = (f) => {
      if (!f) return;
      files[key] = f; zone.classList.add("ok");
      $(".fname", zone).textContent = `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
      $("#job-run").disabled = !(files.source1 && files.source2 && files.source3);
    };
    input.addEventListener("change", () => set(input.files[0]));
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); set(e.dataTransfer.files[0]); });
  });
  $("#job-run").addEventListener("click", async () => {
    const fd = new FormData();
    for (const k of ["source1", "source2", "source3", "ground_truth"]) if (files[k]) fd.append(k, files[k]);
    startJob(() => api("/api/jobs", { method: "POST", body: fd }));
  });
  $("#job-sample").addEventListener("click", () => startJob(() => api("/api/jobs/sample", { method: "POST" })));
}

const STAGES = ["normalising text", "partitioning and blocking", "computing pair features", "scoring and decoding", "writing and validating files"];
async function startJob(create) {
  const out = $("#job-out");
  $("#job-run").disabled = $("#job-sample").disabled = true;
  try {
    let job = await create();
    while (job.status === "queued" || job.status === "running") {
      const i = STAGES.indexOf(job.stage);
      out.innerHTML = `<div class="card pad"><h2><span class="spinner" style="color:var(--accent)"></span>&nbsp; Resolving ${esc(job.name)}</h2>
        <div class="steps">${STAGES.map((s, j) => `<span class="step ${j < i ? "done" : j === i ? "on" : ""}">${esc(s)}</span>`).join("")}</div>
        <p class="muted small">${job.elapsed_s}s elapsed</p></div>`;
      await new Promise((r) => setTimeout(r, 700));
      job = await api("/api/jobs/" + job.id);
    }
    if (job.status === "failed") throw new Error(job.error);
    renderJob(job);
  } catch (e) {
    out.innerHTML = `<div class="card pad"><span class="badge bad">Job failed</span><p>${esc(e.message)}</p></div>`;
  } finally {
    $("#job-sample").disabled = false;
    $("#job-run").disabled = !(files.source1 && files.source2 && files.source3);
  }
}

function renderJob(job) {
  const r = job.result, m = r.metrics, v = r.validation;
  const tiles = [
    ["References", fmtInt(r.dataset.s1), `${fmtInt(r.dataset.s2 + r.dataset.s3)} records in sources 2 + 3`],
    ["Matched pairs", fmtInt(r.matched_pairs), `${pct(r.matched_s1 / r.dataset.s1)} of references matched`],
    ["Candidates", fmtInt(r.candidate_pairs), `${(r.candidate_pairs / r.dataset.s1).toFixed(1)} per reference`],
    m ? ["Macro F0.5", m.f05.toFixed(4), `P ${pct(m.precision)} · R ${pct(m.recall)}`] : ["Run time", job.elapsed_s + "s", "end to end"],
  ];
  const byC = (r.metrics_by_country || r.by_country).map((c) => `<tr><td>${esc(c.country)}</td>
      ${m ? `<td class="num">${fmtInt(c.n_s1)}</td><td class="num">${c.f05.toFixed(4)}</td><td class="num">${pct(c.precision)}</td><td class="num">${pct(c.recall)}</td>`
          : `<td class="num">${fmtInt(c.s1)}</td><td class="num">${pct(c.matched_share)}</td><td class="num">${c.matches_per_s1.toFixed(2)}</td>`}</tr>`).join("");
  const head = m ? "<th>Country</th><th class='num'>References</th><th class='num'>F0.5</th><th class='num'>Precision</th><th class='num'>Recall</th>"
    : "<th>Country</th><th class='num'>References</th><th class='num'>Matched</th><th class='num'>Matches / ref.</th>";
  const preview = r.preview.slice(0, 12).map((p) => `<tr><td><div style="font-weight:560">${esc(p.name)}</div><div class="muted small mono">${esc(p.s1)}</div></td>
      <td>${p.matches.map((x) => `<div>${esc(x.name)} <span class="muted small">${pct(x.p, 0)}</span></div>`).join("")}</td></tr>`).join("");
  $("#job-out").innerHTML = `<div class="stack">
    <div class="card pad row" style="justify-content:space-between">
      <div><h2>${esc(job.name)} resolved in ${job.elapsed_s}s</h2>
        ${v.pass ? '<span class="badge good">✓ submission format valid</span>' : `<span class="badge bad">validation issues</span> <span class="small">${esc(v.issues.join("; "))}</span>`}
        ${m && r.blocking ? `<span class="badge plain">blocking recall ${pct(r.blocking.cand_recall)}</span>` : ""}</div>
      <a class="btn" href="/api/jobs/${esc(job.id)}/download">Download results (.zip)</a></div>
    <div class="kpis" style="margin:0">${tiles.map(([l, val, n]) => `<div class="card kpi"><div class="label">${l}</div><div class="value">${val}</div><div class="note">${n}</div></div>`).join("")}</div>
    <div class="card pad"><h3>By country</h3><div class="table-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${byC}</tbody></table></div></div>
    <div class="card pad"><h3>Preview: highest-confidence clusters</h3><div class="table-wrap"><table class="data"><thead><tr><th>Reference</th><th>Matched records</th></tr></thead><tbody>${preview}</tbody></table></div></div>
  </div>`;
}

// ------------------------------------------------------------------ MODEL (charts)
const clip = (s, n) => (s.length > n ? s.slice(0, Math.max(1, Math.floor(n) - 1)) + "…" : s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function hbars(el, rows, { fmt = (v) => v.toFixed(3), domain = null, color = "--s1", labelW: lw = 170, tipFn } = {}) {
  let labelW = lw;
  const W = Math.max(300, el.clientWidth || 560), rowH = 26, H = rows.length * rowH + 8;
  labelW = Math.min(labelW, W * 0.45);
  const vals = rows.map((r) => r.value);
  const lo = domain ? domain[0] : 0, hi = domain ? domain[1] : Math.max(...vals);
  const x = (v) => labelW + ((v - lo) / (hi - lo || 1)) * (W - labelW - 56);
  const bars = rows.map((r, i) => {
    const y = 4 + i * rowH, x1 = x(lo), x2 = Math.max(x(r.value), x1 + 2);
    return `<g data-tip="${esc(tipFn ? tipFn(r) : `<b>${esc(r.label)}</b><br>${fmt(r.value)}`)}">
      <rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>
      <text x="${labelW - 8}" y="${y + 17}" text-anchor="end">${esc(clip(r.label, (labelW - 10) / 6.4))}</text>
      <path d="M${x1},${y + 6} H${x2 - 4} a4,4 0 0 1 4,4 V${y + 16} a4,4 0 0 1 -4,4 H${x1} Z" fill="${css(r.color || color)}"/>
      <text x="${x2 + 6}" y="${y + 17}" style="fill:var(--text);font-variant-numeric:tabular-nums">${fmt(r.value)}</text></g>`;
  }).join("");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${bars}</svg>`;
  bindTips(el);
}

function lineChart(el, legendEl, curve, series, efPoint) {
  const W = Math.max(300, el.clientWidth || 560), H = 270, m = { l: 40, r: 16, t: 10, b: 44 };
  const xs = curve.map((d) => d.t);
  const all = curve.flatMap((d) => series.map((s) => d[s.key]));
  const lo = Math.max(0, Math.floor(Math.min(...all) * 20) / 20), hi = 1;
  const x = (v) => m.l + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - m.l - m.r);
  const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
  const ticks = []; for (let v = lo; v <= hi + 1e-9; v += (hi - lo) / 4) ticks.push(v);
  const grid = ticks.map((v) => `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(3)}</text>`).join("");
  const xt = [0.1, 0.3, 0.5, 0.7, 0.9].map((v) => `<text x="${x(v)}" y="${H - m.b + 16}" text-anchor="middle">${v}</text>`).join("");
  const lines = series.map((s) => `<path d="${curve.map((d, i) => `${i ? "L" : "M"}${x(d.t).toFixed(1)},${y(d[s.key]).toFixed(1)}`).join("")}" fill="none" stroke="${css(s.color)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`).join("");
  const ef = efPoint ? `<line x1="${m.l}" x2="${W - m.r}" y1="${y(efPoint)}" y2="${y(efPoint)}" stroke="${css("--muted")}" stroke-dasharray="4 4"/>
      <text x="${W - m.r}" y="${y(efPoint) - 6}" text-anchor="end">expected-F decoding ${efPoint.toFixed(4)}</text>` : "";
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision, recall and F0.5 by threshold">${grid}
      <line class="axis-line" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>${xt}${ef}${lines}
      <line id="xh" y1="${m.t}" y2="${H - m.b}" stroke="${css("--muted")}" class="hidden"/>
      <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" id="hit"/>
      <text x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle" style="fill:var(--muted)">threshold on match probability</text></svg>`;
  legendEl.innerHTML = series.map((s) => `<span><i style="background:${css(s.color)}"></i>${s.label}</span>`).join("");
  const svg = $("svg", el), hit = $("#hit", el), xh = $("#xh", el);
  hit.addEventListener("mousemove", (e) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const d = curve.reduce((a, b) => (Math.abs(x(b.t) - p.x) < Math.abs(x(a.t) - p.x) ? b : a));
    xh.setAttribute("x1", x(d.t)); xh.setAttribute("x2", x(d.t)); xh.classList.remove("hidden");
    showTip(e, `<b>t = ${d.t.toFixed(2)}</b><br>` + series.map((s) => `${s.label}: ${d[s.key].toFixed(4)}`).join("<br>"));
  });
  hit.addEventListener("mouseleave", () => { hideTip(); xh.classList.add("hidden"); });
}

async function loadModel() {
  state.report = await api("/api/report");
  renderModel();
}

function renderModel() {
  const { train, test } = state.report;
  if (!train || !test) { $("#kpis").innerHTML = '<div class="card empty">No training report found in artifacts/.</div>'; return; }
  const fr = test.slices.by_country.find((c) => c.country === "France");
  const tiles = [
    ["Test macro F0.5", test.final.f05.toFixed(4), `${fmtInt(test.final.n_s1)} held-out references`],
    ["Precision", pct(test.final.precision), "pairs predicted that are right"],
    ["Recall", pct(test.final.recall), "true pairs found"],
    ["France (zero-shot)", fr ? fr.f05.toFixed(4) : "–", "country never seen in training"],
    ["Out-of-fold F0.5", train.final.f05.toFixed(4), `train split, ${train.protocol.folds} folds by state`],
  ];
  $("#kpis").innerHTML = tiles.map(([l, v, n]) => `<div class="card kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="note">${n}</div></div>`).join("");

  const ladder = train.ladder.filter((r) => !r.step.startsWith("Stage 2:"));
  hbars($("#ch-ladder"), ladder.map((r) => ({ label: r.step, value: r.f05, r, color: r.step === "Stage 2 ensemble" ? "--s1" : "--muted" })),
    { fmt: (v) => v.toFixed(4), domain: [0.5, 1], labelW: 160, tipFn: (d) => `<b>${esc(d.r.step)}</b><br>${esc(d.r.detail)}<br>F0.5 ${d.r.f05.toFixed(4)} · P ${pct(d.r.precision)} · R ${pct(d.r.recall)}` });

  lineChart($("#ch-curve"), $("#lg-curve"), test.threshold_curve,
    [{ key: "precision", label: "Precision", color: "--s1" }, { key: "recall", label: "Recall", color: "--s2" }, { key: "f05", label: "Macro F0.5", color: "--s3" }],
    test.final.f05);

  hbars($("#ch-country"), test.slices.by_country.map((c) => ({ label: c.country + (c.country === "France" ? " (unseen)" : ""), value: c.f05, c })),
    { fmt: (v) => v.toFixed(4), domain: [0.9, 1], labelW: 120, tipFn: (d) => `<b>${esc(d.c.country)}</b><br>F0.5 ${d.c.f05.toFixed(4)}<br>precision ${pct(d.c.precision)} · recall ${pct(d.c.recall)}<br>${fmtInt(d.c.n_s1)} references` });

  $("#tb-slices").innerHTML = `<thead><tr><th>Records</th><th class="num">True pairs</th><th class="num">Precision</th><th class="num">Recall</th></tr></thead><tbody>` +
    test.slices.by_record.map((s) => `<tr><td>${esc(s.slice)}</td><td class="num">${fmtInt(s.true_pairs)}</td><td class="num">${pct(s.precision)}</td><td class="num">${pct(s.recall)}</td></tr>`).join("") + "</tbody>";

  hbars($("#ch-imp"), train.importance.slice(0, 12).map((f) => ({ label: f.label || f.feature, value: f.gain, f })),
    { fmt: (v) => pct(v, 1), labelW: 250, tipFn: (d) => `<b>${esc(d.f.label || d.f.feature)}</b><br><span class="mono">${esc(d.f.feature)}</span> · ${esc(d.f.group_label || "")}<br>${pct(d.f.gain, 2)} of total gain` });

  const b = test.blocking;
  $("#blocking").innerHTML = `<table class="data"><tbody>
    <tr><td>Recall of true pairs</td><td class="num">${pct(b.cand_recall, 2)}</td></tr>
    <tr><td>Oracle F0.5 (perfect matcher on these candidates)</td><td class="num">${b.oracle_f05.toFixed(4)}</td></tr>
    <tr><td>Candidates per reference</td><td class="num">${b.cand_per_s1.toFixed(1)}</td></tr>
    <tr><td>Reduction vs. all pairs</td><td class="num">${pct(b.reduction_ratio, 3)}</td></tr></tbody></table>`;
}

// ------------------------------------------------------------------ boot
let resizeT;
addEventListener("resize", () => {
  clearTimeout(resizeT);
  resizeT = setTimeout(() => { if (state.report && !$("#tab-model").classList.contains("hidden")) renderModel(); }, 150);
});

async function boot() {
  theme.init();
  $$("nav.tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#match-form").addEventListener("submit", runMatch);
  initBatch();
  try {
    state.meta = await api("/api/meta");
  } catch (e) {
    $("#match-out").innerHTML = `<div class="card pad"><span class="badge bad">Server not ready</span> ${esc(e.message)}</div>`;
    return;
  }
  const m = state.meta;
  $("#ver").textContent = "v" + m.version;
  $("#ref-count").textContent = `${fmtInt(m.dataset.s1)} reference businesses (US, India and France)`;
  $("#q-country").innerHTML = m.countries.map((c) => `<option>${esc(c)}</option>`).join("");
  $("#examples").innerHTML = m.examples.map((x, i) => `<button type="button" class="chip" data-i="${i}" title="${esc(x.name + " | " + x.address)}"><b>${esc(x.tag)}</b>${esc(x.name)}</button>`).join("");
  $$("#examples .chip").forEach((c) => c.addEventListener("click", () => {
    const x = m.examples[+c.dataset.i];
    $("#q-name").value = x.name; $("#q-addr").value = x.address; $("#q-country").value = x.country;
    runMatch();
  }));
  const h = location.hash.slice(1);
  showTab(["match", "explore", "batch", "model"].includes(h) ? h : "match");
}
boot();
