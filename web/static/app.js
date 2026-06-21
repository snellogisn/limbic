// limbic web console — vanilla JS, no framework. Drives both pages.

function esc(s) {
  return String(s == null ? "" : s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function badge(status) {
  const known = ["completed", "cannot_complete", "error", "stopped"];
  const cls = known.includes(status) ? status : "unknown";
  const label = (status || "unknown").replace("_", " ");
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function planHtml(plan) {
  if (!plan || !plan.length) return "<p class='meta'>(no steps)</p>";
  const items = plan.map(s => {
    const args = Object.entries(s.args || {}).map(([k, v]) => `${k}=${v}`).join(", ");
    return `<li><code>${esc(s.primitive)}(${esc(args)})</code></li>`;
  }).join("");
  return `<ol>${items}</ol>`;
}

// ----- Ask page -------------------------------------------------------------
function initAskPage() {
  const taskEl = document.getElementById("task");
  const runBtn = document.getElementById("run");
  const stopBtn = document.getElementById("stop");
  const busy = document.getElementById("busy");
  const resultEl = document.getElementById("result");

  async function submit(task) {
    if (!task.trim()) { taskEl.focus(); return; }
    runBtn.disabled = true;
    stopBtn.classList.remove("hidden");
    stopBtn.disabled = false;
    busy.textContent = "running the pipeline…";
    busy.classList.remove("hidden");
    resultEl.classList.add("hidden");
    try {
      const resp = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
      });
      const r = await resp.json();
      renderResult(r);
    } catch (e) {
      resultEl.innerHTML = `<p class="badge error">request failed</p><pre>${esc(e)}</pre>`;
      resultEl.classList.remove("hidden");
    } finally {
      runBtn.disabled = false;
      stopBtn.classList.add("hidden");
      busy.classList.add("hidden");
    }
  }

  // Emergency stop: a separate request that freezes the arm mid-motion. The
  // pending /api/run above then returns with status "stopped".
  async function stop() {
    stopBtn.disabled = true;
    busy.textContent = "stopping…";
    try {
      await fetch("/api/stop", { method: "POST" });
    } catch (e) { /* the run will still return; nothing to do here */ }
  }

  function renderResult(r) {
    const counts = r.counts || {};
    const reason = r.status !== "completed" && r.error
      ? `<p><strong>Why it couldn't:</strong> ${esc(r.error)}</p>` : "";
    const rationale = r.rationale ? `<p class="meta">${esc(r.rationale)}</p>` : "";
    resultEl.innerHTML = `
      <div class="row" style="justify-content:space-between">
        <div>${badge(r.status)} <strong>${esc(r.task)}</strong></div>
        <div class="meta">${esc(r.mode || "")} · ${esc(r.model || "")}</div>
      </div>
      ${rationale}
      ${reason}
      <h3>Plan</h3>
      ${planHtml(r.plan)}
      <p class="meta">logged ${counts.movements || 0} movements ·
        ${counts.data || 0} data reads · ${counts.thinking || 0} thoughts</p>
      <p><a class="link" href="/runs?run=${encodeURIComponent(r.run_id || "")}">View full log →</a></p>
    `;
    resultEl.classList.remove("hidden");
  }

  runBtn.addEventListener("click", () => submit(taskEl.value));
  stopBtn.addEventListener("click", stop);
  document.querySelectorAll(".test").forEach(btn => {
    btn.addEventListener("click", () => { taskEl.value = btn.dataset.task; submit(btn.dataset.task); });
  });
}

// ----- Logs page ------------------------------------------------------------
function initLogsPage() {
  const listEl = document.getElementById("runs-list");
  const detailEl = document.getElementById("detail");
  const emptyEl = document.getElementById("empty");
  const refreshBtn = document.getElementById("refresh");
  const wanted = new URLSearchParams(location.search).get("run");

  async function loadRuns() {
    const { runs } = await (await fetch("/api/runs")).json();
    emptyEl.classList.toggle("hidden", runs.length > 0);
    listEl.innerHTML = runs.map(r => `
      <div class="run-item" data-id="${esc(r.run_id)}">
        <div class="task">${badge(r.status)} ${esc(r.task)}</div>
        <div class="when">${esc(r.started_at || r.run_id)} · ${esc(r.mode || "")}</div>
      </div>`).join("");

    listEl.querySelectorAll(".run-item").forEach(item => {
      item.addEventListener("click", () => selectRun(item.dataset.id));
    });

    const first = wanted || (runs[0] && runs[0].run_id);
    if (first) selectRun(first);
  }

  async function selectRun(runId) {
    listEl.querySelectorAll(".run-item").forEach(i =>
      i.classList.toggle("selected", i.dataset.id === runId));
    const d = await (await fetch(`/api/runs/${encodeURIComponent(runId)}`)).json();
    if (d.error) { detailEl.innerHTML = `<pre>${esc(d.error)}</pre>`; return; }

    const res = d.result || {};
    const reason = res.status !== "completed" && res.error
      ? `<p><strong>Why it couldn't:</strong> ${esc(res.error)}</p>` : "";
    detailEl.innerHTML = `
      <div class="row" style="justify-content:space-between">
        <div>${badge(res.status)} <strong>${esc(res.task || d.run_id)}</strong></div>
        <div class="meta">${esc(res.mode || "")} · ${esc(res.model || "")}</div>
      </div>
      ${res.rationale ? `<p class="meta">${esc(res.rationale)}</p>` : ""}
      ${reason}
      <h3>Plan</h3>${planHtml(res.plan)}
      ${stream("Thinking (decisions)", d.thinking)}
      ${stream("Data (what it sensed)", d.data)}
      ${stream("Movements (what it did)", d.movements)}
    `;
  }

  function stream(title, records) {
    records = records || [];
    const body = records.length
      ? `<pre>${esc(records.map(r => JSON.stringify(r)).join("\n"))}</pre>`
      : `<p class="meta">(none)</p>`;
    return `<div class="stream"><h3>${esc(title)} <span class="count">· ${records.length}</span></h3>${body}</div>`;
  }

  refreshBtn.addEventListener("click", loadRuns);
  loadRuns();
}

// ----- boot -----------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("task")) initAskPage();
  if (document.getElementById("runs-list")) initLogsPage();
});
