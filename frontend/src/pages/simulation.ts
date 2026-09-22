import { api, poll } from "../api";
import type { DemoRow } from "../types";

const STATUS_INTERVAL_MS = 2000;
const LOG_INTERVAL_MS = 2000;
const LOG_LIMIT = 30;

export function renderSimulation(container: HTMLElement): () => void {
  container.innerHTML = `
    <h2>Simulation</h2>
    <p class="muted">Play one of the pre-defined scenarios; the backend reports it exactly as the real sensors
      would. Starting another one while this one is still playing resolves it and takes over immediately.</p>
    <div id="demo-list"></div>
    <div id="run-status"></div>
    <h3>Recent log</h3>
    <div id="sim-log"></div>
  `;
  const demoListEl = container.querySelector<HTMLElement>("#demo-list")!;
  const statusEl = container.querySelector<HTMLElement>("#run-status")!;
  const logEl = container.querySelector<HTMLElement>("#sim-log")!;

  let demos: DemoRow[] = [];
  let message = "";
  let messageIsError = false;

  function renderButtons(): void {
    demoListEl.innerHTML = "";
    for (const demo of demos) {
      const row = document.createElement("div");
      row.className = "demo-row";
      const button = document.createElement("button");
      button.textContent = demo.name;
      button.addEventListener("click", () => play(demo.name));
      row.appendChild(button);
      const desc = document.createElement("span");
      desc.className = "muted";
      desc.textContent = demo.description;
      row.appendChild(desc);
      demoListEl.appendChild(row);
    }
  }

  async function play(name: string): Promise<void> {
    message = "";
    try {
      await api.playDemo(name);
      message = `Playing "${name}"…`;
      messageIsError = false;
    } catch (e) {
      message = e instanceof Error ? e.message : String(e);
      messageIsError = true;
    }
    await refreshStatus();
  }

  async function refreshStatus(): Promise<void> {
    const scenarios = await api.scenarios();
    const active = scenarios.find((s) => s.status === "active");

    if (active) {
      const detail = await api.scenario(active.id);
      const latest = detail.aggregated_summaries.at(-1);
      statusEl.innerHTML = `<p><strong>Scenario in progress:</strong> ${detail.situations.length} situation(s)` +
        (latest ? `, latest threat ${latest.threat_score.toFixed(2)} — ${escape(latest.summary)}` : "") + "</p>";
    } else {
      statusEl.innerHTML = message
        ? `<p class="${messageIsError ? "error" : ""}">${escape(message)}</p>`
        : '<p class="muted">No scenario in progress.</p>';
    }
  }

  async function refreshLog(): Promise<void> {
    const log = await api.log({ limit: LOG_LIMIT });
    logEl.innerHTML = log
      .map((l) => `<div class="log-row"><span class="muted">${new Date(l.created_at).toLocaleTimeString()}` +
        `</span> [${escape(l.kind)}] ${escape(l.message)}</div>`)
      .join("") || '<p class="muted">Nothing logged yet.</p>';
  }

  api.demos().then((rows) => {
    demos = rows;
    renderButtons();
  });

  const stopStatus = poll(refreshStatus, STATUS_INTERVAL_MS);
  const stopLog = poll(refreshLog, LOG_INTERVAL_MS);
  return () => {
    stopStatus();
    stopLog();
  };
}

function escape(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
