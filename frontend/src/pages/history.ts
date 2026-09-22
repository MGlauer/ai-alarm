import { api, poll } from "../api";
import { situationDetailView } from "../components/situationDetail";
import type { Scenario, ScenarioDetail, SituationRow } from "../types";
import { escapeHtml } from "../util";

const REFRESH_INTERVAL_MS = 6000;

export function renderHistory(container: HTMLElement): () => void {
  container.innerHTML = "<h2>History</h2><div id=\"scenario-list\"></div>";
  const listEl = container.querySelector<HTMLElement>("#scenario-list")!;

  // which scenarios/situations the user has unfolded -- kept across refreshes, so polling does not collapse them
  const openScenarios = new Set<string>();
  const openSituations = new Set<string>();

  async function refresh(): Promise<void> {
    const scenarios = await api.scenarios();
    listEl.innerHTML = "";
    if (scenarios.length === 0) {
      listEl.innerHTML = '<p class="muted">No situations yet -- play a scenario on the Simulation page.</p>';
      return;
    }
    for (const scenario of scenarios) {
      listEl.appendChild(await scenarioCard(scenario, openScenarios, openSituations, refresh));
    }
  }

  return poll(refresh, REFRESH_INTERVAL_MS);
}

// ---------------------------------------------------------------- scenarios
async function scenarioCard(
  scenario: Scenario, openScenarios: Set<string>, openSituations: Set<string>, onAnswered: () => void,
): Promise<HTMLElement> {
  const card = document.createElement("div");
  card.className = "scenario-card";
  const isOpen = openScenarios.has(scenario.id);
  card.innerHTML = `
    <div class="scenario-header">
      <button class="toggle">${isOpen ? "▾" : "▸"}</button>
      <span class="badge ${scenario.status}">${scenario.status}</span>
      <span>${new Date(scenario.created_at).toLocaleString()}</span>
      <span class="muted">${scenario.situation_ids.length} situation(s)</span>
    </div>
    <div class="scenario-body" ${isOpen ? "" : "hidden"}></div>
  `;
  const toggle = card.querySelector<HTMLButtonElement>(".toggle")!;
  const body = card.querySelector<HTMLElement>(".scenario-body")!;

  const load = async () => {
    body.innerHTML = '<p class="muted">Loading…</p>';
    const detail = await api.scenario(scenario.id);
    body.innerHTML = "";
    body.appendChild(await scenarioDetailView(detail, openSituations, onAnswered));
  };

  toggle.addEventListener("click", async () => {
    if (openScenarios.has(scenario.id)) {
      openScenarios.delete(scenario.id);
      body.hidden = true;
      toggle.textContent = "▸";
    } else {
      openScenarios.add(scenario.id);
      toggle.textContent = "▾";
      body.hidden = false;
      await load();
    }
  });

  if (isOpen) await load();
  return card;
}

async function scenarioDetailView(
  detail: ScenarioDetail, openSituations: Set<string>, onAnswered: () => void,
): Promise<HTMLElement> {
  const el = document.createElement("div");
  const latest = detail.aggregated_summaries.at(-1);
  if (latest) {
    el.innerHTML = `<p><strong>Latest assessment:</strong> ${escapeHtml(latest.summary)} ` +
      `(threat ${latest.threat_score.toFixed(2)})</p>`;
  }
  for (const situation of detail.situations) {
    el.appendChild(await situationCard(situation, openSituations, onAnswered));
  }
  return el;
}

// ---------------------------------------------------------------- situations
async function situationCard(
  row: SituationRow, openSituations: Set<string>, onAnswered: () => void,
): Promise<HTMLElement> {
  const card = document.createElement("div");
  card.className = "situation-card";
  const isOpen = openSituations.has(row.id);
  card.innerHTML = `
    <div class="situation-header">
      <button class="toggle">${isOpen ? "▾" : "▸"}</button>
      <span class="badge ${row.status}">${row.status}</span>
      <span>${escapeHtml(row.area_id)}</span>
      <a href="#situation/${encodeURIComponent(row.id)}" class="muted">open on its own page →</a>
    </div>
    <div class="situation-body" ${isOpen ? "" : "hidden"}></div>
  `;
  const toggle = card.querySelector<HTMLButtonElement>(".toggle")!;
  const body = card.querySelector<HTMLElement>(".situation-body")!;

  const load = async () => {
    body.innerHTML = '<p class="muted">Loading…</p>';
    const detail = await api.situation(row.id);
    body.innerHTML = "";
    body.appendChild(situationDetailView(detail, { onAnswered }));
  };

  toggle.addEventListener("click", async () => {
    if (openSituations.has(row.id)) {
      openSituations.delete(row.id);
      body.hidden = true;
      toggle.textContent = "▸";
    } else {
      openSituations.add(row.id);
      toggle.textContent = "▾";
      body.hidden = false;
      await load();
    }
  });

  if (isOpen) await load();
  return card;
}
