import { api, poll } from "../api";
import { situationDetailView } from "../components/situationDetail";
import { escapeHtml } from "../util";

const REFRESH_INTERVAL_MS = 3000;

/** The page a warning/alarm switches the user to (DESIGN_DOCUMENT.md, Interface): one situation, its evidence,
 * and -- for as long as a warning is open -- the Elevate/Dismiss decision. */
export function renderSituationPage(container: HTMLElement, situationId: string): () => void {
  container.innerHTML = `
    <a href="#history" class="back-link">← Back to history</a>
    <h2>Situation ${escapeHtml(situationId)}</h2>
    <div id="situation-content"><p class="muted">Loading…</p></div>
  `;
  const content = container.querySelector<HTMLElement>("#situation-content")!;

  async function refresh(): Promise<void> {
    try {
      const detail = await api.situation(situationId);
      content.innerHTML = "";
      const heading = document.createElement("p");
      heading.innerHTML = `<span class="badge ${detail.status}">${detail.status}</span> in ` +
        `<strong>${detail.area_id}</strong>`;
      content.appendChild(heading);
      content.appendChild(situationDetailView(detail, { onAnswered: refresh }));
    } catch (e) {
      content.innerHTML = `<p class="error">${e instanceof Error ? e.message : String(e)}</p>`;
    }
  }

  return poll(refresh, REFRESH_INTERVAL_MS);
}
