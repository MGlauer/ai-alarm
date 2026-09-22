import { api, poll } from "../api";
import type { PersonRow } from "../types";

const REFRESH_INTERVAL_MS = 10000;

export function renderPersons(container: HTMLElement): () => void {
  container.innerHTML = `
    <h2>People</h2>
    <p class="muted">Everyone the system currently knows about, with their roles and a representative picture.
      Read-only: people are only added when the system detects and identifies them.</p>
    <div id="person-grid" class="person-grid"></div>
  `;
  const grid = container.querySelector<HTMLElement>("#person-grid")!;

  async function refresh(): Promise<void> {
    const persons = await api.persons();
    grid.innerHTML = "";
    if (persons.length === 0) {
      grid.innerHTML = '<p class="muted">No one in the knowledge base yet.</p>';
      return;
    }
    for (const person of persons) grid.appendChild(personCard(person));
  }

  return poll(refresh, REFRESH_INTERVAL_MS);
}

function personCard(person: PersonRow): HTMLElement {
  const card = document.createElement("div");
  card.className = "person-card";

  const img = document.createElement("img");
  img.src = person.picture_url;
  img.alt = person.name ?? person.id;
  img.height = 270;
  img.addEventListener("error", () => img.replaceWith(avatarFallback(person)), { once: true });
  card.appendChild(img);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = `
    <div class="name">${escape(person.name ?? "Unknown person")}</div>
    <div class="muted">${person.roles.length ? escape(person.roles.join(", ")) : "no known role"}</div>
    <div class="scores">familiarity ${person.familiarity.toFixed(2)} · suspicion ${person.suspicion.toFixed(2)}</div>
  `;
  card.appendChild(meta);
  return card;
}

function avatarFallback(person: PersonRow): HTMLElement {
  const div = document.createElement("div");
  div.className = "avatar-fallback";
  div.textContent = (person.name ?? "?").slice(0, 1).toUpperCase();
  return div;
}

function escape(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
