import { api, poll } from "../api";
import type { AlarmRow, AreaRow, SensorRow, WarningRow } from "../types";

const FRAME_INTERVAL_MS = 2500; // matches the backend's Cache-Control max-age for a sensor frame
const SENSOR_LIST_INTERVAL_MS = 5000;
const WARNINGS_INTERVAL_MS = 3000;

export function renderHome(container: HTMLElement): () => void {
  container.innerHTML = `
    <section id="warnings-panel"></section>
    <h2>Sensors</h2>
    <div id="sensor-grid"></div>
  `;
  const warningsEl = container.querySelector<HTMLElement>("#warnings-panel")!;
  const gridEl = container.querySelector<HTMLElement>("#sensor-grid")!;

  let gridBuilt = false;

  async function refreshWarnings(): Promise<void> {
    const [open, alarms] = await Promise.all([api.warnings("open"), api.alarms()]);
    renderWarnings(warningsEl, open, alarms.slice(0, 5), refreshWarnings);
  }

  async function refreshSensors(): Promise<void> {
    const [sensors, areas] = await Promise.all([api.sensors(), api.areas()]);
    if (!gridBuilt) {
      buildSensorGrid(gridEl, sensors, areas);
      gridBuilt = true;
    } else {
      updatePeoplePresent(gridEl, areas);
    }
  }

  function refreshCameraFrames(): void {
    const now = Date.now();
    gridEl.querySelectorAll<HTMLImageElement>("img[data-sensor]").forEach((img) => {
      img.src = `/api/sensors/${img.dataset.sensor}/frame?t=${now}`;
    });
  }

  const stopWarnings = poll(refreshWarnings, WARNINGS_INTERVAL_MS);
  const stopSensors = poll(refreshSensors, SENSOR_LIST_INTERVAL_MS);
  const stopFrames = poll(refreshCameraFrames, FRAME_INTERVAL_MS);

  return () => {
    stopWarnings();
    stopSensors();
    stopFrames();
  };
}

// ---------------------------------------------------------------- open warnings (HITL: elevate / dismiss)
function renderWarnings(
  container: HTMLElement, open: WarningRow[], recentAlarms: AlarmRow[], onAnswered: () => void,
): void {
  container.innerHTML = "<h2>Open warnings</h2>";

  if (open.length === 0) {
    container.insertAdjacentHTML("beforeend", '<p class="muted">No open warnings.</p>');
  } else {
    const list = document.createElement("div");
    list.className = "warning-list";
    for (const warning of open) list.appendChild(warningCard(warning, onAnswered));
    container.appendChild(list);
  }

  if (recentAlarms.length > 0) {
    const section = document.createElement("div");
    section.innerHTML =
      "<h3>Recent alarms</h3>" +
      recentAlarms
        .map(
          (a) =>
            `<div class="alarm-row">${escape(a.cause)} — ${escape(a.area_id)} (${escape(a.origin)}), ` +
            `${new Date(a.created_at).toLocaleTimeString()}</div>`,
        )
        .join("");
    container.appendChild(section);
  }
}

function warningCard(warning: WarningRow, onAnswered: () => void): HTMLElement {
  const card = document.createElement("div");
  card.className = `warning-card ${warning.colour}`;
  const deadline = warning.answer_deadline
    ? `, deadline ${new Date(warning.answer_deadline).toLocaleTimeString()}`
    : "";
  card.innerHTML = `
    <div><strong>${escape(warning.cause)}</strong> in ${escape(warning.area_id)}
      — suspicion ${warning.suspicion.toFixed(2)}</div>
    <div class="muted">opened ${new Date(warning.created_at).toLocaleTimeString()}${deadline}</div>
    <div class="actions">
      <button data-action="elevate">Elevate to alarm</button>
      <button data-action="dismiss">Dismiss</button>
    </div>
  `;
  const answer = (action: "elevate" | "dismiss", button: HTMLButtonElement) => async () => {
    card.querySelectorAll("button").forEach((b) => (b.disabled = true));
    try {
      await (action === "elevate" ? api.elevate(warning.id) : api.dismiss(warning.id));
      await onAnswered();
    } catch (e) {
      button.disabled = false;
      console.error(e);
    }
  };
  const elevateBtn = card.querySelector<HTMLButtonElement>('[data-action="elevate"]')!;
  const dismissBtn = card.querySelector<HTMLButtonElement>('[data-action="dismiss"]')!;
  elevateBtn.addEventListener("click", answer("elevate", elevateBtn));
  dismissBtn.addEventListener("click", answer("dismiss", dismissBtn));
  return card;
}

// ---------------------------------------------------------------- the sensor grid
function buildSensorGrid(container: HTMLElement, sensors: SensorRow[], areas: AreaRow[]): void {
  const areaById = new Map(areas.map((a) => [a.id, a]));
  const byArea = new Map<string, SensorRow[]>();
  for (const sensor of sensors) {
    if (!byArea.has(sensor.area_id)) byArea.set(sensor.area_id, []);
    byArea.get(sensor.area_id)!.push(sensor);
  }

  container.innerHTML = "";
  for (const [areaId, areaSensors] of byArea) {
    const block = document.createElement("div");
    block.className = "area-block";
    const heading = document.createElement("h3");
    heading.textContent = areaById.get(areaId)?.name ?? areaId;
    const people = document.createElement("span");
    people.className = "people";
    people.dataset.area = areaId;
    heading.appendChild(people);
    block.appendChild(heading);

    const tiles = document.createElement("div");
    tiles.className = "tiles";
    for (const sensor of areaSensors) tiles.appendChild(sensorTile(sensor));
    block.appendChild(tiles);
    container.appendChild(block);
  }
  updatePeoplePresent(container, areas);
}

function sensorTile(sensor: SensorRow): HTMLElement {
  const tile = document.createElement("div");
  tile.className = "tile";

  if (sensor.kind === "camera") {
    const img = document.createElement("img");
    img.dataset.sensor = sensor.id;
    img.alt = sensor.id;
    img.src = `/api/sensors/${sensor.id}/frame?t=${Date.now()}`;
    tile.appendChild(img);
  } else {
    const mic = document.createElement("div");
    mic.className = "mic-tile";
    mic.innerHTML = '<div class="mic-icon">\u{1F3A4}</div>';
    const button = document.createElement("button");
    button.textContent = "Play audio";
    const audio = new Audio();
    button.addEventListener("click", () => {
      audio.src = `/api/sensors/${sensor.id}/audio?t=${Date.now()}`;
      button.disabled = true;
      audio.play().catch(() => {});
    });
    audio.addEventListener("ended", () => (button.disabled = false));
    mic.appendChild(button);
    tile.appendChild(mic);
  }

  const label = document.createElement("div");
  label.className = "tile-label";
  label.textContent = `${sensor.id} · ${sensor.kind}`;
  tile.appendChild(label);
  return tile;
}

function updatePeoplePresent(container: HTMLElement, areas: AreaRow[]): void {
  for (const area of areas) {
    const el = container.querySelector(`.people[data-area="${area.id}"]`);
    if (el) el.textContent = area.people.length ? ` — ${area.people.join(", ")}` : "";
  }
}

function escape(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
