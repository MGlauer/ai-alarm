import { api } from "../api";
import type { AnimalAssessment, BBox, EventRow, PersonAssessment, SituationDetail, SummaryRow } from "../types";
import { escapeHtml } from "../util";

export interface SituationDetailOptions {
  /** Called after the user elevates or dismisses a warning from this view, so the caller can refresh. Omit to
   * render the warnings/alarms read-only (no action buttons). */
  onAnswered?: () => void;
}

/** Everything known about one situation: its evidence (with bounding-box annotations), its summaries, and its
 * warnings/alarms -- with Elevate/Dismiss buttons on any still-open warning when `onAnswered` is given. Shared by
 * the History page (read-only browsing of the past) and the dedicated situation page (the HITL decision view). */
export function situationDetailView(detail: SituationDetail, options: SituationDetailOptions = {}): HTMLElement {
  const el = document.createElement("div");
  const summaryByEvent = new Map(detail.summaries.map((s) => [s.data.event_id, s]));

  const strip = document.createElement("div");
  strip.className = "event-strip";
  for (const event of detail.events) strip.appendChild(eventTile(event, summaryByEvent.get(event.id)));
  el.appendChild(strip);

  if (detail.summaries.length > 0) {
    const list = document.createElement("ul");
    list.className = "summary-list";
    list.innerHTML = detail.summaries
      .map((s) => `<li>${new Date(s.created_at).toLocaleTimeString()} — ${escapeHtml(s.summary)} ` +
        `(threat ${s.threat_score.toFixed(2)})</li>`)
      .join("");
    el.appendChild(list);
  }

  if (detail.warnings.length > 0 || detail.alarms.length > 0) {
    el.appendChild(warningsAndAlarms(detail, options.onAnswered));
  }

  if (detail.log_entries.length > 0) {
    const log = document.createElement("details");
    log.innerHTML = `<summary>Log (${detail.log_entries.length})</summary>` +
      detail.log_entries
        .map((l) => `<div class="log-row"><span class="muted">${new Date(l.created_at).toLocaleTimeString()}` +
          `</span> [${escapeHtml(l.kind)}] ${escapeHtml(l.message)}</div>`)
        .join("");
    el.appendChild(log);
  }

  return el;
}

// ---------------------------------------------------------------- warnings and alarms (the HITL decision)
function warningsAndAlarms(detail: SituationDetail, onAnswered: (() => void) | undefined): HTMLElement {
  const wrap = document.createElement("div");
  wrap.className = "tag-cards";

  for (const warning of detail.warnings) {
    const card = document.createElement("div");
    card.className = `tag-card ${warning.colour}`;
    card.innerHTML = `<span>warning: ${escapeHtml(warning.cause)} — suspicion ${warning.suspicion.toFixed(2)} ` +
      `(${warning.status})</span>`;
    if (warning.status === "open" && onAnswered) {
      const actions = document.createElement("span");
      actions.className = "actions";
      actions.appendChild(actionButton("Elevate to alarm", () => api.elevate(warning.id), onAnswered));
      actions.appendChild(actionButton("Dismiss", () => api.dismiss(warning.id), onAnswered));
      card.appendChild(actions);
    }
    wrap.appendChild(card);
  }

  for (const alarm of detail.alarms) {
    const card = document.createElement("div");
    card.className = "tag-card alarm";
    card.textContent = `alarm: ${alarm.cause} (${alarm.origin})`;
    wrap.appendChild(card);
  }

  return wrap;
}

function actionButton(label: string, action: () => Promise<unknown>, onAnswered: () => void): HTMLButtonElement {
  const button = document.createElement("button");
  button.textContent = label;
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await action();
      onAnswered();
    } catch (e) {
      button.disabled = false;
      console.error(e);
    }
  });
  return button;
}

// ---------------------------------------------------------------- one event, with its annotations
function eventTile(event: EventRow, summary: SummaryRow | undefined): HTMLElement {
  const tile = document.createElement("div");
  tile.className = "event-tile";

  if (event.kind === "video") {
    const frame = document.createElement("div");
    frame.className = "frame";
    const img = document.createElement("img");
    img.src = `/api/media/${event.id}`;
    img.alt = "camera evidence";
    frame.appendChild(img);
    for (const box of annotationsFor(event, summary)) frame.appendChild(bboxElement(box));
    tile.appendChild(frame);
  } else {
    const button = document.createElement("button");
    button.textContent = "▶ Play audio";
    const audio = new Audio();
    button.addEventListener("click", () => {
      audio.src = `/api/media/${event.id}/audio`;
      audio.play().catch(() => {});
    });
    tile.appendChild(button);
  }

  const caption = document.createElement("div");
  caption.className = "tile-label";
  caption.textContent = `${event.sensor_id} · ${new Date(event.start_time).toLocaleTimeString()}`;
  tile.appendChild(caption);
  return tile;
}

interface Annotation {
  bbox: BBox;
  label: string;
}

function annotationsFor(event: EventRow, summary: SummaryRow | undefined): Annotation[] {
  if (summary) {
    const boxes: Annotation[] = [];
    for (const p of summary.data.persons) if (p.bbox) boxes.push({ bbox: p.bbox, label: personLabel(p) });
    for (const a of summary.data.animals) if (a.bbox) boxes.push({ bbox: a.bbox, label: animalLabel(a) });
    if (boxes.length > 0) return boxes;
  }
  // no interpretation yet (or none of its objects carried a box): fall back to the processor's own rough box
  return event.bbox ? [{ bbox: event.bbox, label: "" }] : [];
}

function personLabel(p: PersonAssessment): string {
  const who = p.identity === "known" ? p.name ?? p.person_id ?? "known"
    : p.identity === "unknown" ? "unknown person"
    : `not decidable (${p.cause ?? "unknown"})`;
  return `${who} · suspicion ${p.suspicion.toFixed(2)}`;
}

function animalLabel(a: AnimalAssessment): string {
  return `${a.label} · danger ${a.danger.toFixed(2)}`;
}

function bboxElement(box: Annotation): HTMLElement {
  const div = document.createElement("div");
  div.className = "bbox";
  div.style.left = `${box.bbox.x * 100}%`;
  div.style.top = `${box.bbox.y * 100}%`;
  div.style.width = `${box.bbox.w * 100}%`;
  div.style.height = `${box.bbox.h * 100}%`;
  if (box.label) {
    const tag = document.createElement("span");
    tag.className = "bbox-label";
    tag.textContent = box.label;
    div.appendChild(tag);
  }
  return div;
}
