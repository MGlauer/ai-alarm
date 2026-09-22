import { api, poll } from "./api";
import { showToast } from "./notifications";

const POLL_INTERVAL_MS = 3000;

/** "Upon receiving a warning or an alarm, the user should be notified and the view should switch to a page
 * showing the situation" (DESIGN_DOCUMENT.md, Interface). Runs for the whole session, independent of whichever
 * page is currently mounted -- a warning must be caught even while the user is on, say, the People page.
 *
 * The backend has no push channel, so this polls; a warning/alarm not seen on a previous tick is "received".
 * The very first tick only seeds what is already open/logged -- it must not re-announce and jump away from
 * whatever page the user opened the app on. */
export function startAlertWatcher(onReceived: (situationId: string) => void): () => void {
  const seenWarnings = new Set<string>();
  const seenAlarms = new Set<string>();
  let seeded = false;

  async function tick(): Promise<void> {
    const [openWarnings, alarms] = await Promise.all([api.warnings("open"), api.alarms()]);

    if (!seeded) {
      for (const w of openWarnings) seenWarnings.add(w.id);
      for (const a of alarms) seenAlarms.add(a.id);
      seeded = true;
      return;
    }

    for (const w of openWarnings) {
      if (seenWarnings.has(w.id)) continue;
      seenWarnings.add(w.id);
      showToast(`New warning: ${w.cause} in ${w.area_id} (suspicion ${w.suspicion.toFixed(2)})`, "warning");
      onReceived(w.situation_id);
    }
    for (const a of alarms) {
      if (seenAlarms.has(a.id)) continue;
      seenAlarms.add(a.id);
      showToast(`ALARM: ${a.cause} in ${a.area_id}`, "alarm");
      onReceived(a.situation_id);
    }
  }

  return poll(tick, POLL_INTERVAL_MS);
}
