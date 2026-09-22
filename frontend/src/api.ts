import type {
  AlarmRow, AreaRow, DemoRow, LogEntryRow, PersonRow, Scenario, ScenarioDetail, SensorRow, SituationDetail,
  WarningRow,
} from "./types";

class ApiError extends Error {}

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = (await response.json()).error ?? message;
    } catch {
      // no JSON body: keep the status text
    }
    throw new ApiError(message);
  }
  return (await response.json()) as T;
}

export const api = {
  scenarios: () => req<Scenario[]>("/scenarios"),
  scenario: (id: string) => req<ScenarioDetail>(`/scenarios/${id}`),
  situation: (id: string) => req<SituationDetail>(`/situations/${id}`),
  warnings: (status?: string) => req<WarningRow[]>(`/warnings${status ? `?status=${status}` : ""}`),
  elevate: (id: string) => req<WarningRow>(`/warnings/${id}/elevate`, { method: "POST" }),
  dismiss: (id: string) => req<WarningRow>(`/warnings/${id}/dismiss`, { method: "POST" }),
  alarms: () => req<AlarmRow[]>("/alarms"),
  log: (opts: { situationId?: string; limit?: number } = {}) => {
    const params = new URLSearchParams();
    if (opts.situationId) params.set("situation_id", opts.situationId);
    if (opts.limit) params.set("limit", String(opts.limit));
    const query = params.toString();
    return req<LogEntryRow[]>(`/log${query ? `?${query}` : ""}`);
  },
  sensors: () => req<SensorRow[]>("/sensors"),
  areas: () => req<AreaRow[]>("/areas"),
  persons: () => req<PersonRow[]>("/persons"),
  demos: () => req<DemoRow[]>("/demos"),
  playDemo: (name: string) => req<{ demo: string }>(`/demos/${name}`, { method: "POST" }),
};

/** Runs `fn`, waits `intervalMs`, and repeats -- never overlapping two calls the way `setInterval` would if `fn`
 * is slower than the interval. Returns a function that stops the loop. */
export function poll(fn: () => void | Promise<void>, intervalMs: number): () => void {
  let stopped = false;
  let timer = 0;

  const tick = async () => {
    try {
      await fn();
    } catch (e) {
      console.error(e);
    }
    if (!stopped) timer = window.setTimeout(tick, intervalMs);
  };
  void tick();

  return () => {
    stopped = true;
    window.clearTimeout(timer);
  };
}
