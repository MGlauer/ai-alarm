// Shapes returned by ai_alarm.api (see its module docstring). Rows come back as they are stored: column names as
// keys, timestamps as ISO 8601 UTC strings. Fields not used by any page are left out.

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Scenario {
  id: string;
  status: "active" | "resolved";
  created_at: string;
  resolved_at: string | null;
  situation_ids: string[];
}

export interface AggregatedSummary {
  id: number;
  scenario_id: string;
  created_at: string;
  threat_score: number;
  summary: string;
  data: { situations: Record<string, number>; persons: Record<string, number> };
}

export interface ScenarioDetail extends Scenario {
  situations: SituationRow[];
  aggregated_summaries: AggregatedSummary[];
}

export interface SituationRow {
  id: string;
  scenario_id: string;
  area_id: string;
  status: "active" | "resolved";
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
}

export interface EventRow {
  id: string;
  situation_id: string;
  sensor_id: string;
  kind: "video" | "audio";
  start_time: string;
  evidence: string;
  bbox: BBox | null;
}

export interface PersonAssessment {
  object_id: string;
  identity: "known" | "unknown" | "not_decidable";
  person_id: string | null;
  name: string | null;
  cause: string | null;
  predicted_role: string | null;
  suspicion: number;
  invited: boolean;
  bbox: BBox | null;
}

export interface AnimalAssessment {
  label: string;
  danger: number;
  bbox: BBox | null;
}

export interface SummaryData {
  event_id: string;
  persons: PersonAssessment[];
  animals: AnimalAssessment[];
  weather_suspicion: number;
  unclear_situation: boolean;
}

export interface SummaryRow {
  id: number;
  situation_id: string;
  created_at: string;
  threat_score: number;
  summary: string;
  data: SummaryData;
}

export interface WarningRow {
  id: string;
  situation_id: string;
  area_id: string;
  cause: string;
  suspicion: number;
  colour: "yellow" | "orange";
  created_at: string;
  answer_deadline: string | null;
  status: "open" | "elevated" | "dismissed";
  resolved_by: "user" | "fallback_policy" | null;
  resolved_at: string | null;
}

export interface AlarmRow {
  id: string;
  situation_id: string;
  area_id: string;
  cause: string;
  origin: "rule" | "user_elevation" | "fallback_policy";
  warning_id: string | null;
  created_at: string;
}

export interface LogEntryRow {
  id: number;
  created_at: string;
  situation_id: string | null;
  kind: string;
  message: string;
}

export interface SituationDetail extends SituationRow {
  events: EventRow[];
  summaries: SummaryRow[];
  warnings: WarningRow[];
  alarms: AlarmRow[];
  log_entries: LogEntryRow[];
}

export interface SensorRow {
  id: string;
  kind: "camera" | "microphone";
  area_id: string;
  latest_event: EventRow | null;
}

export interface AreaRow {
  id: string;
  name: string;
  is_entryway: boolean;
  is_drop_off: boolean;
  people: string[];
}

export interface PersonRow {
  id: string;
  name: string | null;
  familiarity: number;
  suspicion: number;
  roles: string[];
  /** A character sprite representing them: their own identity if they are one of the named characters, else a
   * role-based or generic figure. Always present -- there is no "no picture" case. */
  picture_url: string;
}

export interface DemoRow {
  name: string;
  description: string;
}
