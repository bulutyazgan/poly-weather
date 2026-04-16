import type { ScheduleEvent } from "../types";

interface Props {
  events: ScheduleEvent[] | null;
}

const EVENT_ICONS: Record<string, string> = {
  gfs_update: "GFS",
  ecmwf_update: "ECMWF",
  morning_refinement: "HRRR",
  resolution_check: "RESOLVE",
};

export function ScheduleView({ events }: Props) {
  if (!events || events.length === 0) return null;

  return (
    <div className="card">
      <h2>Schedule</h2>
      <div className="schedule-list">
        {events.map((e, i) => (
          <div key={i} className="schedule-item">
            <span className={`schedule-badge ${e.event_type}`}>
              {EVENT_ICONS[e.event_type] ?? e.event_type}
            </span>
            <span className="schedule-time">{e.time} UTC</span>
            <span className="schedule-desc">{e.description}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
