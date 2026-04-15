import type { Station } from "../types";

interface Props {
  stations: Station[] | null;
  selected: string | null;
  onSelect: (id: string | null) => void;
}

export function StationList({ stations, selected, onSelect }: Props) {
  if (!stations) return null;

  return (
    <div className="card">
      <h2>Stations</h2>
      <div className="station-chips">
        <button
          className={`chip ${selected === null ? "active" : ""}`}
          onClick={() => onSelect(null)}
        >
          All
        </button>
        {stations.map((s) => (
          <button
            key={s.station_id}
            className={`chip ${selected === s.station_id ? "active" : ""}`}
            onClick={() => onSelect(s.station_id)}
          >
            {s.city} ({s.station_id})
          </button>
        ))}
      </div>
    </div>
  );
}
