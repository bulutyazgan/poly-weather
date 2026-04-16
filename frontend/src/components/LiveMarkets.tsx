import type { CachedSignalData } from "../types";

interface Props {
  signals: CachedSignalData[] | null;
  loading: boolean;
}

function edgeClass(edge: number | null): string {
  if (edge === null) return "";
  if (edge > 0.08) return "edge-strong";
  if (edge > 0) return "edge-positive";
  return "edge-negative";
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

export function LiveMarkets({ signals, loading }: Props) {
  if (loading && !signals) return <div className="card">Loading markets...</div>;
  if (!signals || signals.length === 0) {
    return (
      <div className="card">
        <h2>Live Markets</h2>
        <p className="muted">No cached signals — waiting for pipeline cycle</p>
      </div>
    );
  }

  // Sort by absolute edge descending
  const sorted = [...signals].sort((a, b) => {
    const ae = Math.abs(a.current_edge ?? 0);
    const be = Math.abs(b.current_edge ?? 0);
    return be - ae;
  });

  return (
    <div className="card live-markets">
      <h2>Live Markets <span className="muted">({signals.length} contracts)</span></h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Station</th>
              <th>Bucket</th>
              <th>Model P</th>
              <th>Bid / Ask</th>
              <th>Edge</th>
              <th>Regime</th>
              <th>Flags</th>
              <th>Age</th>
              <th>Resolves</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <tr key={s.token_id} className={edgeClass(s.current_edge)}>
                <td>
                  <span className="station-city">{s.city}</span>
                </td>
                <td>{s.temp_bucket}°F</td>
                <td>{(s.model_prob * 100).toFixed(1)}%</td>
                <td>
                  {s.live_bid !== null && s.live_ask !== null
                    ? `${(s.live_bid * 100).toFixed(0)}c / ${(s.live_ask * 100).toFixed(0)}c`
                    : "—"}
                </td>
                <td className={edgeClass(s.current_edge)}>
                  {s.current_edge !== null
                    ? `${s.current_edge >= 0 ? "+" : ""}${(s.current_edge * 100).toFixed(1)}%`
                    : "—"}
                </td>
                <td>
                  <span className={`badge confidence-${s.regime_confidence.toLowerCase()}`}>
                    {s.regime_confidence}
                  </span>
                </td>
                <td>
                  {s.active_flags.length > 0
                    ? s.active_flags.map((f) => (
                        <span key={f} className="flag-badge">{f}</span>
                      ))
                    : "—"}
                </td>
                <td>{formatAge(s.forecast_age_s)}</td>
                <td>{s.hours_to_resolution.toFixed(0)}h</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
