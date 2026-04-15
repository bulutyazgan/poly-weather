import type { Signal } from "../types";

interface Props {
  signals: Signal[] | null;
  loading: boolean;
}

export function SignalTable({ signals, loading }: Props) {
  if (loading) return <div className="card">Loading signals...</div>;
  if (!signals || signals.length === 0)
    return <div className="card"><h2>Signals</h2><p>No signals yet</p></div>;

  return (
    <div className="card">
      <h2>Recent Signals</h2>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Station</th>
              <th>Action</th>
              <th>Direction</th>
              <th>Edge</th>
              <th>Size ($)</th>
              <th>Regime</th>
              <th>Model P</th>
              <th>Market P</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((s, i) => (
              <tr key={i} className={s.action === "TRADE" ? "trade-row" : "skip-row"}>
                <td>{new Date(s.signal_timestamp).toLocaleTimeString()}</td>
                <td>{s.station_id}</td>
                <td>
                  <span className={`badge ${s.action.toLowerCase()}`}>
                    {s.action}
                  </span>
                </td>
                <td>{s.direction}</td>
                <td>{(s.edge * 100).toFixed(1)}%</td>
                <td>{s.kelly_size > 0 ? `$${s.kelly_size.toFixed(2)}` : "-"}</td>
                <td>
                  <span className={`badge confidence-${s.regime_confidence.toLowerCase()}`}>
                    {s.regime_confidence}
                  </span>
                </td>
                <td>{(s.model_probability * 100).toFixed(1)}%</td>
                <td>{(s.market_probability * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
