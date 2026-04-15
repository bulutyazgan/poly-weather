import type { Performance } from "../types";

interface Props {
  data: Performance | null;
  loading: boolean;
}

export function PerformanceCard({ data, loading }: Props) {
  if (loading) return <div className="card">Loading performance...</div>;
  if (!data) return <div className="card">No performance data</div>;

  const pnlClass = data.total_pnl >= 0 ? "positive" : "negative";

  return (
    <div className="card">
      <h2>Performance</h2>
      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">Total P&L</span>
          <span className={`metric-value ${pnlClass}`}>
            ${data.total_pnl.toFixed(2)}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Win Rate</span>
          <span className="metric-value">
            {(data.win_rate * 100).toFixed(1)}%
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Trades</span>
          <span className="metric-value">{data.trade_count}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Signals</span>
          <span className="metric-value">{data.signal_count}</span>
        </div>
      </div>
    </div>
  );
}
