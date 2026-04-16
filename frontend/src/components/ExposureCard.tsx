import type { ExposureStatus } from "../types";

interface Props {
  data: ExposureStatus | null;
  loading: boolean;
}

export function ExposureCard({ data, loading }: Props) {
  if (loading && !data) return <div className="card">Loading exposure...</div>;
  if (!data) return null;

  const drawdownMax = data.max_drawdown_pct * 100;
  const drawdownPct = Math.min(data.drawdown_pct, drawdownMax);
  const drawdownFill = drawdownMax > 0 ? (drawdownPct / drawdownMax) * 100 : 0;

  return (
    <div className={`card exposure-card ${data.is_halted ? "halted" : ""}`}>
      <h2>Exposure &amp; Risk</h2>
      {data.is_halted && (
        <div className="halt-banner">TRADING HALTED — Drawdown limit reached</div>
      )}
      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">Exposure</span>
          <span className="metric-value">
            ${data.current_exposure_usd.toFixed(2)}{" "}
            <span className="metric-sub">({data.exposure_pct.toFixed(1)}%)</span>
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Realized P&amp;L</span>
          <span className={`metric-value ${data.realized_pnl >= 0 ? "positive" : "negative"}`}>
            {data.realized_pnl >= 0 ? "+" : ""}${data.realized_pnl.toFixed(2)}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Bankroll</span>
          <span className="metric-value">${data.bankroll.toFixed(0)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Drawdown</span>
          <span className="metric-value">{data.drawdown_pct.toFixed(1)}% / {drawdownMax.toFixed(0)}%</span>
        </div>
      </div>
      <div className="drawdown-bar">
        <div
          className={`drawdown-fill ${drawdownFill > 60 ? "drawdown-warn" : ""} ${drawdownFill > 85 ? "drawdown-danger" : ""}`}
          style={{ width: `${Math.min(drawdownFill, 100)}%` }}
        />
      </div>
    </div>
  );
}
