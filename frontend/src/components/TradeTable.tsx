import type { Trade } from "../types";

interface Props {
  trades: Trade[] | null;
  loading: boolean;
}

function formatBucket(low: number | null, high: number | null): string {
  if (low == null || low === -Infinity) return `${high ?? "?"}°F or below`;
  if (high == null || high === Infinity) return `${low}°F or above`;
  return `${low}-${high}°F`;
}

export function TradeTable({ trades, loading }: Props) {
  if (loading) return <div className="card">Loading trades...</div>;
  if (!trades || trades.length === 0)
    return <div className="card"><h2>Paper Trades</h2><p>No trades yet</p></div>;

  const resolved = trades.filter((t) => t.resolved);
  const pending = trades.filter((t) => !t.resolved);

  return (
    <div className="card">
      <h2>Paper Trades</h2>
      {pending.length > 0 && (
        <>
          <h3>Pending ({pending.length})</h3>
          <TradeRows trades={pending} />
        </>
      )}
      {resolved.length > 0 && (
        <>
          <h3>Resolved ({resolved.length})</h3>
          <TradeRows trades={resolved} />
        </>
      )}
    </div>
  );
}

function TradeRows({ trades }: { trades: Trade[] }) {
  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>City</th>
            <th>Bucket</th>
            <th>Dir</th>
            <th>Entry</th>
            <th>Size</th>
            <th>Model P</th>
            <th>Status</th>
            <th>PnL</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr
              key={t.trade_id}
              className={
                t.resolved
                  ? t.pnl !== null && t.pnl > 0
                    ? "trade-win"
                    : "trade-loss"
                  : ""
              }
            >
              <td>{t.city}</td>
              <td>{formatBucket(t.temp_bucket_low, t.temp_bucket_high)}</td>
              <td>
                <span className={`badge ${t.direction === "BUY_YES" ? "buy-yes" : "buy-no"}`}>
                  {t.direction === "BUY_YES" ? "YES" : "NO"}
                </span>
              </td>
              <td>{(t.entry_price * 100).toFixed(0)}c</td>
              <td>${t.amount_usd.toFixed(2)}</td>
              <td>{t.model_probability !== null ? `${(t.model_probability * 100).toFixed(1)}%` : "-"}</td>
              <td>
                {t.resolved ? (
                  <span className={`badge ${t.outcome ? "outcome-yes" : "outcome-no"}`}>
                    {t.outcome ? "YES" : "NO"}
                  </span>
                ) : (
                  <span className="badge pending">PENDING</span>
                )}
              </td>
              <td className={t.pnl !== null ? (t.pnl > 0 ? "positive" : "negative") : ""}>
                {t.pnl !== null ? `${t.pnl > 0 ? "+" : ""}$${t.pnl.toFixed(2)}` : "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
