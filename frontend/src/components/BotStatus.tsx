import type { BotStatus, PriceMonitorStatus } from "../types";

interface Props {
  data: BotStatus | null;
  loading: boolean;
  priceMonitor?: PriceMonitorStatus | null;
}

function timeAgo(iso: string): string {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  return `${(diff / 3600).toFixed(1)}h ago`;
}

export function BotStatusBar({ data, loading, priceMonitor }: Props) {
  if (loading || !data) return null;

  const sched = data.scheduler;
  const alive = sched.running && !sched.last_error;
  const stale =
    data.signal_cache_age_seconds !== null && data.signal_cache_age_seconds > 7200;

  return (
    <div className={`bot-status ${alive ? "alive" : "unhealthy"}`}>
      <span className="status-dot" />
      <span>
        {sched.running ? "Scheduler running" : "Scheduler stopped"}
      </span>

      {sched.last_run_time && (
        <span className={`status-item ${stale ? "stale" : ""}`}>
          Last cycle: {timeAgo(sched.last_run_time)}
        </span>
      )}

      {sched.next_event_type && sched.next_event_minutes !== null && (
        <span className="status-item">
          Next: {sched.next_event_type} in {Math.round(sched.next_event_minutes)}m
        </span>
      )}

      {sched.last_run_result && (
        <span className="status-item">
          Signals: {sched.last_run_result.signals_generated} |
          Trades: {sched.last_run_result.trades_placed} |
          Errors: {sched.last_run_result.errors}
        </span>
      )}

      {sched.last_error && (
        <span className="status-item error">Error: {sched.last_error}</span>
      )}

      {priceMonitor && (
        <>
          <div className={`status-item ${priceMonitor.ws_connected ? "alive" : "error"}`}>
            <span className="status-label">WebSocket</span>
            <span className="status-value">
              {priceMonitor.ws_connected ? `Connected (${priceMonitor.subscribed_tokens} tokens)` : "Disconnected"}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Monitor</span>
            <span className="status-value">
              {Object.keys(priceMonitor.pending_edges).length} debouncing · {Object.keys(priceMonitor.cooldowns).length} cooldown
            </span>
          </div>
        </>
      )}
    </div>
  );
}
