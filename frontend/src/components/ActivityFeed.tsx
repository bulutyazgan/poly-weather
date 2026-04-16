import { useEffect, useRef } from "react";
import type { ActivityEvent } from "../types";

interface Props {
  events: ActivityEvent[];
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return "";
  }
}

function formatEvent(evt: ActivityEvent): string {
  const t = formatTime(evt.timestamp);
  const d = evt.data;

  switch (evt.event) {
    case "edge_eval":
      return `${t} ${d.station_id || ""} ${d.direction || ""} edge=${((d.edge || 0) * 100).toFixed(1)}% → ${d.action || ""}${d.skip_reason ? ` (${d.skip_reason})` : ""}`;
    case "trade_executed":
      return `${t} TRADE ${d.city || ""} ${d.direction || ""} $${(d.amount_usd || 0).toFixed(2)} @ ${((d.price || 0) * 100).toFixed(0)}c`;
    case "trade_resolved":
      return `${t} RESOLVED ${d.city || ""} ${d.outcome ? "YES" : "NO"} ${(d.pnl || 0) >= 0 ? "+" : ""}$${(d.pnl || 0).toFixed(2)}`;
    case "pipeline_start":
      return `${t} Pipeline starting (${d.event_type || "cycle"})`;
    case "pipeline_complete":
      return `${t} Pipeline done: ${d.signals_generated || 0} signals, ${d.trades_placed || 0} trades`;
    case "cusum_update":
      return `${t} CUSUM ${d.alarm ? "ALARM" : `${(d.pct_of_threshold || 0).toFixed(0)}%`}`;
    case "ws_status":
      return `${t} WebSocket ${d.connected ? "connected" : "disconnected"} (${d.token_count || 0} tokens)`;
    case "exposure_change":
      return `${t} Exposure: $${(d.current_exposure || 0).toFixed(2)} | P&L: $${(d.realized_pnl || 0).toFixed(2)}${d.is_halted ? " [HALTED]" : ""}`;
    default:
      return `${t} ${evt.event}`;
  }
}

function eventClass(evt: ActivityEvent): string {
  switch (evt.event) {
    case "trade_executed":
      return "feed-trade";
    case "trade_resolved":
      return (evt.data.pnl || 0) >= 0 ? "feed-win" : "feed-loss";
    case "pipeline_start":
    case "pipeline_complete":
      return "feed-pipeline";
    case "ws_status":
      return evt.data.connected ? "feed-ok" : "feed-warn";
    case "cusum_update":
      return evt.data.alarm ? "feed-warn" : "";
    case "exposure_change":
      return evt.data.is_halted ? "feed-warn" : "";
    default:
      return "";
  }
}

export function ActivityFeed({ events }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div className="card activity-feed">
      <h2>Activity Feed</h2>
      <div className="feed-scroll">
        {events.length === 0 && (
          <div className="feed-empty">Waiting for events...</div>
        )}
        {events.map((evt, i) => (
          <div key={i} className={`feed-line ${eventClass(evt)}`}>
            {formatEvent(evt)}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
