import { useEffect, useRef, useState } from "react";

type SSEHandler = (data: any) => void;

/**
 * Hook for consuming Server-Sent Events from /api/events.
 * Uses browser-native EventSource with automatic reconnection.
 */
export function useSSE(handlers: Record<string, SSEHandler>) {
  const [connected, setConnected] = useState(false);
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    const es = new EventSource("/api/events");

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    const eventNames = [
      "edge_eval",
      "trade_executed",
      "trade_resolved",
      "exposure_change",
      "pipeline_start",
      "pipeline_complete",
      "cusum_update",
      "ws_status",
      "heartbeat",
    ];

    for (const name of eventNames) {
      es.addEventListener(name, (e: MessageEvent) => {
        try {
          const parsed = JSON.parse(e.data);
          handlersRef.current[name]?.(parsed);
        } catch {
          // ignore parse errors
        }
      });
    }

    return () => es.close();
  }, []);

  return { connected };
}
