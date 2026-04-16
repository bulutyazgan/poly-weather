interface Props {
  healthy: boolean | null;
  error: string | null;
  sseConnected?: boolean;
}

export function StatusBar({ healthy, error, sseConnected }: Props) {
  return (
    <div className={`status-bar ${healthy ? "connected" : "disconnected"}`}>
      <span className="status-dot" />
      {healthy ? "Connected to API" : error ?? "Connecting..."}
      {sseConnected !== undefined && (
        <span className={`status-badge ${sseConnected ? "connected" : "disconnected"}`}>
          {sseConnected ? "Live Feed Connected" : "Live Feed Disconnected"}
        </span>
      )}
    </div>
  );
}
