interface Props {
  healthy: boolean | null;
  error: string | null;
}

export function StatusBar({ healthy, error }: Props) {
  return (
    <div className={`status-bar ${healthy ? "connected" : "disconnected"}`}>
      <span className="status-dot" />
      {healthy ? "Connected to API" : error ?? "Connecting..."}
    </div>
  );
}
