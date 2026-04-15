import type { CusumStatus } from "../types";

interface Props {
  data: CusumStatus | null;
  loading: boolean;
}

export function CusumIndicator({ data, loading }: Props) {
  if (loading || !data) return null;

  const level = data.alarm
    ? "alarm"
    : data.pct_of_threshold > 60
      ? "warning"
      : "ok";

  return (
    <div className={`cusum-indicator cusum-${level}`}>
      <span className="cusum-label">CUSUM</span>
      <span className="cusum-value">
        {data.alarm ? "ALARM" : `${data.pct_of_threshold.toFixed(0)}%`}
      </span>
      <div className="cusum-bar">
        <div
          className="cusum-fill"
          style={{ width: `${Math.min(data.pct_of_threshold, 100)}%` }}
        />
      </div>
    </div>
  );
}
