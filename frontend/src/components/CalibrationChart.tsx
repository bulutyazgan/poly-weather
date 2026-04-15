import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { Calibration } from "../types";

interface Props {
  data: Calibration | null;
  loading: boolean;
}

export function CalibrationChart({ data, loading }: Props) {
  if (loading) return <div className="card">Loading calibration...</div>;

  const hasMetrics = data && data.brier_score !== null;

  return (
    <div className="card">
      <h2>Calibration</h2>
      {!hasMetrics ? (
        <p className="muted">
          Requires 10+ resolved trades. Currently: {data?.resolved_count ?? 0}
        </p>
      ) : (
        <>
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">Brier Score</span>
              <span className="metric-value">
                {data.brier_score!.toFixed(4)}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Brier Skill Score</span>
              <span className={`metric-value ${data.brier_skill_score! > 0 ? "positive" : "negative"}`}>
                {data.brier_skill_score!.toFixed(4)}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Resolved Trades</span>
              <span className="metric-value">{data.resolved_count}</span>
            </div>
          </div>

          {data.reliability_diagram && data.reliability_diagram.length > 0 && (
            <>
              <h3>Reliability Diagram</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data.reliability_diagram}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    dataKey="bin_center"
                    label={{ value: "Forecast Probability", position: "bottom" }}
                    stroke="#888"
                  />
                  <YAxis
                    label={{
                      value: "Observed Frequency",
                      angle: -90,
                      position: "insideLeft",
                    }}
                    stroke="#888"
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1a1a2e", border: "1px solid #333" }}
                  />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                    stroke="#555"
                    strokeDasharray="5 5"
                    label="Perfect"
                  />
                  <Line
                    type="monotone"
                    dataKey="observed_frequency"
                    stroke="#4fc3f7"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </>
          )}
        </>
      )}
    </div>
  );
}
