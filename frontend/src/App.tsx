import { useState } from "react";
import { useApi } from "./hooks/useApi";
import { PerformanceCard } from "./components/PerformanceCard";
import { StationList } from "./components/StationList";
import { SignalTable } from "./components/SignalTable";
import { CalibrationChart } from "./components/CalibrationChart";
import { ScheduleView } from "./components/ScheduleView";
import { StatusBar } from "./components/StatusBar";
import { TradeTable } from "./components/TradeTable";
import type { Calibration, Performance, ScheduleEvent, Signal, Station, Trade } from "./types";
import "./App.css";

function App() {
  const [selectedStation, setSelectedStation] = useState<string | null>(null);

  const health = useApi<{ status: string }>("/health", 10000);
  const performance = useApi<Performance>("/api/performance", 15000);
  const stations = useApi<Station[]>("/api/stations", 60000);
  const calibration = useApi<Calibration>("/api/calibration", 30000);
  const schedule = useApi<ScheduleEvent[]>("/api/schedule", 60000);

  const trades = useApi<Trade[]>("/api/trades?limit=50", 15000);

  const signalPath = selectedStation
    ? `/api/signals?station=${selectedStation}&limit=50`
    : "/api/signals?limit=50";
  const signals = useApi<Signal[]>(signalPath, 15000);

  return (
    <div className="dashboard">
      <header>
        <h1>TradeBot Dashboard</h1>
        <p className="subtitle">Polymarket Weather Prediction System</p>
        <StatusBar
          healthy={health.data?.status === "ok"}
          error={health.error}
        />
      </header>

      <main>
        <div className="top-row">
          <PerformanceCard data={performance.data} loading={performance.loading} />
          <CalibrationChart data={calibration.data} loading={calibration.loading} />
        </div>

        <StationList
          stations={stations.data}
          selected={selectedStation}
          onSelect={setSelectedStation}
        />

        <TradeTable trades={trades.data} loading={trades.loading} />

        <SignalTable signals={signals.data} loading={signals.loading} />

        <ScheduleView events={schedule.data} />
      </main>

      <footer>
        <p>Paper Trading Mode | Auto-refreshing every 15s</p>
      </footer>
    </div>
  );
}

export default App;
