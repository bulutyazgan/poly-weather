import { useCallback, useRef, useState } from "react";
import { useApi } from "./hooks/useApi";
import { useSSE } from "./hooks/useSSE";
import { PerformanceCard } from "./components/PerformanceCard";
import { StationList } from "./components/StationList";
import { SignalTable } from "./components/SignalTable";
import { CalibrationChart } from "./components/CalibrationChart";
import { CusumIndicator } from "./components/CusumIndicator";
import { ScheduleView } from "./components/ScheduleView";
import { BotStatusBar } from "./components/BotStatus";
import { StatusBar } from "./components/StatusBar";
import { TradeTable } from "./components/TradeTable";
import { ActivityFeed } from "./components/ActivityFeed";
import { ExposureCard } from "./components/ExposureCard";
import { LiveMarkets } from "./components/LiveMarkets";
import type {
  ActivityEvent,
  BotStatus,
  CachedSignalData,
  Calibration,
  CusumStatus,
  ExposureStatus,
  Performance,
  PriceMonitorStatus,
  ScheduleEvent,
  Signal,
  Station,
  Trade,
} from "./types";
import "./App.css";

const MAX_FEED_EVENTS = 50;

function App() {
  const [selectedStation, setSelectedStation] = useState<string | null>(null);
  const [activityEvents, setActivityEvents] = useState<ActivityEvent[]>([]);
  const [lastHeartbeat, setLastHeartbeat] = useState<string | null>(null);

  // REST polling for base data
  const health = useApi<{ status: string }>("/health", 10000);
  const botStatus = useApi<BotStatus>("/api/status", 30000);
  const performance = useApi<Performance>("/api/performance", 30000);
  const stations = useApi<Station[]>("/api/stations", 300000);
  const calibration = useApi<Calibration>("/api/calibration", 60000);
  const cusum = useApi<CusumStatus>("/api/cusum", 30000);
  const schedule = useApi<ScheduleEvent[]>("/api/schedule", 300000);
  const trades = useApi<Trade[]>("/api/trades?limit=50", 30000);
  const exposure = useApi<ExposureStatus>("/api/exposure", 30000);
  const cachedSignals = useApi<CachedSignalData[]>("/api/cached-signals", 30000);
  const priceMonitor = useApi<PriceMonitorStatus>("/api/price-monitor", 15000);

  const signalPath = selectedStation
    ? `/api/signals?station=${selectedStation}&limit=50`
    : "/api/signals?limit=50";
  const signals = useApi<Signal[]>(signalPath, 30000);

  // Stable refs for SSE-driven state updates
  const exposureRef = useRef(exposure);
  exposureRef.current = exposure;
  const cusumRef = useRef(cusum);
  cusumRef.current = cusum;

  // Push activity event to feed
  const pushEvent = useCallback((evt: ActivityEvent) => {
    setActivityEvents((prev) => {
      const next = [...prev, evt];
      return next.length > MAX_FEED_EVENTS ? next.slice(-MAX_FEED_EVENTS) : next;
    });
  }, []);

  // SSE event handlers
  const sseHandlers = {
    edge_eval: (msg: any) => {
      pushEvent(msg);
    },
    trade_executed: (msg: any) => {
      pushEvent(msg);
      // Trigger refetch of trades and performance
      trades.refetch();
      performance.refetch();
    },
    trade_resolved: (msg: any) => {
      pushEvent(msg);
      // Trigger refetch of trades, performance, and calibration
      trades.refetch();
      performance.refetch();
      calibration.refetch();
    },
    exposure_change: (msg: any) => {
      pushEvent(msg);
      exposure.refetch();
    },
    pipeline_start: (msg: any) => {
      pushEvent(msg);
    },
    pipeline_complete: (msg: any) => {
      pushEvent(msg);
      // Full refresh after pipeline cycle
      botStatus.refetch();
      signals.refetch();
      cachedSignals.refetch();
      performance.refetch();
      priceMonitor.refetch();
    },
    cusum_update: (msg: any) => {
      pushEvent(msg);
      cusum.refetch();
    },
    ws_status: (msg: any) => {
      pushEvent(msg);
      priceMonitor.refetch();
    },
    heartbeat: (msg: any) => {
      setLastHeartbeat(msg.timestamp || msg.data?.ts || new Date().toISOString());
    },
  };

  const { connected: sseConnected } = useSSE(sseHandlers);

  return (
    <div className="dashboard">
      <header>
        <h1>TradeBot Dashboard</h1>
        <p className="subtitle">Polymarket Weather Prediction System</p>
        <StatusBar
          healthy={health.data?.status === "ok"}
          error={health.error}
          sseConnected={sseConnected}
        />
        <BotStatusBar
          data={botStatus.data}
          loading={botStatus.loading}
          priceMonitor={priceMonitor.data}
        />
      </header>

      <main>
        <div className="top-row">
          <div>
            <ExposureCard data={exposure.data} loading={exposure.loading} />
            <CusumIndicator data={cusum.data} loading={cusum.loading} />
          </div>
          <div>
            <PerformanceCard data={performance.data} loading={performance.loading} />
          </div>
        </div>

        <LiveMarkets signals={cachedSignals.data} loading={cachedSignals.loading} />

        <ActivityFeed events={activityEvents} />

        <StationList
          stations={stations.data}
          selected={selectedStation}
          onSelect={setSelectedStation}
        />

        <TradeTable trades={trades.data} loading={trades.loading} />

        <SignalTable signals={signals.data} loading={signals.loading} />

        <div className="top-row">
          <CalibrationChart data={calibration.data} loading={calibration.loading} />
          <ScheduleView events={schedule.data} />
        </div>
      </main>

      <footer>
        <p>
          Paper Trading Mode | {sseConnected ? "Live Feed Active" : "Live Feed Disconnected"}
          {lastHeartbeat && ` | Last heartbeat: ${Math.round((Date.now() - new Date(lastHeartbeat).getTime()) / 1000)}s ago`}
        </p>
      </footer>
    </div>
  );
}

export default App;
