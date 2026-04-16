export interface Station {
  station_id: string;
  city: string;
  lat: number;
  lon: number;
  elevation_ft: number;
  model_grid_elevation_ft: number;
  lapse_rate_correction_f: number;
  flags: string[];
}

export interface Performance {
  total_pnl: number;
  win_rate: number;
  trade_count: number;
  signal_count: number;
  fill_rate: number | null;
  adverse_selection_ratio: number | null;
  being_picked_off: boolean;
}

export interface Signal {
  market_id: string;
  direction: string;
  action: string;
  edge: number;
  kelly_size: number;
  signal_timestamp: string;
  station_id: string;
  regime: string;
  regime_confidence: string;
  model_probability: number;
  market_probability: number;
  logged_at: string;
  skip_reason: string;
}

export interface Trade {
  trade_id: string;
  direction: string;
  question: string;
  city: string;
  resolution_date: string;
  temp_bucket_low: number | null;
  temp_bucket_high: number | null;
  entry_price: number;
  amount_usd: number;
  model_probability: number | null;
  resolved: boolean;
  outcome: boolean | null;
  pnl: number | null;
}

export interface CusumStatus {
  alarm: boolean;
  cusum_pos: number;
  cusum_neg: number;
  threshold: number;
  pct_of_threshold: number;
}

export interface ScheduleEvent {
  time: string;
  event_type: string;
  description: string;
}

export interface ReliabilityBin {
  bin_center: number;
  observed_frequency: number;
  count: number;
}

export interface Calibration {
  brier_score: number | null;
  brier_skill_score: number | null;
  reliability_diagram: ReliabilityBin[] | null;
  resolved_count: number;
}

export interface SchedulerStatus {
  running: boolean;
  last_run_time: string | null;
  last_run_result: Record<string, number> | null;
  last_error: string | null;
  next_event_type: string;
  next_event_time: string | null;
  next_event_minutes: number | null;
}

export interface BotStatus {
  scheduler: SchedulerStatus;
  signal_cache_age_seconds: number | null;
  signal_count: number;
}

export interface ExposureStatus {
  current_exposure_usd: number;
  realized_pnl: number;
  is_halted: boolean;
  bankroll: number;
  max_drawdown_pct: number;
  exposure_pct: number;
  drawdown_pct: number;
}

export interface CachedSignalData {
  token_id: string;
  station_id: string;
  city: string;
  question: string;
  temp_bucket: string;
  model_prob: number;
  regime: string;
  regime_confidence: string;
  active_flags: string[];
  ensemble_spread_pctile: number;
  forecast_time: string;
  forecast_age_s: number;
  live_bid: number | null;
  live_ask: number | null;
  live_mid: number | null;
  current_edge: number | null;
  resolution_date: string;
  hours_to_resolution: number;
}

export interface PriceMonitorStatus {
  running: boolean;
  pending_edges: Record<string, { first_seen: string; elapsed_s: number }>;
  cooldowns: Record<string, { expires: string; remaining_s: number }>;
  ws_connected: boolean;
  subscribed_tokens: number;
}

export interface ActivityEvent {
  event: string;
  data: Record<string, any>;
  timestamp: string;
}
