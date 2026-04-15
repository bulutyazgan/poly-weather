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
}

export interface Trade {
  trade_id: string;
  direction: string;
  question: string;
  city: string;
  resolution_date: string;
  temp_bucket_low: number;
  temp_bucket_high: number;
  entry_price: number;
  amount_usd: number;
  model_probability: number | null;
  resolved: boolean;
  outcome: boolean | null;
  pnl: number | null;
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
