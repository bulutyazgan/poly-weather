"""Tests for SignalCache."""
from datetime import date, datetime, timezone

from src.data.models import MarketContract, RegimeClassification
from src.orchestrator.signal_cache import CachedSignal, SignalCache


def _make_cached_signal(
    token_id: str = "token_abc",
    model_prob: float = 0.40,
    station_id: str = "KNYC",
) -> CachedSignal:
    return CachedSignal(
        model_prob=model_prob,
        regime=RegimeClassification(
            station_id=station_id,
            valid_date=date(2026, 4, 15),
            regime="normal",
            confidence="HIGH",
            ensemble_spread_percentile=30.0,
        ),
        contract=MarketContract(
            token_id=token_id,
            no_token_id=f"no_{token_id}",
            condition_id="cond_1",
            question="Will NYC high be 72-73°F?",
            city="NYC",
            resolution_date=date(2026, 4, 16),
            end_date_utc=datetime(2026, 4, 16, 23, 59, 59, tzinfo=timezone.utc),
            temp_bucket_low=72.0,
            temp_bucket_high=73.0,
            outcome="Yes",
            volume_24h=5000.0,
        ),
        station_id=station_id,
        forecast_time=datetime(2026, 4, 15, 10, 30, tzinfo=timezone.utc),
    )


class TestSignalCache:
    def test_empty_cache_returns_none(self):
        cache = SignalCache()
        assert cache.get("nonexistent") is None

    def test_empty_cache_get_all_empty(self):
        cache = SignalCache()
        assert cache.get_all() == {}

    def test_update_and_get(self):
        cache = SignalCache()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        assert cache.get("tok_1") is sig

    def test_update_replaces_entirely(self):
        cache = SignalCache()
        sig1 = _make_cached_signal("tok_1")
        sig2 = _make_cached_signal("tok_2")
        cache.update({"tok_1": sig1})
        cache.update({"tok_2": sig2})
        assert cache.get("tok_1") is None
        assert cache.get("tok_2") is sig2

    def test_get_all_returns_copy(self):
        cache = SignalCache()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        all_signals = cache.get_all()
        all_signals["tok_99"] = sig
        assert cache.get("tok_99") is None

    def test_forecast_age_seconds(self):
        cache = SignalCache()
        sig = _make_cached_signal()
        cache.update({"tok_1": sig})
        age = cache.forecast_age_seconds
        assert 0.0 <= age < 2.0

    def test_forecast_age_before_any_update(self):
        cache = SignalCache()
        assert cache.forecast_age_seconds == float("inf")

    def test_updated_event_is_set_on_update(self):
        cache = SignalCache()
        assert not cache.updated.is_set()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        assert cache.updated.is_set()

    def test_updated_event_can_be_cleared_and_reset(self):
        cache = SignalCache()
        sig = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig})
        cache.updated.clear()
        assert not cache.updated.is_set()
        cache.update({"tok_2": _make_cached_signal("tok_2")})
        assert cache.updated.is_set()

    def test_no_token_reverse_lookup(self):
        """Looking up a NO token_id returns the parent YES token's cache entry."""
        cache = SignalCache()
        sig = _make_cached_signal("tok_yes")
        # The helper sets no_token_id to "no_tok_yes"
        cache.update({"tok_yes": sig})
        result = cache.get("no_tok_yes")
        assert result is sig

    def test_no_token_reverse_lookup_cleared_on_update(self):
        """Reverse lookup is rebuilt on each update — stale entries don't persist."""
        cache = SignalCache()
        sig1 = _make_cached_signal("tok_1")
        cache.update({"tok_1": sig1})
        assert cache.get("no_tok_1") is sig1

        sig2 = _make_cached_signal("tok_2")
        cache.update({"tok_2": sig2})
        # Old NO token mapping is gone
        assert cache.get("no_tok_1") is None
        assert cache.get("no_tok_2") is sig2
