"""Tests for ExposureTracker."""
import pytest

from src.trading.exposure_tracker import ExposureTracker


class TestExposureTracker:
    def test_initial_exposure_defaults_to_zero(self):
        tracker = ExposureTracker()
        assert tracker.current == 0.0

    def test_initial_exposure_custom(self):
        tracker = ExposureTracker(initial=50.0)
        assert tracker.current == 50.0

    def test_add_increases_exposure(self):
        tracker = ExposureTracker()
        tracker.add(10.0)
        tracker.add(5.0)
        assert tracker.current == 15.0

    def test_reset_sets_to_value(self):
        tracker = ExposureTracker(initial=100.0)
        tracker.reset(25.0)
        assert tracker.current == 25.0

    def test_add_after_reset(self):
        tracker = ExposureTracker(initial=100.0)
        tracker.reset(0.0)
        tracker.add(7.0)
        assert tracker.current == 7.0


class TestDrawdownCircuitBreaker:
    """Tests for drawdown-based trading halt."""

    def test_not_halted_initially(self):
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        assert tracker.is_halted is False
        assert tracker.realized_pnl == 0.0

    def test_not_halted_after_wins(self):
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        tracker.record_pnl(5.0)
        tracker.record_pnl(3.0)
        assert tracker.realized_pnl == pytest.approx(8.0)
        assert tracker.is_halted is False

    def test_not_halted_small_loss(self):
        """Loss within threshold -> still trading."""
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        # 15% of 300 = $45 threshold
        tracker.record_pnl(-40.0)
        assert tracker.is_halted is False

    def test_halted_at_drawdown_limit(self):
        """Loss exceeding 15% of $300 = $45 -> halted."""
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        tracker.record_pnl(-46.0)
        assert tracker.is_halted is True

    def test_halted_after_cumulative_losses(self):
        """Multiple small losses accumulate past threshold."""
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        # 10 losses of $5 = -$50 > $45 threshold
        for _ in range(10):
            tracker.record_pnl(-5.0)
        assert tracker.realized_pnl == pytest.approx(-50.0)
        assert tracker.is_halted is True

    def test_wins_offset_losses(self):
        """Wins reduce cumulative loss, keeping below threshold."""
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        tracker.record_pnl(-30.0)  # -30
        tracker.record_pnl(10.0)   # -20
        tracker.record_pnl(-20.0)  # -40
        assert tracker.realized_pnl == pytest.approx(-40.0)
        assert tracker.is_halted is False  # -40 > -45 threshold

    def test_reset_halt_clears_state(self):
        """Manual reset allows trading to resume."""
        tracker = ExposureTracker(bankroll=300.0, max_drawdown_pct=0.15)
        tracker.record_pnl(-50.0)
        assert tracker.is_halted is True
        tracker.reset_halt()
        assert tracker.is_halted is False
        assert tracker.realized_pnl == 0.0

    def test_custom_drawdown_pct(self):
        """10% drawdown limit on $100 bankroll = $10 threshold."""
        tracker = ExposureTracker(bankroll=100.0, max_drawdown_pct=0.10)
        tracker.record_pnl(-9.0)
        assert tracker.is_halted is False
        tracker.record_pnl(-2.0)  # total -11 > $10 threshold
        assert tracker.is_halted is True

    def test_record_pnl_without_amount_leaves_exposure(self):
        """record_pnl with no amount_usd only affects drawdown, not exposure."""
        tracker = ExposureTracker(bankroll=300.0, initial=10.0, max_drawdown_pct=0.15)
        tracker.add(5.0)
        assert tracker.current == 15.0
        tracker.record_pnl(-50.0)
        assert tracker.is_halted is True
        assert tracker.current == 15.0  # no amount_usd → exposure unchanged


class TestExposureRelease:
    """Resolved trades must release their capital from exposure."""

    def test_record_pnl_releases_exposure(self):
        """Resolving a $10 trade releases $10 of exposure."""
        tracker = ExposureTracker()
        tracker.add(10.0)
        assert tracker.current == 10.0
        tracker.record_pnl(pnl=5.0, amount_usd=10.0)
        assert tracker.current == 0.0

    def test_multiple_trades_partial_release(self):
        """Two trades, one resolves → only that trade's exposure released."""
        tracker = ExposureTracker()
        tracker.add(10.0)  # trade A
        tracker.add(8.0)   # trade B
        assert tracker.current == 18.0
        tracker.record_pnl(pnl=2.0, amount_usd=10.0)  # trade A resolves
        assert tracker.current == pytest.approx(8.0)

    def test_exposure_does_not_go_negative(self):
        """Exposure floors at zero even if over-released."""
        tracker = ExposureTracker()
        tracker.add(5.0)
        tracker.record_pnl(pnl=0.0, amount_usd=10.0)
        assert tracker.current == 0.0

    def test_full_cycle_trade_and_resolve(self):
        """Full lifecycle: trade → exposure up → resolve → exposure back to zero."""
        tracker = ExposureTracker(bankroll=300.0)
        tracker.add(3.0)   # place trade
        tracker.add(2.5)   # place another
        assert tracker.current == 5.5
        tracker.record_pnl(pnl=1.0, amount_usd=3.0)   # first resolves win
        tracker.record_pnl(pnl=-2.5, amount_usd=2.5)   # second resolves loss
        assert tracker.current == 0.0
        assert tracker.realized_pnl == pytest.approx(-1.5)


class TestEventBusIntegration:
    """ExposureTracker publishes events when event_bus is provided."""

    def test_add_publishes_exposure_change(self):
        from src.api.event_bus import EventBus
        bus = EventBus()
        q = bus.subscribe()
        tracker = ExposureTracker(bankroll=300.0, event_bus=bus)
        tracker.add(10.0)
        assert not q.empty()
        msg = q.get_nowait()
        assert msg["event"] == "exposure_change"
        assert msg["data"]["current_exposure"] == 10.0

    def test_record_pnl_publishes_exposure_change(self):
        from src.api.event_bus import EventBus
        bus = EventBus()
        q = bus.subscribe()
        tracker = ExposureTracker(bankroll=300.0, event_bus=bus)
        tracker.add(10.0)
        q.get_nowait()  # discard add event
        tracker.record_pnl(-5.0, amount_usd=10.0)
        msg = q.get_nowait()
        assert msg["event"] == "exposure_change"
        assert msg["data"]["realized_pnl"] == -5.0

    def test_no_event_bus_still_works(self):
        """Without event_bus, ExposureTracker works normally."""
        tracker = ExposureTracker(bankroll=300.0)
        tracker.add(10.0)
        tracker.record_pnl(-5.0)
        assert tracker.current == 10.0
        assert tracker.realized_pnl == -5.0
