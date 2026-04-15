"""Tests for ExposureTracker."""
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
