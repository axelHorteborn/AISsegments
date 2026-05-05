"""Tests for Track / Segment dataclasses and the to_segments helper."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aissegments import Segment, Track, to_segments


class TestTrackConstruction:
    def test_from_arrays_coerces_lists(self):
        track = Track.from_arrays(
            mmsi=12345,
            t=[0, 1, 2],
            lon=[10.0, 10.1, 10.2],
            lat=[55.0, 55.0, 55.0],
            sog=[10.0, 10.0, 10.0],
            cog=[90.0, 90.0, 90.0],
        )
        assert track.mmsi == 12345
        assert len(track) == 3
        assert track.t.dtype == np.float64
        assert track.lon.dtype == np.float64

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length 4, expected 3"):
            Track(
                mmsi=1,
                t=np.array([0.0, 1.0, 2.0]),
                lon=np.array([10.0, 10.1, 10.2, 10.3]),
                lat=np.array([55.0, 55.0, 55.0]),
                sog=np.array([10.0, 10.0, 10.0]),
                cog=np.array([90.0, 90.0, 90.0]),
            )

    def test_non_monotonic_time_raises(self):
        with pytest.raises(ValueError, match="monotonically non-decreasing"):
            Track.from_arrays(
                mmsi=1,
                t=[0.0, 2.0, 1.0],
                lon=[0.0, 0.1, 0.2],
                lat=[55.0, 55.0, 55.0],
                sog=[10.0, 10.0, 10.0],
                cog=[90.0, 90.0, 90.0],
            )

    def test_single_point_track_is_valid(self):
        track = Track.from_arrays(
            mmsi=1, t=[0.0], lon=[10.0], lat=[55.0], sog=[10.0], cog=[90.0]
        )
        assert len(track) == 1

    def test_empty_track_is_valid(self):
        track = Track.from_arrays(mmsi=1, t=[], lon=[], lat=[], sog=[], cog=[])
        assert len(track) == 0


class TestTrackTake:
    def test_take_subset(self):
        track = Track.from_arrays(
            mmsi=1,
            t=[0, 1, 2, 3, 4],
            lon=[0.0, 0.1, 0.2, 0.3, 0.4],
            lat=[55.0] * 5,
            sog=[10.0] * 5,
            cog=[90.0] * 5,
        )
        sub = track.take([0, 2, 4])
        assert len(sub) == 3
        assert sub.mmsi == 1
        np.testing.assert_array_equal(sub.t, [0.0, 2.0, 4.0])

    def test_take_preserves_order(self):
        track = Track.from_arrays(
            mmsi=1,
            t=[0, 1, 2, 3, 4],
            lon=[0.0, 0.1, 0.2, 0.3, 0.4],
            lat=[55.0] * 5,
            sog=[10.0] * 5,
            cog=[90.0] * 5,
        )
        # take() does not re-sort; the caller is expected to pass sorted indices.
        sub = track.take([0, 1, 4])
        np.testing.assert_array_equal(sub.t, [0.0, 1.0, 4.0])


class TestSegment:
    def test_segment_default_n_points(self):
        seg = Segment(
            mmsi=1,
            t_start=0.0, t_end=60.0,
            lon_start=0.0, lat_start=55.0, lon_end=0.1, lat_end=55.0,
            cog_mean=90.0, sog_mean=10.0,
        )
        assert seg.n_points == 2

    def test_segment_is_frozen(self):
        seg = Segment(
            mmsi=1,
            t_start=0.0, t_end=60.0,
            lon_start=0.0, lat_start=55.0, lon_end=0.1, lat_end=55.0,
            cog_mean=90.0, sog_mean=10.0,
        )
        with pytest.raises(FrozenInstanceError):
            seg.mmsi = 2  # type: ignore[misc]


class TestToSegments:
    def test_empty_track_yields_no_segments(self):
        track = Track.from_arrays(mmsi=1, t=[], lon=[], lat=[], sog=[], cog=[])
        assert to_segments(track) == []

    def test_single_point_yields_no_segments(self):
        track = Track.from_arrays(
            mmsi=1, t=[0.0], lon=[0.0], lat=[55.0], sog=[10.0], cog=[90.0]
        )
        assert to_segments(track) == []

    def test_pairs_consecutive_points(self):
        track = Track.from_arrays(
            mmsi=42,
            t=[0, 60, 120],
            lon=[0.0, 0.001, 0.002],
            lat=[55.0, 55.0, 55.0],
            sog=[10.0, 10.0, 10.0],
            cog=[90.0, 90.0, 90.0],
        )
        segs = to_segments(track)
        assert len(segs) == 2
        assert all(s.mmsi == 42 for s in segs)
        assert all(s.n_points == 2 for s in segs)
        assert segs[0].t_start == 0.0 and segs[0].t_end == 60.0
        assert segs[1].t_start == 60.0 and segs[1].t_end == 120.0

    def test_cog_wraps_at_360(self):
        """Mean of 350° and 10° should be 0° (or 360°) — not 180°."""
        track = Track.from_arrays(
            mmsi=1,
            t=[0, 60],
            lon=[0.0, 0.001],
            lat=[55.0, 55.0],
            sog=[10.0, 10.0],
            cog=[350.0, 10.0],
        )
        seg = to_segments(track)[0]
        # ((10 - 350 + 180) mod 360) - 180 = -340 + 180 = -160? Let me recompute.
        # diff = ((10 - 350 + 180) % 360) - 180 = ((-160) % 360) - 180 = 200 - 180 = 20
        # mean = (350 + 20/2) % 360 = 360 % 360 = 0
        assert seg.cog_mean == pytest.approx(0.0, abs=1e-6)
