"""End-to-end tests on real AIS data from the ``aisdb`` package.

These exercise the full pipeline ``aisdb decoder → TrackGen → from_aisdb_track
→ tdkc/tdkc_segments`` against the 2,372-message corpus shipped with aisdb's
test suite.  Synthetic tests in ``test_tdkc.py`` validate algorithm
correctness; this file validates the *integration* — that the dtypes,
field names, and value ranges produced by aisdb compose with aissegments
without surprises.
"""
from __future__ import annotations

import numpy as np
import pytest

from aissegments import Segment, Track, tdkc, tdkc_segments
from aissegments.adapters import from_aisdb_track

pytestmark = pytest.mark.aisdb


class TestRealAisCorpus:
    def test_corpus_yields_at_least_one_track(self, aisdb_tracks: list[dict]):
        assert len(aisdb_tracks) >= 1

    def test_track_dicts_have_required_keys(self, aisdb_tracks: list[dict]):
        for t in aisdb_tracks[:5]:
            for key in ("mmsi", "time", "lon", "lat", "sog", "cog"):
                assert key in t, f"missing {key!r} in aisdb track dict"


class TestAdapterOnRealData:
    def test_round_trips_first_few(self, aisdb_tracks: list[dict]):
        for raw in aisdb_tracks[:5]:
            track = from_aisdb_track(raw)
            assert isinstance(track, Track)
            assert track.mmsi == int(raw["mmsi"])
            assert len(track) == len(raw["time"])

    def test_dtypes_are_float64(self, aisdb_tracks: list[dict]):
        # aisdb returns time as uint32 and coords as float32; our adapter
        # must coerce to the float64 contract that TDKC expects.
        track = from_aisdb_track(aisdb_tracks[0])
        for arr in (track.t, track.lon, track.lat, track.sog, track.cog):
            assert arr.dtype == np.float64

    def test_time_is_monotonic(self, aisdb_tracks: list[dict]):
        for raw in aisdb_tracks[:10]:
            track = from_aisdb_track(raw)
            if len(track) > 1:
                assert np.all(np.diff(track.t) >= 0)


class TestTdkcOnRealTrack:
    def test_compresses_long_track(self, aisdb_multi_point_track: dict):
        track = from_aisdb_track(aisdb_multi_point_track)
        compressed = tdkc(track)
        assert 2 <= len(compressed) <= len(track)

    def test_endpoints_are_preserved(self, aisdb_multi_point_track: dict):
        track = from_aisdb_track(aisdb_multi_point_track)
        compressed = tdkc(track)
        assert compressed.t[0] == track.t[0]
        assert compressed.t[-1] == track.t[-1]
        assert compressed.lon[0] == track.lon[0]
        assert compressed.lon[-1] == track.lon[-1]

    def test_compressed_time_is_monotonic(self, aisdb_multi_point_track: dict):
        track = from_aisdb_track(aisdb_multi_point_track)
        compressed = tdkc(track)
        assert np.all(np.diff(compressed.t) >= 0)

    def test_paper_faithful_threshold_keeps_more_points(self, aisdb_multi_point_track: dict):
        """min_sed_m=0 (paper-faithful) should keep at least as many as the default."""
        track = from_aisdb_track(aisdb_multi_point_track)
        n_default = len(tdkc(track))
        n_paper = len(tdkc(track, min_sed_m=0.0, min_svd_kn=0.0))
        assert n_paper >= n_default

    def test_high_floor_compresses_aggressively(self, aisdb_multi_point_track: dict):
        """A very high SED floor should produce fewer key points than the default."""
        track = from_aisdb_track(aisdb_multi_point_track)
        n_default = len(tdkc(track))
        n_high = len(tdkc(track, min_sed_m=10_000.0, min_svd_kn=100.0))
        assert n_high <= n_default


class TestTdkcSegmentsOnRealTrack:
    def test_segments_cover_input(self, aisdb_multi_point_track: dict):
        track = from_aisdb_track(aisdb_multi_point_track)
        segs = tdkc_segments(track)
        assert all(isinstance(s, Segment) for s in segs)
        # Sum of (n_points - 1) across segments == (len(track) - 1).
        total = sum(s.n_points - 1 for s in segs)
        assert total == len(track) - 1

    def test_segments_are_time_contiguous(self, aisdb_multi_point_track: dict):
        track = from_aisdb_track(aisdb_multi_point_track)
        segs = tdkc_segments(track)
        from itertools import pairwise

        for a, b in pairwise(segs):
            assert a.t_end == b.t_start

    def test_segments_preserve_input_endpoints(self, aisdb_multi_point_track: dict):
        track = from_aisdb_track(aisdb_multi_point_track)
        segs = tdkc_segments(track)
        assert segs[0].t_start == track.t[0]
        assert segs[-1].t_end == track.t[-1]
        assert segs[0].mmsi == track.mmsi


class TestAggregateCompression:
    def test_corpus_wide_compression_is_meaningful(self, aisdb_tracks: list[dict]):
        """Across the whole bundled corpus, default TDKC must drop >0 points."""
        total_orig = 0
        total_kept = 0
        for raw in aisdb_tracks:
            if len(raw["time"]) < 3:
                continue
            track = from_aisdb_track(raw)
            total_orig += len(track)
            total_kept += len(tdkc(track))
        if total_orig == 0:
            pytest.skip("No aisdb tracks of length >= 3 available")
        assert total_kept < total_orig
