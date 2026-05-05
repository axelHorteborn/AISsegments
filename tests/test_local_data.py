"""Integration tests against locally-dropped AIS CSV data.

These run only when at least one CSV file is present under ``tests/data/``;
otherwise the fixture skips and these tests are reported as skipped.

See ``tests/data/README.md`` for the expected CSV format.
"""
from __future__ import annotations

import numpy as np
import pytest

from aissegments import Track, tdkc, tdkc_segments

pytestmark = pytest.mark.local_csv


class TestLocalCsvTracks:
    def test_at_least_one_track_loaded(self, local_csv_tracks: list[Track]):
        assert len(local_csv_tracks) >= 1

    def test_each_track_is_well_formed(self, local_csv_tracks: list[Track]):
        for track in local_csv_tracks[:20]:
            assert track.t.dtype == np.float64
            assert len(track.t) == len(track.lon) == len(track.lat)
            assert len(track.sog) == len(track.cog) == len(track.t)
            if len(track) > 1:
                assert np.all(np.diff(track.t) >= 0)


class TestTdkcOnLocalData:
    def test_compresses_longest_track(self, local_csv_tracks: list[Track]):
        track = local_csv_tracks[0]
        if len(track) < 3:
            pytest.skip(f"Longest local track has only {len(track)} pts")
        compressed = tdkc(track)
        assert 2 <= len(compressed) <= len(track)

    def test_endpoints_preserved(self, local_csv_tracks: list[Track]):
        track = local_csv_tracks[0]
        if len(track) < 3:
            pytest.skip()
        compressed = tdkc(track)
        assert compressed.t[0] == track.t[0]
        assert compressed.t[-1] == track.t[-1]

    def test_segments_cover_input(self, local_csv_tracks: list[Track]):
        track = local_csv_tracks[0]
        if len(track) < 3:
            pytest.skip()
        segs = tdkc_segments(track)
        total = sum(s.n_points - 1 for s in segs)
        assert total == len(track) - 1

    def test_high_floor_drops_more_points(self, local_csv_tracks: list[Track]):
        """A meaningfully high SED floor should compress at least as aggressively."""
        track = local_csv_tracks[0]
        if len(track) < 10:
            pytest.skip(f"Need ≥10 pts to demonstrate floor effect; got {len(track)}")
        n_default = len(tdkc(track))
        n_high = len(tdkc(track, min_sed_m=1_000.0, min_svd_kn=10.0))
        assert n_high <= n_default


class TestAggregateLocalCompression:
    def test_corpus_compresses_meaningfully(self, local_csv_tracks: list[Track]):
        total_orig = 0
        total_kept = 0
        for track in local_csv_tracks:
            if len(track) < 3:
                continue
            total_orig += len(track)
            total_kept += len(tdkc(track))
        if total_orig == 0:
            pytest.skip("No local tracks of length >= 3")
        assert total_kept < total_orig
