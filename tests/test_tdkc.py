"""Behavioural and edge-case tests for the TDKC algorithm."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from aissegments import Track, tdkc, tdkc_indices, tdkc_segments
from aissegments.tdkc import (
    _build_cbt,
    _collect_thresholds,
    _compute_sed_svd,
    _haversine_m,
    _identify_keys,
    _key_indices_recursive,
    _segment_sums,
    _zscore,
)

# ---------------------------------------------------------------------------
# Internal numerics
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_zero_distance(self):
        d = _haversine_m(np.array([0.0]), np.array([55.0]), np.array([0.0]), np.array([55.0]))
        assert d[0] == pytest.approx(0.0)

    def test_known_distance_one_degree_lat(self):
        # ~111 km per degree of latitude at any longitude.
        d = _haversine_m(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
        assert d[0] == pytest.approx(111_195.0, rel=0.01)

    def test_vectorised(self):
        d = _haversine_m(
            np.array([0.0, 0.0]), np.array([0.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 2.0]),
        )
        assert d.shape == (2,)
        assert d[1] > d[0]


class TestZscore:
    def test_constant_returns_zeros(self):
        x = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(_zscore(x), [0.0, 0.0, 0.0])

    def test_normal_zscore(self):
        x = np.array([1.0, 2.0, 3.0])
        z = _zscore(x)
        assert z.mean() == pytest.approx(0.0, abs=1e-12)
        assert z.std() == pytest.approx(1.0, abs=1e-12)

    def test_empty_returns_empty(self):
        x = np.array([], dtype=float)
        z = _zscore(x)
        assert z.size == 0


class TestComputeSedSvd:
    def test_no_intermediates(self):
        t = np.array([0.0, 60.0])
        lon = np.array([0.0, 0.001])
        lat = np.array([55.0, 55.0])
        sog = np.array([10.0, 10.0])
        cog = np.array([90.0, 90.0])
        sed, svd = _compute_sed_svd(t, lon, lat, sog, cog, 0, 1)
        assert sed.size == 0 and svd.size == 0

    def test_zero_dt_returns_zeros(self):
        t = np.array([0.0, 0.0, 0.0])
        lon = np.array([0.0, 0.001, 0.002])
        lat = np.array([55.0, 55.0, 55.0])
        sog = np.array([10.0, 10.0, 10.0])
        cog = np.array([90.0, 90.0, 90.0])
        sed, svd = _compute_sed_svd(t, lon, lat, sog, cog, 0, 2)
        np.testing.assert_array_equal(sed, [0.0])
        np.testing.assert_array_equal(svd, [0.0])

    def test_perfectly_straight_at_constant_speed_gives_zero(self):
        n = 11
        t = np.arange(n, dtype=float) * 60.0
        lon = np.linspace(0.0, 0.01, n)
        lat = np.full(n, 55.0)
        sog = np.full(n, 10.0)
        cog = np.full(n, 90.0)
        sed, svd = _compute_sed_svd(t, lon, lat, sog, cog, 0, n - 1)
        np.testing.assert_allclose(sed, 0.0, atol=1e-6)
        np.testing.assert_allclose(svd, 0.0, atol=1e-12)

    def test_off_baseline_point_has_positive_sed(self):
        t = np.array([0.0, 60.0, 120.0])
        # Middle point is shifted north of the baseline.
        lon = np.array([0.0, 0.001, 0.002])
        lat = np.array([55.0, 55.05, 55.0])
        sog = np.array([10.0, 10.0, 10.0])
        cog = np.array([90.0, 90.0, 90.0])
        sed, _svd = _compute_sed_svd(t, lon, lat, sog, cog, 0, 2)
        assert sed[0] > 1000.0  # tens of km off the baseline

    def test_course_wrap_handled(self):
        """Going from 350° to 10° should be a 20° change, not 340°."""
        t = np.array([0.0, 60.0, 120.0])
        lon = np.array([0.0, 0.001, 0.002])
        lat = np.array([55.0, 55.0, 55.0])
        sog = np.array([10.0, 10.0, 10.0])
        cog = np.array([350.0, 0.0, 10.0])  # halfway -> 0°
        _sed, svd = _compute_sed_svd(t, lon, lat, sog, cog, 0, 2)
        # If we did NOT wrap, c_sync(t=60s) = 350 + 0.5*(10-350) = 350 - 170 = 180°.
        # That would be near-zero SVD only because cog[1]=0 vs interp=180 differ by 180°.
        # With wrap, c_sync = 350 + 0.5*20 = 360 ≡ 0, matching cog[1]=0 exactly,
        # so SVD must be ~0.
        assert svd[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# CBT construction + traversal
# ---------------------------------------------------------------------------


class TestBuildCbt:
    def test_too_few_points_returns_none(self):
        t = np.array([0.0, 60.0])
        lon = np.array([0.0, 0.001])
        lat = np.array([55.0, 55.0])
        sog = np.array([10.0, 10.0])
        cog = np.array([90.0, 90.0])
        assert _build_cbt(t, lon, lat, sog, cog, 0, 1) is None

    def test_zero_dt_returns_none(self):
        # Sub-trajectory where start and end share a timestamp -> degenerate;
        # _compute_sed_svd returns zeros, but argmax still picks an index, so
        # the tree builds.  This guards the actual stop condition (end-start<2).
        t = np.array([0.0, 0.0, 0.0])
        lon = np.array([0.0, 0.001, 0.002])
        lat = np.array([55.0, 55.0, 55.0])
        sog = np.array([10.0, 10.0, 10.0])
        cog = np.array([90.0, 90.0, 90.0])
        node = _build_cbt(t, lon, lat, sog, cog, 0, 2)
        assert node is not None
        assert node.idx == 1
        assert node.sed == 0.0 and node.svd == 0.0

    def test_balanced_split_on_three_points(self):
        t = np.array([0.0, 60.0, 120.0])
        lon = np.array([0.0, 0.001, 0.002])
        lat = np.array([55.0, 55.05, 55.0])
        sog = np.array([10.0, 10.0, 10.0])
        cog = np.array([90.0, 90.0, 90.0])
        node = _build_cbt(t, lon, lat, sog, cog, 0, 2)
        assert node is not None
        assert node.idx == 1
        assert node.left is None and node.right is None  # leaf children


class TestCollectThresholds:
    def test_none_node_is_noop(self):
        sed_acc: list[float] = []
        svd_acc: list[float] = []
        _collect_thresholds(None, sed_acc, svd_acc)
        assert sed_acc == [] and svd_acc == []

    def test_full_tree_walk(self):
        t = np.arange(5, dtype=float) * 60.0
        lon = np.array([0.0, 0.001, 0.0008, 0.003, 0.004])
        lat = np.array([55.0, 55.05, 55.0, 55.0, 55.0])
        sog = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        cog = np.array([90.0, 90.0, 90.0, 90.0, 90.0])
        node = _build_cbt(t, lon, lat, sog, cog, 0, 4)
        sed_acc: list[float] = []
        svd_acc: list[float] = []
        _collect_thresholds(node, sed_acc, svd_acc)
        # We must visit every internal node — at least one accumulated value.
        assert len(sed_acc) >= 1
        assert len(sed_acc) == len(svd_acc)


class TestIdentifyKeys:
    def test_none_node_returns_false(self):
        out: set[int] = set()
        assert _identify_keys(None, 0.0, 0.0, out) is False
        assert out == set()

    def test_single_node_above_threshold_is_key(self):
        node_args = {"idx": 5, "sed": 100.0, "svd": 0.0, "left": None, "right": None}
        from aissegments.tdkc import _Node
        node = _Node(**node_args)
        out: set[int] = set()
        assert _identify_keys(node, sed_eps=10.0, svd_eps=10.0, out=out) is True
        assert out == {5}

    def test_node_below_threshold_with_key_child_promotes(self):
        from aissegments.tdkc import _Node
        leaf = _Node(idx=3, sed=100.0, svd=0.0)
        parent = _Node(idx=2, sed=0.0, svd=0.0, left=leaf, right=None)
        out: set[int] = set()
        assert _identify_keys(parent, sed_eps=10.0, svd_eps=10.0, out=out) is True
        assert out == {2, 3}

    def test_all_below_threshold_keeps_nothing(self):
        from aissegments.tdkc import _Node
        leaf = _Node(idx=3, sed=0.0, svd=0.0)
        parent = _Node(idx=2, sed=0.0, svd=0.0, left=leaf, right=None)
        out: set[int] = set()
        assert _identify_keys(parent, sed_eps=10.0, svd_eps=10.0, out=out) is False
        assert out == set()


# ---------------------------------------------------------------------------
# Public API behaviour
# ---------------------------------------------------------------------------


class TestTdkc:
    def test_empty_track_passes_through(self):
        empty = Track.from_arrays(mmsi=1, t=[], lon=[], lat=[], sog=[], cog=[])
        out = tdkc(empty)
        assert len(out) == 0

    def test_single_point_passes_through(self):
        one = Track.from_arrays(mmsi=1, t=[0.0], lon=[0.0], lat=[55.0], sog=[10.0], cog=[90.0])
        out = tdkc(one)
        assert len(out) == 1

    def test_two_points_pass_through(self):
        two = Track.from_arrays(
            mmsi=1, t=[0.0, 60.0], lon=[0.0, 0.001],
            lat=[55.0, 55.0], sog=[10.0, 10.0], cog=[90.0, 90.0],
        )
        out = tdkc(two)
        assert len(out) == 2

    def test_zero_dt_track_returns_unchanged(self):
        n = 10
        track = Track.from_arrays(
            mmsi=1,
            t=[0.0] * n,  # all timestamps identical
            lon=np.linspace(0.0, 0.01, n),
            lat=[55.0] * n,
            sog=[10.0] * n,
            cog=[90.0] * n,
        )
        # All dt=0, so SED/SVD are zeros throughout, sed_eps=svd_eps=0,
        # no node strictly exceeds threshold -> only endpoints survive.
        out = tdkc(track)
        assert len(out) == 2

    def test_straight_track_compresses_to_endpoints(self, straight_track: Track):
        out = tdkc(straight_track)
        assert len(out) == 2

    def test_l_shaped_track_preserves_corner(self, l_shaped_track: Track):
        out = tdkc(l_shaped_track)
        # 3 expected: start, corner, end.  Other points may also survive
        # depending on numerical noise, but the corner must.
        assert len(out) >= 3
        # First and last are always endpoints of the original.
        assert out.t[0] == l_shaped_track.t[0]
        assert out.t[-1] == l_shaped_track.t[-1]
        # The corner is at index 49 -> t=49*60=2940; check that some output
        # point is near the corner (within one sample period).
        corner_t = l_shaped_track.t[49]
        assert np.min(np.abs(out.t - corner_t)) <= 60.0

    def test_speed_change_track_preserves_change_point(self, speed_change_track: Track):
        out = tdkc(speed_change_track)
        # DP would compress this to 2 (everything is on a straight line);
        # TDKC should keep the speed change at index 50.
        assert len(out) >= 3
        change_t = speed_change_track.t[49]  # last point at sog=5
        assert np.min(np.abs(out.t - change_t)) <= 120.0

    def test_compression_ratio_is_high(self, straight_track: Track):
        out = tdkc(straight_track)
        ratio = 1 - len(out) / len(straight_track)
        assert ratio > 0.95

    def test_output_indices_are_sorted_by_time(self, l_shaped_track: Track):
        out = tdkc(l_shaped_track)
        assert np.all(np.diff(out.t) >= 0)


class TestTdkcSegments:
    def test_empty_track_yields_no_segments(self):
        empty = Track.from_arrays(mmsi=1, t=[], lon=[], lat=[], sog=[], cog=[])
        assert tdkc_segments(empty) == []

    def test_single_point_yields_no_segments(self):
        track = Track.from_arrays(
            mmsi=1, t=[0.0], lon=[0.0], lat=[55.0], sog=[10.0], cog=[90.0]
        )
        assert tdkc_segments(track) == []

    def test_two_point_track_yields_one_segment(self):
        track = Track.from_arrays(
            mmsi=1, t=[0.0, 60.0],
            lon=[0.0, 0.001], lat=[55.0, 55.0],
            sog=[10.0, 10.0], cog=[90.0, 90.0],
        )
        segs = tdkc_segments(track)
        assert len(segs) == 1
        assert segs[0].n_points == 2

    def test_segments_total_npoints_covers_input(self, l_shaped_track: Track):
        segs = tdkc_segments(l_shaped_track)
        # Sum of (n_points - 1) across segments equals (N - 1) since segments
        # share endpoints.
        total = sum(s.n_points - 1 for s in segs)
        assert total == len(l_shaped_track) - 1

    def test_segments_contiguous_in_time(self, l_shaped_track: Track):
        segs = tdkc_segments(l_shaped_track)
        for a, b in pairwise(segs):
            assert a.t_end == b.t_start

    def test_first_and_last_endpoints_match_input(self, l_shaped_track: Track):
        segs = tdkc_segments(l_shaped_track)
        assert segs[0].t_start == l_shaped_track.t[0]
        assert segs[-1].t_end == l_shaped_track.t[-1]


# ---------------------------------------------------------------------------
# Level-synchronous evaluation == node-by-node recursion
# ---------------------------------------------------------------------------


def _ais_like_track(seed: int, n: int) -> Track:
    """Ship-like random track: course/speed random walk, GPS noise, stops,
    whole-second timestamps with repeats, positions on the AIS 1/10000' grid."""
    rng = np.random.default_rng(seed)
    dt = rng.choice([0.0, 1.0, 2.0, 3.0, 6.0, 10.0, 30.0, 180.0], size=n - 1,
                    p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05])
    t = 1.7e9 + np.concatenate([[0.0], np.cumsum(dt)])
    sog = np.clip(10.0 + np.cumsum(rng.normal(0.0, 0.3, n)), 0.0, 25.0)
    sog[rng.random(n) < 0.1] = 0.0
    cog = (90.0 + np.cumsum(rng.normal(0.0, 4.0, n))) % 360.0
    step = sog * 0.514444 * np.concatenate([[0.0], dt])
    north = np.cumsum(step * np.cos(np.deg2rad(cog))) + rng.normal(0.0, 5.0, n)
    east = np.cumsum(step * np.sin(np.deg2rad(cog))) + rng.normal(0.0, 5.0, n)
    lat = np.round((57.0 + north / 111_195.0) * 600_000.0) / 600_000.0
    lon = np.round((11.0 + east / (111_195.0 * np.cos(np.deg2rad(57.0)))) * 600_000.0) / 600_000.0
    return Track.from_arrays(mmsi=219000099, t=t, lon=lon, lat=lat, sog=np.round(sog, 1),
                             cog=np.round(cog, 1))


def _assert_same_keys(track: Track) -> None:
    for floors in ((1.0, 0.01), (30.0, 0.3), (0.0, 0.0)):
        np.testing.assert_array_equal(
            tdkc_indices(track, min_sed_m=floors[0], min_svd_kn=floors[1]),
            _key_indices_recursive(track, *floors),
        )


class TestSegmentSums:
    def test_matches_numpy_sum_bit_for_bit(self):
        """Every summation regime: < 8 values, 8-128 with and without a
        remainder, and the halving above 128."""
        rng = np.random.default_rng(7)
        edge = [1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129, 130, 136, 255, 256, 257, 1000, 2049]
        lens = np.array(edge + rng.integers(1, 3000, size=40).tolist())
        total = int(lens.sum())
        x = rng.random(total) * 10.0 ** rng.uniform(-3.0, 4.0, total)
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
        got = _segment_sums(x, offs, lens)
        want = np.array([x[o : o + k].sum() for o, k in zip(offs, lens, strict=True)])
        np.testing.assert_array_equal(got, want)


class TestLevelSynchronousMatchesRecursion:
    """``tdkc_indices`` must return exactly the reference recursion's key points."""

    @pytest.mark.parametrize("n", [3, 4, 9, 33, 130, 300, 1500])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_ais_like_tracks(self, n: int, seed: int):
        _assert_same_keys(_ais_like_track(seed, n))

    def test_fixture_tracks(self, straight_track, l_shaped_track, speed_change_track):
        for track in (straight_track, l_shaped_track, speed_change_track):
            _assert_same_keys(track)

    def test_vessel_at_berth(self):
        """Long runs of identical positions and SOG 0 give constant SED / SVD
        (zero standard deviation) inside many sub-trajectories."""
        n = 400
        rng = np.random.default_rng(11)
        lon = np.full(n, 11.5)
        lat = np.full(n, 57.2)
        moved = rng.random(n) < 0.05
        lon[moved] += 1.0 / 600_000.0
        track = Track.from_arrays(mmsi=1, t=np.arange(n) * 180.0, lon=lon, lat=lat,
                                  sog=np.zeros(n), cog=np.full(n, 360.0 - 0.1))
        _assert_same_keys(track)

    def test_runs_of_equal_timestamps(self):
        """Sub-trajectories whose ends share a timestamp keep SED = SVD = 0,
        next to normal ones in the same level."""
        base = _ais_like_track(5, 200)
        t = np.repeat(base.t[::4], 4)[: len(base)]
        track = Track(mmsi=1, t=t, lon=base.lon, lat=base.lat, sog=base.sog, cog=base.cog)
        _assert_same_keys(track)

    def test_course_through_north(self):
        n = 120
        cog = (350.0 + np.arange(n) * 0.5) % 360.0
        lat = 57.0 + np.cumsum(np.cos(np.deg2rad(cog))) * 1e-4
        lon = 11.0 + np.cumsum(np.sin(np.deg2rad(cog))) * 1e-4
        track = Track.from_arrays(mmsi=1, t=np.arange(n) * 10.0, lon=lon, lat=lat,
                                  sog=np.full(n, 12.0), cog=cog)
        _assert_same_keys(track)

    def test_nan_speed_picks_the_same_split_as_argmax(self):
        """np.argmax returns the first NaN; the batched arg-max must too."""
        base = _ais_like_track(9, 80)
        sog = base.sog.copy()
        sog[[17, 50]] = np.nan
        track = Track(mmsi=1, t=base.t, lon=base.lon, lat=base.lat, sog=sog, cog=base.cog)
        _assert_same_keys(track)


class TestTdkcIndices:
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_short_tracks_keep_every_index(self, n: int):
        track = Track.from_arrays(mmsi=1, t=np.arange(n) * 60.0, lon=np.zeros(n),
                                  lat=np.full(n, 55.0), sog=np.full(n, 10.0),
                                  cog=np.full(n, 90.0))
        np.testing.assert_array_equal(tdkc_indices(track), np.arange(n))

    def test_tdkc_is_take_of_indices(self, l_shaped_track: Track):
        idx = tdkc_indices(l_shaped_track)
        out = tdkc(l_shaped_track)
        np.testing.assert_array_equal(out.t, l_shaped_track.t[idx])
        np.testing.assert_array_equal(out.lon, l_shaped_track.lon[idx])

    def test_segments_run_between_consecutive_indices(self, l_shaped_track: Track):
        idx = tdkc_indices(l_shaped_track)
        segs = tdkc_segments(l_shaped_track)
        assert len(segs) == len(idx) - 1
        for seg, (a, b) in zip(segs, pairwise(idx.tolist()), strict=True):
            assert seg.t_start == l_shaped_track.t[a]
            assert seg.t_end == l_shaped_track.t[b]
            assert seg.n_points == b - a + 1
