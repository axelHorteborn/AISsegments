"""Top-Down Kinematic Compression (TDKC) on AIS tracks.

Implements the algorithm of:

    Guo, S., Bolbot, V., & Valdez Banda, O. (2024). An adaptive trajectory
    compression and feature preservation method for maritime traffic analysis.
    *Ocean Engineering*, 312, 119189.  https://doi.org/10.1016/j.oceaneng.2024.119189

Pipeline
--------
1. Recursively build a *Compression Binary Tree* (CBT).  Each node records the
   intermediate point with the maximum aggregated z-score of Synchronous
   Euclidean Distance (SED) and Synchronous Velocity Difference (SVD), and
   splits the sub-trajectory at that point.  Recursion exhausts down to
   length-2 leaves; the threshold is *not* applied during construction.
2. Compute adaptive thresholds ``sed_eps`` and ``svd_eps`` as the mean SED and
   mean SVD across all CBT nodes (paper Eqs. 27-28), each clamped to a
   user-configurable floor that defends against floating-point noise on clean
   inputs.  Real AIS data is unaffected (mean SED ~10²-10³ m, well above the
   1 m default floor).
3. A node is a *key node* if either of its measurements exceeds its threshold,
   *or* any of its children is a key node — this is the recursion-termination
   fix from the paper.  Indices of key-node split points are retained.
4. Output: original first/last point + every key-node split point.

Implementation
--------------
The public functions evaluate the tree **one level at a time**: every
sub-trajectory of a level is handled in one set of numpy operations
(:func:`_key_indices`), instead of one Python call per node.  The output is
identical to the node-by-node recursion (:func:`_key_indices_recursive`, kept
as the reference): SED, SVD and z-scores are computed with the same
element-wise operations, the per-sub-trajectory sums follow numpy's own
pairwise summation order (:func:`_segment_sums`), and the threshold means are
taken over the nodes in the recursion's pre-order.  It is about 8x faster on
real AIS tracks and has no recursion-depth limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from aissegments._types import Segment, Track, to_segments

# Earth radius in metres for haversine.  WGS84 mean radius.
_EARTH_RADIUS_M = 6_371_008.8

# Default threshold floors.  Below GPS accuracy (~10 m for Class A AIS) but
# well above double-precision arithmetic noise on lat/lon arithmetic.
_DEFAULT_MIN_SED_M = 1.0
_DEFAULT_MIN_SVD_KN = 0.01


# ---------------------------------------------------------------------------
# Internal numerics
# ---------------------------------------------------------------------------


def _haversine_m(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in metres, vectorised over numpy arrays."""
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalise; return zeros if std is zero (constant input)."""
    if x.size == 0:
        return x
    mean = x.mean()
    std = x.std()
    if std == 0.0:
        return np.zeros_like(x)
    return (x - mean) / std


def _compute_sed_svd(
    t: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    sog: np.ndarray,
    cog: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SED and SVD for every intermediate point in ``[start, end]``.

    Returns two arrays of shape ``(end - start - 1,)``.  Caller guarantees
    ``end - start >= 2`` so there is at least one intermediate.  Zero-fills
    if the start and end share a timestamp (synchronous interpolation
    degenerates).
    """
    inter = np.arange(start + 1, end)
    n_inter = inter.size
    t_s, t_e = float(t[start]), float(t[end])
    dt_total = t_e - t_s
    if dt_total <= 0.0:
        return np.zeros(n_inter), np.zeros(n_inter)
    frac = (t[inter] - t_s) / dt_total

    # SED: distance between the actual point and the synchronous-interpolated
    # point on the start-end great circle (paper Eqs. 4-7, generalised to the
    # sphere via haversine — see paper Sec. 3.2.1 par. on Cartesian distortion).
    lon_sync = lon[start] + frac * (lon[end] - lon[start])
    lat_sync = lat[start] + frac * (lat[end] - lat[start])
    sed = _haversine_m(lon_sync, lat_sync, lon[inter], lat[inter])

    # SVD: magnitude of the velocity-vector difference between the actual
    # point and the synchronous-interpolated velocity (paper Eqs. 11-19).
    delta_s = float(sog[end]) - float(sog[start])
    delta_c = ((float(cog[end]) - float(cog[start]) + 180.0) % 360.0) - 180.0
    s_sync = sog[start] + frac * delta_s
    c_sync = cog[start] + frac * delta_c
    vx_sync = s_sync * np.sin(np.deg2rad(c_sync))
    vy_sync = s_sync * np.cos(np.deg2rad(c_sync))
    vx_pt = sog[inter] * np.sin(np.deg2rad(cog[inter]))
    vy_pt = sog[inter] * np.cos(np.deg2rad(cog[inter]))
    svd = np.hypot(vx_sync - vx_pt, vy_sync - vy_pt)
    return sed, svd


# ---------------------------------------------------------------------------
# Compression Binary Tree
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    idx: int
    sed: float
    svd: float
    left: _Node | None = None
    right: _Node | None = None


def _build_cbt(
    t: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    sog: np.ndarray,
    cog: np.ndarray,
    start: int,
    end: int,
) -> _Node | None:
    """Recursive CBT construction (paper Algorithm 3).  No threshold applied."""
    if end - start < 2:
        return None
    sed_arr, svd_arr = _compute_sed_svd(t, lon, lat, sog, cog, start, end)
    aggregated = _zscore(sed_arr) + _zscore(svd_arr)
    rel_max = int(np.argmax(aggregated))
    split_idx = start + 1 + rel_max
    return _Node(
        idx=split_idx,
        sed=float(sed_arr[rel_max]),
        svd=float(svd_arr[rel_max]),
        left=_build_cbt(t, lon, lat, sog, cog, start, split_idx),
        right=_build_cbt(t, lon, lat, sog, cog, split_idx, end),
    )


def _collect_thresholds(
    node: _Node | None, sed_acc: list[float], svd_acc: list[float]
) -> None:
    if node is None:
        return
    sed_acc.append(node.sed)
    svd_acc.append(node.svd)
    _collect_thresholds(node.left, sed_acc, svd_acc)
    _collect_thresholds(node.right, sed_acc, svd_acc)


def _identify_keys(
    node: _Node | None,
    sed_eps: float,
    svd_eps: float,
    out: set[int],
) -> bool:
    """Return True if ``node`` (or any descendant) is a key node.  Mutates ``out``."""
    if node is None:
        return False
    self_key = (node.sed > sed_eps) or (node.svd > svd_eps)
    left_key = _identify_keys(node.left, sed_eps, svd_eps, out)
    right_key = _identify_keys(node.right, sed_eps, svd_eps, out)
    if self_key or left_key or right_key:
        out.add(node.idx)
        return True
    return False


def _adaptive_thresholds(
    cbt: _Node, min_sed_m: float, min_svd_kn: float
) -> tuple[float, float]:
    """Mean SED / SVD across the tree, clamped to the configured floors."""
    sed_acc: list[float] = []
    svd_acc: list[float] = []
    _collect_thresholds(cbt, sed_acc, svd_acc)
    sed_eps = max(float(np.mean(sed_acc)), float(min_sed_m))
    svd_eps = max(float(np.mean(svd_acc)), float(min_svd_kn))
    return sed_eps, svd_eps


def _key_indices_recursive(track: Track, min_sed_m: float, min_svd_kn: float) -> np.ndarray:
    """Reference implementation: key-point indices from the node-by-node recursion.

    :func:`_key_indices` must return exactly the same array; the test suite
    compares the two.  Needs ``len(track) >= 3``.
    """
    n = len(track)
    cbt = _build_cbt(track.t, track.lon, track.lat, track.sog, track.cog, 0, n - 1)
    assert cbt is not None  # n >= 3 guarantees a non-empty intermediate set
    sed_eps, svd_eps = _adaptive_thresholds(cbt, min_sed_m, min_svd_kn)
    keys: set[int] = {0, n - 1}
    _identify_keys(cbt, sed_eps, svd_eps, keys)
    return np.array(sorted(keys), dtype=np.int64)


# ---------------------------------------------------------------------------
# Level-synchronous evaluation
# ---------------------------------------------------------------------------

# numpy sums float64 arrays pairwise: plain loop below 8 values, eight
# interleaved accumulators up to this block size, halving above it.
_PAIRWISE_BLOCK = 128


def _segment_sums(x: np.ndarray, offs: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """``out[k] == x[offs[k]:offs[k] + lens[k]].sum()``, bit for bit.

    Reproduces numpy's pairwise summation for many slices at once (a plain
    ``np.add.reduceat`` adds in another order and differs in the last bits,
    which can flip the arg-max between two near-equal split candidates).
    Every ``lens[k]`` must be >= 1.
    """
    out = np.empty(len(offs))
    last = len(x) - 1
    small = lens < 8
    if small.any():
        o = offs[small]
        n = lens[small]
        acc = np.zeros(len(o))
        for j in range(int(n.max())):
            acc += np.where(j < n, x[np.minimum(o + j, last)], 0.0)
        out[small] = acc
    big = lens > _PAIRWISE_BLOCK
    mid = ~small & ~big
    if mid.any():
        o = offs[mid]
        n = lens[mid]
        stop = n - (n % 8)
        blocks = stop // 8
        res = np.empty(len(o))
        for nb in np.unique(blocks).tolist():
            sel = blocks == nb
            g = x[o[sel][:, None] + np.arange(8 * nb)].reshape(-1, nb, 8)
            r = g[:, 0, :].copy()
            for b in range(1, nb):
                r += g[:, b, :]
            res[sel] = ((r[:, 0] + r[:, 1]) + (r[:, 2] + r[:, 3])) + (
                (r[:, 4] + r[:, 5]) + (r[:, 6] + r[:, 7])
            )
        rem = n - stop
        for k in range(int(rem.max())):
            res += np.where(k < rem, x[np.minimum(o + stop + k, last)], 0.0)
        out[mid] = res
    if big.any():
        o = offs[big]
        n = lens[big]
        half = n // 2
        half -= half % 8
        both = _segment_sums(x, np.concatenate([o, o + half]), np.concatenate([half, n - half]))
        out[big] = both[: len(o)] + both[len(o) :]
    return out


def _segment_zscores(
    x: np.ndarray, offs: np.ndarray, lens: np.ndarray, seg_of: np.ndarray
) -> np.ndarray:
    """:func:`_zscore` applied to every slice of ``x`` (same arithmetic)."""
    count = lens.astype(np.float64)
    mean = (_segment_sums(x, offs, lens) / count)[seg_of]
    dev = x - mean
    std = np.sqrt(_segment_sums(dev * dev, offs, lens) / count)[seg_of]
    ok = std != 0.0
    if ok.all():
        return dev / std
    z = np.zeros_like(x)
    z[ok] = dev[ok] / std[ok]
    return z


def _key_indices(
    t: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    sog: np.ndarray,
    cog: np.ndarray,
    min_sed_m: float,
    min_svd_kn: float,
) -> np.ndarray:
    """Key-point indices (sorted, endpoints included) for ``n >= 3`` points.

    Builds the Compression Binary Tree breadth-first: ``starts`` / ``ends``
    hold the sub-trajectories of the current level, and one pass computes
    SED, SVD, z-scores and the split point for all of them.  Nodes are stored
    level by level with their parent and side, which is enough to rebuild the
    recursion's pre-order (threshold means) and to propagate key status up.
    """
    n = len(t)
    # Per-point terms the recursion recomputes at every node.
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    cos_lat = np.cos(lat_r)
    cog_r = np.deg2rad(cog)
    vx_pt = sog * np.sin(cog_r)
    vy_pt = sog * np.cos(cog_r)

    starts = np.array([0])
    ends = np.array([n - 1])
    parent = np.array([-1])
    is_left = np.array([False])
    lv_idx: list[np.ndarray] = []
    lv_sed: list[np.ndarray] = []
    lv_svd: list[np.ndarray] = []
    lv_parent: list[np.ndarray] = []
    lv_left: list[np.ndarray] = []
    n_nodes = 0
    while len(starts):
        lens = ends - starts - 1
        offs = np.zeros(len(lens), dtype=np.int64)
        np.cumsum(lens[:-1], out=offs[1:])
        seg_of = np.repeat(np.arange(len(lens)), lens)
        inter = np.arange(int(lens.sum())) + np.repeat(starts + 1 - offs, lens)
        dt_total = t[ends] - t[starts]
        sed = np.zeros(len(inter))
        svd = np.zeros(len(inter))
        # Sub-trajectories whose ends share a timestamp keep SED = SVD = 0.
        good = (dt_total > 0.0)[seg_of]
        ip = inter[good]
        if len(ip):
            sg = seg_of[good]
            frac = (t[ip] - t[starts][sg]) / dt_total[sg]
            lon_s = lon[starts][sg]
            lat_s = lat[starts][sg]
            lon_sync = lon_s + frac * (lon[ends] - lon[starts])[sg]
            lat_sync = lat_s + frac * (lat[ends] - lat[starts])[sg]
            lat1r = np.deg2rad(lat_sync)
            dlon = lon_r[ip] - np.deg2rad(lon_sync)
            dlat = lat_r[ip] - lat1r
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * cos_lat[ip] * np.sin(dlon / 2.0) ** 2
            sed[good] = 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
            sog_s = sog[starts]
            cog_s = cog[starts]
            delta_s = (sog[ends] - sog_s)[sg]
            delta_c = (((cog[ends] - cog_s + 180.0) % 360.0) - 180.0)[sg]
            s_sync = sog_s[sg] + frac * delta_s
            c_sync = np.deg2rad(cog_s[sg] + frac * delta_c)
            svd[good] = np.hypot(
                s_sync * np.sin(c_sync) - vx_pt[ip], s_sync * np.cos(c_sync) - vy_pt[ip]
            )
        agg = _segment_zscores(sed, offs, lens, seg_of) + _segment_zscores(svd, offs, lens, seg_of)
        # First maximum per sub-trajectory, like np.argmax: a NaN wins, and the
        # first NaN when there are several (np.maximum propagates NaN).
        seg_max = np.maximum.reduceat(agg, offs)[seg_of]
        candidate = np.isnan(agg) | (agg == seg_max)
        split = np.minimum.reduceat(np.where(candidate, inter, n), offs)
        at = offs + (split - starts - 1)
        ids = np.arange(n_nodes, n_nodes + len(starts))
        lv_idx.append(split)
        lv_sed.append(sed[at])
        lv_svd.append(svd[at])
        lv_parent.append(parent)
        lv_left.append(is_left)
        n_nodes += len(starts)
        go_left = split - starts >= 2
        go_right = ends - split >= 2
        starts = np.concatenate([starts[go_left], split[go_right]])
        ends = np.concatenate([split[go_left], ends[go_right]])
        parent = np.concatenate([ids[go_left], ids[go_right]])
        is_left = np.concatenate(
            [np.ones(int(go_left.sum()), dtype=bool), np.zeros(int(go_right.sum()), dtype=bool)]
        )

    node_idx = np.concatenate(lv_idx)
    node_sed = np.concatenate(lv_sed)
    node_svd = np.concatenate(lv_svd)
    node_parent = np.concatenate(lv_parent)
    node_left = np.concatenate(lv_left)
    bounds = np.cumsum([0, *(len(a) for a in lv_idx)])
    levels = range(len(lv_idx) - 1, 0, -1)  # deepest first, root level excluded

    # Pre-order position of every node: the recursion's threshold lists are
    # built in that order, and np.mean's result depends on it in the last bits.
    size = np.ones(n_nodes, dtype=np.int64)
    left_size = np.zeros(n_nodes, dtype=np.int64)
    for lv in levels:
        a, b = bounds[lv], bounds[lv + 1]
        np.add.at(size, node_parent[a:b], size[a:b])
        lft = node_left[a:b]
        left_size[node_parent[a:b][lft]] = size[a:b][lft]
    preorder = np.zeros(n_nodes, dtype=np.int64)
    for lv in range(1, len(lv_idx)):
        a, b = bounds[lv], bounds[lv + 1]
        p = node_parent[a:b]
        preorder[a:b] = preorder[p] + 1 + np.where(node_left[a:b], 0, left_size[p])
    order = np.empty(n_nodes, dtype=np.int64)
    order[preorder] = np.arange(n_nodes)
    sed_eps = max(float(np.mean(node_sed[order])), float(min_sed_m))
    svd_eps = max(float(np.mean(node_svd[order])), float(min_svd_kn))

    key = (node_sed > sed_eps) | (node_svd > svd_eps)
    for lv in levels:
        a, b = bounds[lv], bounds[lv + 1]
        key[node_parent[a:b][key[a:b]]] = True
    return np.unique(np.concatenate([[0], node_idx[key], [n - 1]]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tdkc_indices(
    track: Track,
    *,
    min_sed_m: float = _DEFAULT_MIN_SED_M,
    min_svd_kn: float = _DEFAULT_MIN_SVD_KN,
) -> np.ndarray:
    """Indices of the points TDKC keeps, sorted, endpoints included.

    The array form of :func:`tdkc`: ``tdkc(track) == track.take(tdkc_indices(track))``.
    Useful when the caller builds its own output from the original arrays
    (bulk database loads) and does not need :class:`Track` / :class:`Segment`
    objects.  Tracks of length ``<= 2`` return every index.

    See :func:`tdkc` for ``min_sed_m`` / ``min_svd_kn``.
    """
    n = len(track)
    if n <= 2:
        return np.arange(n, dtype=np.int64)
    return _key_indices(
        track.t, track.lon, track.lat, track.sog, track.cog, min_sed_m, min_svd_kn
    )


def tdkc(
    track: Track,
    *,
    min_sed_m: float = _DEFAULT_MIN_SED_M,
    min_svd_kn: float = _DEFAULT_MIN_SVD_KN,
) -> Track:
    """Compress ``track`` using TDKC; return a new Track of only the key points.

    Endpoints are always retained.  Tracks of length ``<= 2`` pass through
    unchanged.

    Parameters
    ----------
    track : Track
        Input AIS track for a single vessel.
    min_sed_m : float, optional
        Lower bound for the adaptive SED threshold, in metres.  Defaults to
        ``1.0`` (below typical GPS accuracy, above floating-point noise).
        Set to ``0`` for paper-faithful behaviour at the cost of FP-driven
        artefacts on perfectly clean inputs.
    min_svd_kn : float, optional
        Lower bound for the adaptive SVD threshold, in knots.  Defaults to
        ``0.01``.  Set to ``0`` for paper-faithful behaviour.

    Returns
    -------
    Track
        New track containing the original endpoints plus every key-node split
        point identified by TDKC.  Always sorted by ``t``.
    """
    if len(track) <= 2:
        return track
    return track.take(tdkc_indices(track, min_sed_m=min_sed_m, min_svd_kn=min_svd_kn))


def tdkc_segments(
    track: Track,
    *,
    min_sed_m: float = _DEFAULT_MIN_SED_M,
    min_svd_kn: float = _DEFAULT_MIN_SVD_KN,
) -> list[Segment]:
    """Compress with TDKC and emit one ``Segment`` per consecutive key-pair.

    Each segment's ``n_points`` is the count of *original* points spanned,
    including both endpoints — useful when the segments are written into a
    PostGIS table that wants to preserve the underlying observation density.

    See :func:`tdkc` for ``min_sed_m`` / ``min_svd_kn``.
    """
    n = len(track)
    if n < 2:
        return []
    if n == 2:
        return to_segments(track)
    sorted_idx = tdkc_indices(track, min_sed_m=min_sed_m, min_svd_kn=min_svd_kn).tolist()
    compressed = track.take(sorted_idx)
    base_segments = to_segments(compressed)
    enriched: list[Segment] = []
    for seg, (i_start, i_end) in zip(base_segments, pairwise(sorted_idx), strict=True):
        enriched.append(
            Segment(
                mmsi=seg.mmsi,
                t_start=seg.t_start,
                t_end=seg.t_end,
                lon_start=seg.lon_start,
                lat_start=seg.lat_start,
                lon_end=seg.lon_end,
                lat_end=seg.lat_end,
                cog_mean=seg.cog_mean,
                sog_mean=seg.sog_mean,
                n_points=int(i_end - i_start + 1),
            )
        )
    return enriched
