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

Recursion depth scales with tree depth (≈ ``log2(N)`` for balanced inputs).
For pathological inputs that produce a degenerate tree, raise the system
recursion limit before calling :func:`tdkc`.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    n = len(track)
    if n <= 2:
        return track
    cbt = _build_cbt(track.t, track.lon, track.lat, track.sog, track.cog, 0, n - 1)
    assert cbt is not None  # n >= 3 guarantees a non-empty intermediate set
    sed_eps, svd_eps = _adaptive_thresholds(cbt, min_sed_m, min_svd_kn)
    keys: set[int] = {0, n - 1}
    _identify_keys(cbt, sed_eps, svd_eps, keys)
    return track.take(sorted(keys))


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
    cbt = _build_cbt(track.t, track.lon, track.lat, track.sog, track.cog, 0, n - 1)
    assert cbt is not None
    sed_eps, svd_eps = _adaptive_thresholds(cbt, min_sed_m, min_svd_kn)
    keys: set[int] = {0, n - 1}
    _identify_keys(cbt, sed_eps, svd_eps, keys)
    sorted_idx = sorted(keys)
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
