"""Typed containers for AIS tracks and the linestring-segment records they yield."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_FLOAT_FIELDS = ("t", "lon", "lat", "sog", "cog")


@dataclass
class Track:
    """An ordered AIS track for a single vessel.

    All array fields must be the same length and ``t`` must be
    monotonically non-decreasing.  Construct via :meth:`from_arrays` to get
    consistent ``float64`` dtypes; the constructor itself only validates.

    Attributes
    ----------
    mmsi : int
        Maritime Mobile Service Identity for the vessel.
    t : numpy.ndarray
        Unix timestamps in seconds since epoch (UTC).  Shape ``(N,)``.
    lon, lat : numpy.ndarray
        WGS84 longitude/latitude in degrees.  Shape ``(N,)``.
    sog : numpy.ndarray
        Speed over ground in knots.  Shape ``(N,)``.
    cog : numpy.ndarray
        Course over ground in degrees, ``[0, 360)``.  Shape ``(N,)``.
    """

    mmsi: int
    t: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    sog: np.ndarray
    cog: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.t)
        for name in _FLOAT_FIELDS[1:]:
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"Track field {name!r} has length {len(arr)}, expected {n}"
                )
        if n > 1:
            diffs = np.diff(self.t)
            if np.any(diffs < 0):
                raise ValueError("Track.t must be monotonically non-decreasing")

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def from_arrays(
        cls,
        mmsi: int,
        t: Iterable[float],
        lon: Iterable[float],
        lat: Iterable[float],
        sog: Iterable[float],
        cog: Iterable[float],
    ) -> Track:
        """Coerce Python iterables / mixed dtypes into a validated ``Track``."""
        return cls(
            mmsi=int(mmsi),
            t=np.asarray(list(t), dtype=np.float64),
            lon=np.asarray(list(lon), dtype=np.float64),
            lat=np.asarray(list(lat), dtype=np.float64),
            sog=np.asarray(list(sog), dtype=np.float64),
            cog=np.asarray(list(cog), dtype=np.float64),
        )

    def take(self, indices: Iterable[int]) -> Track:
        """Return a new ``Track`` containing only the points at ``indices``."""
        idx = np.asarray(list(indices), dtype=np.int64)
        return Track(
            mmsi=self.mmsi,
            t=self.t[idx],
            lon=self.lon[idx],
            lat=self.lat[idx],
            sog=self.sog[idx],
            cog=self.cog[idx],
        )


@dataclass(frozen=True)
class Segment:
    """A constant-COG/SOG linestring between two key AIS points.

    Attributes
    ----------
    mmsi : int
        Vessel identifier.
    t_start, t_end : float
        Unix-second timestamps of the segment endpoints.
    lon_start, lat_start, lon_end, lat_end : float
        WGS84 coordinates (degrees) of the endpoints.
    cog_mean, sog_mean : float
        Mean course (degrees) and speed (knots) across the segment endpoints.
    n_points : int
        Number of *original* AIS points represented by the segment, including
        both endpoints.  ``2`` if the segment was built without a backing
        original track; ``>=2`` after enrichment by :func:`tdkc_segments`.
    """

    mmsi: int
    t_start: float
    t_end: float
    lon_start: float
    lat_start: float
    lon_end: float
    lat_end: float
    cog_mean: float
    sog_mean: float
    n_points: int = 2


def to_segments(track: Track) -> list[Segment]:
    """Pair consecutive points of ``track`` into ``Segment`` records.

    Each segment carries ``n_points = 2`` since the source track is treated as
    already-compressed (or already-key-point-only).  Use
    :func:`aissegments.tdkc_segments` if you need the original-point-count
    enrichment.
    """
    n = len(track)
    if n < 2:
        return []
    segments: list[Segment] = []
    for i in range(n - 1):
        # Course wrap: take the shorter of the two arc directions when the
        # endpoints straddle 0/360.  Mean is then re-wrapped to [0, 360).
        cog_diff = ((track.cog[i + 1] - track.cog[i] + 180.0) % 360.0) - 180.0
        cog_mean = (track.cog[i] + cog_diff / 2.0) % 360.0
        segments.append(
            Segment(
                mmsi=track.mmsi,
                t_start=float(track.t[i]),
                t_end=float(track.t[i + 1]),
                lon_start=float(track.lon[i]),
                lat_start=float(track.lat[i]),
                lon_end=float(track.lon[i + 1]),
                lat_end=float(track.lat[i + 1]),
                cog_mean=float(cog_mean),
                sog_mean=float((track.sog[i] + track.sog[i + 1]) / 2.0),
                n_points=2,
            )
        )
    return segments
