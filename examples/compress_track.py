"""Run TDKC on a short synthetic AIS track and print the compression result.

Usage::

    python -m examples.compress_track

This is intentionally dependency-free (only numpy + aissegments) so it works
out of the box right after ``pip install -e .``.
"""
from __future__ import annotations

import numpy as np

from aissegments import Track, tdkc, tdkc_segments


def make_demo_track() -> Track:
    """An L-shaped track with 100 points and a 90° turn at the midpoint."""
    n_leg = 50
    t = np.arange(2 * n_leg, dtype=float) * 60.0
    lon = np.concatenate(
        [12.0 + np.arange(n_leg) * 0.001, np.full(n_leg, 12.0 + (n_leg - 1) * 0.001)]
    )
    lat = np.concatenate(
        [np.full(n_leg, 55.0), 55.0 + np.arange(1, n_leg + 1) * 0.001]
    )
    sog = np.full(2 * n_leg, 10.0)
    cog = np.concatenate([np.full(n_leg, 90.0), np.full(n_leg, 0.0)])
    return Track.from_arrays(mmsi=219000999, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


def main() -> None:
    track = make_demo_track()
    compressed = tdkc(track)
    segments = tdkc_segments(track)
    cr = 1.0 - len(compressed) / len(track)
    print(f"Original points : {len(track)}")
    print(f"Compressed     : {len(compressed)}")
    print(f"Compression rate: {cr:.1%}")
    print(f"Segments       : {len(segments)}")
    for i, s in enumerate(segments):
        print(
            f"  [{i}] t=[{s.t_start:.0f},{s.t_end:.0f}] "
            f"COG~{s.cog_mean:.0f} SOG~{s.sog_mean:.1f} kn  n_points={s.n_points}"
        )


if __name__ == "__main__":
    main()
