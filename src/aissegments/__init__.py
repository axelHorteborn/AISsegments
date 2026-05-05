"""AIS trajectory segmentation and feature-preserving compression.

Public API:

- :class:`Track` / :class:`Segment` — typed input/output containers.
- :func:`to_segments` — pair consecutive points of a Track into Segment records.
- :func:`tdkc` — Top-Down Kinematic Compression; returns a compressed Track.
- :func:`tdkc_segments` — convenience wrapper: TDKC plus segment construction
  with original-point-count enrichment.

Reference
---------
Guo, S., Bolbot, V., & Valdez Banda, O. (2024). An adaptive trajectory
compression and feature preservation method for maritime traffic analysis.
*Ocean Engineering*, 312, 119189.
"""
from aissegments._types import Segment, Track, to_segments
from aissegments.adapters import (
    from_aisdb_track,
    read_csv_static_records,
    read_csv_tracks,
)
from aissegments.tdkc import tdkc, tdkc_segments

__version__ = "0.2.0"

__all__ = [
    "Segment",
    "Track",
    "__version__",
    "from_aisdb_track",
    "read_csv_static_records",
    "read_csv_tracks",
    "tdkc",
    "tdkc_segments",
    "to_segments",
]
