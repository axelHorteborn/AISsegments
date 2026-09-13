"""AIS trajectory segmentation and feature-preserving compression.

Public API:

- :class:`Track` / :class:`Segment` — typed input/output containers.
- :func:`to_segments` — pair consecutive points of a Track into Segment records.
- :func:`tdkc` — Top-Down Kinematic Compression; returns a compressed Track.
- :func:`tdkc_segments` — convenience wrapper: TDKC plus segment construction
  with original-point-count enrichment.
- :func:`read_parquet_tracks` / :func:`read_parquet_static_records` —
  streaming Parquet loaders (require the optional ``pyarrow`` dependency).
- :func:`ship_type_to_toc`, :data:`DEFAULT_SHIP_TYPE_NAMES`,
  :data:`DEFAULT_MOBILE_TYPES` — provider-layout helpers shared by the
  CSV and Parquet readers.

Reference
---------
Guo, S., Bolbot, V., & Valdez Banda, O. (2024). An adaptive trajectory
compression and feature preservation method for maritime traffic analysis.
*Ocean Engineering*, 312, 119189.
"""

from aissegments._schema import (
    DEFAULT_MOBILE_TYPES,
    DEFAULT_SHIP_TYPE_NAMES,
    ship_type_to_toc,
)
from aissegments._types import Segment, Track, to_segments
from aissegments.adapters import (
    from_aisdb_track,
    read_csv_static_records,
    read_csv_tracks,
)
from aissegments.parquet import read_parquet_static_records, read_parquet_tracks
from aissegments.tdkc import tdkc, tdkc_segments

__version__ = "0.3.0"

__all__ = [
    "DEFAULT_MOBILE_TYPES",
    "DEFAULT_SHIP_TYPE_NAMES",
    "Segment",
    "Track",
    "__version__",
    "from_aisdb_track",
    "read_csv_static_records",
    "read_csv_tracks",
    "read_parquet_static_records",
    "read_parquet_tracks",
    "ship_type_to_toc",
    "tdkc",
    "tdkc_segments",
    "to_segments",
]
