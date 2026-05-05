"""Adapters for getting AIS data into ``aissegments``.

- :func:`from_aisdb_track` — single AISdb track-dict → :class:`Track`.
- :func:`read_csv_tracks` — generic CSV loader; groups by MMSI, returns one
  :class:`Track` per vessel.  Recognises common column name variants
  (Marine Cadastre's ``BaseDateTime``/``LAT``/``LON``, etc.) and parses
  ISO 8601 timestamps as well as unix seconds.  Handles ``.gz`` transparently.
- :func:`read_csv_static_records` — companion to ``read_csv_tracks`` that
  surfaces per-vessel static fields (VesselType, Length, Width, Draft,
  IMO, …) when the source CSV carries them.  Output is shaped to match
  AISdb's static-row dict so downstream consumers can treat aisdb-decoded
  data and rich-CSV data uniformly.

None of these pull in extra runtime dependencies beyond ``numpy``.
"""
from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aissegments._types import Track

# Column aliases (all lower-cased before matching).  First element is the
# canonical name used in error messages.
_CSV_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "mmsi": ("mmsi",),
    "time": ("time", "basedatetime", "timestamp", "datetime", "date_time"),
    "lon": ("lon", "longitude"),
    "lat": ("lat", "latitude"),
    "sog": ("sog", "speed"),
    "cog": ("cog", "course"),
}

# Optional static-info columns recognised by :func:`read_csv_static_records`.
# All aliases are matched lowercase.  Output keys mirror AISdb's static-row
# dict so the same downstream code path can consume aisdb-decoded rows and
# rich-CSV rows uniformly.
_STATIC_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "vessel_name": ("vesselname", "vessel_name", "name"),
    "call_sign": ("callsign", "call_sign"),
    "imo": ("imo", "imo_num"),
    "ship_type": ("vesseltype", "ship_type", "shiptype"),
    "destination": ("destination",),
    # Source columns that get split into AISdb's per-quadrant antenna offsets.
    "length": ("length", "loa", "ship_length"),
    "width": ("width", "beam", "breadth"),
    "draught": ("draft", "draught"),
}


def _resolve_csv_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map each canonical name to the actual header it found, or raise."""
    lower_to_actual = {c.strip().lower(): c for c in fieldnames}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in _CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_actual:
                resolved[canonical] = lower_to_actual[alias]
                break
        else:
            missing.append(canonical)
    if missing:
        raise KeyError(f"CSV missing required columns (any of these aliases): {missing}")
    return resolved


def _to_unix_seconds(value: str) -> float:
    """Parse either a numeric unix-seconds timestamp or an ISO 8601 string.

    Naive ISO timestamps are interpreted as UTC, the de-facto convention for
    AIS data feeds.
    """
    s = value.strip()
    try:
        return float(s)
    except ValueError:
        pass
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _open_text_csv(path: Path):
    """Open a CSV file transparently, gunzipping if the suffix says so."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def from_aisdb_track(track_dict: Mapping[str, Any]) -> Track:
    """Convert an AISdb ``TrackGen`` track dictionary into an :class:`aissegments.Track`.

    AISdb yields tracks as dictionaries with at least these keys:

    - ``mmsi``  (scalar int)
    - ``time``  (1-D array of unix seconds)
    - ``lon``   (1-D array, degrees)
    - ``lat``   (1-D array, degrees)
    - ``sog``   (1-D array, knots)
    - ``cog``   (1-D array, degrees)

    Tracks are passed through with no resampling or smoothing — TDKC is
    designed to operate on the raw AIS time series.

    Parameters
    ----------
    track_dict : Mapping[str, Any]
        A track dict in AISdb's column-array layout.

    Returns
    -------
    Track
        A validated :class:`aissegments.Track`.

    Raises
    ------
    KeyError
        If the dict is missing any required key.
    ValueError
        If the array fields disagree on length or ``time`` is not monotonic.
    """
    required = ("mmsi", "time", "lon", "lat", "sog", "cog")
    missing = [k for k in required if k not in track_dict]
    if missing:
        raise KeyError(f"AISdb track dict is missing keys: {missing}")
    return Track.from_arrays(
        mmsi=int(track_dict["mmsi"]),
        t=np.asarray(track_dict["time"], dtype=np.float64),
        lon=np.asarray(track_dict["lon"], dtype=np.float64),
        lat=np.asarray(track_dict["lat"], dtype=np.float64),
        sog=np.asarray(track_dict["sog"], dtype=np.float64),
        cog=np.asarray(track_dict["cog"], dtype=np.float64),
    )


def read_csv_tracks(path: Path | str) -> list[Track]:
    """Load AIS pings from a CSV (optionally ``.gz``) and group them by MMSI.

    The header row must contain (in any order, case-insensitive) one
    column for each of the six required fields below.  Common aliases are
    accepted so that files exported by different data providers work
    out-of-the-box:

    | Field | Recognised header names                                          |
    |-------|------------------------------------------------------------------|
    | mmsi  | ``mmsi``                                                         |
    | time  | ``time``, ``BaseDateTime``, ``timestamp``, ``datetime``, ``date_time`` |
    | lon   | ``lon``, ``longitude``, ``LON``                                  |
    | lat   | ``lat``, ``latitude``, ``LAT``                                   |
    | sog   | ``sog``, ``speed``                                               |
    | cog   | ``cog``, ``course``                                              |

    Time values are accepted as either:

    - Unix seconds (numeric, e.g. ``1625097600``), or
    - ISO 8601 strings (e.g. ``2019-01-01T14:15:12``, ``...Z``, or
      ``...+00:00``).  Naive timestamps are interpreted as UTC.

    Rows are grouped by ``mmsi`` and sorted by time within each group, so
    the order of rows in the source file does not matter.

    Parameters
    ----------
    path : Path or str
        Path to a ``.csv`` or ``.csv.gz`` file.

    Returns
    -------
    list[Track]
        One :class:`Track` per unique MMSI, sorted by descending track length
        so callers can pick the densest tracks easily.

    Raises
    ------
    KeyError
        If a required column is missing from the header.
    ValueError
        If the file is empty, or any required field is missing / non-numeric
        / un-parseable.
    """
    path = Path(path)
    by_mmsi: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)

    with _open_text_csv(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV (no header row): {path}")
        col_map = _resolve_csv_columns(list(reader.fieldnames))

        for line_no, row in enumerate(reader, start=2):
            try:
                mmsi = int(row[col_map["mmsi"]])
                t = _to_unix_seconds(row[col_map["time"]])
                lon = float(row[col_map["lon"]])
                lat = float(row[col_map["lat"]])
                sog = float(row[col_map["sog"]])
                cog = float(row[col_map["cog"]])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row {line_no} in {path}: {exc}") from exc
            by_mmsi[mmsi].append((t, lon, lat, sog, cog))

    tracks: list[Track] = []
    for mmsi, rows in by_mmsi.items():
        rows.sort(key=lambda r: r[0])
        cols = np.asarray(rows, dtype=np.float64).T
        tracks.append(
            Track(
                mmsi=mmsi,
                t=cols[0],
                lon=cols[1],
                lat=cols[2],
                sog=cols[3],
                cog=cols[4],
            )
        )
    tracks.sort(key=len, reverse=True)
    return tracks


def _to_int_or_none(value: str) -> int | None:
    """Parse a CSV cell as int, tolerating empties, whitespace, and floats."""
    s = (value or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: str) -> float | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_csv_static_records(path: Path | str) -> dict[int, dict[str, Any]]:
    """Extract per-MMSI static info from a CSV (e.g. Marine Cadastre).

    The CSV must have ``mmsi`` and a recognised time column (see
    :func:`read_csv_tracks` for aliases).  Optional static columns are
    detected case-insensitively:

    | Output key  | Recognised headers             |
    |-------------|---------------------------------|
    | vessel_name | ``VesselName``, ``name``        |
    | call_sign   | ``CallSign``                    |
    | imo         | ``IMO``, ``imo_num``            |
    | ship_type   | ``VesselType``, ``ShipType``    |
    | destination | ``Destination``                 |
    | dim_bow / dim_stern | (split from ``Length`` / ``LOA``)  — half each |
    | dim_port / dim_star | (split from ``Width`` / ``Beam``)  — half each |
    | draught     | ``Draft``, ``Draught``          |

    Marine Cadastre publishes overall ``Length``/``Width`` rather than the
    per-quadrant antenna offsets that AIS Type-5 carries.  The split is
    halved into ``dim_bow``/``dim_stern`` (and similarly for width) — a
    centred-antenna approximation, which is what most AIS feeds report
    when the antenna position is unknown.

    Multiple rows per MMSI are merged with **last non-NULL value wins per
    field**, and the latest ``time`` is recorded.  Returns ``{}`` if no
    recognised static columns are present.

    Output dict is shaped to match AISdb's static-row dict so the same
    downstream code (e.g. OMRAT's ``_ensure_static``/``_ensure_state``)
    can consume both data sources uniformly.

    Raises
    ------
    ValueError
        If the file is empty (no header).
    KeyError
        If ``mmsi`` or ``time`` columns are missing.
    """
    path = Path(path)
    out: dict[int, dict[str, Any]] = {}

    with _open_text_csv(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV (no header row): {path}")
        kinematic_map = _resolve_csv_columns(list(reader.fieldnames))
        # Build the static column map; absent columns are simply skipped.
        lower_to_actual = {c.strip().lower(): c for c in reader.fieldnames}
        static_map: dict[str, str] = {}
        for canonical, aliases in _STATIC_COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower_to_actual:
                    static_map[canonical] = lower_to_actual[alias]
                    break
        if not static_map:
            return {}

        for row in reader:
            try:
                mmsi = int(row[kinematic_map["mmsi"]])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                t = _to_unix_seconds(row[kinematic_map["time"]])
            except (TypeError, ValueError, KeyError):
                continue

            entry = out.setdefault(mmsi, {"mmsi": mmsi, "time": t})
            # Track latest timestamp per MMSI.
            if t > entry.get("time", t):
                entry["time"] = t

            for canonical, src_col in static_map.items():
                # csv.DictReader yields "" for missing/blank cells; the
                # parsers below all return None for empty input so no
                # explicit None-guard is needed here.
                raw = row.get(src_col, "")
                if canonical in ("imo", "ship_type"):
                    val = _to_int_or_none(raw)
                elif canonical in ("length", "width", "draught"):
                    val = _to_float_or_none(raw)
                else:
                    s = str(raw).strip()
                    val = s or None
                if val is not None:
                    entry[canonical] = val

    # Convert overall Length / Width into AISdb's per-quadrant antenna
    # offsets (halved — centred-antenna approximation).
    for entry in out.values():
        length = entry.pop("length", None)
        width = entry.pop("width", None)
        if length is not None:
            half = length / 2.0
            entry["dim_bow"] = half
            entry["dim_stern"] = half
        if width is not None:
            half = width / 2.0
            entry["dim_port"] = half
            entry["dim_star"] = half

    return out
