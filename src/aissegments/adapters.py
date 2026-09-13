"""Adapters for getting AIS data into ``aissegments``.

- :func:`from_aisdb_track` — single AISdb track-dict → :class:`Track`.
- :func:`read_csv_tracks` — generic CSV loader; groups by MMSI, returns one
  :class:`Track` per vessel.  Recognises common column name variants
  (Marine Cadastre's ``BaseDateTime``/``LAT``/``LON``, DMA's
  ``# Timestamp``, etc.) and parses ISO 8601, ``dd/mm/yyyy`` and unix
  timestamps.  Handles ``.gz`` transparently.
- :func:`read_csv_static_records` — companion to ``read_csv_tracks`` that
  surfaces per-vessel static fields (ship type, dimensions, draught,
  IMO, …) when the source CSV carries them.  Output is shaped to match
  AISdb's static-row dict so downstream consumers can treat aisdb-decoded
  data and rich-CSV data uniformly.

Column-layout knowledge is shared with the Parquet adapters through
:mod:`aissegments._schema`.  None of these pull in extra runtime
dependencies beyond ``numpy``.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from aissegments._schema import (
    DEFAULT_MOBILE_TYPES,
    MOBILE_ALIASES,
    convert_static_value,
    normalise_mobile_type,
    parse_time_string,
    resolve_column,
    resolve_kinematics,
    resolve_static,
    split_length_width,
    warn_if_mobile_filter_dropped_everything,
)
from aissegments._types import Track


def _mobile_keep(mobile_types: tuple[str, ...] | None):
    """Return ``cell -> bool`` for the transceiver-class filter (``None`` = keep all)."""
    if mobile_types is None:
        return None
    wanted = {normalise_mobile_type(m) for m in mobile_types}
    return lambda cell: normalise_mobile_type(cell) in wanted


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


def read_csv_tracks(
    path: Path | str,
    *,
    mobile_types: tuple[str, ...] | None = DEFAULT_MOBILE_TYPES,
    time_format: str | None = None,
    dayfirst: bool = True,
    skip_invalid: bool = False,
) -> list[Track]:
    """Load AIS pings from a CSV (optionally ``.gz``) and group them by MMSI.

    The header row must contain (in any order, case-insensitive) one
    column for each of the six required fields below.  Common aliases are
    accepted so that files exported by different data providers work
    out-of-the-box:

    | Field | Recognised header names                                          |
    |-------|------------------------------------------------------------------|
    | mmsi  | ``mmsi``                                                         |
    | time  | ``time``, ``BaseDateTime``, ``timestamp``, ``# Timestamp``, ``datetime``, ``date_time`` |
    | lon   | ``lon``, ``longitude``, ``LON``                                  |
    | lat   | ``lat``, ``latitude``, ``LAT``                                   |
    | sog   | ``sog``, ``speed``                                               |
    | cog   | ``cog``, ``course``                                              |

    Time values are accepted as:

    - Unix seconds (numeric, e.g. ``1625097600``),
    - ISO 8601 strings (e.g. ``2019-01-01T14:15:12``, ``...Z``, or
      ``...+00:00``), or
    - ``dd/mm/yyyy HH:MM:SS`` (``mm/dd`` with ``dayfirst=False``).

    Naive timestamps are interpreted as UTC.  Any other layout needs an
    explicit ``time_format``.

    Rows are grouped by ``mmsi`` and sorted by time within each group, so
    the order of rows in the source file does not matter.

    Parameters
    ----------
    path : Path or str
        Path to a ``.csv`` or ``.csv.gz`` file.
    mobile_types : tuple of str, optional
        Transceiver classes to keep when a class column (``Type of mobile``,
        ``TransceiverClass``, ...) is present.  Compared case-insensitively
        with a leading ``"Class"`` ignored, so the default
        ``("Class A", "Class B")`` keeps ``A``/``B`` too and drops base
        stations and AtoN.  Pass ``None`` to keep every row.  A warning is
        emitted if the filter removes every row of a non-empty file.
    time_format : str, optional
        strptime pattern for the time column.  Skips auto-detection.
    dayfirst : bool
        How to read ambiguous ``xx/xx/yyyy`` strings.
    skip_invalid : bool
        Drop rows with missing or non-numeric kinematic fields instead of
        raising.  Useful for feeds where Class B or static-only rows leave
        SOG/COG blank.

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
        If the file is empty, or (unless ``skip_invalid``) any required
        field is missing / non-numeric / un-parseable.
    """
    path = Path(path)
    by_mmsi: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    keep_mobile = _mobile_keep(mobile_types)
    n_rows = n_mobile_kept = 0

    with _open_text_csv(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV (no header row): {path}")
        names = list(reader.fieldnames)
        col_map = resolve_kinematics(names, what="CSV")
        mobile_col = resolve_column(names, MOBILE_ALIASES)

        for line_no, row in enumerate(reader, start=2):
            n_rows += 1
            if mobile_col is not None and keep_mobile is not None:
                if not keep_mobile(row.get(mobile_col)):
                    continue
                n_mobile_kept += 1
            try:
                mmsi = int(row[col_map["mmsi"]])
                t = parse_time_string(
                    row[col_map["time"]], time_format=time_format, dayfirst=dayfirst
                )
                lon = float(row[col_map["lon"]])
                lat = float(row[col_map["lat"]])
                sog = float(row[col_map["sog"]])
                cog = float(row[col_map["cog"]])
            except (TypeError, ValueError) as exc:
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid row {line_no} in {path}: {exc}") from exc
            by_mmsi[mmsi].append((t, lon, lat, sog, cog))

    warn_if_mobile_filter_dropped_everything(
        n_rows=n_rows,
        n_kept=n_mobile_kept,
        mobile_col=mobile_col,
        mobile_types=mobile_types,
        path=path,
    )

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


def read_csv_static_records(
    path: Path | str,
    *,
    mobile_types: tuple[str, ...] | None = DEFAULT_MOBILE_TYPES,
    time_format: str | None = None,
    dayfirst: bool = True,
    ship_type_names: Mapping[str, int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Extract per-MMSI static info from a CSV (Marine Cadastre, DMA aisdk, ...).

    The CSV must have ``mmsi`` and a recognised time column (see
    :func:`read_csv_tracks` for aliases).  Optional static columns are
    detected case-insensitively:

    | Output key  | Recognised headers                                  |
    |-------------|-----------------------------------------------------|
    | vessel_name | ``VesselName``, ``name``                             |
    | call_sign   | ``CallSign``                                         |
    | imo         | ``IMO``, ``imo_num``                                 |
    | ship_type   | ``VesselType``, ``ShipType``, ``Ship type``          |
    | destination | ``Destination``                                      |
    | dim_bow / dim_stern | ``dim_a``/``dim_b``, ``to_bow``/``to_stern``, or a full ``A``/``B``/``C``/``D`` set; else half of ``Length`` / ``LOA`` |
    | dim_port / dim_star | ``dim_c``/``dim_d``, ``to_port``/``to_starboard``, or ``C``/``D``; else half of ``Width`` / ``Beam`` |
    | draught     | ``Draft``, ``Draught``                               |

    Marine Cadastre publishes overall ``Length``/``Width`` rather than the
    per-quadrant antenna offsets that AIS Type-5 carries; those are halved
    into ``dim_bow``/``dim_stern`` (and similarly for width) as a
    centred-antenna approximation.  True offsets win when present.

    Ship types may be numeric codes or decoded names (``"Cargo"``); names
    are mapped through ``ship_type_names`` (default
    :data:`aissegments.DEFAULT_SHIP_TYPE_NAMES`) and unknown names are
    skipped.  Non-numeric IMO cells (``"Unknown"``) are skipped.

    Multiple rows per MMSI are merged with **last non-NULL value wins per
    field**, and the latest ``time`` is recorded.  Returns ``{}`` if no
    recognised static columns are present.  See :func:`read_csv_tracks`
    for ``mobile_types``, ``time_format`` and ``dayfirst``.

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
    keep_mobile = _mobile_keep(mobile_types)
    n_rows = n_mobile_kept = 0

    with _open_text_csv(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV (no header row): {path}")
        names = list(reader.fieldnames)
        kinematic_map = resolve_kinematics(names, what="CSV")
        mobile_col = resolve_column(names, MOBILE_ALIASES)
        static_map = resolve_static(names)
        if not static_map:
            return {}

        for row in reader:
            n_rows += 1
            if mobile_col is not None and keep_mobile is not None:
                if not keep_mobile(row.get(mobile_col)):
                    continue
                n_mobile_kept += 1
            try:
                mmsi = int(row[kinematic_map["mmsi"]])
            except (TypeError, ValueError, KeyError):
                continue
            try:
                t = parse_time_string(
                    row[kinematic_map["time"]], time_format=time_format, dayfirst=dayfirst
                )
            except (TypeError, ValueError, KeyError):
                continue

            entry = out.setdefault(mmsi, {"mmsi": mmsi, "time": t})
            # Track latest timestamp per MMSI.
            if t > entry.get("time", t):
                entry["time"] = t

            for canonical, src_col in static_map.items():
                # csv.DictReader yields "" for missing/blank cells;
                # convert_static_value returns None for those.
                raw = (row.get(src_col) or "").strip()
                if not raw:
                    continue
                val = convert_static_value(canonical, raw, ship_type_names)
                if val is not None:
                    entry[canonical] = val

    warn_if_mobile_filter_dropped_everything(
        n_rows=n_rows,
        n_kept=n_mobile_kept,
        mobile_col=mobile_col,
        mobile_types=mobile_types,
        path=path,
    )
    for entry in out.values():
        split_length_width(entry)
    return out
