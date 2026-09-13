"""Streaming Parquet adapters for getting AIS data into ``aissegments``.

- :func:`read_parquet_tracks` — batch-streamed loader; groups pings by MMSI
  and yields one :class:`Track` per vessel.  Designed for monthly AIS dumps
  in the hundred-million-ping range: pings are accumulated in compact
  dtypes (``int32`` MMSI, ``float32`` kinematics) so a 135M-row month fits
  in a few GB of RAM instead of tens.
- :func:`read_parquet_static_records` — companion that surfaces per-vessel
  static fields (ship type, dimensions, draught, IMO, …).  Output is shaped
  to match AISdb's static-row dict, exactly like
  :func:`aissegments.read_csv_static_records`.

Column-layout knowledge (aliases, transceiver classes, ship-type names,
timestamp formats) is shared with the CSV adapters through
:mod:`aissegments._schema`, so any provider layout that works through one
reader works through the other.

Requires the optional ``pyarrow`` dependency (``pip install
aissegments[parquet]``); everything is imported lazily so the rest of the
package stays numpy-only.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from aissegments._schema import (
    DEFAULT_MOBILE_TYPES,
    FLOAT_STATIC_FIELDS,
    MOBILE_ALIASES,
    convert_static_value,
    normalise_mobile_type,
    resolve_column,
    resolve_kinematics,
    resolve_static,
    sniff_time_format,
    split_length_width,
    warn_if_mobile_filter_dropped_everything,
)
from aissegments._types import Track

__all__ = [
    "read_parquet_static_records",
    "read_parquet_tracks",
]

# Largest assignable MMSI (9 digits).  Anything outside (0, this] is a
# decoding artefact and gets dropped.
_MAX_MMSI = 999_999_999


def _import_pyarrow():
    """Import pyarrow lazily; raise a helpful error when it is missing."""
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading Parquet AIS files requires the optional 'pyarrow' "
            "dependency: pip install pyarrow (or aissegments[parquet])"
        ) from exc
    return pa, pc, pq


def _first_time_sample(pf, time_col: str) -> str:
    """Return the first non-null value of a string time column."""
    for batch in pf.iter_batches(batch_size=4096, columns=[time_col]):
        for value in batch.column(0):
            if value.is_valid:
                return str(value.as_py())
    raise ValueError(f"Timestamp column {time_col!r} has no non-null values")


def _make_time_parser(pf, time_col: str, *, time_format: str | None, dayfirst: bool):
    """Build ``arrow array -> float64 unix-seconds (NaN = invalid)``."""
    pa, pc, _ = _import_pyarrow()
    field_type = pf.schema_arrow.field(time_col).type

    def _ts_to_seconds(ts_arr) -> np.ndarray:
        secs = pc.cast(pc.cast(ts_arr, pa.timestamp("s"), safe=False), pa.int64())
        out = secs.to_numpy(zero_copy_only=False).astype(np.float64)
        mask = pc.is_null(secs).to_numpy(zero_copy_only=False)
        out[mask] = np.nan
        return out

    if pa.types.is_timestamp(field_type):
        return _ts_to_seconds
    if not (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
        # Plain numeric column: unix seconds already.
        return lambda arr: _float_np(pc, pa, arr)

    if time_format is None:
        time_format = sniff_time_format(_first_time_sample(pf, time_col), dayfirst=dayfirst)
    if time_format == "unix":
        return lambda arr: _float_np(pc, pa, arr)
    fmt = time_format
    return lambda arr: _ts_to_seconds(pc.strptime(arr, format=fmt, unit="s", error_is_null=True))


def _column(batch, name: str):
    """Fetch a column from a RecordBatch by name (index-based for pyarrow compat)."""
    return batch.column(batch.schema.get_field_index(name))


def _float_np(pc, pa, col) -> np.ndarray:
    """Arrow column → float64 numpy with NaN for nulls."""
    arr = pc.cast(col, pa.float64(), safe=False)
    out = arr.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    mask = pc.is_null(arr).to_numpy(zero_copy_only=False)
    if mask.any():
        out = out.copy()
        out[mask] = np.nan
    return out


def _string_valid_np(pc, pa, col) -> np.ndarray:
    """Bool mask: cell is non-null (and not an empty string, for string cols)."""
    ok = pc.is_valid(col)
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        ok = pc.and_(
            ok,
            pc.fill_null(pc.not_equal(col, pa.scalar("", type=col.type)), False),
        )
    return ok.to_numpy(zero_copy_only=False).astype(bool)


def _mobile_mask(
    pc, pa, batch, mobile_col: str | None, mobile_types: tuple[str, ...] | None, n: int
) -> np.ndarray:
    """Row mask keeping only the requested transceiver classes.

    Values are compared after :func:`normalise_mobile_type` on both sides,
    so ``("Class A", "Class B")`` also keeps Marine Cadastre's ``A``/``B``.
    """
    if mobile_col is None or mobile_types is None:
        return np.ones(n, dtype=bool)
    col = _column(batch, mobile_col)
    if pa.types.is_dictionary(col.type):
        col = pc.cast(col, col.type.value_type)
    wanted = {normalise_mobile_type(m) for m in mobile_types}
    # The class column has a handful of distinct values, so normalise those
    # in Python and match the raw cells against the survivors.
    raw_matches = [
        v
        for v in pc.unique(col).to_pylist()
        if v is not None and normalise_mobile_type(v) in wanted
    ]
    if not raw_matches:
        return np.zeros(n, dtype=bool)
    ok = pc.fill_null(pc.is_in(col, value_set=pa.array(raw_matches, type=col.type)), False)
    return ok.to_numpy(zero_copy_only=False).astype(bool)


def read_parquet_tracks(
    path: Path | str,
    *,
    mobile_types: tuple[str, ...] | None = DEFAULT_MOBILE_TYPES,
    time_format: str | None = None,
    dayfirst: bool = True,
    batch_size: int = 131_072,
) -> Iterator[Track]:
    """Stream AIS pings from a Parquet file, grouped into per-MMSI tracks.

    Unlike :func:`aissegments.read_csv_tracks` (which returns a list sorted
    by track density), this is a **generator** yielding tracks in ascending
    MMSI order — a 135M-ping monthly dump would not fit in memory as
    float64 lists.  Pings are accumulated in compact dtypes (~28 bytes per
    ping) and only widened to the ``float64`` a :class:`Track` requires
    when each vessel's slice is yielded.

    Required columns (case-insensitive, same aliases as the CSV reader):
    mmsi, time, lon/latitude, sog, cog.  Timestamps may be native Parquet
    timestamps, unix seconds, ISO 8601 strings, or ``dd/mm/yyyy HH:MM:SS``
    strings (``mm/dd`` with ``dayfirst=False``); any other string layout
    needs an explicit ``time_format``.

    Rows are dropped when the transceiver class is filtered out, the MMSI
    is not a plausible 9-digit vessel identity, the position is outside
    WGS84 bounds, or any kinematic field is null.

    Parameters
    ----------
    path : Path or str
        Path to the ``.parquet`` file.
    mobile_types : tuple of str, optional
        Transceiver classes to keep when a class column (``Type of mobile``,
        ``TransceiverClass``, ...) is present.  Compared case-insensitively
        with a leading ``"Class"`` ignored, so the default
        ``("Class A", "Class B")`` keeps ``A``/``B`` too and drops base
        stations and AtoN.  Pass ``None`` to keep every row.  A warning is
        emitted if the filter removes every row of a non-empty file.
    time_format : str, optional
        strptime pattern for a string time column.  Skips format sniffing.
    dayfirst : bool
        How to read ambiguous ``xx/xx/yyyy`` strings when sniffing.
    batch_size : int
        Arrow record-batch size for the streaming read.

    Yields
    ------
    Track
        One validated, time-sorted :class:`Track` per unique MMSI.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If the timestamp format cannot be recognised.
    ImportError
        If ``pyarrow`` is not installed.
    """
    pa, pc, pq = _import_pyarrow()
    pf = pq.ParquetFile(Path(path))
    names = list(pf.schema_arrow.names)
    cols = resolve_kinematics(names, what="Parquet")
    mobile_col = resolve_column(names, MOBILE_ALIASES)
    parse_time = _make_time_parser(pf, cols["time"], time_format=time_format, dayfirst=dayfirst)

    read_cols = list(cols.values())
    if mobile_col is not None and mobile_types is not None:
        read_cols.append(mobile_col)

    mmsi_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    lon_chunks: list[np.ndarray] = []
    lat_chunks: list[np.ndarray] = []
    sog_chunks: list[np.ndarray] = []
    cog_chunks: list[np.ndarray] = []
    n_rows = n_mobile_kept = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        n = batch.num_rows
        n_rows += n
        mobile_ok = _mobile_mask(pc, pa, batch, mobile_col, mobile_types, n)
        n_mobile_kept += int(mobile_ok.sum())

        mmsi = _float_np(pc, pa, _column(batch, cols["mmsi"]))
        t = np.asarray(parse_time(_column(batch, cols["time"])), dtype=np.float64)
        lon = _float_np(pc, pa, _column(batch, cols["lon"]))
        lat = _float_np(pc, pa, _column(batch, cols["lat"]))
        sog = _float_np(pc, pa, _column(batch, cols["sog"]))
        cog = _float_np(pc, pa, _column(batch, cols["cog"]))

        keep = (
            mobile_ok
            & np.isfinite(mmsi)
            & (mmsi > 0)
            & (mmsi <= _MAX_MMSI)
            & np.isfinite(t)
            & np.isfinite(lon)
            & (np.abs(lon) <= 180.0)
            & np.isfinite(lat)
            & (np.abs(lat) <= 90.0)
            & np.isfinite(sog)
            & np.isfinite(cog)
        )
        if not keep.any():
            continue
        mmsi_chunks.append(mmsi[keep].astype(np.int32))
        t_chunks.append(t[keep])
        lon_chunks.append(lon[keep].astype(np.float32))
        lat_chunks.append(lat[keep].astype(np.float32))
        sog_chunks.append(sog[keep].astype(np.float32))
        cog_chunks.append(cog[keep].astype(np.float32))

    warn_if_mobile_filter_dropped_everything(
        n_rows=n_rows,
        n_kept=n_mobile_kept,
        mobile_col=mobile_col,
        mobile_types=mobile_types,
        path=path,
    )
    if not mmsi_chunks:
        return

    def _consume(chunks: list[np.ndarray]) -> np.ndarray:
        merged = np.concatenate(chunks)
        chunks.clear()  # free per-batch chunks before the next concat
        return merged

    mmsi_all = _consume(mmsi_chunks)
    t_all = _consume(t_chunks)
    lon_all = _consume(lon_chunks)
    lat_all = _consume(lat_chunks)
    sog_all = _consume(sog_chunks)
    cog_all = _consume(cog_chunks)

    order = np.lexsort((t_all, mmsi_all))
    mmsi_all = mmsi_all[order]
    t_all = t_all[order]
    lon_all = lon_all[order]
    lat_all = lat_all[order]
    sog_all = sog_all[order]
    cog_all = cog_all[order]
    del order

    starts = np.concatenate(
        ([0], np.flatnonzero(mmsi_all[1:] != mmsi_all[:-1]) + 1, [len(mmsi_all)])
    )
    for i in range(len(starts) - 1):
        s, e = int(starts[i]), int(starts[i + 1])
        yield Track(
            mmsi=int(mmsi_all[s]),
            t=t_all[s:e].astype(np.float64, copy=True),
            lon=lon_all[s:e].astype(np.float64),
            lat=lat_all[s:e].astype(np.float64),
            sog=sog_all[s:e].astype(np.float64),
            cog=cog_all[s:e].astype(np.float64),
        )


def _last_per_mmsi(mmsi_subset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(unique_mmsi, index_of_last_occurrence)`` within the subset."""
    unique, rev_first = np.unique(mmsi_subset[::-1], return_index=True)
    return unique, len(mmsi_subset) - 1 - rev_first


def read_parquet_static_records(
    path: Path | str,
    *,
    mobile_types: tuple[str, ...] | None = DEFAULT_MOBILE_TYPES,
    time_format: str | None = None,
    dayfirst: bool = True,
    ship_type_names: Mapping[str, int] | None = None,
    batch_size: int = 131_072,
) -> dict[int, dict[str, Any]]:
    """Extract per-MMSI static info from an AIS Parquet file.

    The Parquet companion to :func:`aissegments.read_csv_static_records`;
    output dicts use the same AISdb-shaped keys (``dim_bow``/``dim_stern``/
    ``dim_port``/``dim_star``, ``imo``, ``ship_type``, ``draught``,
    ``destination``, ``vessel_name``, ``call_sign``, ``time``) so the same
    downstream code consumes both.

    - Per-quadrant antenna offsets (``dim_a``.. or a full set of
      ``A``/``B``/``C``/``D`` columns) are used directly when present;
      halved ``Length``/``Width`` remain the fallback.
    - Ship types may be numeric codes or decoded names; names are mapped
      to AIS ``type_and_cargo`` codes through ``ship_type_names``
      (default :data:`aissegments.DEFAULT_SHIP_TYPE_NAMES`).  Names that
      carry no information (``"Undefined"``, ...) are skipped.
    - Non-numeric ``IMO`` cells like ``"Unknown"`` are skipped.

    Merging is **last usable value wins per field** in file order; ``time``
    records the latest timestamp seen per MMSI.  Returns ``{}`` when no
    recognised static column exists.  See :func:`read_parquet_tracks` for
    ``mobile_types``, ``time_format`` and ``dayfirst``.
    """
    pa, pc, pq = _import_pyarrow()
    pf = pq.ParquetFile(Path(path))
    names = list(pf.schema_arrow.names)
    cols = resolve_kinematics(names, what="Parquet")  # raises if mmsi/time missing
    mobile_col = resolve_column(names, MOBILE_ALIASES)
    parse_time = _make_time_parser(pf, cols["time"], time_format=time_format, dayfirst=dayfirst)

    static_map = resolve_static(names)
    if not static_map:
        return {}

    read_cols = [cols["mmsi"], cols["time"], *static_map.values()]
    if mobile_col is not None and mobile_types is not None:
        read_cols.append(mobile_col)

    out: dict[int, dict[str, Any]] = {}
    n_rows = n_mobile_kept = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        n = batch.num_rows
        n_rows += n
        mobile_ok = _mobile_mask(pc, pa, batch, mobile_col, mobile_types, n)
        n_mobile_kept += int(mobile_ok.sum())

        mmsi_f = _float_np(pc, pa, _column(batch, cols["mmsi"]))
        t = np.asarray(parse_time(_column(batch, cols["time"])), dtype=np.float64)
        base = (
            mobile_ok & np.isfinite(mmsi_f) & (mmsi_f > 0) & (mmsi_f <= _MAX_MMSI) & np.isfinite(t)
        )
        if not base.any():
            continue
        mmsi = mmsi_f.astype(np.int64)

        # Latest timestamp per MMSI (any surviving row).
        base_idx = np.flatnonzero(base)
        unique, last = _last_per_mmsi(mmsi[base_idx])
        for m, tv in zip(unique.tolist(), t[base_idx[last]].tolist(), strict=True):
            entry = out.setdefault(int(m), {"mmsi": int(m), "time": float(tv)})
            if float(tv) > entry["time"]:
                entry["time"] = float(tv)

        for canonical, src_col in static_map.items():
            col = _column(batch, src_col)
            if canonical in FLOAT_STATIC_FIELDS:
                vals_f = _float_np(pc, pa, col)
                valid = base & np.isfinite(vals_f)
                if not valid.any():
                    continue
                idx = np.flatnonzero(valid)
                unique, last = _last_per_mmsi(mmsi[idx])
                picked: list[Any] = vals_f[idx[last]].tolist()
            else:
                valid = base & _string_valid_np(pc, pa, col)
                if not valid.any():
                    continue
                idx = np.flatnonzero(valid)
                unique, last = _last_per_mmsi(mmsi[idx])
                # Materialise only the handful of picked cells, not the
                # whole 100k-row string column.
                picked = col.take(pa.array(idx[last])).to_pylist()
            for m, v in zip(unique.tolist(), picked, strict=True):
                converted = convert_static_value(canonical, v, ship_type_names)
                if converted is not None:
                    out[int(m)][canonical] = converted

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
