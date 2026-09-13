"""Column-layout knowledge shared by the CSV and Parquet adapters.

Everything that is *about the data source* rather than *about the file
format* lives here: which header names map to which canonical field, how
transceiver classes are spelled, how decoded ship-type names map back to
AIS ``type_and_cargo`` codes, and how string timestamps are parsed.  The
CSV and Parquet readers are thin format layers on top of this module, so
a provider layout that works through one works through the other.

Recognised layouts out of the box:

- Marine Cadastre (``BaseDateTime``, ``LAT``/``LON``, ``VesselType`` as a
  numeric code, overall ``Length``/``Width``, ``TransceiverClass`` = ``A``/``B``).
- Danish Maritime Authority ``aisdk`` (``# Timestamp`` as day-first
  ``dd/mm/yyyy HH:MM:SS``, ``Type of mobile`` = ``Class A``/``Class B``,
  ``Ship type`` as a decoded name, per-quadrant antenna offsets
  ``A``/``B``/``C``/``D``).
- Anything using the plain lower-case canonical names (``mmsi``, ``time``,
  ``lon``, ``lat``, ``sog``, ``cog``, ...).

Other layouts are reached through the reader keyword arguments
(``time_format``, ``dayfirst``, ``mobile_types``, ``ship_type_names``).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "DEFAULT_MOBILE_TYPES",
    "DEFAULT_SHIP_TYPE_NAMES",
    "DIM_ALIASES",
    "FLOAT_STATIC_FIELDS",
    "KINEMATIC_ALIASES",
    "MOBILE_ALIASES",
    "STATIC_ALIASES",
    "convert_static_value",
    "normalise_mobile_type",
    "parse_time_string",
    "resolve_column",
    "resolve_kinematics",
    "resolve_static",
    "ship_type_to_toc",
    "split_length_width",
    "warn_if_mobile_filter_dropped_everything",
]

# ---------------------------------------------------------------------------
# Column aliases (all matched against lower-cased, stripped header names)
# ---------------------------------------------------------------------------

#: Required kinematic fields → accepted header names.
KINEMATIC_ALIASES: dict[str, tuple[str, ...]] = {
    "mmsi": ("mmsi",),
    "time": ("time", "basedatetime", "timestamp", "# timestamp", "datetime", "date_time"),
    "lon": ("lon", "longitude"),
    "lat": ("lat", "latitude"),
    "sog": ("sog", "speed"),
    "cog": ("cog", "course"),
}

#: Optional static fields → accepted header names.  Output keys mirror
#: AISdb's static-row dict so downstream code can consume aisdb-decoded
#: rows and file-sourced rows uniformly.
STATIC_ALIASES: dict[str, tuple[str, ...]] = {
    "vessel_name": ("vesselname", "vessel_name", "name"),
    "call_sign": ("callsign", "call_sign"),
    "imo": ("imo", "imo_num"),
    "ship_type": ("vesseltype", "ship_type", "shiptype", "ship type"),
    "destination": ("destination",),
    # Overall dimensions; split into halved per-quadrant offsets when the
    # true offsets (``DIM_ALIASES``) are absent.
    "length": ("length", "loa", "ship_length"),
    "width": ("width", "beam", "breadth"),
    "draught": ("draft", "draught"),
}

#: Per-quadrant antenna offsets as AIS Type-5 actually reports them.
#: Preferred over the halved Length/Width split when present.
DIM_ALIASES: dict[str, tuple[str, ...]] = {
    "dim_bow": ("dim_bow", "dim_a", "to_bow"),
    "dim_stern": ("dim_stern", "dim_b", "to_stern"),
    "dim_port": ("dim_port", "dim_c", "to_port"),
    "dim_star": ("dim_star", "dim_d", "to_starboard"),
}

# Bare single-letter offset columns (DMA layout).  Too ambiguous to trust
# individually, so they are only accepted when all four appear together.
_DIM_SHORT: dict[str, str] = {
    "dim_bow": "a",
    "dim_stern": "b",
    "dim_port": "c",
    "dim_star": "d",
}

#: Column carrying the AIS transceiver class ("Class A", "B", "Base
#: Station", "AtoN", ...).  Used to drop non-vessel rows.
MOBILE_ALIASES: tuple[str, ...] = (
    "type of mobile",
    "mobile type",
    "transceiver class",
    "transceiverclass",
)

#: Transceiver classes kept by default.  Compared after
#: :func:`normalise_mobile_type`, so ``"A"`` / ``"class a"`` match too.
DEFAULT_MOBILE_TYPES: tuple[str, ...] = ("Class A", "Class B")

#: Static fields parsed as floats (everything else is int- or string-like).
FLOAT_STATIC_FIELDS = frozenset(
    {"draught", "length", "width", "dim_bow", "dim_stern", "dim_port", "dim_star"}
)

#: Decoded ship-type name (lower-case) → AIS ``type_and_cargo`` code.
#: Spellings follow the DMA ``aisdk`` dumps; first-digit-only categories
#: map to the ``x0`` representative (every "Cargo" subtype reports as 70).
#: Pass ``ship_type_names={**DEFAULT_SHIP_TYPE_NAMES, ...}`` to a reader
#: to extend or override it for another provider.
DEFAULT_SHIP_TYPE_NAMES: Mapping[str, int] = {
    "wig": 20,
    "fishing": 30,
    "towing": 31,
    "towing long/wide": 32,
    "dredging": 33,
    "diving": 34,
    "military": 35,
    "sailing": 36,
    "pleasure": 37,
    "hsc": 40,
    "pilot": 50,
    "sar": 51,
    "tug": 52,
    "port tender": 53,
    "anti-pollution": 54,
    "law enforcement": 55,
    "medical": 58,
    "passenger": 60,
    "cargo": 70,
    "tanker": 80,
    "other": 90,
}

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ])\d{2}:\d{2}:\d{2}")
_SLASH_RE = re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$")
_CLASS_PREFIX_RE = re.compile(r"^class\s*")


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


def resolve_column(names: list[str], aliases: tuple[str, ...]) -> str | None:
    """Return the first actual column whose lowered name is in ``aliases``."""
    lower_to_actual = {c.strip().lower(): c for c in names}
    for alias in aliases:
        if alias in lower_to_actual:
            return lower_to_actual[alias]
    return None


def resolve_kinematics(names: list[str], *, what: str = "File") -> dict[str, str]:
    """Map the six required canonical fields to actual columns, or raise ``KeyError``."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in KINEMATIC_ALIASES.items():
        col = resolve_column(names, aliases)
        if col is None:
            missing.append(canonical)
        else:
            resolved[canonical] = col
    if missing:
        raise KeyError(f"{what} missing required columns (any of these aliases): {missing}")
    return resolved


def resolve_static(names: list[str]) -> dict[str, str]:
    """Map every recognised optional static field to its actual column.

    Per-quadrant offsets are taken from the explicit ``DIM_ALIASES`` names,
    or from bare ``A``/``B``/``C``/``D`` columns when all four are present.
    Absent fields are simply omitted.
    """
    static_map: dict[str, str] = {}
    for canonical, aliases in STATIC_ALIASES.items():
        col = resolve_column(names, aliases)
        if col is not None:
            static_map[canonical] = col
    for canonical, aliases in DIM_ALIASES.items():
        col = resolve_column(names, aliases)
        if col is not None:
            static_map[canonical] = col
    unresolved = [k for k in DIM_ALIASES if k not in static_map]
    if unresolved:
        short = {k: resolve_column(names, (_DIM_SHORT[k],)) for k in DIM_ALIASES}
        if all(v is not None for v in short.values()):
            for k in unresolved:
                static_map[k] = short[k]  # type: ignore[assignment]
    return static_map


# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------


def normalise_mobile_type(value: Any) -> str:
    """Canonical form for transceiver-class comparison.

    Lower-cases, strips, and drops a leading ``"class"`` so that DMA's
    ``"Class A"`` and Marine Cadastre's ``"A"`` compare equal.
    """
    if value is None:
        return ""
    return _CLASS_PREFIX_RE.sub("", str(value).strip().lower())


def ship_type_to_toc(value: Any, names: Mapping[str, int] | None = None) -> int | None:
    """Convert a ship-type cell to an AIS ``type_and_cargo`` code.

    Accepts the numeric code itself (int, float, or numeric string — the
    Marine Cadastre convention) or a decoded name such as ``"Cargo"`` /
    ``"Law enforcement"`` looked up case-insensitively in ``names``
    (default :data:`DEFAULT_SHIP_TYPE_NAMES`).  Unknown or unavailable
    values (``"Undefined"``, ``"Reserved"``, empty) return ``None``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value) if value == value else None  # NaN guard
    s = str(value).strip()
    if not s:
        return None
    if _NUMERIC_RE.match(s):
        return int(float(s))
    table = DEFAULT_SHIP_TYPE_NAMES if names is None else names
    return table.get(s.lower())


def convert_static_value(
    canonical: str, value: Any, ship_type_names: Mapping[str, int] | None = None
) -> Any:
    """Coerce one static cell to its output type; ``None`` means skip it."""
    if value is None:
        return None
    if canonical == "ship_type":
        return ship_type_to_toc(value, ship_type_names)
    if canonical == "imo":
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None
    if canonical in FLOAT_STATIC_FIELDS:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if f == f else None
    s = str(value).strip()
    return s or None


def split_length_width(entry: dict[str, Any]) -> None:
    """Replace ``length``/``width`` with halved per-quadrant offsets, in place.

    Only fills offsets that are not already present (true ``A``/``B``/``C``/``D``
    values win).  The halving is a centred-antenna approximation.
    """
    length = entry.pop("length", None)
    width = entry.pop("width", None)
    if length is not None and "dim_bow" not in entry:
        entry["dim_bow"] = length / 2.0
        entry["dim_stern"] = length / 2.0
    if width is not None and "dim_port" not in entry:
        entry["dim_port"] = width / 2.0
        entry["dim_star"] = width / 2.0


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def slash_time_format(dayfirst: bool) -> str:
    """strptime format for ``dd/mm/yyyy HH:MM:SS`` (or ``mm/dd/yyyy``)."""
    return "%d/%m/%Y %H:%M:%S" if dayfirst else "%m/%d/%Y %H:%M:%S"


def sniff_time_format(sample: str, *, dayfirst: bool = True) -> str:
    """Return a strptime format (or ``"unix"``) matching one sample string.

    Recognises ISO 8601 (``2025-01-01 00:00:00`` / ``...T...``), the
    slash layout ``01/01/2025 00:00:00`` (resolved by ``dayfirst``), and
    numeric unix seconds.  Raises ``ValueError`` otherwise.
    """
    s = sample.strip()
    m = _ISO_RE.match(s)
    if m:
        return f"%Y-%m-%d{m.group(1)}%H:%M:%S"
    if _SLASH_RE.match(s):
        return slash_time_format(dayfirst)
    if _NUMERIC_RE.match(s):
        return "unix"
    raise ValueError(
        f"Unrecognised timestamp format {s!r}; pass time_format= with a strptime pattern"
    )


def parse_time_string(
    value: str, *, time_format: str | None = None, dayfirst: bool = True
) -> float:
    """Parse one timestamp cell to unix seconds (naive stamps are UTC).

    With ``time_format`` set, that strptime pattern is used and nothing
    else is tried.  Otherwise numeric unix seconds, ISO 8601 (including a
    ``Z`` suffix or offset), and the slash layout are accepted.
    """
    s = value.strip()
    if time_format is not None:
        dt = datetime.strptime(s, time_format)
        return dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp()
    try:
        return float(s)
    except ValueError:
        pass
    if _SLASH_RE.match(s):
        dt = datetime.strptime(s, slash_time_format(dayfirst))
        return dt.replace(tzinfo=timezone.utc).timestamp()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def warn_if_mobile_filter_dropped_everything(
    *,
    n_rows: int,
    n_kept: int,
    mobile_col: str | None,
    mobile_types: tuple[str, ...] | None,
    path: Any,
) -> None:
    """Warn when a transceiver-class filter removed every row of a non-empty file.

    This is almost always a spelling mismatch between ``mobile_types`` and
    the values in the file rather than a file with no vessels.
    """
    if mobile_col is None or mobile_types is None or n_rows == 0 or n_kept > 0:
        return
    warnings.warn(
        f"{path}: mobile_types={mobile_types!r} matched none of the {n_rows} values in "
        f"column {mobile_col!r}; check the spelling or pass mobile_types=None",
        stacklevel=3,
    )
