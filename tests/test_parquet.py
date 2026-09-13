"""Tests for the streaming Parquet adapters (DMA aisdk, Marine Cadastre, generic layouts)."""

from __future__ import annotations

import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aissegments import (
    DEFAULT_SHIP_TYPE_NAMES,
    Track,
    read_parquet_static_records,
    read_parquet_tracks,
    tdkc_segments,
)
from aissegments.parquet import _first_time_sample, _import_pyarrow

T0 = 1_735_689_600.0  # 2025-01-01 00:00:00 UTC


def _write(tmp_path, columns: dict, name: str = "sample.parquet"):
    """Write a Parquet file from a plain column dict; return its path."""
    path = tmp_path / name
    pq.write_table(pa.table(columns), path)
    return path


def _dma_columns(n_extra_static: bool = True) -> dict:
    """A tiny two-vessel file in DMA aisdk column layout (shuffled in time)."""
    cols = {
        "# Timestamp": [
            "01/01/2025 00:00:10",
            "01/01/2025 00:00:00",
            "01/01/2025 00:00:00",
            "01/01/2025 00:00:20",
            "01/01/2025 00:00:10",
            "01/01/2025 00:00:05",
        ],
        "Type of mobile": [
            "Class A",
            "Class A",
            "Base Station",
            "Class A",
            "Class B",
            "AtoN",
        ],
        "MMSI": [219000001, 219000001, 2190064, 219000001, 265000002, 992191234],
        "Latitude": [57.01, 57.0, 57.5, 57.02, 56.5, 57.9],
        "Longitude": [11.01, 11.0, 11.5, 11.02, 11.6, 11.9],
        "SOG": [10.0, 10.0, None, 10.0, 5.0, None],
        "COG": [45.0, 45.0, None, 45.0, 90.0, None],
    }
    if n_extra_static:
        cols.update(
            {
                "Ship type": ["Cargo", "Cargo", None, "Cargo", "Sailing", None],
                "Cargo type": [None] * 6,
                "IMO": ["9612325", "9612325", None, "9612325", "Unknown", None],
                "Name": ["IMAVERE", "IMAVERE", None, "IMAVERE", "SVALAN", None],
                "Callsign": ["5BWV4", "5BWV4", None, "5BWV4", None, None],
                "Destination": ["EEBEK", "EEBEK", None, "EEBEK", "  ", None],
                "Draught": [6.1, 6.0, None, 6.1, None, None],
                "A": [154.0, 154.0, None, 154.0, None, None],
                "B": [32.0, 32.0, None, 32.0, None, None],
                "C": [6.0, 6.0, None, 6.0, None, None],
                "D": [21.0, 21.0, None, 21.0, None, None],
                "Length": [186.0, 186.0, None, 186.0, 12.0, None],
                "Width": [27.0, 27.0, None, 27.0, 4.0, None],
            }
        )
    return cols


class TestTimeFormats:
    def test_dma_day_first(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": ["01/01/2025 00:00:00"],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        (track,) = read_parquet_tracks(f)
        assert track.t[0] == T0

    def test_iso_with_space_and_t(self, tmp_path):
        for stamp in ("2025-01-01 00:00:00", "2025-01-01T00:00:00"):
            f = _write(
                tmp_path,
                {
                    "mmsi": [1001],
                    "time": [stamp],
                    "lon": [11.0],
                    "lat": [57.0],
                    "sog": [1.0],
                    "cog": [0.0],
                },
            )
            (track,) = read_parquet_tracks(f)
            assert track.t[0] == T0

    def test_numeric_string_unix(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": [str(T0)],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        (track,) = read_parquet_tracks(f)
        assert track.t[0] == T0

    def test_numeric_column_unix(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": [T0],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        (track,) = read_parquet_tracks(f)
        assert track.t[0] == T0

    def test_native_timestamp_column(self, tmp_path):
        stamps = pa.array([int(T0) * 1000], type=pa.timestamp("ms"))
        f = tmp_path / "native.parquet"
        pq.write_table(
            pa.table(
                {
                    "mmsi": [1001],
                    "time": stamps,
                    "lon": [11.0],
                    "lat": [57.0],
                    "sog": [1.0],
                    "cog": [0.0],
                }
            ),
            f,
        )
        (track,) = read_parquet_tracks(f)
        assert track.t[0] == T0

    def test_unparseable_stamp_becomes_nan_and_row_dropped(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001, 1001],
                "time": ["01/01/2025 00:00:00", "99/99/9999 00:00:00"],
                "lon": [11.0, 11.0],
                "lat": [57.0, 57.0],
                "sog": [1.0, 1.0],
                "cog": [0.0, 0.0],
            },
        )
        (track,) = read_parquet_tracks(f)
        assert len(track) == 1

    def test_unrecognised_format_raises(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": ["Jan 1 2025"],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        with pytest.raises(ValueError, match="Unrecognised timestamp format"):
            list(read_parquet_tracks(f))

    def test_all_null_time_column_raises(self, tmp_path):
        f = tmp_path / "nulltime.parquet"
        pq.write_table(
            pa.table(
                {
                    "mmsi": pa.array([1001], type=pa.int64()),
                    "time": pa.array([None], type=pa.string()),
                    "lon": [11.0],
                    "lat": [57.0],
                    "sog": [1.0],
                    "cog": [0.0],
                }
            ),
            f,
        )
        with pytest.raises(ValueError, match="no non-null values"):
            list(read_parquet_tracks(f))

    def test_sample_skips_leading_nulls(self, tmp_path):
        f = tmp_path / "leadingnull.parquet"
        pq.write_table(
            pa.table(
                {
                    "time": pa.array([None, "01/01/2025 00:00:00"], type=pa.string()),
                }
            ),
            f,
        )
        assert _first_time_sample(pq.ParquetFile(f), "time") == "01/01/2025 00:00:00"

    def test_month_first_with_dayfirst_false(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": ["01/02/2025 00:00:00"],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        (day_first,) = read_parquet_tracks(f)
        (month_first,) = read_parquet_tracks(f, dayfirst=False)
        assert day_first.t[0] == T0 + 31 * 86400  # 1 Feb
        assert month_first.t[0] == T0 + 86400  # 2 Jan

    def test_explicit_time_format_overrides_sniffing(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": ["2025.01.01 00-00-00"],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        with pytest.raises(ValueError, match="time_format="):
            list(read_parquet_tracks(f))
        (track,) = read_parquet_tracks(f, time_format="%Y.%m.%d %H-%M-%S")
        assert track.t[0] == T0


class TestReadParquetTracks:
    def test_dma_grouping_sorting_and_filtering(self, tmp_path):
        f = _write(tmp_path, _dma_columns())
        tracks = list(read_parquet_tracks(f))
        # Base Station and AtoN rows dropped; two vessels remain, MMSI order.
        assert [t.mmsi for t in tracks] == [219000001, 265000002]
        assert len(tracks[0]) == 3
        assert np.all(np.diff(tracks[0].t) >= 0)  # time-sorted despite shuffle
        assert tracks[0].t[0] == T0
        assert isinstance(tracks[0], Track)
        assert tracks[0].t.dtype == np.float64

    def test_mobile_types_none_keeps_everything_with_valid_kinematics(self, tmp_path):
        f = _write(tmp_path, _dma_columns())
        tracks = list(read_parquet_tracks(f, mobile_types=None))
        # Base Station / AtoN rows still drop out (NULL SOG/COG), but the
        # filter itself is off.
        assert [t.mmsi for t in tracks] == [219000001, 265000002]

    def test_custom_mobile_filter(self, tmp_path):
        f = _write(tmp_path, _dma_columns())
        tracks = list(read_parquet_tracks(f, mobile_types=("Class B",)))
        assert [t.mmsi for t in tracks] == [265000002]

    def test_marine_cadastre_transceiver_class_kept_by_default(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "MMSI": [1001, 1002, 1003],
                "BaseDateTime": ["2025-01-01T00:00:00"] * 3,
                "LAT": [57.0] * 3,
                "LON": [11.0] * 3,
                "SOG": [1.0] * 3,
                "COG": [0.0] * 3,
                "TransceiverClass": ["A", "B", "base station"],
            },
        )
        tracks = list(read_parquet_tracks(f))
        assert [t.mmsi for t in tracks] == [1001, 1002]

    def test_dictionary_encoded_mobile_column(self, tmp_path):
        cols = _dma_columns(False)
        cols["Type of mobile"] = pa.array(cols["Type of mobile"]).dictionary_encode()
        f = tmp_path / "dict.parquet"
        pq.write_table(pa.table(cols), f)
        assert [t.mmsi for t in read_parquet_tracks(f)] == [219000001, 265000002]

    def test_mismatched_mobile_types_warns(self, tmp_path):
        f = _write(tmp_path, _dma_columns(False))
        with pytest.warns(UserWarning, match="matched none"):
            assert list(read_parquet_tracks(f, mobile_types=("Vessel",))) == []

    def test_no_mobile_column_is_fine(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001, 1002],
                "time": [T0, T0],
                "lon": [11.0, 11.1],
                "lat": [57.0, 57.1],
                "sog": [1.0, 2.0],
                "cog": [0.0, 10.0],
            },
        )
        assert len(list(read_parquet_tracks(f))) == 2

    def test_invalid_rows_dropped(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001, 0, 1_234_567_890, 1001, 1001, 1001],
                "time": [T0, T0, T0, T0 + 1, T0 + 2, T0 + 3],
                "lon": [11.0, 11.0, 11.0, 200.0, 11.0, 11.0],
                "lat": [57.0, 57.0, 57.0, 57.0, 95.0, 57.0],
                "sog": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "cog": [0.0, 0.0, 0.0, 0.0, 0.0, None],
            },
        )
        (track,) = read_parquet_tracks(f)
        assert track.mmsi == 1001
        assert len(track) == 1  # bad mmsi x2, out-of-range lon/lat, null cog

    def test_empty_file_yields_nothing(self, tmp_path):
        f = tmp_path / "empty.parquet"
        pq.write_table(
            pa.table(
                {
                    "mmsi": pa.array([], type=pa.int64()),
                    "time": pa.array([], type=pa.float64()),
                    "lon": pa.array([], type=pa.float64()),
                    "lat": pa.array([], type=pa.float64()),
                    "sog": pa.array([], type=pa.float64()),
                    "cog": pa.array([], type=pa.float64()),
                }
            ),
            f,
        )
        assert list(read_parquet_tracks(f)) == []

    def test_all_rows_filtered_yields_nothing(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [0],
                "time": [T0],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        assert list(read_parquet_tracks(f)) == []

    def test_small_batches_merge_across_boundaries(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1002, 1001, 1002, 1001, 1002, 1001],
                "time": [T0 + 5, T0, T0 + 15, T0 + 10, T0 + 25, T0 + 20],
                "lon": [11.0] * 6,
                "lat": [57.0] * 6,
                "sog": [1.0] * 6,
                "cog": [0.0] * 6,
            },
        )
        tracks = list(read_parquet_tracks(f, batch_size=2))
        assert [t.mmsi for t in tracks] == [1001, 1002]
        assert all(len(t) == 3 for t in tracks)
        assert all(np.all(np.diff(t.t) > 0) for t in tracks)

    def test_output_feeds_tdkc(self, tmp_path):
        rng = np.random.default_rng(7)
        n = 50
        t = T0 + np.arange(n) * 10.0
        lon = 11.0 + np.cumsum(rng.normal(0.0005, 0.0001, n))
        lat = 57.0 + np.cumsum(rng.normal(0.0005, 0.0001, n))
        f = _write(
            tmp_path,
            {
                "mmsi": [1001] * n,
                "time": t,
                "lon": lon,
                "lat": lat,
                "sog": np.full(n, 10.0),
                "cog": np.full(n, 45.0),
            },
        )
        (track,) = read_parquet_tracks(f)
        segments = tdkc_segments(track, min_sed_m=30.0, min_svd_kn=0.3)
        assert segments
        assert sum(s.n_points for s in segments) >= n


class TestReadParquetStaticRecords:
    def test_dma_full_record(self, tmp_path):
        f = _write(tmp_path, _dma_columns())
        recs = read_parquet_static_records(f)
        rec = recs[219000001]
        assert rec["vessel_name"] == "IMAVERE"
        assert rec["call_sign"] == "5BWV4"
        assert rec["imo"] == 9612325
        assert rec["ship_type"] == 70  # "Cargo" mapped to TOC
        assert rec["destination"] == "EEBEK"
        assert rec["draught"] == 6.1
        # Per-quadrant offsets used directly, NOT the halved Length/Width.
        assert (rec["dim_bow"], rec["dim_stern"]) == (154.0, 32.0)
        assert (rec["dim_port"], rec["dim_star"]) == (6.0, 21.0)
        assert rec["time"] == T0 + 20

    def test_imo_unknown_and_blank_destination_skipped(self, tmp_path):
        f = _write(tmp_path, _dma_columns())
        rec = read_parquet_static_records(f)[265000002]
        assert "imo" not in rec  # "Unknown"
        assert "destination" not in rec  # whitespace-only
        assert rec["ship_type"] == 36  # Sailing

    def test_length_width_fallback_halved(self, tmp_path):
        f = _write(tmp_path, _dma_columns())
        rec = read_parquet_static_records(f)[265000002]
        # No A/B/C/D for this vessel: halved Length/Width kicks in.
        assert (rec["dim_bow"], rec["dim_stern"]) == (6.0, 6.0)
        assert (rec["dim_port"], rec["dim_star"]) == (2.0, 2.0)

    def test_last_value_wins_in_file_order(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001, 1001],
                "time": [T0, T0 + 10],
                "lon": [11.0, 11.0],
                "lat": [57.0, 57.0],
                "sog": [1.0, 1.0],
                "cog": [0.0, 0.0],
                "draught": [5.0, 7.5],
            },
        )
        rec = read_parquet_static_records(f)[1001]
        assert rec["draught"] == 7.5
        assert rec["time"] == T0 + 10

    def test_time_advances_across_batches(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001, 1001, 1001],
                "time": [T0, T0 + 10, T0 + 20],
                "lon": [11.0] * 3,
                "lat": [57.0] * 3,
                "sog": [1.0] * 3,
                "cog": [0.0] * 3,
                "draught": [5.0, None, None],
            },
        )
        rec = read_parquet_static_records(f, batch_size=1)[1001]
        assert rec["time"] == T0 + 20  # later batch bumps the entry's time
        assert rec["draught"] == 5.0

    def test_no_static_columns_returns_empty(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": [T0],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
            },
        )
        assert read_parquet_static_records(f) == {}

    def test_missing_required_column_raises(self, tmp_path):
        f = _write(tmp_path, {"mmsi": [1001], "draught": [5.0]})
        with pytest.raises(KeyError):
            read_parquet_static_records(f)

    def test_mobile_filter_applies(self, tmp_path):
        f = _write(tmp_path, _dma_columns())
        recs = read_parquet_static_records(f, mobile_types=("Class A",))
        assert set(recs) == {219000001}

    def test_all_rows_filtered_returns_empty(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [0],
                "time": [T0],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
                "draught": [5.0],
            },
        )
        assert read_parquet_static_records(f) == {}

    def test_all_null_static_batch_is_skipped(self, tmp_path):
        f = tmp_path / "nullstatics.parquet"
        pq.write_table(
            pa.table(
                {
                    "mmsi": pa.array([1001], type=pa.int64()),
                    "time": pa.array([T0], type=pa.float64()),
                    "lon": [11.0],
                    "lat": [57.0],
                    "sog": [1.0],
                    "cog": [0.0],
                    "draught": pa.array([None], type=pa.float64()),
                    "destination": pa.array([None], type=pa.string()),
                }
            ),
            f,
        )
        rec = read_parquet_static_records(f)[1001]
        assert "draught" not in rec
        assert "destination" not in rec

    def test_ship_type_names_override(self, tmp_path):
        cols = _dma_columns()
        cols["Ship type"] = ["Cargo ship"] * 6
        f = _write(tmp_path, cols)
        assert "ship_type" not in read_parquet_static_records(f)[219000001]
        names = {**DEFAULT_SHIP_TYPE_NAMES, "cargo ship": 79}
        recs = read_parquet_static_records(f, ship_type_names=names)
        assert recs[219000001]["ship_type"] == 79

    def test_lone_a_column_is_not_an_offset(self, tmp_path):
        cols = _dma_columns()
        for k in ("B", "C", "D"):
            del cols[k]
        f = _write(tmp_path, cols)
        rec = read_parquet_static_records(f)[219000001]
        # Falls back to the halved Length/Width split.
        assert (rec["dim_bow"], rec["dim_stern"]) == (93.0, 93.0)
        assert (rec["dim_port"], rec["dim_star"]) == (13.5, 13.5)

    def test_numeric_ship_type_column(self, tmp_path):
        f = _write(
            tmp_path,
            {
                "mmsi": [1001],
                "time": [T0],
                "lon": [11.0],
                "lat": [57.0],
                "sog": [1.0],
                "cog": [0.0],
                "shiptype": [80],
            },
        )
        assert read_parquet_static_records(f)[1001]["ship_type"] == 80


class TestPyarrowOptionality:
    def test_missing_pyarrow_raises_helpful_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        with pytest.raises(ImportError, match="pip install pyarrow"):
            _import_pyarrow()
