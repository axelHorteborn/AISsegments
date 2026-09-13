"""Unit tests for the input adapters.

- ``from_aisdb_track`` is exercised here with synthetic AISdb-shaped dicts;
  real end-to-end integration with aisdb's bundled test corpus lives in
  ``test_aisdb_integration.py`` (gated on the ``aisdb`` pytest marker).
- ``read_csv_tracks`` is exercised here with inline CSVs written via
  ``tmp_path``.  Integration with files dropped under ``tests/data/`` is
  covered by ``test_local_data.py``.
- ``read_csv_static_records`` is exercised here with inline CSVs.
"""

from __future__ import annotations

import numpy as np
import pytest

from aissegments import tdkc
from aissegments.adapters import (
    from_aisdb_track,
    read_csv_static_records,
    read_csv_tracks,
)

# ---------------------------------------------------------------------------
# Adapter unit tests with synthetic dicts
# ---------------------------------------------------------------------------


class TestFromAisdbTrack:
    def test_minimal_dict_round_trips(self):
        d = {
            "mmsi": 219000123,
            "time": np.array([0.0, 60.0, 120.0]),
            "lon": np.array([12.0, 12.001, 12.002]),
            "lat": np.array([55.0, 55.0, 55.0]),
            "sog": np.array([10.0, 10.0, 10.0]),
            "cog": np.array([90.0, 90.0, 90.0]),
        }
        track = from_aisdb_track(d)
        assert track.mmsi == 219000123
        assert len(track) == 3
        np.testing.assert_array_equal(track.t, [0.0, 60.0, 120.0])

    def test_lists_are_coerced(self):
        d = {
            "mmsi": 1,
            "time": [0, 60, 120],
            "lon": [0.0, 0.001, 0.002],
            "lat": [55.0, 55.0, 55.0],
            "sog": [10.0, 10.0, 10.0],
            "cog": [90.0, 90.0, 90.0],
        }
        track = from_aisdb_track(d)
        assert track.t.dtype == np.float64

    def test_missing_key_raises(self):
        d = {
            "mmsi": 1,
            "time": [0],
            "lon": [0.0],
            "lat": [55.0],
            "sog": [10.0],
            # cog missing
        }
        with pytest.raises(KeyError, match="cog"):
            from_aisdb_track(d)

    def test_invalid_track_propagates_value_error(self):
        d = {
            "mmsi": 1,
            "time": [0, 1, 2],
            "lon": [0.0, 0.1],  # wrong length
            "lat": [55.0, 55.0, 55.0],
            "sog": [10.0, 10.0, 10.0],
            "cog": [90.0, 90.0, 90.0],
        }
        with pytest.raises(ValueError):
            from_aisdb_track(d)

    def test_adapter_output_works_with_tdkc(self):
        n = 50
        d = {
            "mmsi": 1,
            "time": np.arange(n) * 60.0,
            "lon": np.linspace(0.0, 0.05, n),
            "lat": np.full(n, 55.0),
            "sog": np.full(n, 10.0),
            "cog": np.full(n, 90.0),
        }
        track = from_aisdb_track(d)
        compressed = tdkc(track)
        assert len(compressed) == 2  # straight at constant speed -> endpoints only


# ---------------------------------------------------------------------------
# read_csv_tracks
# ---------------------------------------------------------------------------


class TestReadCsvTracks:
    def _write(self, path, header_and_rows: str) -> None:
        path.write_text(header_and_rows, encoding="utf-8")

    def test_basic_round_trip(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog\n"
            "100,0,12.0,55.0,10.0,90.0\n"
            "100,60,12.001,55.0,10.0,90.0\n"
            "200,0,13.0,56.0,5.0,180.0\n",
        )
        tracks = read_csv_tracks(f)
        assert len(tracks) == 2
        # Sorted by descending length: mmsi 100 has 2 pts, mmsi 200 has 1.
        assert tracks[0].mmsi == 100
        assert len(tracks[0]) == 2
        assert tracks[1].mmsi == 200
        assert len(tracks[1]) == 1

    def test_string_path_is_accepted(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(f, "mmsi,time,lon,lat,sog,cog\n100,0,12,55,10,90\n")
        tracks = read_csv_tracks(str(f))
        assert tracks[0].mmsi == 100

    def test_rows_are_sorted_by_time_per_mmsi(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog\n100,60,12.001,55,10,90\n100,0,12.0,55,10,90\n",
        )
        tracks = read_csv_tracks(f)
        np.testing.assert_array_equal(tracks[0].t, [0.0, 60.0])

    def test_case_insensitive_columns(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(f, "MMSI,Time,Lon,Lat,SOG,COG\n100,0,12,55,10,90\n")
        tracks = read_csv_tracks(f)
        assert tracks[0].mmsi == 100

    def test_extra_columns_are_ignored(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog,heading,name\n100,0,12,55,10,90,89,FOO\n",
        )
        tracks = read_csv_tracks(f)
        assert tracks[0].mmsi == 100

    def test_dtypes_are_float64(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(f, "mmsi,time,lon,lat,sog,cog\n100,0,12,55,10,90\n100,60,12.001,55,10,90\n")
        tracks = read_csv_tracks(f)
        for arr in (tracks[0].t, tracks[0].lon, tracks[0].lat, tracks[0].sog, tracks[0].cog):
            assert arr.dtype == np.float64

    def test_missing_column_raises_key_error(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(f, "mmsi,time,lon,lat,sog\n100,0,12,55,10\n")  # cog missing
        with pytest.raises(KeyError, match="cog"):
            read_csv_tracks(f)

    def test_empty_file_raises_value_error(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty CSV"):
            read_csv_tracks(f)

    def test_invalid_value_raises_with_line_number(self, tmp_path):
        f = tmp_path / "t.csv"
        self._write(f, "mmsi,time,lon,lat,sog,cog\nabc,0,12,55,10,90\n")
        with pytest.raises(ValueError, match="row 2"):
            read_csv_tracks(f)

    def test_marine_cadastre_columns(self, tmp_path):
        """Marine Cadastre publishes AIS with BaseDateTime / LAT / LON columns."""
        f = tmp_path / "mc.csv"
        f.write_text(
            "MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName\n"
            "366330480,2019-01-01T14:15:12,26.10411,-80.12838,0.0,360.0,511.0,TOTHILL\n"
            "366330480,2019-01-01T14:16:12,26.10411,-80.12838,0.5,360.0,511.0,TOTHILL\n",
            encoding="utf-8",
        )
        tracks = read_csv_tracks(f)
        assert len(tracks) == 1
        assert tracks[0].mmsi == 366330480
        # 60 s between the two rows.
        assert tracks[0].t[1] - tracks[0].t[0] == pytest.approx(60.0)

    def test_iso_with_z_suffix(self, tmp_path):
        f = tmp_path / "z.csv"
        f.write_text(
            "mmsi,time,lon,lat,sog,cog\n"
            "100,2021-07-01T00:00:00Z,12,55,10,90\n"
            "100,2021-07-01T00:01:00+00:00,12.001,55,10,90\n",
            encoding="utf-8",
        )
        tracks = read_csv_tracks(f)
        assert tracks[0].t[1] - tracks[0].t[0] == pytest.approx(60.0)

    def test_gzip_loads_transparently(self, tmp_path):
        import gzip

        f = tmp_path / "t.csv.gz"
        with gzip.open(f, "wt", encoding="utf-8") as g:
            g.write("mmsi,time,lon,lat,sog,cog\n100,0,12,55,10,90\n100,60,12.001,55,10,90\n")
        tracks = read_csv_tracks(f)
        assert len(tracks) == 1
        assert len(tracks[0]) == 2

    def test_compresses_with_tdkc(self, tmp_path):
        # Three points in a straight line at constant speed: TDKC should keep 2.
        f = tmp_path / "t.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog\n"
            "100,0,12.0,55,10,90\n"
            "100,60,12.001,55,10,90\n"
            "100,120,12.002,55,10,90\n",
        )
        track = read_csv_tracks(f)[0]
        assert len(tdkc(track)) == 2


# ---------------------------------------------------------------------------
# read_csv_static_records
# ---------------------------------------------------------------------------


class TestReadCsvStaticRecords:
    def _write(self, path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_marine_cadastre_extracts_full_static_record(self, tmp_path):
        f = tmp_path / "mc.csv"
        self._write(
            f,
            "MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,"
            "VesselType,Status,Length,Width,Draft,Cargo,TransceiverClass\n"
            "366330480,2019-01-01T14:15:12,26.10411,-80.12838,0.0,360.0,511.0,"
            "TOTHILL,9120144,WDH2932,37,,28.0,8.0,2.5,,B\n",
        )
        out = read_csv_static_records(f)
        assert 366330480 in out
        rec = out[366330480]
        assert rec["mmsi"] == 366330480
        assert rec["vessel_name"] == "TOTHILL"
        assert rec["imo"] == 9120144
        assert rec["call_sign"] == "WDH2932"
        assert rec["ship_type"] == 37
        assert rec["draught"] == 2.5
        # Length 28 → dim_bow=14, dim_stern=14
        assert rec["dim_bow"] == 14.0
        assert rec["dim_stern"] == 14.0
        # Width 8 → dim_port=4, dim_star=4
        assert rec["dim_port"] == 4.0
        assert rec["dim_star"] == 4.0

    def test_returns_empty_when_no_static_columns(self, tmp_path):
        f = tmp_path / "minimal.csv"
        self._write(f, "mmsi,time,lon,lat,sog,cog\n100,0,12,55,10,90\n")
        assert read_csv_static_records(f) == {}

    def test_partial_static_columns(self, tmp_path):
        # Only IMO and Draft present; no Length/Width/VesselType/Name.
        f = tmp_path / "partial.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog,imo,draft\n100,0,12,55,10,90,12345,4.5\n",
        )
        rec = read_csv_static_records(f)[100]
        assert rec["imo"] == 12345
        assert rec["draught"] == 4.5
        assert "dim_bow" not in rec  # length not in source
        assert "ship_type" not in rec

    def test_multiple_rows_merge_with_latest_wins(self, tmp_path):
        # Two rows for the same MMSI; later row has different draft.
        f = tmp_path / "merge.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog,Draft\n"
            "100,2019-01-01T00:00:00,12,55,10,90,3.0\n"
            "100,2019-01-01T12:00:00,12.001,55,10,90,5.0\n",
        )
        rec = read_csv_static_records(f)[100]
        assert rec["draught"] == 5.0  # latest non-null wins

    def test_blank_static_values_are_ignored(self, tmp_path):
        f = tmp_path / "blanks.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog,VesselName,IMO,Length\n"
            "100,2019-01-01T00:00:00,12,55,10,90,FIRST,9999,30\n"
            "100,2019-01-01T01:00:00,12,55,10,90,,,\n",  # blanks should NOT overwrite
        )
        rec = read_csv_static_records(f)[100]
        assert rec["vessel_name"] == "FIRST"
        assert rec["imo"] == 9999
        assert rec["dim_bow"] == 15.0  # 30/2

    def test_invalid_numeric_values_skipped(self, tmp_path):
        f = tmp_path / "bad_nums.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog,IMO,Draft\n100,0,12,55,10,90,not_a_number,bogus\n",
        )
        rec = read_csv_static_records(f)[100]
        assert "imo" not in rec
        assert "draught" not in rec

    def test_invalid_mmsi_row_skipped(self, tmp_path):
        f = tmp_path / "bad_mmsi.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog,IMO\nabc,0,12,55,10,90,9999\n100,0,12,55,10,90,1234\n",
        )
        out = read_csv_static_records(f)
        assert 100 in out and out[100]["imo"] == 1234
        assert len(out) == 1

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty CSV"):
            read_csv_static_records(f)

    def test_missing_mmsi_or_time_raises(self, tmp_path):
        f = tmp_path / "no_mmsi.csv"
        # Has a static column but no mmsi/time.
        self._write(f, "VesselName,IMO\nTEST,9999\n")
        with pytest.raises(KeyError):
            read_csv_static_records(f)

    def test_ignores_invalid_time_rows(self, tmp_path):
        f = tmp_path / "bad_time.csv"
        self._write(
            f,
            "mmsi,time,lon,lat,sog,cog,IMO\n"
            "100,not-a-date,12,55,10,90,9999\n"
            "100,2019-01-01T00:00:00,12,55,10,90,8888\n",
        )
        rec = read_csv_static_records(f)[100]
        assert rec["imo"] == 8888  # only the valid-time row contributed

    def test_gz_supported(self, tmp_path):
        import gzip

        f = tmp_path / "x.csv.gz"
        with gzip.open(f, "wt", encoding="utf-8") as g:
            g.write("mmsi,time,lon,lat,sog,cog,IMO\n100,0,12,55,10,90,1234\n")
        out = read_csv_static_records(f)
        assert out[100]["imo"] == 1234


DMA_CSV = (
    "# Timestamp,Type of mobile,MMSI,Latitude,Longitude,Navigational status,ROT,SOG,COG,"
    "Heading,IMO,Callsign,Name,Ship type,Cargo type,Width,Length,Type of position fixing device,"
    "Draught,Destination,ETA,Data source type,A,B,C,D\n"
    "01/01/2025 00:00:10,Class A,219000001,57.01,11.01,Under way using engine,0,10.0,45.0,"
    "45,9612325,5BWV4,IMAVERE,Cargo,,27,186,GPS,6.1,EEBEK,,AIS,154,32,6,21\n"
    "01/01/2025 00:00:00,Class A,219000001,57.0,11.0,Under way using engine,0,10.0,45.0,"
    "45,9612325,5BWV4,IMAVERE,Cargo,,27,186,GPS,6.0,EEBEK,,AIS,154,32,6,21\n"
    "01/01/2025 00:00:00,Base Station,2190064,57.5,11.5,Unknown value,,,,"
    ",Unknown,,,Undefined,,,,Undefined,,,,AIS,,,,\n"
    "01/01/2025 00:00:10,Class B,265000002,56.5,11.6,Unknown value,,5.0,90.0,"
    ",Unknown,,SVALAN,Sailing,,4,12,GPS,,,,AIS,,,,\n"
    "01/01/2025 00:00:05,AtoN,992191234,57.9,11.9,Unknown value,,,,"
    ",Unknown,,,Undefined,,,,Undefined,,,,AIS,,,,\n"
)

DMA_T0 = 1_735_689_600.0  # 2025-01-01 00:00:00 UTC


class TestReadCsvTracksProviderLayouts:
    def test_dma_layout_filters_non_vessels_and_parses_day_first(self, tmp_path):
        f = tmp_path / "aisdk.csv"
        f.write_text(DMA_CSV, encoding="utf-8")
        tracks = read_csv_tracks(f)
        assert [t.mmsi for t in tracks] == [219000001, 265000002]
        np.testing.assert_allclose(tracks[0].t, [DMA_T0, DMA_T0 + 10])

    def test_dma_layout_without_filter_needs_skip_invalid(self, tmp_path):
        f = tmp_path / "aisdk.csv"
        f.write_text(DMA_CSV, encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid row 4"):
            read_csv_tracks(f, mobile_types=None)
        tracks = read_csv_tracks(f, mobile_types=None, skip_invalid=True)
        assert [t.mmsi for t in tracks] == [219000001, 265000002]

    def test_marine_cadastre_transceiver_class_kept_by_default(self, tmp_path):
        f = tmp_path / "mc.csv"
        f.write_text(
            "MMSI,BaseDateTime,LAT,LON,SOG,COG,TransceiverClass\n"
            "1001,2025-01-01T00:00:00,57.0,11.0,1.0,0.0,A\n"
            "1002,2025-01-01T00:00:00,57.0,11.0,1.0,0.0,B\n"
            "1003,2025-01-01T00:00:00,57.0,11.0,1.0,0.0,base station\n",
            encoding="utf-8",
        )
        assert sorted(t.mmsi for t in read_csv_tracks(f)) == [1001, 1002]

    def test_mismatched_mobile_types_warns(self, tmp_path):
        f = tmp_path / "aisdk.csv"
        f.write_text(DMA_CSV, encoding="utf-8")
        with pytest.warns(UserWarning, match="matched none"):
            assert read_csv_tracks(f, mobile_types=("Vessel",)) == []

    def test_month_first_and_explicit_format(self, tmp_path):
        f = tmp_path / "us.csv"
        f.write_text(
            "mmsi,time,lon,lat,sog,cog\n1,01/02/2025 00:00:00,11,57,1,0\n", encoding="utf-8"
        )
        (day_first,) = read_csv_tracks(f)
        (month_first,) = read_csv_tracks(f, dayfirst=False)
        assert day_first.t[0] == DMA_T0 + 31 * 86400
        assert month_first.t[0] == DMA_T0 + 86400

        g = tmp_path / "odd.csv"
        g.write_text(
            "mmsi,time,lon,lat,sog,cog\n1,2025.01.01 00-00-00,11,57,1,0\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Invalid row 2"):
            read_csv_tracks(g)
        (track,) = read_csv_tracks(g, time_format="%Y.%m.%d %H-%M-%S")
        assert track.t[0] == DMA_T0


class TestReadCsvStaticRecordsProviderLayouts:
    def test_dma_full_record(self, tmp_path):
        f = tmp_path / "aisdk.csv"
        f.write_text(DMA_CSV, encoding="utf-8")
        recs = read_csv_static_records(f)
        assert set(recs) == {219000001, 265000002}
        rec = recs[219000001]
        assert rec["vessel_name"] == "IMAVERE"
        assert rec["call_sign"] == "5BWV4"
        assert rec["imo"] == 9612325
        assert rec["ship_type"] == 70  # "Cargo" mapped to TOC
        assert rec["destination"] == "EEBEK"
        assert rec["draught"] == 6.0  # last value in file order wins
        # Per-quadrant offsets used directly, NOT the halved Length/Width.
        assert (rec["dim_bow"], rec["dim_stern"]) == (154.0, 32.0)
        assert (rec["dim_port"], rec["dim_star"]) == (6.0, 21.0)
        assert rec["time"] == DMA_T0 + 10

        sail = recs[265000002]
        assert sail["ship_type"] == 36
        assert "imo" not in sail  # "Unknown" skipped
        # No A/B/C/D for this row → halved Length/Width fallback.
        assert (sail["dim_bow"], sail["dim_port"]) == (6.0, 2.0)

    def test_ship_type_names_override(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text(
            "mmsi,time,lon,lat,sog,cog,shiptype\n1,0,11,57,1,0,Cargo ship\n", encoding="utf-8"
        )
        assert "ship_type" not in read_csv_static_records(f)[1]
        recs = read_csv_static_records(f, ship_type_names={"cargo ship": 79})
        assert recs[1]["ship_type"] == 79

    def test_lone_a_column_is_not_an_offset(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text(
            "mmsi,time,lon,lat,sog,cog,A,Length\n1,0,11,57,1,0,154,186\n", encoding="utf-8"
        )
        rec = read_csv_static_records(f)[1]
        assert (rec["dim_bow"], rec["dim_stern"]) == (93.0, 93.0)

    def test_mobile_filter_applies(self, tmp_path):
        f = tmp_path / "aisdk.csv"
        f.write_text(DMA_CSV, encoding="utf-8")
        assert set(read_csv_static_records(f, mobile_types=("Class A",))) == {219000001}
        assert 2190064 in read_csv_static_records(f, mobile_types=None)


# Real end-to-end integration with aisdb's bundled test corpus lives in
# ``test_aisdb_integration.py`` (gated on the ``aisdb`` marker).
