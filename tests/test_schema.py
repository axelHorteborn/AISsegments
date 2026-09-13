"""Tests for the provider-layout helpers shared by the CSV and Parquet readers."""

from __future__ import annotations

import pytest

from aissegments import DEFAULT_SHIP_TYPE_NAMES, ship_type_to_toc
from aissegments._schema import (
    convert_static_value,
    normalise_mobile_type,
    parse_time_string,
    resolve_column,
    resolve_kinematics,
    resolve_static,
    sniff_time_format,
    split_length_width,
)

T0 = 1_735_689_600.0  # 2025-01-01 00:00:00 UTC


class TestShipTypeToToc:
    def test_none_and_empty(self):
        assert ship_type_to_toc(None) is None
        assert ship_type_to_toc("") is None
        assert ship_type_to_toc("   ") is None

    def test_numeric_passthrough(self):
        assert ship_type_to_toc(70) == 70
        assert ship_type_to_toc(80.0) == 80
        assert ship_type_to_toc("60") == 60
        assert ship_type_to_toc("30.0") == 30

    def test_nan_is_none(self):
        assert ship_type_to_toc(float("nan")) is None

    def test_default_names(self):
        assert ship_type_to_toc("Cargo") == 70
        assert ship_type_to_toc("  tanker ") == 80
        assert ship_type_to_toc("Law enforcement") == 55
        assert ship_type_to_toc("Towing long/wide") == 32

    def test_unknown_names_are_none(self):
        assert ship_type_to_toc("Undefined") is None
        assert ship_type_to_toc("Reserved") is None

    def test_custom_names_replace_defaults(self):
        names = {"cargo ship": 70}
        assert ship_type_to_toc("Cargo ship", names) == 70
        assert ship_type_to_toc("Cargo", names) is None  # not in the custom table
        assert ship_type_to_toc("Cargo", {**DEFAULT_SHIP_TYPE_NAMES, **names}) == 70


class TestColumnResolution:
    def test_resolve_found_and_missing(self):
        assert resolve_column(["MMSI", "# Timestamp"], ("# timestamp",)) == "# Timestamp"
        assert resolve_column(["MMSI"], ("# timestamp",)) is None

    def test_kinematics_missing_raises_with_label(self):
        with pytest.raises(KeyError, match=r"CSV missing.*sog"):
            resolve_kinematics(["MMSI", "# Timestamp", "Latitude", "Longitude", "COG"], what="CSV")

    def test_kinematics_accepts_marine_cadastre_and_dma_headers(self):
        mc = resolve_kinematics(["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"])
        assert mc["time"] == "BaseDateTime"
        dma = resolve_kinematics(["# Timestamp", "MMSI", "Latitude", "Longitude", "SOG", "COG"])
        assert dma["time"] == "# Timestamp"

    def test_static_full_abcd_set_is_used(self):
        m = resolve_static(["A", "B", "C", "D", "Length"])
        assert m["dim_bow"] == "A" and m["dim_star"] == "D"
        assert m["length"] == "Length"

    def test_static_partial_abcd_is_ignored(self):
        m = resolve_static(["A", "Length", "Width"])
        assert "dim_bow" not in m
        assert set(m) == {"length", "width"}

    def test_static_explicit_dim_names_win_over_letters(self):
        m = resolve_static(["to_bow", "A", "B", "C", "D"])
        assert m["dim_bow"] == "to_bow"
        assert m["dim_stern"] == "B"

    def test_static_ship_type_space_alias(self):
        assert resolve_static(["Ship type"]) == {"ship_type": "Ship type"}


class TestMobileTypeNormalisation:
    @pytest.mark.parametrize("raw", ["Class A", "class a", " CLASS A ", "A", "a", "ClassA"])
    def test_class_a_spellings_agree(self, raw):
        assert normalise_mobile_type(raw) == "a"

    def test_non_vessel_classes_are_distinct(self):
        assert normalise_mobile_type("Base Station") == "base station"
        assert normalise_mobile_type("AtoN") == "aton"
        assert normalise_mobile_type(None) == ""


class TestTimeParsing:
    def test_sniff_formats(self):
        assert sniff_time_format("2025-01-01 00:00:00") == "%Y-%m-%d %H:%M:%S"
        assert sniff_time_format("2025-01-01T00:00:00") == "%Y-%m-%dT%H:%M:%S"
        assert sniff_time_format("01/01/2025 00:00:00") == "%d/%m/%Y %H:%M:%S"
        assert sniff_time_format("01/01/2025 00:00:00", dayfirst=False) == "%m/%d/%Y %H:%M:%S"
        assert sniff_time_format("1735689600") == "unix"
        with pytest.raises(ValueError, match="time_format="):
            sniff_time_format("Jan 1 2025")

    def test_parse_unix_iso_and_slash(self):
        assert parse_time_string(str(T0)) == T0
        assert parse_time_string("2025-01-01T00:00:00") == T0
        assert parse_time_string("2025-01-01T00:00:00Z") == T0
        assert parse_time_string("2025-01-01T01:00:00+01:00") == T0
        assert parse_time_string("01/02/2025 00:00:00") == T0 + 31 * 86400
        assert parse_time_string("01/02/2025 00:00:00", dayfirst=False) == T0 + 86400

    def test_parse_explicit_format(self):
        assert parse_time_string("2025.01.01 00-00-00", time_format="%Y.%m.%d %H-%M-%S") == T0
        with pytest.raises(ValueError):
            parse_time_string("2025-01-01T00:00:00", time_format="%Y.%m.%d %H-%M-%S")


class TestStaticValues:
    def test_convert_static_value(self):
        assert convert_static_value("imo", "9612325") == 9612325
        assert convert_static_value("imo", "Unknown") is None
        assert convert_static_value("ship_type", "Cargo") == 70
        assert convert_static_value("draught", "6.1") == 6.1
        assert convert_static_value("draught", "n/a") is None
        assert convert_static_value("destination", "  ") is None
        assert convert_static_value("vessel_name", " IMAVERE ") == "IMAVERE"

    def test_split_length_width_fallback_only(self):
        entry = {"length": 100.0, "width": 20.0, "dim_bow": 70.0, "dim_stern": 30.0}
        split_length_width(entry)
        assert (entry["dim_bow"], entry["dim_stern"]) == (70.0, 30.0)
        assert (entry["dim_port"], entry["dim_star"]) == (10.0, 10.0)
        assert "length" not in entry and "width" not in entry
