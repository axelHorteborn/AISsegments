# Local AIS test data

Drop CSV files here to feed the test suite and visualisations with denser
real-AIS data than [aisdb's bundled corpus](https://github.com/AISViz/AISdb/tree/master/aisdb/tests/testdata)
provides (the bundled corpus is a one-day worldwide snapshot, capped at 3
pings per vessel).

When at least one valid file is present:

- the `local_csv` pytest mark activates and runs integration tests in
  [tests/test_local_data.py](../test_local_data.py);
- a sixth visualisation figure (`06_local_csv_overview.png`) is produced
  by [examples/visualize.py](../../examples/visualize.py).

When the directory is empty, those tests skip cleanly and the visualisation
function is a no-op — nothing else changes.

## Required CSV format

Header row with these columns (case-insensitive; extra columns are ignored):

| Column | Type    | Unit                       |
| ------ | ------- | -------------------------- |
| mmsi   | integer | AIS MMSI                   |
| time   | float   | Unix seconds since epoch   |
| lon    | float   | degrees, WGS84 (EPSG:4326) |
| lat    | float   | degrees, WGS84 (EPSG:4326) |
| sog    | float   | knots                      |
| cog    | float   | degrees, `[0, 360)`        |

Rows are grouped by `mmsi` and sorted by `time` automatically — the order
within the file does not matter.

Example (one vessel, two pings):

```csv
mmsi,time,lon,lat,sog,cog
257123456,1625097600,12.345,55.678,10.5,90.0
257123456,1625097660,12.347,55.678,10.5,90.0
```

## License — read before committing

Files placed here become part of the **AISsegments source tree** when
committed.  AISsegments is MIT-licensed; data added here must be
**permissively licensed** (public domain, CC0, CC-BY, MIT, BSD).

Do **not** commit:

- AGPL-licensed test data (e.g., aisdb's bundled CSV, since aisdb is AGPL).
- Data with explicit "non-commercial use only" or "research use only"
  restrictions.
- Data with redistribution restrictions.

Recommended public-domain / permissive sources:

- **US Marine Cadastre AIS** — https://marinecadastre.gov/ais/
  US Government works are public domain (17 USC §105).  Daily files include
  thousands of pings per vessel; a small bbox-extracted slice is ideal.
- **Norwegian Coastal Administration** historical AIS samples (varies; check).
- **HELCOM Baltic AIS** open-data exports (varies; check).
- Hand-anonymised AIS exports from the user's own institutional data, with
  permission.

If unsure about a file's licence, leave it out.

## Multiple files

Anything matching `*.csv` is loaded.  Files are concatenated by MMSI before
the per-vessel grouping, so splitting one large dataset across several files
(e.g. by month) works as expected.
