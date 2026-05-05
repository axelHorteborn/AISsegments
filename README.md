# AISsegments

**A Python toolkit for compressing AIS vessel trajectories into compact, query-friendly linestring segments — without losing the kinematic features that maritime safety and risk analysis depend on.**

## What is this?

`aissegments` takes raw AIS position reports (millions of pings per day in a busy sea area) and outputs a small number of **constant-COG/SOG linestring segments per vessel**, one row per (vessel, time-window, course/speed). The output is shaped to drop straight into a PostGIS `LINESTRING` table.

It is the reference Python implementation of the **Top-Down Kinematic Compression (TDKC)** algorithm of Guo, Bolbot & Valdez Banda (Ocean Engineering 312 (2024), 119189), with the recursion-termination and adaptive-threshold fixes described in the paper.

## Goals

1. **Shrink AIS data without losing the maritime-relevant features.** A continent-scale AIS feed easily produces hundreds of millions of position reports per month. Most of those points are uninformative — a vessel cruising in a straight line at a steady speed needs only its endpoints. TDKC keeps the points where vessels actually do something interesting (turn, accelerate, stop, manoeuvre) and drops the rest.
2. **Treat AIS as a sequence of *kinematic states*, not just positions.** The classical Douglas-Peucker simplification looks only at how far points stray from a straight line. That throws away course and speed changes that lie on a straight track, which are exactly the events maritime risk analysis cares about. TDKC uses both position (Synchronous Euclidean Distance) **and** velocity (Synchronous Velocity Difference) with adaptive per-track thresholds.
3. **Produce database-ready output.** The output isn't a smaller list of points — it's a list of `Segment` records with start/end coordinates, mean COG/SOG, and the count of original observations spanned. Each segment is a 2-point `LINESTRING` ready for `ST_Intersects` and other PostGIS operations.
4. **Stay framework-neutral.** The core depends only on NumPy. AISdb is an optional adapter (`pip install "aissegments[aisdb]"`); other input paths (CSV from Marine Cadastre / institutional exports / your own pipeline) work via [`read_csv_tracks`](src/aissegments/adapters.py) without any extra dependencies.

## Who is this for?

- **Maritime risk analysts** who need to run collision/grounding/allision queries against millions of vessel positions per area-year and want the spatial+temporal index to fit in memory.
- **AIS data engineers** maintaining a Postgres/PostGIS warehouse of vessel tracks and looking for a principled way to densify ingestion without overwhelming storage.
- **Researchers** reproducing or extending Guo et al.'s adaptive trajectory compression work.

## What it produces

- `tdkc(track)` — same `Track` interface but with only the *key points* retained (typically 1-5% of the input, depending on track shape and threshold tuning).
- `tdkc_segments(track)` — a list of `Segment` records, one per consecutive key-point pair, ready for direct insertion as PostGIS `LINESTRING(start_lon start_lat, end_lon end_lat)` geometries with `cog_mean`, `sog_mean`, and `n_points` (count of original AIS pings each segment represents).

## Why not just use Douglas-Peucker?

DP and its variants throw away every point that lies on a straight line, regardless of whether the vessel's *behaviour* is changing. A vessel slowing from 15 to 5 knots while continuing to head east — DP keeps two points (start, end) and you lose the entire speed change. TDKC keeps the deceleration point because its **velocity vector** has shifted. See [`docs/algorithm.md`](docs/algorithm.md) for the precise math, and [`examples/output/03_min_svd_sweep.png`](examples/output/) for a visual side-by-side.

## Companion package

[OMRAT](https://github.com/axelande/OMRAT) (Open Maritime Risk Analysis Tool) — a QGIS plugin for collision/grounding/allision risk modelling — uses AISsegments as its segment-ingestion backend. The OMRAT pipeline shows a complete end-to-end flow: NMEA / CSV → aisdb decode → TDKC compression → bulk-load into a year-partitioned PostGIS schema.

## Install

```bash
pip install aissegments

# with the optional AISdb adapter for ingestion from raw NMEA / CSV
pip install "aissegments[aisdb]"
```

For development with full test + coverage tooling:

```bash
git clone https://github.com/axelHorteborn/AISsegments
cd AISsegments
pip install -e ".[dev,aisdb]"
pytest --cov
```

## Quickstart

```python
import numpy as np
from aissegments import Track, tdkc, tdkc_segments

# Build a Track from your own arrays (lat/lon in degrees, sog in knots, cog in degrees).
track = Track.from_arrays(
    mmsi=219000123,
    t=np.array([0, 60, 120, 180, 240], dtype=float),     # unix seconds
    lon=np.array([12.0, 12.001, 12.002, 12.003, 12.004]),
    lat=np.array([55.0, 55.0, 55.0, 55.0, 55.0]),
    sog=np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
    cog=np.array([90.0, 90.0, 90.0, 90.0, 90.0]),
)

# Compress: returns a Track containing only the key points.
compressed = tdkc(track)
print(len(compressed), "key points kept out of", len(track))

# Or go straight to segment records (one per consecutive key-point pair).
segments = tdkc_segments(track)
for s in segments:
    print(s.t_start, s.t_end, s.cog_mean, s.sog_mean, s.n_points)
```

## Using AISdb as an input adapter

`aissegments` can consume the per-vessel track dicts produced by [AISdb](https://github.com/AISViz/AISdb)'s `TrackGen()`:

```python
import aisdb
from aissegments.adapters import from_aisdb_track
from aissegments import tdkc_segments

with aisdb.SQLiteDBConn(dbpath="ais.db") as conn:
    qry = aisdb.DBQuery(start=..., end=..., callback=aisdb.sql_query_strs.in_bbox_time)
    tracks = aisdb.TrackGen(qry.gen_qry(), decimate=False)
    for t_dict in tracks:
        track = from_aisdb_track(t_dict)
        for seg in tdkc_segments(track):
            ...  # write seg to your PostGIS table
```

## What's in the package

| Module | Purpose |
| --- | --- |
| [`aissegments.tdkc`](src/aissegments/tdkc.py) | TDKC algorithm: SED + SVD, Compression Binary Tree, adaptive thresholds, key-node identification |
| [`aissegments._types`](src/aissegments/_types.py) | `Track` and `Segment` dataclasses, `to_segments` helper |
| [`aissegments.adapters`](src/aissegments/adapters.py) | Input adapters: `from_aisdb_track`, `read_csv_tracks` (Marine Cadastre etc.), `read_csv_static_records` for vessel-info extraction |

## Algorithm details

See [docs/algorithm.md](docs/algorithm.md) for the mathematical formulation, with equation references back to the source paper.

## Citation

If you use this package in academic work, please cite both the software and the underlying paper:

> Guo, S., Bolbot, V., & Valdez Banda, O. (2024). An adaptive trajectory compression and feature preservation method for maritime traffic analysis. *Ocean Engineering*, 312, 119189. https://doi.org/10.1016/j.oceaneng.2024.119189

A `CITATION.cff` is included so GitHub renders a "Cite this repository" widget.

## License

MIT — see [LICENSE](LICENSE).
