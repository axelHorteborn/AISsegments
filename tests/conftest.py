"""Shared pytest fixtures: synthetic + real-AIS-data tracks for the test suite.

The real-AIS fixtures decode the test data shipped with the ``aisdb`` package
(``aisdb/tests/testdata/test_data_20210701.csv`` — 2,372 dynamic AIS messages
from 2021-07-01).  They are guarded by ``pytest.importorskip`` so the suite
still runs when ``aisdb`` is not installed (the fixture skips, dependent
tests skip in turn).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from aissegments import Track


@pytest.fixture
def straight_track() -> Track:
    """100 evenly-sampled points moving due east at constant 10 knots.

    All intermediate points lie exactly on the start-end baseline so SED is
    zero throughout, and SOG/COG are constant so SVD is zero — TDKC must
    compress this to just the two endpoints.
    """
    n = 100
    t = np.arange(n, dtype=float) * 60.0
    lon = 12.0 + np.arange(n) * 0.001
    lat = np.full(n, 55.0)
    sog = np.full(n, 10.0)
    cog = np.full(n, 90.0)
    return Track.from_arrays(mmsi=219000001, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


@pytest.fixture
def l_shaped_track() -> Track:
    """Two straight legs of 50 points each, joined by a sharp 90 degree turn.

    The turn point at index 50 is a strong feature in both position and
    course; it must survive compression.
    """
    n_leg = 50
    t = np.arange(2 * n_leg, dtype=float) * 60.0
    # Leg 1: east at 10 kn.
    lon1 = 12.0 + np.arange(n_leg) * 0.001
    lat1 = np.full(n_leg, 55.0)
    cog1 = np.full(n_leg, 90.0)
    # Leg 2: north at 10 kn (continuing from end of leg 1).
    lon2 = np.full(n_leg, lon1[-1])
    lat2 = lat1[-1] + np.arange(1, n_leg + 1) * 0.001
    cog2 = np.full(n_leg, 0.0)
    lon = np.concatenate([lon1, lon2])
    lat = np.concatenate([lat1, lat2])
    cog = np.concatenate([cog1, cog2])
    sog = np.full(2 * n_leg, 10.0)
    return Track.from_arrays(mmsi=219000002, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


@pytest.fixture
def speed_change_track() -> Track:
    """Straight east-bound trajectory but with a step-change in speed at midpoint.

    Position alone (DP) would compress this aggressively because everything
    lies on a straight line; SVD sees the speed step and forces TDKC to keep
    the change-point.
    """
    n = 100
    t = np.arange(n, dtype=float) * 60.0
    lon = 12.0 + np.arange(n) * 0.001
    lat = np.full(n, 55.0)
    sog = np.where(np.arange(n) < 50, 5.0, 15.0)
    cog = np.full(n, 90.0)
    return Track.from_arrays(mmsi=219000003, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


# ---------------------------------------------------------------------------
# Real AIS data from aisdb's bundled test corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def aisdb_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Decode aisdb's bundled CSV test data into a session-scoped SQLite DB.

    The DB is built once per session.  ``aisdb`` is a hard prerequisite; if
    it is not importable, fixtures that depend on this one are skipped.

    aisdb caches its ``db_daterange`` on the connection at construction time,
    so the decoder uses one connection (closed via ``with``) and queries open
    a fresh connection inside the test body — see ``aisdb_tracks``.
    """
    aisdb = pytest.importorskip("aisdb")
    testdata_root = Path(aisdb.__file__).parent / "tests" / "testdata"
    csv_file = testdata_root / "test_data_20210701.csv"
    if not csv_file.is_file():
        pytest.skip(f"aisdb bundled test data not found at {csv_file}")
    db_path = tmp_path_factory.mktemp("aisdb_corpus") / "test_data.db"
    with aisdb.SQLiteDBConn(str(db_path)) as dbconn:
        aisdb.decode_msgs([str(csv_file)], dbconn=dbconn, source="TEST", verbose=False)
    return db_path


@pytest.fixture(scope="session")
def aisdb_tracks(aisdb_db_path: Path) -> list[dict]:
    """Materialised list of AISdb track dicts from ``aisdb_db_path``.

    Tracks are sorted (descending) by length so dependent tests can pick the
    longest available track without re-scanning.
    """
    aisdb = pytest.importorskip("aisdb")
    from aisdb.database import sqlfcn_callbacks

    with aisdb.SQLiteDBConn(str(aisdb_db_path)) as dbconn:
        q = aisdb.DBQuery(
            callback=sqlfcn_callbacks.in_timerange_validmmsi,
            dbconn=dbconn,
            start=datetime(2021, 7, 1),
            end=datetime(2021, 7, 8),
        )
        tracks = list(aisdb.TrackGen(q.gen_qry(), decimate=False))
    if not tracks:
        pytest.skip("aisdb decoded the test corpus but yielded no tracks")
    tracks.sort(key=lambda t: len(t["time"]), reverse=True)
    return tracks


@pytest.fixture(scope="session")
def local_csv_tracks() -> list[Track]:
    """Tracks loaded from any ``tests/data/*.csv`` files; skipped if none.

    Drop denser AIS data into ``tests/data/`` to activate the integration
    tests in ``test_local_data.py``.  See ``tests/data/README.md`` for the
    expected CSV format and licensing guidance.
    """
    from aissegments.adapters import read_csv_tracks

    data_dir = Path(__file__).parent / "data"
    if not data_dir.is_dir():
        pytest.skip("tests/data/ does not exist")
    csv_files = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.csv.gz"))
    if not csv_files:
        pytest.skip("No CSV / CSV.gz files in tests/data/")
    all_tracks: list[Track] = []
    for f in csv_files:
        all_tracks.extend(read_csv_tracks(f))
    if not all_tracks:
        pytest.skip("CSV files in tests/data/ contained no tracks")
    all_tracks.sort(key=len, reverse=True)
    return all_tracks


@pytest.fixture(scope="session")
def aisdb_multi_point_track(aisdb_tracks: list[dict]) -> dict:
    """The longest track in the bundled aisdb corpus (>= 3 points).

    The shipped corpus is a one-day snapshot with many vessels seen only once
    or twice, so the longest track is on the order of a few points.  Three
    points is the smallest input that actually exercises TDKC's CBT recursion
    (one intermediate point), so it is the threshold this fixture enforces.
    """
    multi = [t for t in aisdb_tracks if len(t["time"]) >= 3]
    if not multi:
        pytest.skip("No track with >= 3 points in aisdb test corpus")
    return multi[0]
