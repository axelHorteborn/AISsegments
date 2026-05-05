"""Visual smoke-tests for the TDKC algorithm.

Generates up to seven PNG figures into ``examples/output/``:

1. ``01_real_aisdb_overview.png`` — the bundled aisdb test corpus shown as
   it really is: ~2,500 messages from ~2,300 vessels worldwide, mostly one
   ping per ship.  Loaded *at runtime* from the installed ``aisdb`` package
   (no AGPL bytes vendored into this MIT package).
2. ``02_min_sed_sweep.png`` — synthetic L-shape with isolated position noise.
3. ``03_min_svd_sweep.png`` — synthetic straight track with isolated SOG
   steps.  Synthetic because each parameter is *isolated* (one signal at a
   time), which makes the parameter effect easy to read.
4. ``04_kinematics.png`` — synthetic L-shape showing position + SOG + COG
   preservation.
5. ``05_track_types.png`` — synthetic track-type gallery (clean L,
   speed-change, mooring, noisy L) — reference of typical compression
   behaviour on dense data.

If at least one CSV / CSV.gz file is present under ``tests/data/`` (e.g. the
Marine Cadastre US-public-domain ``ais_example.csv``):

6. ``06_real_csv_overview.png`` — corpus-level view of the local data:
   geographic scatter coloured by SOG, track-length histogram, longest
   moving tracks with TDKC overlay, SOG distribution.
7. ``07_real_csv_param_sweep.png`` — ``min_sed_m`` and ``min_svd_kn``
   sweeps on a single dense moving real track.  In real data both signals
   are active simultaneously, so each sweep also clamps the *other*
   parameter to a high value to isolate the one being varied.

Run::

    pip install -e ".[dev]"      # matplotlib + aisdb come in via the dev extra
    python examples/visualize.py

These are *not* pytest tests — visual review by a human is the validation.
Algorithm correctness is covered to 100% by ``tests/``.
"""
from __future__ import annotations

import contextlib
import gc
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aissegments import Track, tdkc
from aissegments.adapters import from_aisdb_track

# ---------------------------------------------------------------------------
# Synthetic track generators
# ---------------------------------------------------------------------------


def make_noisy_l_track(
    n_per_leg: int = 50, noise_m: float = 10.0, seed: int = 42
) -> Track:
    """L-shape with noise on **position only**.

    SOG and COG are exact constants per leg, so SVD is identically zero and
    the parameter sweeps for ``min_sed_m`` show *only* the SED effect.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_leg
    t = np.arange(n, dtype=float) * 60.0
    lon_clean = np.concatenate(
        [12.0 + np.arange(n_per_leg) * 0.001, np.full(n_per_leg, 12.0 + (n_per_leg - 1) * 0.001)]
    )
    lat_clean = np.concatenate(
        [np.full(n_per_leg, 55.0), 55.0 + np.arange(1, n_per_leg + 1) * 0.001]
    )
    deg_per_m = 1.0 / 111_000.0
    lon = lon_clean + rng.normal(0, noise_m * deg_per_m, n)
    lat = lat_clean + rng.normal(0, noise_m * deg_per_m, n)
    sog = np.full(n, 10.0)
    cog = np.concatenate([np.full(n_per_leg, 90.0), np.full(n_per_leg, 0.0)])
    return Track.from_arrays(mmsi=219000111, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


def make_realistic_l_track(
    n_per_leg: int = 50, pos_noise_m: float = 10.0, seed: int = 42
) -> Track:
    """L-shape with noise on **all** of position, SOG and COG.

    Used in the qualitative track-types figure to show how TDKC behaves on
    realistic AIS data where every channel carries some noise.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_leg
    t = np.arange(n, dtype=float) * 60.0
    lon_clean = np.concatenate(
        [12.0 + np.arange(n_per_leg) * 0.001, np.full(n_per_leg, 12.0 + (n_per_leg - 1) * 0.001)]
    )
    lat_clean = np.concatenate(
        [np.full(n_per_leg, 55.0), 55.0 + np.arange(1, n_per_leg + 1) * 0.001]
    )
    deg_per_m = 1.0 / 111_000.0
    lon = lon_clean + rng.normal(0, pos_noise_m * deg_per_m, n)
    lat = lat_clean + rng.normal(0, pos_noise_m * deg_per_m, n)
    sog = np.full(n, 10.0) + rng.normal(0, 0.1, n)
    cog_clean = np.concatenate([np.full(n_per_leg, 90.0), np.full(n_per_leg, 0.0)])
    cog = (cog_clean + rng.normal(0, 1.0, n)) % 360.0
    return Track.from_arrays(mmsi=219000444, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


def make_clean_speed_change_track() -> Track:
    """Perfectly straight line with **discrete SOG steps and zero noise**.

    SED is identically zero, so the ``min_svd_kn`` sweep cleanly isolates
    the velocity-difference behaviour.
    """
    n = 200
    t = np.arange(n, dtype=float) * 30.0
    lon = 12.0 + np.linspace(0, 0.05, n)
    lat = np.full(n, 55.0)
    sog = np.zeros(n)
    sog[:50] = 5.0
    sog[50:100] = 10.0
    sog[100:150] = 5.0
    sog[150:] = 15.0
    cog = np.full(n, 90.0)
    return Track.from_arrays(mmsi=219000222, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


def make_jittery_speed_track(seed: int = 11) -> Track:
    """Speed steps **plus small jitter** in SOG only.

    Used to show that ``min_svd_kn`` filters jitter while preserving the
    real change-points.  Position is clean, COG is constant.
    """
    rng = np.random.default_rng(seed)
    base = make_clean_speed_change_track()
    sog = base.sog + rng.normal(0, 0.3, len(base))
    return Track.from_arrays(
        mmsi=base.mmsi, t=base.t, lon=base.lon, lat=base.lat, sog=sog, cog=base.cog
    )


def make_mooring_track(seed: int = 3) -> Track:
    """Cruise → station-keeping → cruise: realistic harbour-approach pattern.

    The stationary section is dominated by GPS jitter; a good algorithm
    represents it with a tight cluster of points (or just two), not by
    preserving every jitter sample.
    """
    rng = np.random.default_rng(seed)
    n_cruise = 30
    n_moored = 60
    n = 2 * n_cruise + n_moored
    t = np.arange(n, dtype=float) * 60.0
    deg_per_m = 1.0 / 111_000.0
    # Cruise in
    lon_c1 = 12.0 + np.arange(n_cruise) * 0.001
    lat_c1 = np.full(n_cruise, 55.0)
    # Moored: small jitter around the arrival point
    lon_m = lon_c1[-1] + rng.normal(0, 5.0 * deg_per_m, n_moored)
    lat_m = lat_c1[-1] + rng.normal(0, 5.0 * deg_per_m, n_moored)
    # Cruise out (different direction)
    lon_c2 = lon_m[-1] + np.arange(1, n_cruise + 1) * 0.001
    lat_c2 = lat_m[-1] + np.arange(1, n_cruise + 1) * 0.0005
    lon = np.concatenate([lon_c1, lon_m, lon_c2])
    lat = np.concatenate([lat_c1, lat_m, lat_c2])
    sog = np.concatenate(
        [np.full(n_cruise, 10.0), np.full(n_moored, 0.2), np.full(n_cruise, 8.0)]
    ) + rng.normal(0, 0.1, n)
    cog = np.concatenate(
        [np.full(n_cruise, 90.0), rng.uniform(0, 360, n_moored), np.full(n_cruise, 60.0)]
    )
    return Track.from_arrays(mmsi=219000333, t=t, lon=lon, lat=lat, sog=sog, cog=cog)


# ---------------------------------------------------------------------------
# Real CSV data loader (tests/data/*.csv*) and helpers
# ---------------------------------------------------------------------------


def _load_local_csv_tracks() -> list[Track]:
    """Load all CSV / CSV.gz files from tests/data/, sorted longest-first.

    Returns ``[]`` if no files are present.
    """
    from aissegments import read_csv_tracks
    data_dir = Path(__file__).resolve().parent.parent / "tests" / "data"
    if not data_dir.is_dir():
        return []
    files = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.csv.gz"))
    if not files:
        return []
    all_tracks: list[Track] = []
    for f in files:
        all_tracks.extend(read_csv_tracks(f))
    all_tracks.sort(key=len, reverse=True)
    return all_tracks


def _pick_moving_track(
    tracks: list[Track],
    *,
    min_pts: int = 30,
    min_max_sog: float = 5.0,
    min_span_deg: float = 0.05,
) -> Track | None:
    """Find the longest track that is actually moving (not anchored)."""
    for t in tracks:
        if len(t) < min_pts:
            continue
        if float(t.sog.max()) < min_max_sog:
            continue
        span = max(float(t.lon.max() - t.lon.min()), float(t.lat.max() - t.lat.min()))
        if span < min_span_deg:
            continue
        return t
    return None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_track_panel(
    ax: plt.Axes,
    original: Track,
    compressed: Track,
    title: str,
) -> None:
    ax.plot(
        original.lon, original.lat,
        marker=".", markersize=3, linestyle="none",
        color="lightgray", label=f"original ({len(original)} pts)",
    )
    ax.plot(
        compressed.lon, compressed.lat,
        marker="o", markersize=5, linestyle="-",
        color="crimson", linewidth=1.2,
        label=f"compressed ({len(compressed)} pts)",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Real AIS data loader (reads aisdb's installed test corpus at runtime)
# ---------------------------------------------------------------------------


def _load_aisdb_tracks() -> list[dict] | None:
    """Decode aisdb's bundled CSV into a list of track dicts; ``None`` if
    aisdb is not installed."""
    try:
        import aisdb
        from aisdb.database import sqlfcn_callbacks
    except ImportError:
        return None
    csv_file = Path(aisdb.__file__).parent / "tests" / "testdata" / "test_data_20210701.csv"
    if not csv_file.is_file():
        return None
    db_path = Path(tempfile.gettempdir()) / "aissegments_viz_aisdb.db"
    if db_path.exists():
        with contextlib.suppress(OSError):
            db_path.unlink()
    with aisdb.SQLiteDBConn(str(db_path)) as dbconn:
        aisdb.decode_msgs([str(csv_file)], dbconn=dbconn, source="VIZ", verbose=False)
    with aisdb.SQLiteDBConn(str(db_path)) as dbconn:
        q = aisdb.DBQuery(
            callback=sqlfcn_callbacks.in_timerange_validmmsi,
            dbconn=dbconn,
            start=datetime(2021, 7, 1),
            end=datetime(2021, 7, 8),
        )
        tracks = list(aisdb.TrackGen(q.gen_qry(), decimate=False))
    gc.collect()
    with contextlib.suppress(OSError):
        if db_path.exists():
            db_path.unlink()  # Windows may hold the lock briefly; tempdir cleans up
    return tracks


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_real_aisdb_overview(outdir: Path) -> Path | None:
    """Four-panel overview of aisdb's bundled real-AIS test corpus.

    Returns ``None`` (and prints a warning) if aisdb is not installed.
    """
    tracks = _load_aisdb_tracks()
    if not tracks:
        print(
            "WARN: aisdb not installed (or bundled CSV missing); skipping real-data figure. "
            "Install with: pip install -e \".[dev]\""
        )
        return None

    all_lons = np.concatenate([t["lon"] for t in tracks if len(t["lon"]) > 0])
    all_lats = np.concatenate([t["lat"] for t in tracks if len(t["lat"]) > 0])
    sizes = np.array([len(t["time"]) for t in tracks])
    long_tracks = sorted(tracks, key=lambda t: len(t["time"]), reverse=True)
    multi_point = [t for t in long_tracks if len(t["time"]) >= 2][:8]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -- Top-left: world scatter ------------------------------------------
    ax = axes[0, 0]
    ax.scatter(all_lons, all_lats, s=3, color="steelblue", alpha=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"All AIS positions in the corpus\n"
        f"{len(all_lons)} pings from {len(tracks)} unique vessels"
    )
    ax.grid(True, alpha=0.3)

    # -- Top-right: track-length histogram --------------------------------
    ax = axes[0, 1]
    bins = np.arange(1, sizes.max() + 2)
    counts, _, _ = ax.hist(sizes, bins=bins, edgecolor="black", color="steelblue")
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(bins[i] + 0.5, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Messages per vessel (track length)")
    ax.set_ylabel("Count of vessels")
    ax.set_title(
        f"Track-length distribution\n"
        f"max = {sizes.max()} pts; "
        f"only {(sizes >= 3).sum()}/{len(sizes)} tracks have ≥3 pts"
    )
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # -- Bottom-left: longest tracks with TDKC overlay --------------------
    ax = axes[1, 0]
    cmap = plt.get_cmap("tab10")
    for i, raw in enumerate(multi_point):
        track = from_aisdb_track(raw)
        compressed = tdkc(track)
        color = cmap(i)
        ax.plot(track.lon, track.lat, ".", color=color, markersize=8, alpha=0.4)
        ax.plot(
            compressed.lon, compressed.lat,
            "o-", color=color, markersize=6, linewidth=1.2,
            label=f"mmsi {track.mmsi} ({len(track)}→{len(compressed)})",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Longest {len(multi_point)} tracks in the corpus\n"
        f"original (faded) vs. TDKC output (solid)"
    )
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # -- Bottom-right: geographic density (hex bins) ----------------------
    ax = axes[1, 1]
    hb = ax.hexbin(all_lons, all_lats, gridsize=40, cmap="Blues", mincnt=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic density (hex bins, log colourbar)")
    cb = fig.colorbar(hb, ax=ax, orientation="vertical", shrink=0.8)
    cb.set_label("messages per bin")

    fig.suptitle(
        "Real AIS data from the aisdb test corpus "
        "(test_data_20210701.csv — 2021-07-01, worldwide)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout()
    out = outdir / "01_real_aisdb_overview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_track_types(outdir: Path) -> Path:
    """Reference: TDKC behaviour on four synthetic track shapes (dense data).

    Kept alongside the real-data figure because aisdb's bundled corpus has at
    most 3 points per vessel, which doesn't visualise compression well.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    tracks = [
        ("Realistic L-shape (noise on position, SOG and COG)", make_realistic_l_track()),
        ("Speed changes on a perfectly straight line", make_clean_speed_change_track()),
        ("Cruise → moored → cruise", make_mooring_track()),
        ("Clean L-shape (no noise anywhere)", make_realistic_l_track(pos_noise_m=0.0, seed=99)),
    ]
    for ax, (title, track) in zip(axes.flat, tracks, strict=True):
        compressed = tdkc(track)
        cr = 1 - len(compressed) / len(track)
        _plot_track_panel(ax, track, compressed, f"{title}\nCR = {cr:.0%}")
    fig.suptitle("TDKC default behaviour on synthetic track types (reference)", fontsize=13, y=0.995)
    fig.tight_layout()
    out = outdir / "05_track_types.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_min_sed_sweep(outdir: Path) -> Path:
    """``min_sed_m`` trades GPS-noise rejection for spatial detail.

    Uses a track with noise on **position only** (SOG/COG are exact constants),
    so SVD is identically zero and only ``min_sed_m`` drives the result.
    """
    track = make_noisy_l_track(noise_m=10.0)
    floors = [0.0, 5.0, 50.0, 500.0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, floor in zip(axes, floors, strict=True):
        compressed = tdkc(track, min_sed_m=floor, min_svd_kn=0.0)
        cr = 1 - len(compressed) / len(track)
        _plot_track_panel(
            ax, track, compressed,
            f"min_sed_m = {floor:g} m\n{len(compressed)} pts kept (CR = {cr:.0%})",
        )
    fig.suptitle(
        "Effect of min_sed_m on an L-shape with 10 m position noise "
        "(min_svd_kn = 0; SOG/COG exact)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = outdir / "02_min_sed_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_min_svd_sweep(outdir: Path) -> Path:
    """``min_svd_kn`` trades speed-jitter rejection for kinematic detail.

    Uses a clean straight-line track with discrete SOG steps plus small
    jitter, so SED is identically zero and only ``min_svd_kn`` drives the
    result.
    """
    track = make_jittery_speed_track()
    floors = [0.0, 0.5, 1.5, 5.0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, floor in zip(axes, floors, strict=True):
        compressed = tdkc(track, min_sed_m=0.0, min_svd_kn=floor)
        cr = 1 - len(compressed) / len(track)
        ax.plot(track.t / 60.0, track.sog, color="lightgray", linewidth=0.7, label=f"orig ({len(track)})")
        ax.plot(
            compressed.t / 60.0, compressed.sog,
            "o-", color="crimson", markersize=5, linewidth=1.2,
            label=f"kept ({len(compressed)})",
        )
        ax.set_title(
            f"min_svd_kn = {floor:g} kn\nCR = {cr:.0%}",
            fontsize=10,
        )
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("SOG (knots)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Effect of min_svd_kn on a straight track with SOG steps + jitter "
        "(min_sed_m = 0; position exact)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = outdir / "03_min_svd_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_kinematics(outdir: Path) -> Path:
    """Position + SOG + COG side-by-side: what's preserved through compression."""
    track = make_realistic_l_track(pos_noise_m=8.0)
    compressed = tdkc(track)
    cr = 1 - len(compressed) / len(track)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    ax = axes[0]
    ax.plot(track.lon, track.lat, ".", markersize=3, color="lightgray", label="original")
    ax.plot(compressed.lon, compressed.lat, "o-", color="crimson", markersize=5, label="kept")
    ax.set_title(f"Position (CR = {cr:.0%})", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(track.t / 60.0, track.sog, color="lightgray", label="original")
    ax.plot(
        compressed.t / 60.0, compressed.sog, "o-", color="crimson", markersize=5, label="kept",
    )
    ax.set_title("Speed over time", fontsize=11)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SOG (knots)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(track.t / 60.0, track.cog, color="lightgray", label="original")
    ax.plot(
        compressed.t / 60.0, compressed.cog, "o-", color="crimson", markersize=5, label="kept",
    )
    ax.set_title("Course over time", fontsize=11)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("COG (degrees)")
    ax.set_ylim(-10, 370)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Position, speed, and course preservation through TDKC", fontsize=13, y=0.995
    )
    fig.tight_layout()
    out = outdir / "04_kinematics.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def fig_real_csv_overview(outdir: Path) -> Path | None:
    """Four-panel overview of locally-dropped real AIS data (tests/data/*.csv*)."""
    tracks = _load_local_csv_tracks()
    if not tracks:
        print("INFO: no tests/data/*.csv files; skipping local-data overview")
        return None

    sizes = np.array([len(t) for t in tracks])
    all_lons = np.concatenate([t.lon for t in tracks])
    all_lats = np.concatenate([t.lat for t in tracks])
    all_sogs = np.concatenate([t.sog for t in tracks])
    long_tracks = [t for t in tracks if len(t) >= 30 and float(t.sog.max()) >= 5][:6]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -- Top-left: scatter coloured by speed --------------------------------
    ax = axes[0, 0]
    sc = ax.scatter(all_lons, all_lats, s=2, c=all_sogs, cmap="viridis", alpha=0.4, vmin=0, vmax=20)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"All AIS positions ({len(all_lons):,} pings, {len(tracks):,} vessels)\n"
        f"colour = SOG (knots)"
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("SOG (knots)")
    ax.grid(True, alpha=0.3)

    # -- Top-right: track-length histogram -----------------------------------
    ax = axes[0, 1]
    ax.hist(sizes, bins=50, color="steelblue", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("Messages per vessel")
    ax.set_ylabel("Count of vessels (log)")
    ax.set_title(
        f"Track-length distribution\n"
        f"max={sizes.max()}, median={int(np.median(sizes))}, "
        f"mean={sizes.mean():.1f}"
    )
    ax.grid(True, alpha=0.3)

    # -- Bottom-left: 6 longest moving tracks with TDKC overlay --------------
    ax = axes[1, 0]
    cmap = plt.get_cmap("tab10")
    for i, track in enumerate(long_tracks):
        compressed = tdkc(track)
        cr = 1 - len(compressed) / len(track)
        color = cmap(i)
        ax.plot(track.lon, track.lat, ".", color=color, markersize=3, alpha=0.4)
        ax.plot(
            compressed.lon, compressed.lat,
            "o-", color=color, markersize=4, linewidth=1.0,
            label=f"mmsi {track.mmsi}: {len(track)}→{len(compressed)} (CR={cr:.0%})",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Six longest moving tracks (≥30 pts, max SOG ≥ 5 kn)\n"
        "original (faded) vs. TDKC output (solid)"
    )
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # -- Bottom-right: SOG distribution --------------------------------------
    ax = axes[1, 1]
    ax.hist(all_sogs, bins=60, color="steelblue", edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel("SOG (knots)")
    ax.set_ylabel("Count of pings (log)")
    ax.set_title("Speed distribution across all pings")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Real AIS data from tests/data/ "
        "(Marine Cadastre US public-domain example)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout()
    out = outdir / "06_real_csv_overview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_real_csv_param_sweep(outdir: Path) -> Path | None:
    """Parameter sweep on a single dense real moving track from tests/data/."""
    tracks = _load_local_csv_tracks()
    if not tracks:
        return None
    track = _pick_moving_track(tracks)
    if track is None:
        print("INFO: no moving track found in tests/data/; skipping param-sweep figure")
        return None

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # To isolate one parameter on real data we have to *suppress* the other —
    # otherwise both SED- and SVD-driven keys are active simultaneously and a
    # sweep on one parameter shows no visible effect.  We set the suppressed
    # one to a value far above any plausible AIS feature.
    svd_suppress_kn = 1e6   # well above any real velocity step
    sed_suppress_m = 1e9    # well above any real position deviation

    # Top row: min_sed_m sweep, position view (SVD path suppressed)
    sed_floors = [0.0, 50.0, 500.0, 5000.0]
    for col, floor in enumerate(sed_floors):
        ax = axes[0, col]
        compressed = tdkc(track, min_sed_m=floor, min_svd_kn=svd_suppress_kn)
        cr = 1 - len(compressed) / len(track)
        ax.plot(track.lon, track.lat, ".", markersize=3, color="lightgray",
                label=f"orig ({len(track)})")
        ax.plot(compressed.lon, compressed.lat, "o-", color="crimson", markersize=4,
                label=f"kept ({len(compressed)})")
        ax.set_title(f"min_sed_m = {floor:g} m\nCR = {cr:.0%}", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom row: min_svd_kn sweep, SOG over time (SED path suppressed)
    svd_floors = [0.0, 0.5, 2.0, 10.0]
    for col, floor in enumerate(svd_floors):
        ax = axes[1, col]
        compressed = tdkc(track, min_sed_m=sed_suppress_m, min_svd_kn=floor)
        cr = 1 - len(compressed) / len(track)
        t_min = (track.t - track.t[0]) / 60.0
        ct_min = (compressed.t - track.t[0]) / 60.0
        ax.plot(t_min, track.sog, color="lightgray", linewidth=0.7,
                label=f"orig ({len(track)})")
        ax.plot(ct_min, compressed.sog, "o-", color="crimson", markersize=4,
                label=f"kept ({len(compressed)})")
        ax.set_title(f"min_svd_kn = {floor:g} kn\nCR = {cr:.0%}", fontsize=10)
        ax.set_xlabel("Time (min, since track start)")
        ax.set_ylabel("SOG (knots)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Parameter sweeps on a real moving track  "
        f"(MMSI {track.mmsi}, {len(track)} pts, "
        f"max SOG = {float(track.sog.max()):.1f} kn)\n"
        "top row: only min_sed_m varies (SVD path suppressed); "
        "bottom row: only min_svd_kn varies (SED path suppressed)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    out = outdir / "07_real_csv_param_sweep.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True)
    figures = [
        fig_real_aisdb_overview(outdir),
        fig_min_sed_sweep(outdir),
        fig_min_svd_sweep(outdir),
        fig_kinematics(outdir),
        fig_track_types(outdir),
        fig_real_csv_overview(outdir),
        fig_real_csv_param_sweep(outdir),
    ]
    figures = [f for f in figures if f is not None]
    print(f"Wrote {len(figures)} figures to {outdir}:")
    for f in figures:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
