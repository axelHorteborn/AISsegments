# Algorithm reference

This document complements [the source paper](https://doi.org/10.1016/j.oceaneng.2024.119189) with implementation-level notes specific to `aissegments`.

## Synchronous Euclidean Distance (SED)

For an intermediate point at index `i` in a sub-trajectory `[start, end]`:

1. Compute the time fraction `frac = (t[i] - t[start]) / (t[end] - t[start])`.
2. Linearly interpolate position in lat/lon space:
   - `lon_sync = lon[start] + frac * (lon[end] - lon[start])`
   - `lat_sync = lat[start] + frac * (lat[end] - lat[start])`
3. SED is the great-circle (haversine) distance between `(lon[i], lat[i])` and `(lon_sync, lat_sync)`, in metres.

The paper's Cartesian formulation introduces distortion that grows with latitude. We use haversine throughout (paper Sec. 3.2.1) so SED is meaningful in any sea area.

## Synchronous Velocity Difference (SVD)

For the same intermediate point:

1. Compute the start-to-end speed step `delta_s = sog[end] - sog[start]`.
2. Compute the start-to-end course step **using the shorter arc** so that 350° → 10° is +20°, not –340°:
   `delta_c = ((cog[end] - cog[start] + 180) mod 360) - 180`
3. Linearly interpolate speed and course to time `t[i]`:
   - `s_sync = sog[start] + frac * delta_s`
   - `c_sync = cog[start] + frac * delta_c`
4. Convert both `(s_sync, c_sync)` and the actual `(sog[i], cog[i])` into 2-D velocity vectors (`vx = s sin(c)`, `vy = s cos(c)`) and take the magnitude of the difference.

Working in velocity-vector space (rather than separate speed and course thresholds) lets a single SVD scalar capture both speed and course deviation — see paper Fig. 7 for the motivating example.

## Aggregated split-point selection

Within each sub-trajectory, SED and SVD are z-score normalised across the intermediate points and **summed** (paper Eq. 24). The split point is the one with the maximum aggregated value. Z-scoring within the sub-trajectory (rather than globally) means the split is responsive to local feature density: a tight cluster of stationary points won't get spuriously split because everything in it has near-zero raw SED/SVD.

If the sub-trajectory has constant SED (or constant SVD), the corresponding z-score returns zeros — so the other dimension drives the split. If both are constant, the algorithm picks index `start + 1` deterministically (`np.argmax` on an all-zero array).

## Compression Binary Tree

Construction is recursive (paper Algorithm 3) and deliberately **does not consult any threshold** — it exhausts down to length-2 leaves. This decouples tree-building from threshold determination and fixes the *recursion termination problem*: classical Douglas-Peucker stops as soon as one node falls below threshold, which can drop a high-feature node hidden behind a low-feature ancestor.

The tree is built once; the threshold is then computed as the mean SED and mean SVD across all tree nodes (Eqs. 27-28).

## Key-node identification

A node is a *key node* if either of its measurements exceeds the corresponding threshold, **or** if any of its descendants is a key node. The recursion keeps the high-feature descendants reachable even when their parents are below threshold — this is the central correctness fix of TDKC over DP/TD-TR.

## What `aissegments` outputs

`tdkc(track)` returns a new `Track` containing:

- the original endpoints (always),
- the index of every key node identified by the procedure above.

`tdkc_segments(track)` additionally pairs consecutive key points into `Segment` records and back-fills `n_points` with the count of original AIS observations spanned by each segment, including both endpoints. Sum of `(n_points - 1)` across all segments equals `len(track) - 1`.

## Complexity

- CBT construction: `O(N log N)` for balanced inputs (each level is `O(N)` work, depth is `O(log N)`).
- Threshold collection: `O(N)`.
- Key-node identification: `O(N)`.

For pathological inputs the tree degenerates and the construction phase is `O(N²)`; recursion depth scales with tree depth so very long degenerate inputs may need an explicit `sys.setrecursionlimit()` bump.

## Coordinates and units

| Quantity | Unit | Convention |
| --- | --- | --- |
| `t`   | seconds since Unix epoch | UTC |
| `lon`, `lat` | degrees | WGS84 (EPSG:4326) |
| `sog` | knots | per AIS spec |
| `cog` | degrees | nautical compass (0° = N, 90° = E), `[0, 360)` |
