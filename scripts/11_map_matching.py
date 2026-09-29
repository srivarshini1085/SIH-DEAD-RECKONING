"""
Advanced Map-Matching & Non-Holonomic Constraint (NHC) Engine
=============================================================
Takes a fused (lat, lng) trajectory and snaps it onto real road geometry.
A vehicle cannot drive through buildings or slide sideways off a road.

Pipeline:
  1. Accept trajectory as list of (lat, lng, timestamp) dicts or a CSV path.
  2. Download (or load cached) OSM drivable road network for the bounding box.
  3. For each point: find nearest road edge → project onto that edge (shapely).
  4. Apply NHC: if implied lateral movement between consecutive points exceeds
     threshold, pull the point toward the road centerline.
  5. Only accept snapped position if within MAX_SNAP_DIST_M of original point.
  6. Save before/after comparison plot → models/map_matching_comparison.png.
  7. Return snapped trajectory as list of dicts.

Usage (standalone):
    py scripts/11_map_matching.py --csv data/processed/final_trajectory.csv
    py scripts/11_map_matching.py --csv data/processed/final_trajectory.csv --plot

Required packages:
    pip install osmnx networkx shapely geopandas matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRAPH_CACHE_DIR = ROOT / "data" / "osm_cache"
PLOT_OUT = ROOT / "models" / "map_matching_comparison.png"

# ── Tuning constants ──────────────────────────────────────────────────────────
MAX_SNAP_DIST_M   = 30.0    # reject snap if road is farther than this
NHC_LATERAL_MAX_M = 3.0     # max allowed lateral (sideways) displacement per step
OSM_PADDING_DEG   = 0.002   # ~200 m padding around bounding box
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map-match a fused trajectory onto OSM road network.")
    p.add_argument("--csv",  type=Path, default=ROOT / "data" / "processed" / "final_trajectory.csv",
                   help="CSV with columns: time, ai_lat, ai_lng (output of 09_run_fusion_pipeline.py)")
    p.add_argument("--plot", action="store_true", help="Save before/after comparison plot")
    p.add_argument("--no-osm", action="store_true",
                   help="Skip OSM download (use NHC-only correction, useful offline)")
    return p.parse_args()


# ── Geometry helpers ──────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2)**2
    return float(2 * R * np.arcsin(np.sqrt(min(1.0, a))))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360) % 360)


def lateral_displacement_m(lat1, lon1, lat2, lon2, road_bearing_deg: float) -> float:
    """
    Compute the lateral (sideways) component of displacement from point1 to point2
    relative to the road bearing direction.
    """
    dist = haversine_m(lat1, lon1, lat2, lon2)
    if dist < 1e-6:
        return 0.0
    move_bearing = bearing_deg(lat1, lon1, lat2, lon2)
    angle_diff = np.radians((move_bearing - road_bearing_deg + 180) % 360 - 180)
    return abs(dist * np.sin(angle_diff))


# ── OSM graph loading ─────────────────────────────────────────────────────────

def load_or_download_graph(lats: np.ndarray, lons: np.ndarray):
    """Load cached OSM graph or download if not present."""
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError("osmnx not installed. Run: pip install osmnx")

    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    lat_min = float(lats.min()) - OSM_PADDING_DEG
    lat_max = float(lats.max()) + OSM_PADDING_DEG
    lon_min = float(lons.min()) - OSM_PADDING_DEG
    lon_max = float(lons.max()) + OSM_PADDING_DEG

    # Cache key based on rounded bounding box
    cache_key = f"{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}"
    cache_path = GRAPH_CACHE_DIR / f"road_graph_{cache_key}.graphml"

    if cache_path.exists():
        print(f"  Loading cached OSM graph: {cache_path.name}")
        G = ox.load_graphml(cache_path)
    else:
        print(f"  Downloading OSM road network for bbox "
              f"[{lat_min:.4f},{lat_max:.4f},{lon_min:.4f},{lon_max:.4f}]...")
        G = ox.graph_from_bbox(
            lat_max, lat_min, lon_max, lon_min,
            network_type="drive",
            simplify=True,
        )
        ox.save_graphml(G, cache_path)
        print(f"  Cached → {cache_path}")

    return G


# ── Project point onto nearest road edge ─────────────────────────────────────

def snap_point_to_road(G, lat: float, lon: float, ox_module):
    """
    Find nearest edge and project (lat, lon) onto it.
    Returns (snapped_lat, snapped_lon, road_bearing_deg, dist_to_road_m).
    """
    from shapely.geometry import Point, LineString

    try:
        u, v, key = ox_module.distance.nearest_edges(G, lon, lat)
    except Exception:
        return lat, lon, 0.0, float("inf")

    edge_data = G.edges[u, v, key]
    if "geometry" in edge_data:
        line = edge_data["geometry"]
    else:
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        line = LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])

    pt = Point(lon, lat)
    proj_dist = line.project(pt)
    snapped_pt = line.interpolate(proj_dist)
    snapped_lon, snapped_lat = snapped_pt.x, snapped_pt.y

    dist_m = haversine_m(lat, lon, snapped_lat, snapped_lon)

    # Road bearing from edge geometry
    coords = list(line.coords)
    if len(coords) >= 2:
        road_bearing = bearing_deg(coords[0][1], coords[0][0], coords[-1][1], coords[-1][0])
    else:
        road_bearing = 0.0

    return snapped_lat, snapped_lon, road_bearing, dist_m


# ── NHC-only correction (no OSM, offline fallback) ───────────────────────────

def apply_nhc_only(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-Holonomic Constraint: smooth out lateral jumps without OSM.
    If lateral displacement between consecutive points exceeds NHC_LATERAL_MAX_M,
    blend the point back toward the previous position.
    """
    out_lats = lats.copy()
    out_lons = lons.copy()

    for i in range(1, len(lats)):
        if i < 2:
            continue
        road_bearing = bearing_deg(lats[i-2], lons[i-2], lats[i-1], lons[i-1])
        lateral = lateral_displacement_m(lats[i-1], lons[i-1], lats[i], lons[i], road_bearing)
        if lateral > NHC_LATERAL_MAX_M:
            # Pull point toward previous position proportionally
            alpha = NHC_LATERAL_MAX_M / (lateral + 1e-6)
            out_lats[i] = lats[i-1] + alpha * (lats[i] - lats[i-1])
            out_lons[i] = lons[i-1] + alpha * (lons[i] - lons[i-1])

    return out_lats, out_lons


# ── Main map-matching function ────────────────────────────────────────────────

def map_match(
    lats: np.ndarray,
    lons: np.ndarray,
    use_osm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap trajectory to road network with NHC.

    Args:
        lats, lons : raw fused trajectory arrays
        use_osm    : if False, apply NHC-only (offline mode)

    Returns:
        snapped_lats, snapped_lons
    """
    if not use_osm:
        print("  OSM disabled — applying NHC-only correction.")
        return apply_nhc_only(lats, lons)

    try:
        import osmnx as ox
    except ImportError:
        print("  osmnx not available — falling back to NHC-only.")
        return apply_nhc_only(lats, lons)

    try:
        G = load_or_download_graph(lats, lons)
    except Exception as e:
        print(f"  OSM graph load failed ({e}) — falling back to NHC-only.")
        return apply_nhc_only(lats, lons)

    snapped_lats = lats.copy()
    snapped_lons = lons.copy()
    n_snapped = 0
    n_nhc_corrected = 0

    print(f"  Snapping {len(lats)} trajectory points to road network...")
    for i in range(len(lats)):
        s_lat, s_lon, road_bearing, dist_m = snap_point_to_road(G, lats[i], lons[i], ox)

        # Only accept snap if road is close enough
        if dist_m <= MAX_SNAP_DIST_M:
            snapped_lats[i] = s_lat
            snapped_lons[i] = s_lon
            n_snapped += 1

            # NHC check: lateral displacement vs road bearing
            if i > 0:
                lateral = lateral_displacement_m(
                    snapped_lats[i-1], snapped_lons[i-1],
                    snapped_lats[i],   snapped_lons[i],
                    road_bearing,
                )
                if lateral > NHC_LATERAL_MAX_M:
                    alpha = NHC_LATERAL_MAX_M / (lateral + 1e-6)
                    snapped_lats[i] = snapped_lats[i-1] + alpha * (snapped_lats[i] - snapped_lats[i-1])
                    snapped_lons[i] = snapped_lons[i-1] + alpha * (snapped_lons[i] - snapped_lons[i-1])
                    n_nhc_corrected += 1

    print(f"  Snapped: {n_snapped}/{len(lats)} points  |  NHC corrections: {n_nhc_corrected}")
    return snapped_lats, snapped_lons


# ── Plot ──────────────────────────────────────────────────────────────────────

def save_comparison_plot(
    raw_lats, raw_lons,
    snapped_lats, snapped_lons,
    G=None,
    outage_mask: np.ndarray = None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: trajectory comparison ──
    ax = axes[0]
    ax.plot(raw_lons, raw_lats, "b-", lw=1.5, alpha=0.7, label="Raw fused (EKF)")
    ax.plot(snapped_lons, snapped_lats, "g-", lw=2, label="Map-matched")
    if outage_mask is not None:
        ax.plot(raw_lons[outage_mask], raw_lats[outage_mask], "r.", ms=4, label="GPS denied (raw)")
        ax.plot(snapped_lons[outage_mask], snapped_lats[outage_mask], "m.", ms=4, label="GPS denied (snapped)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Raw EKF vs Map-Matched Trajectory")
    ax.legend(fontsize=8)

    # ── Right: per-point snap distance ──
    ax = axes[1]
    snap_dist = np.array([
        haversine_m(raw_lats[i], raw_lons[i], snapped_lats[i], snapped_lons[i])
        for i in range(len(raw_lats))
    ])
    ax.plot(snap_dist, "g-", lw=1)
    ax.axhline(MAX_SNAP_DIST_M, color="r", ls="--", label=f"Max snap dist ({MAX_SNAP_DIST_M}m)")
    ax.set_xlabel("Window index"); ax.set_ylabel("Snap distance (m)")
    ax.set_title("Point-to-Road Distance")
    ax.legend(fontsize=8)

    plt.tight_layout()
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=120)
    plt.close()
    print(f"  Plot saved → {PLOT_OUT}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"Trajectory CSV not found: {args.csv}\n"
            "Run 09_run_fusion_pipeline.py first."
        )

    df = pd.read_csv(args.csv)
    required = {"ai_lat", "ai_lng"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Expected output of 09_run_fusion_pipeline.py")

    raw_lats = df["ai_lat"].to_numpy(dtype=np.float64)
    raw_lons = df["ai_lng"].to_numpy(dtype=np.float64)

    print(f"\nMap-matching {len(raw_lats)} trajectory points...")
    snapped_lats, snapped_lons = map_match(raw_lats, raw_lons, use_osm=not args.no_osm)

    # Update CSV in-place
    df["ai_lat"] = snapped_lats
    df["ai_lng"] = snapped_lons
    df.to_csv(args.csv, index=False)
    print(f"  Updated CSV → {args.csv}")

    if args.plot or True:   # always save plot when run standalone
        outage_mask = None
        if "gps_status" in df.columns:
            outage_mask = (df["gps_status"] == "DENIED").to_numpy()
        save_comparison_plot(raw_lats, raw_lons, snapped_lats, snapped_lons,
                             outage_mask=outage_mask)

    total_snap = np.mean([
        haversine_m(raw_lats[i], raw_lons[i], snapped_lats[i], snapped_lons[i])
        for i in range(len(raw_lats))
    ])
    print(f"\n  Mean snap correction: {total_snap:.2f} m")


if __name__ == "__main__":
    main()
