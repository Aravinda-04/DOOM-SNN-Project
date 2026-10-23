"""Universal navigation route graph and causal progress tracker for any Doom level.

Generates dynamic waypoint routes and interactive trigger contexts from WAD geometry
without requiring human demonstration recordings. Provides 100% drop-in compatibility
with the frozen 84-dimensional SNN engine-state contract.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from map01_route import RouteContext, angle_delta_degrees
from wad_navmesh import NavGrid, NavMeshConfig, build_navmesh
from wad_parser import DoomMap, load_map_from_wad
from wad_pathfinder import PathfinderConfig, find_nearest_health_item, plan_level_route


class GeneralizedRoute:
    """Universal level route graph compatible with the SNN RouteContext contract."""

    def __init__(
        self,
        points: np.ndarray | Sequence[tuple[float, float]],
        *,
        use_points: np.ndarray | Sequence[tuple[float, float]] | None = None,
        source_name: str = "wad_autonomous_route",
        navgrid: NavGrid | None = None,
    ):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("Route points must have shape (N, 2), with N >= 2.")
        if not np.all(np.isfinite(pts)):
            raise ValueError("Route points contain non-finite coordinates.")

        self.points = pts
        self.edges = tuple((i, i + 1) for i in range(len(pts) - 1))
        self.source_name = str(source_name)
        self.navgrid = navgrid

        if use_points is not None and len(use_points) > 0:
            self.use_points = np.asarray(use_points, dtype=np.float32)
        else:
            self.use_points = np.empty((0, 2), dtype=np.float32)

        # Content hash
        digest = hashlib.sha256()
        digest.update(b"generalized-doom-route-v1")
        digest.update(self.points.tobytes())
        digest.update(self.use_points.tobytes())
        digest.update(self.source_name.encode("utf-8"))
        self.sha256 = digest.hexdigest()

        # Cache visited items for dynamic health seeking
        self._visited_items: set[int] = set()

    @classmethod
    def from_navgrid(
        cls,
        grid: NavGrid,
        *,
        start_pos: tuple[float, float] | None = None,
        health_detour: bool = False,
        pathfinder_config: PathfinderConfig | None = None,
    ) -> "GeneralizedRoute":
        """Construct GeneralizedRoute directly from a pre-built NavGrid."""
        points = plan_level_route(
            grid,
            start_pos=start_pos,
            health_detour=health_detour,
            config=pathfinder_config,
        )

        # Consolidate interaction trigger centers (doors and exit switches)
        use_pts: list[tuple[float, float]] = []
        use_pts.extend(grid.door_midpoints)
        use_pts.extend(grid.exit_midpoints)

        return cls(
            points=points,
            use_points=use_pts,
            source_name=f"{grid.doom_map.name}_AStar",
            navgrid=grid,
        )

    @classmethod
    def from_map(
        cls,
        doom_map: DoomMap,
        *,
        mesh_config: NavMeshConfig | None = None,
        pathfinder_config: PathfinderConfig | None = None,
        health_detour: bool = False,
    ) -> "GeneralizedRoute":
        """Construct GeneralizedRoute directly from a parsed DoomMap level."""
        grid = build_navmesh(doom_map, mesh_config)
        return cls.from_navgrid(
            grid,
            health_detour=health_detour,
            pathfinder_config=pathfinder_config,
        )

    @classmethod
    def from_wad(
        cls,
        wad_path: Path | str = "freedoom2.wad",
        map_name: str = "MAP01",
        *,
        mesh_config: NavMeshConfig | None = None,
        pathfinder_config: PathfinderConfig | None = None,
        health_detour: bool = False,
    ) -> "GeneralizedRoute":
        """Construct GeneralizedRoute from WAD archive and level marker."""
        doom_map = load_map_from_wad(wad_path, map_name)
        return cls.from_map(
            doom_map,
            mesh_config=mesh_config,
            pathfinder_config=pathfinder_config,
            health_detour=health_detour,
        )

    def context(
        self,
        position: np.ndarray | Sequence[float],
        progress_index: int,
        *,
        capture_radius: float = 72.0,
        waypoint_radius: float = 38.0,
        off_track_distance: float = 128.0,
        use_radius: float = 64.0,
    ) -> RouteContext:
        """Produce the standard RouteContext for SNN observation vectors."""
        pos = np.asarray(position, dtype=np.float32)
        if pos.shape != (2,) or not np.all(np.isfinite(pos)):
            raise ValueError("position must be a finite vector with shape (2,).")

        n_points = len(self.points)
        progress = int(np.clip(progress_index, 0, n_points - 1))

        # Lookahead capture window
        stop = min(n_points, progress + 5)
        local_dists = np.linalg.norm(self.points[progress:stop] - pos, axis=1)
        nearest_offset = int(np.argmin(local_dists))
        if float(local_dists[nearest_offset]) < capture_radius:
            progress += nearest_offset

        # Advance along waypoints within capture radius
        while (
            progress + 1 < n_points
            and float(np.linalg.norm(self.points[progress] - pos)) < waypoint_radius
        ):
            progress += 1

        # Off-track / deviation detection
        prev_idx = max(0, progress - 1)
        edge_start = self.points[prev_idx]
        edge = self.points[progress] - edge_start
        edge_len_sq = float(np.dot(edge, edge))

        if edge_len_sq <= 1e-9:
            dist_to_edge = float(np.linalg.norm(pos - edge_start))
        else:
            proj = float(np.clip(np.dot(pos - edge_start, edge) / edge_len_sq, 0.0, 1.0))
            closest = edge_start + proj * edge
            dist_to_edge = float(np.linalg.norm(pos - closest))

        recovery = dist_to_edge > off_track_distance
        target_idx = progress

        # Recovery lookahead
        if recovery:
            r_start = max(0, progress - 2)
            r_stop = min(n_points, progress + 4)
            rec_dists = np.linalg.norm(self.points[r_start:r_stop] - pos, axis=1)
            candidate = r_start + int(np.argmin(rec_dists))
            target_idx = max(progress, candidate)

        target = self.points[target_idx].copy()
        delta = target - pos
        distance = float(np.linalg.norm(delta))
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)

        # Interactive trigger recommendation
        if len(self.use_points) > 0:
            use_dist = float(np.min(np.linalg.norm(self.use_points - pos, axis=1)))
            use_recommended = use_dist < use_radius
        else:
            use_recommended = False

        return RouteContext(
            progress_index=progress,
            progress_fraction=progress / max(n_points - 1, 1),
            target=target,
            delta=delta,
            distance=distance,
            bearing_degrees=bearing,
            off_track=recovery,
            use_recommended=use_recommended,
            recovery=recovery,
        )

    def request_health_detour(
        self,
        current_pos: tuple[float, float] | Sequence[float],
        min_heal_amount: int = 10,
    ) -> bool:
        """Dynamically reroute via nearest uncollected Medikit if available."""
        if self.navgrid is None:
            return False

        cx, cy = float(current_pos[0]), float(current_pos[1])
        nearest = find_nearest_health_item(
            self.navgrid,
            (cx, cy),
            visited_items=self._visited_items,
            min_heal_amount=min_heal_amount,
        )
        if nearest is None:
            return False

        self._visited_items.add(nearest.index)
        new_route_points = plan_level_route(
            self.navgrid,
            start_pos=(cx, cy),
            health_detour=True,
            visited_items=self._visited_items,
        )
        self.points = np.asarray(new_route_points, dtype=np.float32)
        return True


def angle_delta_degrees(target: float, current: float) -> float:
    """Return signed shortest target-current angle in [-180, 180)."""
    return float((target - current + 180.0) % 360.0 - 180.0)
