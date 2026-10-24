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
from wad_pathfinder import (
    PathfinderConfig,
    find_nearest_health_item,
    find_path,
    line_of_sight,
    plan_level_route,
)


def filter_route_doors(
    door_midpoints: Sequence[tuple[float, float]],
    route_points: np.ndarray | Sequence[tuple[float, float]],
    max_dist: float = 48.0,
) -> list[tuple[float, float]]:
    """Return only door midpoints that lie within max_dist of any segment of the route."""
    pts = np.asarray(route_points, dtype=np.float32)
    filtered: list[tuple[float, float]] = []
    for dx, dy in door_midpoints:
        d_pt = np.array([dx, dy], dtype=np.float32)
        min_d = float("inf")
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            seg = p2 - p1
            l2 = float(np.dot(seg, seg))
            if l2 <= 1e-6:
                d = float(np.linalg.norm(d_pt - p1))
            else:
                t = max(0.0, min(1.0, float(np.dot(d_pt - p1, seg) / l2)))
                proj = p1 + t * seg
                d = float(np.linalg.norm(d_pt - proj))
            if d < min_d:
                min_d = d
        if min_d <= max_dist:
            filtered.append((dx, dy))
    return filtered


class GeneralizedRoute:
    """Universal level route graph compatible with the SNN RouteContext contract."""

    def __init__(
        self,
        points: np.ndarray | Sequence[tuple[float, float]],
        *,
        use_points: np.ndarray | Sequence[tuple[float, float]] | None = None,
        route_doors: Sequence[tuple[float, float]] | None = None,
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
        self.route_doors = list(route_doors) if route_doors is not None else []

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

        # Identify mandatory inflection/anchor waypoints (sharp turns, hairpins, alcoves)
        self.mandatory_waypoints: set[int] = set()
        for i in range(1, len(self.points) - 1):
            v1 = self.points[i] - self.points[i - 1]
            v2 = self.points[i + 1] - self.points[i]
            l1 = float(np.linalg.norm(v1))
            l2 = float(np.linalg.norm(v2))
            if l1 > 1e-6 and l2 > 1e-6:
                cos_theta = float(np.dot(v1, v2) / (l1 * l2))
                if cos_theta <= 0.3:
                    self.mandatory_waypoints.add(i)

    @classmethod
    def from_navgrid(
        cls,
        grid: NavGrid,
        *,
        start_pos: tuple[float, float] | None = None,
        health_detour: bool = False,
        tactical: bool = True,
        pathfinder_config: PathfinderConfig | None = None,
    ) -> "GeneralizedRoute":
        """Construct GeneralizedRoute directly from a pre-built NavGrid."""
        points = plan_level_route(
            grid,
            start_pos=start_pos,
            health_detour=health_detour,
            tactical=tactical,
            config=pathfinder_config,
        )

        # Consolidate interaction trigger centers (doors along route and exit switches)
        route_doors = filter_route_doors(grid.door_midpoints, points, max_dist=48.0)
        use_pts: list[tuple[float, float]] = []
        use_pts.extend(route_doors)
        use_pts.extend(grid.exit_midpoints)

        return cls(
            points=points,
            use_points=use_pts,
            route_doors=route_doors,
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
        tactical: bool = True,
    ) -> "GeneralizedRoute":
        """Construct GeneralizedRoute directly from a parsed DoomMap level."""
        grid = build_navmesh(doom_map, mesh_config)
        return cls.from_navgrid(
            grid,
            health_detour=health_detour,
            tactical=tactical,
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
        tactical: bool = True,
    ) -> "GeneralizedRoute":
        """Construct GeneralizedRoute from WAD archive and level marker."""
        doom_map = load_map_from_wad(wad_path, map_name)
        return cls.from_map(
            doom_map,
            mesh_config=mesh_config,
            pathfinder_config=pathfinder_config,
            health_detour=health_detour,
            tactical=tactical,
        )

    def context(
        self,
        position: np.ndarray | Sequence[float],
        progress_index: int,
        *,
        capture_radius: float = 96.0,
        waypoint_radius: float = 56.0,
        mandatory_radius: float = 56.0,
        off_track_distance: float = 128.0,
        use_radius: float = 64.0,
        max_progress_index: int | None = None,
    ) -> RouteContext:
        """Produce the standard RouteContext for SNN observation vectors."""
        pos = np.asarray(position, dtype=np.float32)
        if pos.shape != (2,) or not np.all(np.isfinite(pos)):
            raise ValueError("position must be a finite vector with shape (2,).")

        n_points = len(self.points)
        limit = n_points if max_progress_index is None else min(n_points, max(1, max_progress_index + 1))
        progress = int(np.clip(progress_index, 0, limit - 1))

        # Lookahead capture window
        stop = min(limit, progress + 5)
        local_dists = np.linalg.norm(self.points[progress:stop] - pos, axis=1)
        nearest_offset = int(np.argmin(local_dists))

        # Clamp lookahead so it cannot jump past a corner through a solid wall
        for offset in range(1, nearest_offset + 1):
            idx = progress + offset - 1
            if idx in self.mandatory_waypoints and float(np.linalg.norm(self.points[idx] - pos)) >= mandatory_radius:
                target_pt = self.points[progress + offset]
                wall_found = False
                if self.navgrid is not None and hasattr(self.navgrid, "solid_wall_mask"):
                    dx = target_pt[0] - pos[0]
                    dy = target_pt[1] - pos[1]
                    dist = math.hypot(dx, dy)
                    if dist > 1.0:
                        n_s = int(math.ceil(dist / 16.0))
                        for s in range(1, n_s + 1):
                            t = s / n_s
                            c, r = self.navgrid.world_to_grid(pos[0] + t * dx, pos[1] + t * dy)
                            if self.navgrid.in_bounds(c, r) and self.navgrid.solid_wall_mask[r, c]:
                                wall_found = True
                                break
                if wall_found:
                    nearest_offset = offset - 1
                    break

        if float(local_dists[nearest_offset]) < capture_radius and nearest_offset > 0:
            target_pt = self.points[progress + nearest_offset]
            has_wall = False
            if self.navgrid is not None and hasattr(self.navgrid, "solid_wall_mask"):
                dx = target_pt[0] - pos[0]
                dy = target_pt[1] - pos[1]
                dist = math.hypot(dx, dy)
                if dist > 1.0:
                    n_s = int(math.ceil(dist / 16.0))
                    for s in range(1, n_s + 1):
                        t = s / n_s
                        c, r = self.navgrid.world_to_grid(pos[0] + t * dx, pos[1] + t * dy)
                        if self.navgrid.in_bounds(c, r) and self.navgrid.solid_wall_mask[r, c]:
                            has_wall = True
                            break
            if not has_wall:
                progress += nearest_offset

        # Advance along waypoints within capture radius
        while progress + 1 < limit:
            req_radius = mandatory_radius if progress in self.mandatory_waypoints else waypoint_radius
            if float(np.linalg.norm(self.points[progress] - pos)) < req_radius:
                progress += 1
            else:
                break

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
            r_stop = min(limit, progress + 4)
            rec_dists = np.linalg.norm(self.points[r_start:r_stop] - pos, axis=1)
            candidate = r_start + int(np.argmin(rec_dists))
            for m_idx in sorted(self.mandatory_waypoints):
                if progress <= m_idx < candidate:
                    candidate = m_idx
                    break
            target_idx = max(progress, candidate)

        target = self.points[min(target_idx, limit - 1)].copy()


        delta = target - pos
        distance = float(np.linalg.norm(delta))
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)

        # Interactive trigger recommendation
        if len(self.use_points) > 0:
            dists = np.linalg.norm(self.use_points - pos, axis=1)
            min_use_idx = int(np.argmin(dists))
            use_dist = float(dists[min_use_idx])
            use_recommended = bool(use_dist < use_radius)
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
