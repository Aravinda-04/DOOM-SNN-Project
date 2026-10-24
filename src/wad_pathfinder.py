"""Dynamic A* pathfinder and string-pulling waypointer for Doom levels.

Finds optimal collision-free trajectories from any start point to door switches,
health pickups, and level exit triggers using 2D NavGrid geometry.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from wad_navmesh import NavGrid
from wad_parser import Thing


@dataclass(frozen=True)
class PathfinderConfig:
    """Configuration for path planning and waypoint generation."""
    max_edge_length: float = 140.0       # Maximum distance between consecutive waypoints (keeps targets in FOV)
    line_of_sight_step: float = 8.0      # Spatial sampling step for collision-free raycasting (map units)
    health_search_max_cells: int = 4     # Cell search radius to snap entity positions to walkable grid
    health_detour_threshold: int = 50    # Default health value below which detour planning is triggered


def line_of_sight(
    p1: tuple[float, float] | Sequence[float],
    p2: tuple[float, float] | Sequence[float],
    grid: NavGrid,
    step_size: float = 8.0,
    padding_cells: int = 0,
) -> bool:
    """Raycast check between two points. Returns True if segment is completely unblocked."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return True

    steps = int(math.ceil(dist / max(step_size, 1.0)))
    for s in range(1, steps + 1):
        t = s / steps
        x = p1[0] + t * dx
        y = p1[1] + t * dy
        c, r = grid.world_to_grid(x, y)
        if not grid.in_bounds(c, r) or grid.blocked[r, c]:
            return False
        if padding_cells > 0:
            for dr in range(-padding_cells, padding_cells + 1):
                for dc in range(-padding_cells, padding_cells + 1):
                    nc, nr = c + dc, r + dr
                    if not grid.in_bounds(nc, nr) or grid.blocked[nr, nc]:
                        return False

    return True


def find_path(
    grid: NavGrid,
    start_pos: tuple[float, float] | Sequence[float],
    goal_pos: tuple[float, float] | Sequence[float],
) -> list[tuple[float, float]]:
    """Compute optimal raw A* path between two 2D points on the NavGrid."""
    start_x, start_y = float(start_pos[0]), float(start_pos[1])
    goal_x, goal_y = float(goal_pos[0]), float(goal_pos[1])

    start_c, start_r = grid.nearest_walkable(start_x, start_y, max_radius_cells=6)
    goal_c, goal_r = grid.nearest_walkable(goal_x, goal_y, max_radius_cells=8)

    if not grid.is_walkable(start_c, start_r):
        raise ValueError(f"Start position {start_pos} is outside walkable geometry.")
    if not grid.is_walkable(goal_c, goal_r):
        raise ValueError(f"Goal position {goal_pos} is outside walkable geometry.")

    if (start_c, start_r) == (goal_c, goal_r):
        return [(start_x, start_y), (goal_x, goal_y)]

    def heuristic(c: int, r: int) -> float:
        return math.hypot(c - goal_c, r - goal_r)

    # Priority queue: (f_score, step_id, (col, row))
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_set, (heuristic(start_c, start_r), 0, (start_c, start_r)))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {(start_c, start_r): 0.0}

    step_counter = 0
    found = False

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == (goal_c, goal_r):
            found = True
            break

        cur_c, cur_r = current
        cur_g = g_score[current]

        for nc, nr, step_cost in grid.neighbors(cur_c, cur_r, allow_diagonal=True):
            tentative_g = cur_g + step_cost
            if (nc, nr) not in g_score or tentative_g < g_score[(nc, nr)]:
                g_score[(nc, nr)] = tentative_g
                step_counter += 1
                f = tentative_g + heuristic(nc, nr)
                heapq.heappush(open_set, (f, step_counter, (nc, nr)))
                came_from[(nc, nr)] = current

    if not found:
        raise ValueError(f"No valid path found from {start_pos} to {goal_pos}.")

    # Reconstruct path
    curr = (goal_c, goal_r)
    cells = [curr]
    while curr in came_from:
        curr = came_from[curr]
        cells.append(curr)
    cells.reverse()

    # Convert cells to world coordinates
    points: list[tuple[float, float]] = [(start_x, start_y)]
    for c, r in cells[1:-1]:
        points.append(grid.grid_to_world(c, r))
    points.append((goal_x, goal_y))

    return points


def simplify_path(
    raw_points: Sequence[tuple[float, float]],
    grid: NavGrid,
    max_edge_length: float = 140.0,
    line_of_sight_step: float = 8.0,
    padding_cells: int = 0,
) -> list[tuple[float, float]]:
    """Greedy string-pulling simplification with clearance-padded collision checking."""
    if len(raw_points) <= 2:
        return list(raw_points)

    pts = [tuple(p) for p in raw_points]
    simplified: list[tuple[float, float]] = [pts[0]]
    curr_idx = 0
    n = len(pts)

    while curr_idx < n - 1:
        best_next = curr_idx + 1
        for next_idx in range(curr_idx + 1, n):
            p1 = pts[curr_idx]
            p2 = pts[next_idx]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist > max_edge_length:
                break
            if line_of_sight(p1, p2, grid, step_size=line_of_sight_step, padding_cells=padding_cells):
                best_next = next_idx
            else:
                break

        simplified.append(pts[best_next])
        curr_idx = best_next

    return simplified


def find_nearest_health_item(
    grid: NavGrid,
    current_pos: tuple[float, float],
    visited_items: set[int] | None = None,
    min_heal_amount: int = 10,
) -> Thing | None:
    """Find the closest reachable high-value health item in the level."""
    visited = visited_items or set()
    best_item: Thing | None = None
    best_dist = float("inf")

    cx, cy = current_pos[0], current_pos[1]

    for item in grid.doom_map.health_items:
        if item.index in visited or item.heal_amount < min_heal_amount:
            continue

        c, r = grid.world_to_grid(item.x, item.y)
        if not grid.is_walkable(c, r):
            c, r = grid.nearest_walkable(item.x, item.y, max_radius_cells=4)
            if not grid.is_walkable(c, r):
                continue

        dist = math.hypot(item.x - cx, item.y - cy)
        # Prioritize higher heal items (e.g., Medikits/Soulspheres) by effective distance
        weighted_dist = dist / max(item.heal_amount / 25.0, 0.5)

        if weighted_dist < best_dist:
            best_dist = weighted_dist
            best_item = item

    return best_item


TACTICAL_WEAPON_TYPES: set[int] = {
    2001,  # Shotgun
    82,    # Super Shotgun
    2002,  # Chaingun
    2004,  # Plasma rifle
}
TACTICAL_ARMOR_TYPES: set[int] = {
    2018,  # Green Armor
    2019,  # Blue MegaArmor
}
TACTICAL_HEALTH_TYPES: set[int] = {
    2011,  # Medikit
    2013,  # Soulsphere / Supercharge
}
TACTICAL_AMMO_TYPES: set[int] = {
    2008,  # Box of Shells
    2048,  # Box of Bullets
}
TACTICAL_ITEM_TYPES: set[int] = (
    TACTICAL_WEAPON_TYPES
    | TACTICAL_ARMOR_TYPES
    | TACTICAL_HEALTH_TYPES
    | TACTICAL_AMMO_TYPES
)


def plan_tactical_level_route(
    grid: NavGrid,
    start_pos: tuple[float, float] | None = None,
    max_detour_distance: float = 260.0,
    min_waypoint_separation: float = 24.0,
    config: PathfinderConfig | None = None,
) -> list[tuple[float, float]]:
    """Plan trajectory that collects tactical equipment (weapons, armor, ammo, health) near the path."""
    cfg = config or PathfinderConfig()
    start = start_pos or grid.doom_map.player_spawn.position

    if not grid.exit_midpoints:
        raise ValueError(f"No exit lines found in level {grid.doom_map.name}.")

    exit_pos = grid.exit_midpoints[0]

    # Baseline topological path
    base_path = find_path(grid, start, exit_pos)
    base_pts = np.asarray(base_path, dtype=np.float32)

    # Search for reachable high-value tactical items close to baseline trajectory
    candidates: list[tuple[int, float, int, tuple[float, float]]] = []
    spawn_pt = grid.doom_map.player_spawn
    for item in grid.doom_map.things:
        if item.type not in TACTICAL_ITEM_TYPES:
            continue
        # Filter out items that do not spawn in Single Player (flags bit 4 = 16) or skill levels (bits 0-2 = 7)
        if (item.flags & 16) != 0 or (item.flags & 7) == 0:
            continue
        # Avoid backtracking into spawn room for health when already at 100%
        if item.type in TACTICAL_HEALTH_TYPES and math.hypot(item.x - spawn_pt.x, item.y - spawn_pt.y) < 300.0:
            continue
        c, r = grid.world_to_grid(item.x, item.y)
        if grid.is_walkable(c, r):
            pos = (float(item.x), float(item.y))
        else:
            nc, nr = grid.nearest_walkable(item.x, item.y, max_radius_cells=3)
            if not grid.is_walkable(nc, nr):
                continue
            pos = grid.grid_to_world(nc, nr)

        dists = np.hypot(base_pts[:, 0] - item.x, base_pts[:, 1] - item.y)
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])
        max_d = max_detour_distance
        if min_dist <= max_d:
            if line_of_sight(base_pts[min_idx], pos, grid):
                candidates.append((min_idx, min_dist, item.type, pos))
            elif item.type in TACTICAL_WEAPON_TYPES:
                try:
                    p = find_path(grid, base_pts[min_idx], pos)
                    if len(p) <= int(max_d / grid.config.cell_size) + 4:
                        candidates.append((min_idx, min_dist, item.type, pos))
                except Exception:
                    pass

    if not candidates:
        return simplify_path(
            base_path,
            grid,
            max_edge_length=cfg.max_edge_length,
            line_of_sight_step=cfg.line_of_sight_step,
        )

    # Order candidates along route progression
    candidates.sort(key=lambda x: x[0])

    seen_weapons: set[int] = set()
    waypoints_chain: list[tuple[float, float]] = [start]
    last_pos = start
    for _, _, item_type, item_pos in candidates:
        if item_type in TACTICAL_WEAPON_TYPES:
            if item_type in seen_weapons:
                continue
            seen_weapons.add(item_type)
        if math.hypot(item_pos[0] - last_pos[0], item_pos[1] - last_pos[1]) < min_waypoint_separation:
            continue
        waypoints_chain.append(item_pos)
        last_pos = item_pos
    waypoints_chain.append(exit_pos)

    # Connect chained waypoints with collision-free paths
    # Connect and simplify chained waypoints while strictly preserving item anchor points
    full_path: list[tuple[float, float]] = []
    for i in range(len(waypoints_chain) - 1):
        raw_seg = find_path(grid, waypoints_chain[i], waypoints_chain[i + 1])
        simp_seg = simplify_path(
            raw_seg,
            grid,
            max_edge_length=cfg.max_edge_length,
            line_of_sight_step=cfg.line_of_sight_step,
        )
        if i > 0:
            full_path.extend(simp_seg[1:])
        else:
            full_path.extend(simp_seg)

    return full_path


def plan_level_route(
    grid: NavGrid,
    start_pos: tuple[float, float] | None = None,
    health_detour: bool = False,
    tactical: bool = False,
    visited_items: set[int] | None = None,
    config: PathfinderConfig | None = None,
) -> list[tuple[float, float]]:
    """Plan complete sparse route from start position to level exit.

    If tactical is True, gathers along-the-way weapons, armor, and medikits.
    If health_detour is True, detours to the nearest reachable Medikit before
    continuing towards the level exit switch.
    """
    cfg = config or PathfinderConfig()
    start = start_pos or grid.doom_map.player_spawn.position

    if not grid.exit_midpoints:
        raise ValueError(f"No exit lines found in level {grid.doom_map.name}.")

    exit_pos = grid.exit_midpoints[0]

    # Dynamic health detour routing
    if health_detour:
        nearest_med = find_nearest_health_item(grid, start, visited_items)
        if nearest_med is not None:
            med_pos = (nearest_med.x, nearest_med.y)
            path_to_med = find_path(grid, start, med_pos)
            path_to_exit = find_path(grid, med_pos, exit_pos)
            simp_to_med = simplify_path(
                path_to_med,
                grid,
                max_edge_length=cfg.max_edge_length,
                line_of_sight_step=cfg.line_of_sight_step,
            )
            simp_to_exit = simplify_path(
                path_to_exit,
                grid,
                max_edge_length=cfg.max_edge_length,
                line_of_sight_step=cfg.line_of_sight_step,
            )
            return simp_to_med[:-1] + simp_to_exit

    # Tactical along-the-way progression
    if tactical:
        return plan_tactical_level_route(grid, start_pos=start, config=cfg)

    # Direct start-to-exit path
    raw_path = find_path(grid, start, exit_pos)
    return simplify_path(
        raw_path,
        grid,
        max_edge_length=cfg.max_edge_length,
        line_of_sight_step=cfg.line_of_sight_step,
    )
