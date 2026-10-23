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
    max_edge_length: float = 160.0       # Maximum distance between consecutive waypoints (keeps targets in FOV)
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
    for s in range(steps + 1):
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
    padding_cells: int = 1,
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


def plan_level_route(
    grid: NavGrid,
    start_pos: tuple[float, float] | None = None,
    health_detour: bool = False,
    visited_items: set[int] | None = None,
    config: PathfinderConfig | None = None,
) -> list[tuple[float, float]]:
    """Plan complete sparse route from start position to level exit.

    If health_detour is True, detours to the nearest reachable Medikit before
    continuing towards the level exit switch.
    """
    cfg = config or PathfinderConfig()
    start = start_pos or grid.doom_map.player_spawn.position

    if not grid.exit_midpoints:
        raise ValueError(f"No exit lines found in level {grid.doom_map.name}.")

    exit_pos = grid.exit_midpoints[0]

    # Health detour routing
    if health_detour:
        nearest_med = find_nearest_health_item(grid, start, visited_items)
        if nearest_med is not None:
            med_pos = (nearest_med.x, nearest_med.y)
            path_to_med = find_path(grid, start, med_pos)
            path_to_exit = find_path(grid, med_pos, exit_pos)
            combined = path_to_med[:-1] + path_to_exit
            return simplify_path(
                combined,
                grid,
                max_edge_length=cfg.max_edge_length,
                line_of_sight_step=cfg.line_of_sight_step,
            )

    # Direct start-to-exit path
    raw_path = find_path(grid, start, exit_pos)
    return simplify_path(
        raw_path,
        grid,
        max_edge_length=cfg.max_edge_length,
        line_of_sight_step=cfg.line_of_sight_step,
    )
