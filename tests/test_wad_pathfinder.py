"""Unit tests for dynamic A* pathfinder (src/wad_pathfinder.py)."""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from wad_parser import load_map_from_wad
from wad_navmesh import build_navmesh
from wad_pathfinder import (
    PathfinderConfig,
    find_nearest_health_item,
    find_path,
    line_of_sight,
    plan_level_route,
    simplify_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEDOOM2_WAD = PROJECT_ROOT / "freedoom2.wad"


def test_line_of_sight():
    """Verify raycast detects clear corridors vs obstructed walls."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP01")
    grid = build_navmesh(m)

    # Spawn to 50 units east in the open corridor
    p1 = (-192.0, -192.0)
    p2 = (-142.0, -192.0)
    assert line_of_sight(p1, p2, grid)

    # Wall obstacle check: p1 (-192, -192) to points south across the north room wall
    p_blocked = (-192.0, -350.0)
    assert not line_of_sight(p1, p_blocked, grid)


def test_plan_map01_direct_route():
    """Verify complete A* path generation and LOS string-pulling on MAP01."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP01")
    grid = build_navmesh(m)

    route = plan_level_route(grid)

    assert len(route) >= 15
    assert len(route) <= 250

    # Start and End points
    spawn = m.player_spawn
    assert math.hypot(route[0][0] - spawn.x, route[0][1] - spawn.y) < 1.0

    exit_mid = grid.exit_midpoints[0]
    assert math.hypot(route[-1][0] - exit_mid[0], route[-1][1] - exit_mid[1]) < 64.0

    # Every consecutive segment must have clear LOS and bounded edge length
    for i in range(len(route) - 1):
        p1 = route[i]
        p2 = route[i + 1]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        assert dist <= 165.0, f"Segment {i} -> {i+1} length {dist:.1f} exceeds max edge bound"
        assert line_of_sight(p1, p2, grid), f"Segment {i} -> {i+1} clips level geometry"


def test_plan_map02_direct_route():
    """Verify zero-shot level routing on MAP02."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP02")
    grid = build_navmesh(m)

    route = plan_level_route(grid)

    spawn = m.player_spawn
    assert math.hypot(route[0][0] - spawn.x, route[0][1] - spawn.y) < 1.0

    exit_mid = grid.exit_midpoints[0]
    assert math.hypot(route[-1][0] - exit_mid[0], route[-1][1] - exit_mid[1]) < 64.0

    for i in range(len(route) - 1):
        assert line_of_sight(route[i], route[i + 1], grid)


def test_plan_route_with_health_detour():
    """Verify routing detours to pick up a nearby Medikit when requested."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP01")
    grid = build_navmesh(m)

    direct_route = plan_level_route(grid, health_detour=False)
    detour_route = plan_level_route(grid, health_detour=True)

    # Detour route must be non-empty and start at spawn / end at exit
    assert len(detour_route) > 0
    assert math.hypot(detour_route[0][0] - m.player_spawn.x, detour_route[0][1] - m.player_spawn.y) < 1.0
    assert math.hypot(detour_route[-1][0] - grid.exit_midpoints[0][0], detour_route[-1][1] - grid.exit_midpoints[0][1]) < 64.0

    # Nearest medikit should be within pickup range of one of the detour waypoints
    med = find_nearest_health_item(grid, m.player_spawn.position)
    assert med is not None
    min_dist_to_med = min(math.hypot(wp[0] - med.x, wp[1] - med.y) for wp in detour_route)
    assert min_dist_to_med < 64.0
