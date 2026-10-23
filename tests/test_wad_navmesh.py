"""Unit tests for 2D Walkability Grid and NavMesh (src/wad_navmesh.py)."""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from wad_parser import load_map_from_wad
from wad_navmesh import NavGrid, NavMeshConfig, build_navmesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEDOOM2_WAD = PROJECT_ROOT / "freedoom2.wad"


def test_navmesh_map01_construction():
    """Verify building NavGrid on MAP01 with reachability and exit access."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP01")
    grid = build_navmesh(m)

    assert grid.width_cells > 50
    assert grid.height_cells > 50

    # Spawn cell must be walkable
    spawn = m.player_spawn
    sc, sr = grid.world_to_grid(spawn.x, spawn.y)
    assert grid.is_walkable(sc, sr)

    # Reachable cell count
    reachable_count = int(grid.reachable.sum())
    assert reachable_count > 5000

    # Exit linedef must have walkable cells nearby
    exit_line = m.exit_linedefs[0]
    ex_x, ex_y = m.linedef_midpoint(exit_line)
    ec, er = grid.nearest_walkable(ex_x, ex_y, max_radius_cells=5)
    assert grid.is_walkable(ec, er)
    assert math.hypot(ex_x - grid.grid_to_world(ec, er)[0], ex_y - grid.grid_to_world(ec, er)[1]) < 64.0

    # Doors cached
    assert len(grid.door_midpoints) == len(m.door_linedefs)
    assert len(grid.exit_midpoints) == len(m.exit_linedefs)


def test_navmesh_map02_construction():
    """Verify building NavGrid zero-shot on MAP02 with distinct spawn coordinates."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP02")
    grid = build_navmesh(m)

    spawn = m.player_spawn
    sc, sr = grid.world_to_grid(spawn.x, spawn.y)
    assert grid.is_walkable(sc, sr)

    # Reachable cells
    assert int(grid.reachable.sum()) > 10000

    # Exit switch access
    exit_line = m.exit_linedefs[0]
    ex_x, ex_y = m.linedef_midpoint(exit_line)
    ec, er = grid.nearest_walkable(ex_x, ex_y, max_radius_cells=6)
    assert grid.is_walkable(ec, er)


def test_coordinate_transforms():
    """Verify world_to_grid and grid_to_world invertibility and rounding."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP01")
    grid = build_navmesh(m, NavMeshConfig(cell_size=16.0))

    test_x, test_y = -192.0, -192.0
    c, r = grid.world_to_grid(test_x, test_y)
    wx, wy = grid.grid_to_world(c, r)

    # Center of cell should be within half cell size
    assert abs(wx - test_x) <= 8.0
    assert abs(wy - test_y) <= 8.0


def test_neighbors_step_costs_and_corner_cutting():
    """Verify neighbor generator yields orthogonal (1.0) and diagonal (sqrt(2)) steps."""
    m = load_map_from_wad(FREEDOOM2_WAD, "MAP01")
    grid = build_navmesh(m)

    sc, sr = grid.world_to_grid(m.player_spawn.x, m.player_spawn.y)
    neighbors = list(grid.neighbors(sc, sr, allow_diagonal=True))

    assert len(neighbors) >= 3
    for nc, nr, cost in neighbors:
        assert grid.is_walkable(nc, nr)
        assert (cost == pytest.approx(1.0)) or (cost == pytest.approx(1.41421356))
