"""2D Walkability Grid and NavMesh generator from Doom WAD geometry.

Converts vector linedefs, sectors, and doors from a parsed DoomMap into a
high-resolution spatial occupancy grid dilated by the player collision radius.
Supports sector reachability flooding, doorway portals, and hazard weighting.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from wad_parser import DoomMap, Linedef, Vertex


@dataclass(frozen=True)
class NavMeshConfig:
    """Configuration parameters for 2D walkability grid rasterization."""
    cell_size: float = 16.0               # Spatial resolution in Doom map units (16 aligns with Doom grid)
    player_radius: float = 16.0           # Standard Doom player bounding box radius (32x32 box)
    max_step_height: float = 24.0         # Max step height player can ascend without lift (vanilla Doom = 24)
    min_opening_height: float = 56.0      # Minimum vertical clearance player requires (Doom player height = 56)
    margin: float = 32.0                  # Spatial padding around level bounding box in map units
    narrow_sector_threshold: float = 24.0 # Width threshold for floor grates/light slots spanned by player radius
    damaging_cost_multiplier: float = 5.0 # Movement cost multiplier for damaging radioactive nukage sectors


class NavGrid:
    """High-resolution 2D occupancy grid and navigation graph for a Doom level."""

    def __init__(
        self,
        doom_map: DoomMap,
        config: NavMeshConfig | None = None,
    ):
        self.doom_map = doom_map
        self.config = config or NavMeshConfig()

        # Grid bounds and origins
        min_x, min_y, max_x, max_y = self.doom_map.bounding_box
        self.origin_x = min_x - self.config.margin
        self.origin_y = min_y - self.config.margin
        self.width_cells = int(math.ceil((max_x - min_x + 2 * self.config.margin) / self.config.cell_size))
        self.height_cells = int(math.ceil((max_y - min_y + 2 * self.config.margin) / self.config.cell_size))

        # Core spatial arrays (height, width)
        self.blocked = np.zeros((self.height_cells, self.width_cells), dtype=bool)
        self.reachable = np.zeros((self.height_cells, self.width_cells), dtype=bool)
        self.cost_multipliers = np.ones((self.height_cells, self.width_cells), dtype=np.float32)

        # Door & Exit lookup tables
        self.door_cells: dict[tuple[int, int], Linedef] = {}
        self.door_midpoints: list[tuple[float, float]] = self.doom_map.door_interaction_points()
        self.exit_midpoints: list[tuple[float, float]] = self.doom_map.exit_interaction_points()

        # Build grid geometry
        self._rasterize_impassable_walls()
        self._index_doorways()
        self._compute_reachability_from_spawn()

    # -------------------------------------------------------------------------
    # Spatial Coordinate Conversions
    # -------------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (X, Y) coordinates to discrete (col, row) cell indices."""
        col = int((x - self.origin_x) / self.config.cell_size)
        row = int((y - self.origin_y) / self.config.cell_size)
        return (col, row)

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Convert discrete (col, row) cell indices to center world (X, Y) coordinates."""
        cx = self.origin_x + (col + 0.5) * self.config.cell_size
        cy = self.origin_y + (row + 0.5) * self.config.cell_size
        return (cx, cy)

    def in_bounds(self, col: int, row: int) -> bool:
        """Check if grid cell is within spatial boundaries."""
        return 0 <= col < self.width_cells and 0 <= row < self.height_cells

    def is_walkable(self, col: int, row: int) -> bool:
        """Check if cell is navigable (inside bounds, unblocked, and reachable)."""
        return self.in_bounds(col, row) and not self.blocked[row, col] and self.reachable[row, col]

    def nearest_walkable(self, x: float, y: float, max_radius_cells: int = 6) -> tuple[int, int]:
        """Find the closest walkable cell to a given world coordinate."""
        start_c, start_r = self.world_to_grid(x, y)
        if self.is_walkable(start_c, start_r):
            return (start_c, start_r)

        best_cell = (start_c, start_r)
        best_dist = float("inf")

        for r_offset in range(-max_radius_cells, max_radius_cells + 1):
            for c_offset in range(-max_radius_cells, max_radius_cells + 1):
                c = start_c + c_offset
                r = start_r + r_offset
                if self.is_walkable(c, r):
                    cx, cy = self.grid_to_world(c, r)
                    d = math.hypot(cx - x, cy - y)
                    if d < best_dist:
                        best_dist = d
                        best_cell = (c, r)

        return best_cell

    def neighbors(
        self,
        col: int,
        row: int,
        allow_diagonal: bool = True,
    ) -> Iterator[tuple[int, int, float]]:
        """Yield valid (ncol, nrow, step_cost) neighbors for A* pathfinding.

        Diagonal moves verify that neither orthogonal adjacent wall is cut.
        """
        # Cardinal orthogonal directions (cost = 1.0)
        cardinals = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dc, dr in cardinals:
            nc, nr = col + dc, row + dr
            if self.is_walkable(nc, nr):
                cost = 1.0 * float(self.cost_multipliers[nr, nc])
                yield (nc, nr, cost)

        if not allow_diagonal:
            return

        # Diagonal directions (cost = sqrt(2) ~ 1.4142)
        SQRT2 = 1.41421356
        diagonals = [
            (-1, -1, -1, 0, 0, -1),
            (-1, 1, -1, 0, 0, 1),
            (1, -1, 1, 0, 0, -1),
            (1, 1, 1, 0, 0, 1),
        ]
        for dc, dr, oc1_c, oc1_r, oc2_c, oc2_r in diagonals:
            nc, nr = col + dc, row + dr
            if not self.is_walkable(nc, nr):
                continue
            # Prevent corner cutting through solid walls
            if self.blocked[row + oc1_r, col + oc1_c] and self.blocked[row + oc2_r, col + oc2_c]:
                continue
            cost = SQRT2 * float(self.cost_multipliers[nr, nc])
            yield (nc, nr, cost)

    # -------------------------------------------------------------------------
    # Door and Interaction Queries
    # -------------------------------------------------------------------------

    def is_near_door(self, x: float, y: float, radius: float = 64.0) -> bool:
        """True if (X, Y) is within interaction distance of any door linedef."""
        for dx, dy in self.door_midpoints:
            if math.hypot(dx - x, dy - y) <= radius:
                return True
        return False

    def is_near_exit(self, x: float, y: float, radius: float = 64.0) -> bool:
        """True if (X, Y) is within interaction distance of any exit linedef."""
        for ex, ey in self.exit_midpoints:
            if math.hypot(ex - x, ey - y) <= radius:
                return True
        return False

    # -------------------------------------------------------------------------
    # Internal Construction & Rasterization
    # -------------------------------------------------------------------------

    def _find_narrow_slot_sectors(self) -> set[int]:
        """Detect narrow light slots, grates, and door sills that players walk over."""
        sector_lines: dict[int, list[Linedef]] = {}
        for l in self.doom_map.linedefs:
            for s_idx in (
                self.doom_map.sidedefs[l.front_sidedef].sector if l.front_sidedef != -1 else None,
                self.doom_map.sidedefs[l.back_sidedef].sector if l.back_sidedef != -1 else None,
            ):
                if s_idx is not None:
                    sector_lines.setdefault(s_idx, []).append(l)

        narrow_sectors: set[int] = set()
        for s_idx, lines in sector_lines.items():
            xs, ys = [], []
            for l in lines:
                (x1, y1), (x2, y2) = self.doom_map.linedef_segment(l)
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if min(w, h) <= self.config.narrow_sector_threshold:
                narrow_sectors.add(s_idx)

        return narrow_sectors

    def _rasterize_impassable_walls(self) -> None:
        """Dilate solid 1-sided and step/cliff 2-sided linedefs onto occupancy grid."""
        narrow_sectors = self._find_narrow_slot_sectors()

        blocking_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        for l in self.doom_map.linedefs:
            # Interactive doors and level exit lines are traversable portals
            if l.is_door or l.is_exit:
                continue

            # 1-sided lines and explicitly flagged blocking lines
            if l.is_one_sided or bool(l.flags & 0x0001):
                blocking_segments.append(self.doom_map.linedef_segment(l))
            elif l.is_two_sided and not l.is_lift:
                s1 = self.doom_map.sectors[self.doom_map.sidedefs[l.front_sidedef].sector]
                s2 = self.doom_map.sectors[self.doom_map.sidedefs[l.back_sidedef].sector]

                opening = min(s1.ceiling_height, s2.ceiling_height) - max(s1.floor_height, s2.floor_height)
                if opening < self.config.min_opening_height:
                    blocking_segments.append(self.doom_map.linedef_segment(l))
                    continue

                step = abs(s1.floor_height - s2.floor_height)
                if step > self.config.max_step_height:
                    # Narrow depressions/grates with ample vertical clearance are walkable
                    if not (s1.index in narrow_sectors or s2.index in narrow_sectors):
                        blocking_segments.append(self.doom_map.linedef_segment(l))

        # Dilation radius
        radius = self.config.player_radius
        c_size = self.config.cell_size

        for (x1, y1), (x2, y2) in blocking_segments:
            # Bounding box of expanded segment
            bx1 = min(x1, x2) - radius
            bx2 = max(x1, x2) + radius
            by1 = min(y1, y2) - radius
            by2 = max(y1, y2) + radius

            c_min = max(0, int((bx1 - self.origin_x) / c_size))
            c_max = min(self.width_cells - 1, int((bx2 - self.origin_x) / c_size))
            r_min = max(0, int((by1 - self.origin_y) / c_size))
            r_max = min(self.height_cells - 1, int((by2 - self.origin_y) / c_size))

            dx = x2 - x1
            dy = y2 - y1
            l2 = dx * dx + dy * dy

            for r in range(r_min, r_max + 1):
                cy = self.origin_y + (r + 0.5) * c_size
                for c in range(c_min, c_max + 1):
                    if self.blocked[r, c]:
                        continue
                    cx = self.origin_x + (c + 0.5) * c_size
                    if l2 <= 1e-6:
                        dist = math.hypot(cx - x1, cy - y1)
                    else:
                        t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / l2))
                        px = x1 + t * dx
                        py = y1 + t * dy
                        dist = math.hypot(cx - px, cy - py)

                    if dist <= radius:
                        self.blocked[r, c] = True

    def _index_doorways(self) -> None:
        """Index door lines and mark cells within door passage zones."""
        for door in self.doom_map.door_linedefs:
            mid = self.doom_map.linedef_midpoint(door)
            c, r = self.world_to_grid(mid[0], mid[1])
            if self.in_bounds(c, r):
                self.door_cells[(c, r)] = door

    def _compute_reachability_from_spawn(self) -> None:
        """Flood-fill (BFS) from player spawn to identify all reachable playable cells."""
        spawn = self.doom_map.player_spawn
        spawn_c, spawn_r = self.world_to_grid(spawn.x, spawn.y)

        # In case spawn is exactly on cell edge and slightly marked blocked, find closest free cell
        if self.in_bounds(spawn_c, spawn_r) and self.blocked[spawn_r, spawn_c]:
            free_found = False
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    sc, sr = spawn_c + dc, spawn_r + dr
                    if self.in_bounds(sc, sr) and not self.blocked[sr, sc]:
                        spawn_c, spawn_r = sc, sr
                        free_found = True
                        break
                if free_found:
                    break

        if not self.in_bounds(spawn_c, spawn_r):
            raise ValueError(f"Player spawn {spawn.position} is outside grid boundaries.")

        queue: deque[tuple[int, int]] = deque([(spawn_c, spawn_r)])
        self.reachable[spawn_r, spawn_c] = True

        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        while queue:
            c, r = queue.popleft()
            for dc, dr in directions:
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.width_cells and 0 <= nr < self.height_cells:
                    if not self.blocked[nr, nc] and not self.reachable[nr, nc]:
                        self.reachable[nr, nc] = True
                        queue.append((nc, nr))


def build_navmesh(
    doom_map: DoomMap,
    config: NavMeshConfig | None = None,
) -> NavGrid:
    """Factory helper to construct a NavGrid instance for a DoomMap."""
    return NavGrid(doom_map, config)
