"""Unit tests for binary WAD parser (src/wad_parser.py)."""

from __future__ import annotations

import io
import struct
from pathlib import Path
import pytest

from wad_parser import (
    DoomMap,
    WadArchive,
    load_wad,
    load_map_from_wad,
    DOOR_SPECIALS,
    EXIT_SPECIALS,
    HEALTH_ITEM_TYPES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEDOOM2_WAD = PROJECT_ROOT / "freedoom2.wad"


def test_wad_archive_loading():
    """Verify loading freedoom2.wad header and directory table."""
    wad = load_wad(FREEDOOM2_WAD)
    assert wad.wad_type == "IWAD"
    assert "MAP01" in wad.map_names
    assert "MAP02" in wad.map_names
    assert len(wad.map_names) >= 30


def test_load_map01_geometry_and_entities():
    """Verify geometry, sectors, player spawn, and doors extracted from MAP01."""
    doom_map = load_map_from_wad(FREEDOOM2_WAD, "MAP01")

    assert doom_map.name == "MAP01"
    assert len(doom_map.vertices) > 100
    assert len(doom_map.linedefs) > 100
    assert len(doom_map.sidedefs) > 100
    assert len(doom_map.sectors) > 20
    assert len(doom_map.things) > 20

    # Player spawn check
    spawn = doom_map.player_spawn
    assert spawn.is_player_1_start
    assert spawn.x == -192.0
    assert spawn.y == -192.0

    # Bounding box
    min_x, min_y, max_x, max_y = doom_map.bounding_box
    assert min_x < max_x
    assert min_y < max_y
    assert min_x <= spawn.x <= max_x
    assert min_y <= spawn.y <= max_y

    # Exits and doors
    assert len(doom_map.exit_linedefs) >= 1
    assert len(doom_map.door_linedefs) >= 1
    assert len(doom_map.blocking_linedefs) > 50

    # Health items
    health_items = doom_map.health_items
    assert len(health_items) >= 5
    assert all(item.heal_amount > 0 for item in health_items)

    # Interaction points
    door_pts = doom_map.door_interaction_points()
    assert len(door_pts) == len(doom_map.door_linedefs)
    exit_pts = doom_map.exit_interaction_points()
    assert len(exit_pts) == len(doom_map.exit_linedefs)

    # Segment and midpoint math
    first_line = doom_map.linedefs[0]
    p1, p2 = doom_map.linedef_segment(first_line)
    mid = doom_map.linedef_midpoint(first_line)
    assert mid[0] == pytest.approx((p1[0] + p2[0]) / 2.0)
    assert mid[1] == pytest.approx((p1[1] + p2[1]) / 2.0)


def test_load_map02_entities():
    """Verify that MAP02 can be loaded zero-shot with distinct spawn coordinates."""
    wad = load_wad(FREEDOOM2_WAD)
    doom_map = wad.load_map("MAP02")

    assert doom_map.name == "MAP02"
    spawn = doom_map.player_spawn
    assert spawn.x == 640.0
    assert spawn.y == -576.0

    assert len(doom_map.exit_linedefs) >= 1
    assert len(doom_map.door_linedefs) >= 10
    assert len(doom_map.health_items) >= 4


def test_invalid_wad_and_map_errors(tmp_path: Path):
    """Verify informative errors on missing file, corrupt file, or invalid map name."""
    with pytest.raises(FileNotFoundError):
        load_wad(tmp_path / "does_not_exist.wad")

    # Corrupt header
    corrupt_wad = tmp_path / "corrupt.wad"
    corrupt_wad.write_bytes(b"INVALID_HEADER")
    with pytest.raises(ValueError, match="Invalid WAD identification"):
        load_wad(corrupt_wad)

    # Non-existent map
    wad = load_wad(FREEDOOM2_WAD)
    with pytest.raises(KeyError, match="not found in WAD"):
        wad.load_map("MAP99")


def test_synthetic_pwad_parsing(tmp_path: Path):
    """Test parsing a minimal synthetic PWAD generated purely in Python."""
    pwad_path = tmp_path / "test.wad"

    # Build lumps for a 1-sector square map
    # VERTEXES: 4 points: (0,0), (128, 0), (128, 128), (0, 128)
    vertexes_data = struct.pack("<8h", 0, 0, 128, 0, 128, 128, 0, 128)
    # SECTORS: 1 sector (floor_h=0, ceil_h=128, floor_tex="FLOOR", ceil_tex="CEIL", light=160, special=0, tag=0)
    sectors_data = struct.pack("<hh8s8shhh", 0, 128, b"FLOOR0_1", b"CEIL1_1", 160, 0, 0)
    # SIDEDEFS: 4 sidedefs pointing to sector 0
    sidedefs_data = b"".join(struct.pack("<hh8s8s8sh", 0, 0, b"-", b"-", b"STARTAN2", 0) for _ in range(4))
    # LINEDEFS: 4 blocking lines forming a square
    linedefs_data = struct.pack(
        "<28h",
        0, 1, 1, 0, 0, 0, -1,
        1, 2, 1, 0, 0, 1, -1,
        2, 3, 1, 0, 0, 2, -1,
        3, 0, 1, 0, 0, 3, -1,
    )
    # THINGS: 1 player start at (64, 64)
    things_data = struct.pack("<hhhhh", 64, 64, 0, 1, 7)

    lump_defs = [
        ("MAP01", b""),
        ("THINGS", things_data),
        ("LINEDEFS", linedefs_data),
        ("SIDEDEFS", sidedefs_data),
        ("VERTEXES", vertexes_data),
        ("SECTORS", sectors_data),
    ]

    # Assemble WAD file
    header_size = 12
    lump_bytes = b"".join(data for _, data in lump_defs)
    dir_offset = header_size + len(lump_bytes)

    dir_entries = []
    curr_offset = header_size
    for name, data in lump_defs:
        dir_entries.append(struct.pack("<II8s", curr_offset, len(data), name.encode("latin1")))
        curr_offset += len(data)

    wad_bytes = (
        struct.pack("<4sII", b"PWAD", len(lump_defs), dir_offset)
        + lump_bytes
        + b"".join(dir_entries)
    )
    pwad_path.write_bytes(wad_bytes)

    # Now load and assert
    wad = load_wad(pwad_path)
    assert wad.wad_type == "PWAD"
    m = wad.load_map("MAP01")
    assert m.player_spawn.x == 64.0
    assert m.player_spawn.y == 64.0
    assert len(m.vertices) == 4
    assert len(m.linedefs) == 4
    assert len(m.sectors) == 1
    assert m.bounding_box == (0.0, 0.0, 128.0, 128.0)
