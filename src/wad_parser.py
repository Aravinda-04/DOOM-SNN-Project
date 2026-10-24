"""High-performance binary WAD parser for extracting Doom level geometry and entities.

Provides pure-Python binary deserialization for VERTEXES, LINEDEFS, SIDEDEFS,
SECTORS, and THINGS lumps from official IWADs (FreeDoom, Doom II) and custom PWADs.
Zero external C++ dependencies.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


# Standard Doom Special Action IDs
DOOR_SPECIALS = {
    # Manual doors (DR/D1)
    1: "DR_DOOR_OPEN_WAIT_CLOSE",
    26: "DR_DOOR_BLUE_KEY",
    27: "DR_DOOR_YELLOW_KEY",
    28: "DR_DOOR_RED_KEY",
    31: "D1_DOOR_OPEN_STAY_OPEN",
    32: "D1_DOOR_BLUE_KEY_STAY",
    33: "D1_DOOR_RED_KEY_STAY",
    34: "D1_DOOR_YELLOW_KEY_STAY",
    117: "SR_FAST_DOOR_OPEN_WAIT_CLOSE",
    118: "S1_FAST_DOOR_OPEN_WAIT_CLOSE",
    # Switch doors (SR/S1)
    29: "S1_DOOR_OPEN_WAIT_CLOSE",
    63: "SR_DOOR_OPEN_STAY_OPEN",
    99: "SR_DOOR_BLUE_KEY_STAY",
    103: "S1_DOOR_OPEN_STAY_OPEN",
    111: "S1_FAST_DOOR_OPEN_STAY_OPEN",
    112: "S1_FAST_DOOR_OPEN_WAIT_CLOSE",
    114: "SR_FAST_DOOR_OPEN_STAY_OPEN",
    115: "SR_FAST_DOOR_OPEN_WAIT_CLOSE",
    133: "S1_DOOR_BLUE_KEY_STAY",
    134: "SR_DOOR_RED_KEY_STAY",
    135: "S1_DOOR_RED_KEY_STAY",
    136: "SR_DOOR_YELLOW_KEY_STAY",
    137: "S1_DOOR_YELLOW_KEY_STAY",
    # Walkover doors (W1/WR)
    2: "W1_DOOR_OPEN_STAY_OPEN",
    4: "W1_DOOR_OPEN_WAIT_CLOSE",
    86: "WR_DOOR_OPEN_STAY_OPEN",
    90: "WR_DOOR_OPEN_WAIT_CLOSE",
    108: "W1_FAST_DOOR_OPEN_STAY_OPEN",
    109: "W1_FAST_DOOR_OPEN_WAIT_CLOSE",
    110: "WR_FAST_DOOR_OPEN_STAY_OPEN",
    113: "WR_FAST_DOOR_OPEN_WAIT_CLOSE",
}

MANUAL_DOOR_SPECIALS: set[int] = {
    1, 26, 27, 28, 31, 32, 33, 34, 117, 118,
    29, 63, 99, 103, 111, 112, 114, 115, 133, 134, 135, 136, 137,
}

KEY_DOOR_SPECIALS: dict[int, str] = {
    # Blue Key Doors / Switches
    26: "blue",
    32: "blue",
    99: "blue",
    133: "blue",
    # Red Key Doors / Switches
    28: "red",
    33: "red",
    134: "red",
    135: "red",
    # Yellow Key Doors / Switches
    27: "yellow",
    34: "yellow",
    136: "yellow",
    137: "yellow",
}

EXIT_SPECIALS = {
    11: "S1_EXIT_LEVEL_NORMAL",
    51: "S1_EXIT_LEVEL_SECRET",
    52: "W1_EXIT_LEVEL_WALKOVER",
    124: "W1_EXIT_LEVEL_SECRET_WALKOVER",
    197: "G1_EXIT_LEVEL_NORMAL",
}

LIFT_SPECIALS = {
    10: "W1_LIFT_LOWER_WAIT_RAISE",
    21: "S1_LIFT_LOWER_WAIT_RAISE",
    62: "SR_LIFT_LOWER_WAIT_RAISE",
    88: "WR_LIFT_LOWER_WAIT_RAISE",
}

# Standard Doom Thing Types
PLAYER_SPAWN_TYPES = {
    1: "PLAYER_1_START",
    2: "PLAYER_2_START",
    3: "PLAYER_3_START",
    4: "PLAYER_4_START",
    11: "DEATHMATCH_START",
}

HEALTH_ITEM_TYPES = {
    2011: ("MEDIKIT", 25, 1.0),
    2012: ("STIMPACK", 10, 0.7),
    2014: ("HEALTH_POTION", 1, 0.2),
    2013: ("SOULSPHERE", 100, 1.5),
    2024: ("MEGASPHERE", 200, 2.0),
}

ARMOR_ITEM_TYPES = {
    2015: ("ARMOR_HELMET", 1, 0.2),
    2018: ("GREEN_ARMOR", 100, 0.8),
    2019: ("BLUE_ARMOR", 200, 1.5),
    2024: ("MEGASPHERE", 200, 2.0),
}

KEY_TYPES = {
    5: "BLUE_CARD",
    6: "YELLOW_CARD",
    13: "RED_CARD",
    38: "RED_SKULL",
    39: "YELLOW_SKULL",
    40: "BLUE_SKULL",
}

MONSTER_TYPES = {
    3004: "ZOMBIEMAN",
    9: "SHOTGUN_GUY",
    65: "CHAINGUNNER",
    3001: "IMP",
    3002: "DEMON",
    58: "SPECTRE",
    3006: "LOST_SOUL",
    3003: "BARON_OF_HELL",
    3005: "CACODEMON",
    67: "MANCUBUS",
    68: "ARACHNOTRON",
    69: "HELL_KNIGHT",
    71: "PAIN_ELEMENTAL",
    64: "ARCHVILE",
    16: "CYBERDEMON",
    7: "SPIDER_MASTERMIND",
}


@dataclass(frozen=True)
class Vertex:
    """2D coordinate of a polygon corner in map units."""
    x: float
    y: float

    @property
    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Linedef:
    """Line segment connecting two vertices forming walls, doors, or portals."""
    index: int
    v1: int
    v2: int
    flags: int
    special: int
    tag: int
    front_sidedef: int
    back_sidedef: int

    @property
    def is_one_sided(self) -> bool:
        return self.back_sidedef == -1

    @property
    def is_two_sided(self) -> bool:
        return self.back_sidedef != -1

    @property
    def is_blocking(self) -> bool:
        """True if solid impassable wall to players (either 1-sided or flagged ML_BLOCKING)."""
        return self.is_one_sided or bool(self.flags & 0x0001)

    @property
    def is_door(self) -> bool:
        return self.special in DOOR_SPECIALS

    @property
    def required_key(self) -> str | None:
        """Required keycard/skull color ('blue', 'red', 'yellow') to open/activate, else None."""
        return KEY_DOOR_SPECIALS.get(self.special)

    @property
    def is_exit(self) -> bool:
        return self.special in EXIT_SPECIALS

    @property
    def is_lift(self) -> bool:
        return self.special in LIFT_SPECIALS


@dataclass(frozen=True)
class Sidedef:
    """Texture and sector binding for one side of a linedef."""
    x_offset: int
    y_offset: int
    upper_texture: str
    lower_texture: str
    middle_texture: str
    sector: int


@dataclass(frozen=True)
class Sector:
    """Closed floor/ceiling polygon with height, texture, and light attributes."""
    index: int
    floor_height: int
    ceiling_height: int
    floor_texture: str
    ceiling_texture: str
    light_level: int
    special: int
    tag: int

    @property
    def is_damaging(self) -> bool:
        """True if sector contains radioactive nukage/slime."""
        return self.special in (4, 5, 7, 11, 16)


@dataclass(frozen=True)
class Thing:
    """Map entity (Player start, Monster, Medikit, Weapon, Key)."""
    index: int
    x: float
    y: float
    angle: float
    type: int
    flags: int

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def is_player_1_start(self) -> bool:
        return self.type == 1

    @property
    def is_health(self) -> bool:
        return self.type in HEALTH_ITEM_TYPES

    @property
    def is_medikit(self) -> bool:
        return self.type == 2011

    @property
    def is_stimpack(self) -> bool:
        return self.type == 2012

    @property
    def is_armor(self) -> bool:
        return self.type in ARMOR_ITEM_TYPES

    @property
    def is_monster(self) -> bool:
        return self.type in MONSTER_TYPES

    @property
    def is_key(self) -> bool:
        return self.type in KEY_TYPES

    @property
    def key_color(self) -> str | None:
        """Key color ('blue', 'red', 'yellow') if this entity is a key pickup, else None."""
        if self.type in (5, 40):
            return "blue"
        if self.type in (13, 38):
            return "red"
        if self.type in (6, 39):
            return "yellow"
        return None

    @property
    def heal_amount(self) -> int:
        if self.type in HEALTH_ITEM_TYPES:
            return HEALTH_ITEM_TYPES[self.type][1]
        return 0


class DoomMap:
    """Complete parsed representation of a single Doom map level."""

    def __init__(
        self,
        name: str,
        vertices: Sequence[Vertex],
        linedefs: Sequence[Linedef],
        sidedefs: Sequence[Sidedef],
        sectors: Sequence[Sector],
        things: Sequence[Thing],
    ):
        self.name = str(name).upper()
        self.vertices = list(vertices)
        self.linedefs = list(linedefs)
        self.sidedefs = list(sidedefs)
        self.sectors = list(sectors)
        self.things = list(things)

        # Cache geometry lookups
        self._cache_derived_entities()

    def _cache_derived_entities(self) -> None:
        self._player_spawn = next((t for t in self.things if t.is_player_1_start), None)
        self._exit_linedefs = [l for l in self.linedefs if l.is_exit]
        self._door_linedefs = [l for l in self.linedefs if l.is_door]
        self._blocking_linedefs = [l for l in self.linedefs if l.is_blocking]
        self._health_items = [t for t in self.things if t.is_health]

        # Level 2D bounding box
        if self.vertices:
            xs = [v.x for v in self.vertices]
            ys = [v.y for v in self.vertices]
            self._bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            self._bbox = (0.0, 0.0, 0.0, 0.0)

    @property
    def player_spawn(self) -> Thing:
        """Player 1 spawn location and starting heading."""
        if self._player_spawn is None:
            raise ValueError(f"No Player 1 spawn (Thing type 1) found in {self.name}.")
        return self._player_spawn

    @property
    def exit_linedefs(self) -> list[Linedef]:
        """All linedefs triggering level completion."""
        return self._exit_linedefs

    @property
    def door_linedefs(self) -> list[Linedef]:
        """All interactive door linedefs."""
        return self._door_linedefs

    @property
    def blocking_linedefs(self) -> list[Linedef]:
        """All impassable solid wall linedefs."""
        return self._blocking_linedefs

    @property
    def health_items(self) -> list[Thing]:
        """All health pickups (Medikits, Stimpacks, Soulspheres)."""
        return self._health_items

    @property
    def key_items(self) -> list[Thing]:
        """All keycard and skull pickups in the level."""
        return [t for t in self.things if t.is_key and (t.flags & 16) == 0]

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) extent of the level."""
        return self._bbox

    def linedef_segment(self, linedef: Linedef) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return start and end 2D points of a linedef."""
        v1 = self.vertices[linedef.v1]
        v2 = self.vertices[linedef.v2]
        return (v1.as_tuple, v2.as_tuple)

    def linedef_midpoint(self, linedef: Linedef) -> tuple[float, float]:
        """Return exact (X, Y) midpoint of a linedef."""
        v1 = self.vertices[linedef.v1]
        v2 = self.vertices[linedef.v2]
        return ((v1.x + v2.x) / 2.0, (v1.y + v2.y) / 2.0)

    def door_interaction_points(self) -> list[tuple[float, float]]:
        """Return (X, Y) centers of all doors in the level."""
        return [self.linedef_midpoint(l) for l in self.door_linedefs]

    def exit_interaction_points(self) -> list[tuple[float, float]]:
        """Return (X, Y) centers of all exit triggers in the level."""
        return [self.linedef_midpoint(l) for l in self.exit_linedefs]


class WadArchive:
    """Reader for Doom WAD archives (.wad files)."""

    def __init__(self, wad_path: Path | str):
        self.path = Path(wad_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"WAD file not found at: {self.path}")

        self._read_directory()

    def _read_directory(self) -> None:
        with open(self.path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise ValueError("Corrupt WAD file: header too short.")
            id_bytes, num_lumps, infotable_ofs = struct.unpack("<4sII", header)
            self.wad_type = id_bytes.decode("latin1")
            if self.wad_type not in ("IWAD", "PWAD"):
                raise ValueError(f"Invalid WAD identification signature: {self.wad_type}")

            f.seek(infotable_ofs)
            raw_dir = f.read(num_lumps * 16)
            if len(raw_dir) < num_lumps * 16:
                raise ValueError("Corrupt WAD file: directory truncated.")

            self._dir_entries: list[tuple[int, int, str]] = []
            for i in range(num_lumps):
                pos, size, name_bytes = struct.unpack("<II8s", raw_dir[i * 16 : (i + 1) * 16])
                clean_name = name_bytes.decode("latin1").split("\x00")[0].upper()
                self._dir_entries.append((pos, size, clean_name))

    @property
    def map_names(self) -> list[str]:
        """List all map levels present in the WAD archive."""
        return [
            name for _, _, name in self._dir_entries
            if (name.startswith("MAP") and len(name) <= 5)
            or (len(name) == 4 and name[0] == "E" and name[2] == "M")
        ]

    def load_map(self, map_name: str) -> DoomMap:
        """Parse and return a complete DoomMap level."""
        clean_name = str(map_name).upper().strip()
        try:
            map_idx = next(i for i, x in enumerate(self._dir_entries) if x[2] == clean_name)
        except StopIteration:
            raise KeyError(f"Map '{clean_name}' not found in WAD. Available maps: {self.map_names[:10]}...")

        # A map consists of the map marker followed by up to 10 sequential lumps
        lumps: dict[str, tuple[int, int]] = {}
        stop = min(len(self._dir_entries), map_idx + 12)
        for i in range(map_idx + 1, stop):
            pos, size, name = self._dir_entries[i]
            if name in (
                "THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES",
                "SEGS", "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP"
            ):
                lumps[name] = (pos, size)
            elif name.startswith("MAP") or (len(name) == 4 and name[0] == "E" and name[2] == "M"):
                break

        with open(self.path, "rb") as f:
            # 1. Parse VERTEXES
            vertices = []
            if "VERTEXES" in lumps:
                pos, size = lumps["VERTEXES"]
                f.seek(pos)
                raw = f.read(size)
                count = size // 4
                for i in range(count):
                    x, y = struct.unpack("<hh", raw[i * 4 : (i + 1) * 4])
                    vertices.append(Vertex(float(x), float(y)))

            # 2. Parse SECTORS
            sectors = []
            if "SECTORS" in lumps:
                pos, size = lumps["SECTORS"]
                f.seek(pos)
                raw = f.read(size)
                count = size // 26
                for i in range(count):
                    (
                        floor_h, ceil_h, floor_tex, ceil_tex,
                        light, special, tag
                    ) = struct.unpack("<hh8s8shhh", raw[i * 26 : (i + 1) * 26])
                    sectors.append(
                        Sector(
                            index=i,
                            floor_height=int(floor_h),
                            ceiling_height=int(ceil_h),
                            floor_texture=floor_tex.decode("latin1").split("\x00")[0],
                            ceiling_texture=ceil_tex.decode("latin1").split("\x00")[0],
                            light_level=int(light),
                            special=int(special),
                            tag=int(tag),
                        )
                    )

            # 3. Parse SIDEDEFS
            sidedefs = []
            if "SIDEDEFS" in lumps:
                pos, size = lumps["SIDEDEFS"]
                f.seek(pos)
                raw = f.read(size)
                count = size // 30
                for i in range(count):
                    (
                        x_off, y_off, upper, lower, middle, sec
                    ) = struct.unpack("<hh8s8s8sh", raw[i * 30 : (i + 1) * 30])
                    sidedefs.append(
                        Sidedef(
                            x_offset=int(x_off),
                            y_offset=int(y_off),
                            upper_texture=upper.decode("latin1").split("\x00")[0],
                            lower_texture=lower.decode("latin1").split("\x00")[0],
                            middle_texture=middle.decode("latin1").split("\x00")[0],
                            sector=int(sec),
                        )
                    )

            # 4. Parse LINEDEFS
            linedefs = []
            if "LINEDEFS" in lumps:
                pos, size = lumps["LINEDEFS"]
                f.seek(pos)
                raw = f.read(size)
                count = size // 14
                for i in range(count):
                    (
                        v1, v2, flags, special, tag, front, back
                    ) = struct.unpack("<hhhhhhh", raw[i * 14 : (i + 1) * 14])
                    linedefs.append(
                        Linedef(
                            index=i,
                            v1=int(v1),
                            v2=int(v2),
                            flags=int(flags),
                            special=int(special),
                            tag=int(tag),
                            front_sidedef=int(front),
                            back_sidedef=int(back),
                        )
                    )

            # 5. Parse THINGS
            things = []
            if "THINGS" in lumps:
                pos, size = lumps["THINGS"]
                f.seek(pos)
                raw = f.read(size)
                count = size // 10
                for i in range(count):
                    x, y, angle, ttype, flags = struct.unpack("<hhhhh", raw[i * 10 : (i + 1) * 10])
                    things.append(
                        Thing(
                            index=i,
                            x=float(x),
                            y=float(y),
                            angle=float(angle),
                            type=int(ttype),
                            flags=int(flags),
                        )
                    )

        return DoomMap(
            name=clean_name,
            vertices=vertices,
            linedefs=linedefs,
            sidedefs=sidedefs,
            sectors=sectors,
            things=things,
        )


def load_wad(wad_path: Path | str = "freedoom2.wad") -> WadArchive:
    """Convenience factory returning a WadArchive instance."""
    return WadArchive(wad_path)


def load_map_from_wad(wad_path: Path | str, map_name: str = "MAP01") -> DoomMap:
    """Convenience helper to directly parse and return a specific level."""
    return load_wad(wad_path).load_map(map_name)
