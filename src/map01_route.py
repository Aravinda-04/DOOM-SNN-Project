"""MAP01 route extraction and causal progress tracking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demonstrations import load_episode, validate_manifest


DEFAULT_USE_POINTS = np.asarray(
    (
        (680.0, -160.0),
        (1800.0, -145.0),
        (670.0, -1025.0),
        (220.0, -670.0),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class RouteContext:
    progress_index: int
    progress_fraction: float
    target: np.ndarray
    delta: np.ndarray
    distance: float
    bearing_degrees: float
    off_track: bool
    use_recommended: bool


class Map01Route:
    """A reproducible polyline extracted from the shortest successful run."""

    def __init__(
        self,
        points: np.ndarray,
        *,
        source_sha256: str,
        use_points: np.ndarray = DEFAULT_USE_POINTS,
    ):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("Route points must have shape (N, 2), with N >= 2.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Route points contain non-finite coordinates.")
        self.points = points
        self.use_points = np.asarray(use_points, dtype=np.float32)
        self.source_sha256 = str(source_sha256)
        digest = hashlib.sha256()
        digest.update(points.tobytes())
        digest.update(self.use_points.tobytes())
        digest.update(self.source_sha256.encode("ascii"))
        self.sha256 = digest.hexdigest()

    @classmethod
    def from_demonstrations(
        cls,
        dataset_dir: Path,
        *,
        minimum_spacing: float = 10.0,
    ) -> "Map01Route":
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        manifest = validate_manifest(dataset_dir)
        entries = [
            entry
            for entry in manifest["episodes"]
            if entry.get("completed") and entry.get("valid_action_contract")
        ]
        if not entries:
            raise ValueError("No successful demonstrations are available for MAP01.")
        reference = min(entries, key=lambda entry: int(entry["steps"]))
        payload = load_episode(dataset_dir / reference["file"])
        raw = np.asarray(payload["positions"][:, :2], dtype=np.float32)
        retained = [raw[0]]
        for point in raw[1:]:
            if float(np.linalg.norm(point - retained[-1])) >= minimum_spacing:
                retained.append(point)
        if float(np.linalg.norm(raw[-1] - retained[-1])) > 1.0:
            retained.append(raw[-1])
        return cls(
            np.asarray(retained, dtype=np.float32),
            source_sha256=reference["sha256"],
        )

    def context(
        self,
        position: np.ndarray,
        progress_index: int,
        *,
        target_lookahead: int = 2,
        search_ahead: int = 12,
        capture_radius: float = 48.0,
        waypoint_radius: float = 24.0,
        off_track_distance: float = 96.0,
        use_radius: float = 55.0,
    ) -> RouteContext:
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite vector with shape (2,).")
        progress = int(np.clip(progress_index, 0, len(self.points) - 1))
        stop = min(len(self.points), progress + search_ahead)
        distances = np.linalg.norm(self.points[progress:stop] - position, axis=1)
        nearest_offset = int(np.argmin(distances))
        if float(distances[nearest_offset]) < capture_radius:
            progress += nearest_offset
        while (
            progress + 1 < len(self.points)
            and float(np.linalg.norm(self.points[progress] - position))
            < waypoint_radius
        ):
            progress += 1
        target_index = min(progress + target_lookahead, len(self.points) - 1)
        target = self.points[target_index].copy()
        delta = target - position
        distance = float(np.linalg.norm(delta))
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)
        route_distance = float(np.linalg.norm(self.points[progress] - position))
        use_distance = float(
            np.min(np.linalg.norm(self.use_points - position, axis=1))
        )
        return RouteContext(
            progress_index=progress,
            progress_fraction=progress / max(len(self.points) - 1, 1),
            target=target,
            delta=delta,
            distance=distance,
            bearing_degrees=bearing,
            off_track=route_distance > off_track_distance,
            use_recommended=use_distance < use_radius,
        )


def angle_delta_degrees(target: float, current: float) -> float:
    """Return the signed shortest target-current angle in [-180, 180)."""
    return float((target - current + 180.0) % 360.0 - 180.0)
