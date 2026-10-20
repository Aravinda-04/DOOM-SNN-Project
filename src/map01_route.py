"""Sparse MAP01 navigation graph and causal progress tracking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demonstrations import load_episode, validate_manifest


DEFAULT_USE_POINTS = np.asarray(
    ((680.0, -160.0), (1800.0, -145.0), (670.0, -1025.0), (220.0, -670.0)),
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
    recovery: bool = False


def _segment_error(points: np.ndarray, start: int, stop: int) -> tuple[float, int]:
    if stop <= start + 1:
        return 0.0, start
    segment = points[stop] - points[start]
    length_squared = float(np.dot(segment, segment))
    interior = points[start + 1 : stop]
    if length_squared <= 1e-9:
        distances = np.linalg.norm(interior - points[start], axis=1)
    else:
        projection = np.clip(
            ((interior - points[start]) @ segment) / length_squared, 0.0, 1.0
        )
        closest = points[start] + projection[:, None] * segment
        distances = np.linalg.norm(interior - closest, axis=1)
    offset = int(np.argmax(distances))
    return float(distances[offset]), start + 1 + offset


def _rdp_indices(points: np.ndarray, tolerance: float) -> set[int]:
    retained = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start, stop = pending.pop()
        error, index = _segment_error(points, start, stop)
        if error > tolerance:
            retained.add(index)
            pending.extend(((start, index), (index, stop)))
    return retained


def _sparse_indices(
    points: np.ndarray,
    *,
    tolerance: float,
    maximum_path_length: float,
    required_indices: set[int],
) -> list[int]:
    """Simplify a trajectory while bounding distance along every graph edge."""
    retained = _rdp_indices(points, tolerance) | required_indices
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    changed = True
    while changed:
        changed = False
        ordered = sorted(retained)
        for start, stop in zip(ordered, ordered[1:]):
            if cumulative[stop] - cumulative[start] <= maximum_path_length:
                continue
            midpoint_distance = (cumulative[start] + cumulative[stop]) / 2.0
            midpoint = int(np.searchsorted(cumulative, midpoint_distance))
            midpoint = int(np.clip(midpoint, start + 1, stop - 1))
            retained.add(midpoint)
            changed = True
    return sorted(retained)


class Map01Route:
    """Sparse directed graph derived from a successful run's topology."""

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
        self.edges = tuple((index, index + 1) for index in range(len(points) - 1))
        self.use_points = np.asarray(use_points, dtype=np.float32)
        self.source_sha256 = str(source_sha256)
        digest = hashlib.sha256()
        digest.update(b"sparse-map01-route-v2")
        digest.update(points.tobytes())
        digest.update(self.use_points.tobytes())
        digest.update(self.source_sha256.encode("ascii"))
        self.sha256 = digest.hexdigest()

    @classmethod
    def from_demonstrations(
        cls,
        dataset_dir: Path,
        *,
        simplification_tolerance: float = 36.0,
        maximum_path_length: float = 220.0,
    ) -> "Map01Route":
        if simplification_tolerance <= 0 or maximum_path_length <= 0:
            raise ValueError("Route simplification values must be positive.")
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        manifest = validate_manifest(dataset_dir)
        entries = [
            entry for entry in manifest["episodes"]
            if entry.get("completed") and entry.get("valid_action_contract")
        ]
        if not entries:
            raise ValueError("No successful demonstrations are available for MAP01.")
        reference = min(entries, key=lambda entry: int(entry["steps"]))
        payload = load_episode(dataset_dir / reference["file"])
        raw = np.asarray(payload["positions"][:, :2], dtype=np.float32)
        interaction_indices = {
            int(np.argmin(np.linalg.norm(raw - point, axis=1)))
            for point in DEFAULT_USE_POINTS
        }
        indices = _sparse_indices(
            raw,
            tolerance=simplification_tolerance,
            maximum_path_length=maximum_path_length,
            required_indices=interaction_indices,
        )
        sparse = raw[indices]
        keep = np.concatenate(
            ([True], np.linalg.norm(np.diff(sparse, axis=0), axis=1) > 1.0)
        )
        return cls(sparse[keep], source_sha256=reference["sha256"])

    def context(
        self,
        position: np.ndarray,
        progress_index: int,
        *,
        capture_radius: float = 72.0,
        waypoint_radius: float = 38.0,
        off_track_distance: float = 128.0,
        use_radius: float = 58.0,
    ) -> RouteContext:
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite vector with shape (2,).")
        progress = int(np.clip(progress_index, 0, len(self.points) - 1))
        stop = min(len(self.points), progress + 5)
        local = np.linalg.norm(self.points[progress:stop] - position, axis=1)
        nearest_offset = int(np.argmin(local))
        if float(local[nearest_offset]) < capture_radius:
            progress += nearest_offset
        while (
            progress + 1 < len(self.points)
            and float(np.linalg.norm(self.points[progress] - position)) < waypoint_radius
        ):
            progress += 1

        previous_index = max(0, progress - 1)
        edge_start = self.points[previous_index]
        edge = self.points[progress] - edge_start
        edge_length_squared = float(np.dot(edge, edge))
        if edge_length_squared <= 1e-9:
            distance_to_edge = float(np.linalg.norm(position - edge_start))
        else:
            projection = float(
                np.clip(np.dot(position - edge_start, edge) / edge_length_squared, 0, 1)
            )
            closest = edge_start + projection * edge
            distance_to_edge = float(np.linalg.norm(position - closest))
        recovery = distance_to_edge > off_track_distance
        target_index = progress
        if recovery:
            start = max(0, progress - 2)
            stop = min(len(self.points), progress + 4)
            recovery_distances = np.linalg.norm(self.points[start:stop] - position, axis=1)
            candidate = start + int(np.argmin(recovery_distances))
            target_index = max(progress, candidate)

        target = self.points[target_index].copy()
        delta = target - position
        distance = float(np.linalg.norm(delta))
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)
        use_distance = float(np.min(np.linalg.norm(self.use_points - position, axis=1)))
        return RouteContext(
            progress_index=progress,
            progress_fraction=progress / max(len(self.points) - 1, 1),
            target=target,
            delta=delta,
            distance=distance,
            bearing_degrees=bearing,
            off_track=recovery,
            use_recommended=use_distance < use_radius,
            recovery=recovery,
        )


def angle_delta_degrees(target: float, current: float) -> float:
    """Return the signed shortest target-current angle in [-180, 180)."""
    return float((target - current + 180.0) % 360.0 - 180.0)
