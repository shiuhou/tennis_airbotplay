"""Discrete URDF convex-hull preflight for the fixed-base arm."""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .config import BASE_POSITION, RACKET_WRIST_CLEARANCE_M
from .kinematics import origin

CLEARANCE_M = 0.001
BOX_CORNERS = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])


@dataclass(frozen=True)
class Obstacle:
    name: str
    center_m: tuple
    size_m: tuple


def _parse_obstacles(items):
    if not isinstance(items, (list, tuple)):
        raise ValueError("Obstacles must be a JSON list of boxes.")
    boxes = []
    names = {"ground"}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "center_m", "size_m"}:
            raise ValueError("A box requires name, center_m and size_m.")
        name = item["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Obstacle names must be nonempty and unique.")
        center, size = np.asarray(item["center_m"], dtype=float), np.asarray(item["size_m"], dtype=float)
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite([center, size]).all() or np.any(size <= 0):
            raise ValueError(f"Obstacle '{name}' requires finite 3D center and positive 3D size.")
        boxes.append(Obstacle(name, tuple(center), tuple(size)))
        names.add(name)
    return boxes


def stl_triangles(path):
    """Read the source triangle surface without replacing it by a convex hull."""
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count and len(data) == 84 + 50 * count:
            dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            triangles = np.frombuffer(data, dtype=dtype, count=count, offset=84)
            return triangles["vertices"].astype(float)
    if data.startswith(b"solid"):
        vertices = np.array([line.split()[1:] for line in data.splitlines() if line.lstrip().startswith(b"vertex ")], dtype=float)
        if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0 and len(vertices) % 3 == 0:
            return vertices.reshape(-1, 3, 3)
    raise ValueError(f"Invalid collision STL: {path}")


def _stl_points(path):
    return np.unique(stl_triangles(path).reshape(-1, 3), axis=0)


@dataclass(frozen=True)
class _Hull:
    points: np.ndarray
    planes: np.ndarray

    @classmethod
    def from_points(cls, points):
        hull = ConvexHull(points)
        # Qhull triangulates coplanar faces. Keep one plane per face for the LP.
        return cls(points[hull.vertices], np.unique(np.round(hull.equations, 10), axis=0))

    def transform(self, transform):
        rotation, shift = transform[:3, :3], transform[:3, 3]
        normals = self.planes[:, :3] @ rotation.T
        return _Hull(self.points @ rotation.T + shift, np.c_[normals, self.planes[:, 3] - normals @ shift])

    def overlaps(self, other, clearance=CLEARANCE_M):
        if (
            np.any(self.points.max(0) + clearance < other.points.min(0))
            or np.any(other.points.max(0) + clearance < self.points.min(0))
        ):
            return False
        planes = np.vstack([self.planes, other.planes])
        result = linprog(
            np.zeros(3), A_ub=planes[:, :3], b_ub=-planes[:, 3] + clearance / 2,
            bounds=[(None, None)] * 3, method="highs",
        )
        if result.status not in (0, 2):
            raise RuntimeError(f"Collision feasibility calculation failed: {result.message}")
        return result.success


class CollisionChecker:
    """Check ground and boxes in world, and non-adjacent links within the arm."""

    def __init__(self, model, obstacles=(), *, base_position=BASE_POSITION, base_mount_obstacle=None):
        self.model = model
        self.base_position = np.asarray(base_position, dtype=float)
        if self.base_position.shape != (3,) or not np.isfinite(self.base_position).all():
            raise ValueError('Base position must be a finite three-vector.')
        self.obstacles = _parse_obstacles(obstacles)
        self.base_mount_obstacle = base_mount_obstacle
        if base_mount_obstacle is not None and base_mount_obstacle not in {box.name for box in self.obstacles}:
            raise ValueError('Base mount must name an existing obstacle.')
        self._boxes = [_Hull.from_points(BOX_CORNERS * box.size_m + box.center_m) for box in self.obstacles]
        self._shapes = []
        self._excluded = {frozenset((f"link{i}", f"link{i+1}")) for i in range(1, 6)}
        self._excluded.add(frozenset(("base_link", "link1")))
        if model.has_racket:
            # Reviewed rigid assembly: mounting and handle/stop interfaces.
            self._excluded.update(frozenset(pair) for pair in
                                  (("link6", "fixer"), ("fixer", "paddle")))
        for name in model.links:
            elements = model.xml.findall(f"link[@name='{name}']/collision")
            if not elements:
                raise ValueError(f"Missing collision geometry for {name}.")
            shapes = []
            for element in elements:
                mesh = element.find("geometry/mesh")
                box = element.find("geometry/box")
                if mesh is not None and Path(mesh.attrib["filename"]).suffix.lower() == ".stl":
                    points = _stl_points(model.path.parent / mesh.attrib["filename"])
                    points *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                elif box is not None:
                    points = BOX_CORNERS * np.fromstring(box.attrib["size"], sep=" ")
                else:
                    raise ValueError(f"Unsupported collision geometry for {name}.")
                shapes.append(_Hull.from_points(points).transform(origin(element.find("origin"))))
            self._shapes.append(shapes)

    def check(self, q):
        groups = []
        for shapes, frame in zip(self._shapes, self.model.frames(q), strict=True):
            world = frame.copy()
            world[:3, 3] += self.base_position
            groups.append([shape.transform(world) for shape in shapes])
        pairs = []
        for index, (name, shapes) in enumerate(zip(self.model.links, groups, strict=True)):
            if any(shape.points[:, 2].min() <= CLEARANCE_M for shape in shapes):
                pairs.append((name, "ground"))
            for box, hull in zip(self.obstacles, self._boxes, strict=True):
                # Only the fixed base/support interface is intentional contact.
                if name == 'base_link' and box.name == self.base_mount_obstacle:
                    continue
                if any(shape.overlaps(hull) for shape in shapes):
                    pairs.append((name, box.name))
            for other in range(index + 1, len(groups)):
                pair = frozenset((name, self.model.links[other]))
                if pair in self._excluded:
                    continue
                # The reviewed link5/fixer seat has only 0.947 mm nominal separation.
                # Keep 0.5 mm here; the general 1 mm planning margin stays unchanged.
                clearance = RACKET_WRIST_CLEARANCE_M if pair == frozenset(("link5", "fixer")) else CLEARANCE_M
                if any(shape.overlaps(hull, clearance) for shape in shapes for hull in groups[other]):
                    pairs.append((name, self.model.links[other]))
        return pairs

    def require_clear(self, q):
        pairs = self.check(q)
        if pairs:
            raise ValueError(f"Collision or clearance violation: {pairs}")
