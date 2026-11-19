# SPDX-License-Identifier: Apache-2.0
"""The cube, from one wrist RGB-D frame taken at the look pose.

Every valid depth pixel is back-projected to a world point. The table is the
most common height (a 2 mm histogram's peak, not the median: from 0.5 m the
cube can be a fair share of the frame). The cube's top face is what stands 40
to 80 mm over it; its centre is halfway down from the top face, and its yaw is
the turn whose box around the top face is smallest.

A matte black cube can return thin depth on a real D415. When fewer than
``least`` pixels make the top face, the dark pixels -- darker than ``dark`` of
the table's median brightness -- are cast as rays onto the plane of the top
face instead, ``size`` above the table.

Three checks refuse a frame rather than guess from it, each of which a frame
taken from the wrong place has failed: the table must be within ``table_tol``
of ``table`` (the cell's table height in the world); the blob must measure no
more than ``area_ratio`` times a cube's top face at this range; and the cube
must lie inside ``patch``, the search patch in the work frame
(``(x_min, y_min, x_max, y_max)``, ur5-VLA's ``cube.roi_*_m``).
"""

import math
from typing import TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    cube: dict
    table_z: float
    method: str
    pixels: int


def _rotation(cam: dict) -> np.ndarray:
    w, x, y, z = (float(cam["pose"]["rotation"][k]) for k in "wxyz")
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _rays(cam: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """World directions of pixels (u, v), unnormalised, with unit depth along the view."""
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    local = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=np.float64)], axis=1)
    return local @ _rotation(cam).T


def _yaw(xy: np.ndarray) -> float:
    def area(deg: int) -> float:
        a = math.radians(deg)
        return float(np.ptp(xy @ [math.cos(a), math.sin(a)]) * np.ptp(xy @ [-math.sin(a), math.cos(a)]))

    return math.radians(min(range(90), key=area))


def run(
    ctx: NodeContext,
    cameras: list,
    camera: str = "wrist",
    size: float = 0.060,
    least: int = 60,
    dark: float = 0.45,
    table: float = 0.75,
    table_tol: float = 0.03,
    area_ratio: float = 3.0,
    patch: tuple = (-0.10, -0.17, 0.10, 0.17),
) -> Output:
    cam = next(c for c in cameras if c["name"] == camera)
    depth = np.asarray(cam["depth"], dtype=np.float64).reshape(np.shape(cam["depth"])[:2])
    t = cam["pose"]["position"]
    origin = np.array([float(t["x"]), float(t["y"]), float(t["z"])])
    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    ok = np.isfinite(depth) & (depth > 0.0)
    if ok.sum() < 1000:
        raise RuntimeError(f"the {camera} camera returned almost no depth ({int(ok.sum())} pixels)")
    points = origin + _rays(cam, u[ok], v[ok]) * depth[ok][:, None]

    hist, edges = np.histogram(points[:, 2], bins=np.arange(points[:, 2].min(), points[:, 2].max() + 0.002, 0.002))
    peak = int(np.argmax(hist))
    near = np.abs(points[:, 2] - (edges[peak] + 0.001)) < 0.004
    measured = float(np.median(points[near, 2]))
    if abs(measured - table) > table_tol:
        raise RuntimeError(
            f"the most common height in view is {measured:.3f} m, not the table's {table:.3f} m: "
            "is the arm at the look pose?"
        )
    table = measured
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    face_px = (size * K[0, 0] / max(origin[2] - table - size, 0.05)) ** 2  # a top face, seen from here

    band = (points[:, 2] > table + 0.040) & (points[:, 2] < table + 0.080)
    top = points[band]
    method = "depth"
    if len(top) < least:
        rgb = np.asarray(cam["rgb"], dtype=np.float64)[..., :3]
        lum = rgb.mean(axis=2)
        mask = lum < dark * float(np.median(lum[ok]))
        if mask.sum() < least:
            raise RuntimeError(
                f"the {camera} camera does not see the cube ({len(top)} top-face points, {int(mask.sum())} dark pixels)"
            )
        rays = _rays(cam, u[mask], v[mask])
        s = (table + size - origin[2]) / rays[:, 2]
        top = origin + rays * s[:, None]
        method = "dark pixels on the top-face plane"
    # Keep the one blob: points within a cube's diagonal of the robust centre.
    centre = np.median(top[:, :2], axis=0)
    top = top[np.linalg.norm(top[:, :2] - centre, axis=1) < 0.75 * size * math.sqrt(2)]
    if len(top) < least:
        raise RuntimeError(f"the cube's top face is {len(top)} points across, too few to trust")
    if len(top) > area_ratio * face_px:
        raise RuntimeError(
            f"the blob is {len(top)} pixels, {len(top) / face_px:.1f} times a cube's top face from here: not one cube"
        )
    cx, cy = top[:, :2].mean(axis=0)
    z_top = float(np.percentile(top[:, 2], 90)) if method == "depth" else table + size
    if not (patch[0] <= cx <= patch[2] and patch[1] <= cy <= patch[3]):
        raise RuntimeError(f"the cube at ({cx:+.3f}, {cy:+.3f}) m is outside the search patch {patch}")
    yaw = _yaw(top[:, :2] - [cx, cy])
    cube = {
        "position": {"x": float(cx), "y": float(cy), "z": (table + z_top) / 2},
        "rotation": {"w": math.cos(yaw / 2), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2)},
    }
    return {"cube": cube, "table_z": table, "method": method, "pixels": int(len(top))}
