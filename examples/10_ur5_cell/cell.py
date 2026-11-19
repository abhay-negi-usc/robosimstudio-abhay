# SPDX-License-Identifier: Apache-2.0
"""The real UR5 cell, declared once for the sim run and the real run.

A CB3 UR5 on a 3 ft x 6 ft table, with an ATI Axia80-M8 at the flange, a
RealSense D415 on a bracket below it and a Robotiq 2F-85 at the end, picking a
60 mm black cube. Every number here was measured on the cell (the sibling
``ur5-VLA`` and ``ur5-workspace`` checkouts hold the measurements) and says
where it came from.

**One frame convention, the work surface's.** Its origin is the table under the
middle of the patch the wrist camera searches: ``(0.525, -0.20, -0.0173)`` in the
UR ``base`` frame the pendant shows, with the same axes. So a point the pendant
reports as ``(x, y, z)`` is ``(x - 0.525, y + 0.20, z + 0.0173)`` here, and
:func:`base_to_work` / :func:`work_to_base` are the only places that sum is
written.

Importing this module registers the cell, the robot, the camera rig and the
cube; the example scripts import it for that.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import robosimstudio as rss
from robosimstudio.assetlib import register_asset
from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.math import xform_from_matrix
from robosimstudio.domains.rigid import Body
from robosimstudio.robots.registry import register_embodiment
from robosimstudio.scene.registry import register_cell
from robosimstudio.specs import ActuationSpec, CellSpec, StaticSpec
from robosimstudio_bench import register_camera_rig

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- the frames

WORK_IN_BASE = np.array([0.525, -0.20, -0.0173])
"""The work-surface origin in the UR ``base`` frame [m], axes shared.

x and y: the middle of ``cube.roi_*_m`` in ur5-VLA's ``configs/policy/scripted_cube.yaml``
((0.44, -0.35)-(0.61, -0.05)), rounded. z: the table top, from ``pick_cube/scripts/locate.py``
run on the first wrist frame of 40 recorded episodes (``/data/ur5VLA-data/
20261113-142018_collect_loop_s102``, posed by the logged joints and the hand-eye
solve): 3.3 mm (2.5 to 4.0) under ur5-VLA's ``cube.table_z_m`` of -0.014.
"""


def base_to_work(p) -> np.ndarray:
    """A point in the UR base frame to the work-surface frame [m]."""
    return np.asarray(p, dtype=np.float64) - WORK_IN_BASE


def work_to_base(p) -> np.ndarray:
    """A point in the work-surface frame to the UR base frame [m]."""
    return np.asarray(p, dtype=np.float64) + WORK_IN_BASE


# ---------------------------------------------------------------- the table

SURFACE_Z = 0.75
"""World height of the table top [m]. Nothing depends on it; the floor is out of reach."""

TABLE_IN_BASE = ((-0.46, 1.37), (-0.707, 0.207))
"""The 3 ft x 6 ft table's extent in the UR base frame, x then y [m], from the
placement in ur5-workspace's ``scene_real.xml`` (the base 0.4545 m in from one
short edge and 0.25 m off the long axis, facing down the table)."""

_lo = base_to_work((TABLE_IN_BASE[0][0], TABLE_IN_BASE[1][0], 0.0))
_hi = base_to_work((TABLE_IN_BASE[0][1], TABLE_IN_BASE[1][1], 0.0))
_mid, _half = (_lo + _hi) / 2, (_hi - _lo) / 2

UR5_TABLE = register_cell(
    CellSpec(
        name="ur5_table",
        surface_z=SURFACE_Z,
        center=(0.0, 0.0, 0.0),
        half=(float(_half[0]), float(_half[1]), SURFACE_Z / 2),
        statics=(
            StaticSpec(
                name="table",
                half_extents=(float(_half[0]), float(_half[1]), SURFACE_Z / 2),
                xform=(float(_mid[0]), float(_mid[1]), SURFACE_Z / 2, 0.0, 0.0, 0.0, 1.0),
                color=(0.55, 0.52, 0.47),
            ),
        ),
        provenance="the UR5 cell's 3 ft x 6 ft table, work origin under the wrist camera's search patch",
    )
)

# ---------------------------------------------------------------- the robot

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")

CUBE_LOOK_DEG = (-9.23, -110.53, -37.24, -122.27, 90.0, -9.23)
"""``robot.start_poses.cube_look`` in ur5-VLA's ``configs/engineered.yaml``: the
wrist camera looks straight down on the search patch from here. Home."""

HOME_Q = tuple(math.radians(v) for v in CUBE_LOOK_DEG)

JOINT_MIN_DEG = (-37.0, -145.0, -132.0, -150.0, 79.0, -81.0)
JOINT_MAX_DEG = (31.0, -71.0, 8.0, -49.0, 101.0, 63.0)
"""``safety.joint_{min,max}_deg`` in ``engineered.yaml``: the joint box the cube
skill was proven inside on this cell, cables included. The real run refuses a
target outside it."""

GRIPPER_MOUNT = (0.0, 0.0, 0.08155, 0.0, 0.0, 1.0, 0.0)
"""Where the 2F-85 file's frame sits on ``wrist_3_link`` (UR ``tool0``): xyz, xyzw.

81.55 mm along the flange axis -- the Robotiq mount's face, from the as-built
tool CAD (ur5-workspace ``tool_assembly.py``) -- and a half turn about it, as
ur5-workspace mounts it. That puts the pinch point at 237.4 mm, against the
233.9 mm ur5-VLA measured at full open (``safety.tcp_offset_m``).
"""

UR5_CB3_CELL = register_embodiment(
    rss.EmbodimentSpec(
        name="ur5_cb3_cell",
        asset=rss.AssetSpec(path=str(HERE / "assets" / "ur5_cb3_axia_d415.xml")),
        base_offset=(*(-WORK_IN_BASE).tolist(), 0.0, 0.0, 0.0, 1.0),
        arms=(
            rss.ArmSpec(
                name="right",
                joint_names=JOINTS,
                ee_body_name="wrist_3_link",
                base_body_name="base_link",
                gripper="robotiq_2f85_menagerie",
                gripper_mount=GRIPPER_MOUNT,
                home_q=HOME_Q,
            ),
        ),
        # Plain position servos, at the gains twins of UR arms settled on
        # (examples/live/REFERENCE.md, "Pitfalls twins have met").
        arm_actuation=ActuationSpec(
            target_ke=2000.0,
            target_kd=400.0,
            target_ke_per_dof=(2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0),
            target_kd_per_dof=(400.0, 400.0, 400.0, 100.0, 100.0, 100.0),
        ),
        gravcomp=True,
        reset_joint_q=True,
        # The jaws must close across two of the cube's faces, so the IK has to
        # hold the wrist roll the graph asks for. Left free, a cube turned 30
        # degrees was gripped on its corners and squeezed out of the hand.
        honour_roll=True,
    )
)

# ---------------------------------------------------------------- the camera

T_FLANGE_CAM = np.array([
    [0.999977312, 0.003631745, -0.005673227, -7.4628e-05],
    [-0.002156272, 0.970476614, 0.241185597, -0.06956143],
    [0.006381658, -0.241167892, 0.970462427, 0.087525257],
    [0.0, 0.0, 0.0, 1.0],
])  # fmt: skip
"""The D415 colour sensor in the flange frame, OpenCV axes (+z forward, +y down).

ur5-VLA's ``configs/calibration/wrist_handeye.yaml``, solved 2026-11-13 from 52
AprilTag views: 1.5 mm and 0.6 deg rms. The real run reads that file afresh and
says so if it has moved from this copy.
"""

D415_640 = {"fx": 607.3008, "fy": 607.1533, "cx": 324.5818, "cy": 233.7083}
"""The D415's (serial 234422064578) factory colour intrinsics at 640 x 480,
read off the device. Its colour stream is rectified: no distortion."""

WRIST = CameraSpec(
    name="wrist",
    width=640,
    height=480,
    **D415_640,
    near=0.07,
    far=2.0,
    xform=tuple(float(v) for v in xform_from_matrix(T_FLANGE_CAM)),
    # The flange itself, not the role "work": a role resolves to the hand's
    # tool body (the 2F-85's base_mount, 88.55 mm on and a half turn round),
    # and the hand-eye solve is in the flange's frame.
    parent_frame="wrist_3_link",
    channels=("rgb", "depth"),
)
register_camera_rig("ur5_wrist@v1", (WRIST,))

# ---------------------------------------------------------------- the cube

CUBE_HALF = 0.030
"""``cube.cube_size_m`` is 0.060 in scripted_cube.yaml."""

CUBE_AT = (-0.04, -0.06, CUBE_HALF)
"""Where the sim run puts it: inside the search patch, off its centre."""

PLACE_AT = (0.04, 0.10)
"""Where it goes, in the work frame: (0.565, -0.10) in the base frame, inside
the patch the camera searches, so a second run can find it there."""

PARK_AT = (0.80, -0.45, CUBE_HALF)
"""The real run's twin parks its cube here, 1.47 m from the base and out of
reach: the real cube is the one picked, and a twin cube under the twin's hand
would stop the twin short of where the real arm went."""


def register_cube(at=CUBE_AT, yaw_deg: float = 0.0) -> None:
    """Register ``black_cube_60mm``, starting at ``at`` in the work frame, turned ``yaw_deg`` about up."""
    half_turn = math.radians(yaw_deg) / 2
    quat = (0.0, 0.0, math.sin(half_turn), math.cos(half_turn))
    cube = rss.box(name="cube", half=(CUBE_HALF,) * 3, at=at, quat=quat, color=(0.03, 0.03, 0.03))
    register_asset(Body.from_spec(cube, name="black_cube_60mm"))


TASK = f"""
(define (problem ur5_cube_pick_place)
  (:staging (:scene ur5_table) (:embodiment ur5_cb3_cell) (:physics mujoco_500hz@v1) (:max-steps 6000)
            (:cameras ur5_wrist@v1))
  (:objects cube - black_cube_60mm)
  (:regions (place (:ranges (({PLACE_AT[0] - 0.05} {PLACE_AT[1] - 0.05} {PLACE_AT[0] + 0.05} {PLACE_AT[1] + 0.05})))))
  (:goal (once (hold-for 0.5 (and (>= (containment cube place) 1.0) (<= (speed cube) 0.02)
                                  (> (hands-clear cube) 0.08)))))
  (:artifacts (:graph pick_cube "pick_cube")))
"""
