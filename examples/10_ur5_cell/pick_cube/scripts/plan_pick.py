# SPDX-License-Identifier: Apache-2.0
"""Hand poses over and at the cube, the jaws turned to close across two faces.

A cube's faces repeat every 90 degrees and the 2F-85's jaws every 180, so four
headings make the same grasp. This takes the one whose hand rotation is
nearest the hand's rotation now, as ur5-VLA's cube skill does: the wrist then
turns at most 45 degrees. Any one of them, taken blindly, asked this cell's
wrist 3 for 91 degrees -- past the 63 its cables allow -- on the first real dry run.

The tool centre is the 2F-85's pinch point at full open, so ``grasp`` puts it
at the cube's centre height plus ``offset``. The pads ride down about 10 mm as
they close on their four-bar, which still leaves them clear of the table on a
60 mm cube.
"""

import math
from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    above: dict
    grasp: dict
    heading_deg: float
    turn_deg: float


def _angle(a: dict, b: dict) -> float:
    """The angle between two rotations [deg]."""
    dot = abs(sum(float(a[k]) * float(b[k]) for k in "wxyz"))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def run(ctx: NodeContext, cube: dict, clearance: float = 0.12, offset: float = 0.0) -> Output:
    p, q = cube["position"], cube["rotation"]
    yaw = math.degrees(math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]), 1 - 2 * (q["y"] ** 2 + q["z"] ** 2)))
    now = ctx.tool("robot.get_ee_pose")["pose"]["rotation"]
    options = []
    for k in range(-2, 2):
        heading = yaw + 90.0 * k
        rotation = ctx.tool("robot.grasp_frame", close_heading_deg=heading)["rotation"]
        options.append((_angle(rotation, now), heading, rotation))
    turn, heading, rotation = min(options, key=lambda o: o[0])
    at = lambda z: {"position": {"x": p["x"], "y": p["y"], "z": z}, "rotation": rotation}  # noqa: E731
    return {
        "above": at(p["z"] + offset + clearance),
        "grasp": at(p["z"] + offset),
        "heading_deg": heading,
        "turn_deg": turn,
    }
