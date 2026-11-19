# SPDX-License-Identifier: Apache-2.0
"""Raise unless the jaws stopped on something.

1 is open. On the 60 mm cube a 2F-85 stops about 0.67 open; shut on nothing it
reads about 0.1 (the real hand stops near 230 of 255 counts) or 0 (the twin).
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    open_fraction: float


def run(ctx: NodeContext, least: float = 0.25) -> Output:
    reading = float(ctx.tool("robot.get_gripper")["position"])
    if reading < least:
        raise RuntimeError(f"the jaws closed on nothing ({reading:.2f} open)")
    return {"open_fraction": reading}
