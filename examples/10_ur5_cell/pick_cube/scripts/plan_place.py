# SPDX-License-Identifier: Apache-2.0
"""Hand poses over and onto the place spot, keeping the hand's rotation.

The hand is at the pick's ``above`` pose, ``clearance`` over where it gripped,
so the drop height is read off it rather than off a frame: set the cube down
where it was gripped, ``gap`` higher, and let it drop that last few millimetres
rather than press it into the table. ``x`` and ``y`` are the place spot in the
work-surface frame, which is the world's in x and y.
"""

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    over: dict
    drop: dict


def run(
    ctx: NodeContext, hand: dict, x: float = 0.04, y: float = 0.10, clearance: float = 0.12, gap: float = 0.004
) -> Output:
    h = hand["position"]
    at = lambda z: {"position": {"x": x, "y": y, "z": z}, "rotation": hand["rotation"]}  # noqa: E731
    return {"over": at(h["z"]), "drop": at(h["z"] - clearance + gap)}
