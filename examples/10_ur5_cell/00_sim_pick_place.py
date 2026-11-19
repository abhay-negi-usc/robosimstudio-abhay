# SPDX-License-Identifier: Apache-2.0
"""UR5 cell: pick and place, in the twin

The real cell's CB3 UR5, Axia80, wrist D415 and 2F-85 pick a 60 mm black cube
and set it down on the place spot, scored as the benchmark scores it.

``pick_cube/`` sees through the wrist camera alone (``privileged=False``: no
``sim.*`` tool), and every pose it asks for reaches the arm as one ``moveJ`` at
the speed the real arm is driven at (``twin.MoveJFacade``). So a pass here has
been tried on the commands ``run_real.py`` sends the hardware. The
cell's numbers -- the hand-eye calibration, the camera's intrinsics, the gripper
mount, the table under the base -- are in ``cell.py`` with where each was
measured. Needs the `[gap]` extra.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cell  # registers the cell, the robot, the camera and the cube
from twin import CellGraph

import robosimstudio as rss
import robosimstudio_bench as bench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--at", type=float, nargs=2, default=cell.CUBE_AT[:2], metavar=("X", "Y"),
                        help="where the cube starts, work-surface frame [m]")  # fmt: skip
    parser.add_argument("--yaw", type=float, default=0.0, help="the cube's starting turn about up [deg]")
    parser.add_argument("--no-view", action="store_true", help="exit after the episode instead of opening the page")
    args = parser.parse_args()

    cell.register_cube(at=(args.at[0], args.at[1], cell.CUBE_HALF), yaw_deg=args.yaw)
    env = bench.Env(bench.task(cell.TASK), renderer="auto")
    result = bench.rollout(env, CellGraph(env, graph="pick_cube", privileged=False))
    print(f"success={result.success} after {result.steps} steps")

    if not args.no_view:
        rss.view(env)


if __name__ == "__main__":
    main()
