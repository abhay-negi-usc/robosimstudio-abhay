# SPDX-License-Identifier: Apache-2.0
"""UR5 cell: pick and place, on the real arm

The graph ``00_sim_pick_place.py`` passes in the twin, run on the hardware: the
UR5 over ur_rtde, the 2F-85 over Modbus, the Axia80 as a 25 N contact stop and
the wrist D415 for the cube. Device code is ur5-VLA's own, imported from its
checkout (``--ur5vla``, beside this repository by default).

**It moves nothing unless asked.** Without ``--execute`` the arm is read, never
commanded: the graph sees the real cube through the real camera at the real
arm's pose, every ``moveJ`` it would send is safety-checked and printed, and the
twin carries it out (the twin's hand closes on air there, so the dry run ends at
the grip check -- that is expected). With ``--execute`` it asks before the first
motion, and before every one unless ``--yes``. Keep the pendant's stop in reach.

Before running: the arm powered with brakes released and no External Control
program running, the cube inside ur5-VLA's search patch, nothing else on the
table near it, and Axia80Monitor closed (it holds the sensor's stream). Needs the
`[gap]` extra, and ur_rtde, pyrealsense2, minimalmodbus, pyserial and pyyaml
beside it (the README's command adds them with ``--with``).
"""

import argparse
import functools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cell  # registers the cell, the robot, the camera and the cube
from real import CONTACT_N, Hardware, RealFacade
from twin import CellGraph

import robosimstudio as rss
import robosimstudio_bench as bench


def ask(what: str) -> bool:
    return input(f"{what}\n  move? [y/N] ").strip().lower() == "y"


def graph_outcome(trace_dir: Path) -> str:
    """``placed`` if the place subgraph finished, else the first error the graph recorded."""
    for path in sorted(trace_dir.glob("*/dag_trace.json")):
        events = json.loads(path.read_text()).get("events", [])
        if any(e.get("node_name") == "place_sg.placed" and e.get("status") == "ok" for e in events):
            return "placed"
        errors = [e["detail"].get("error", "") for e in events if e.get("event_type") == "error_recorded"]
        if errors:
            return f"failed: {errors[0]}"
    return "no trace"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ur5vla", type=Path, default=HERE.parents[2] / "ur5-VLA", help="the ur5-VLA checkout")
    parser.add_argument("--config", default="configs/engineered.yaml", help="ur5-VLA config, relative to --ur5vla")
    parser.add_argument("--execute", action="store_true", help="move the real arm (default: a dry run)")
    parser.add_argument("--yes", action="store_true", help="with --execute: ask once, not before every motion")
    parser.add_argument("--speed", type=float, default=20.0, help="moveJ joint speed [deg/s]")
    parser.add_argument("--contact-n", type=float, default=CONTACT_N, help="Axia80 contact stop [N]")
    parser.add_argument("--no-ft", action="store_true", help="run without the Axia80 (dry run only)")
    parser.add_argument("--no-view", action="store_true", help="exit after the episode instead of opening the page")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    if args.execute and args.no_ft:
        parser.error("--execute needs the Axia80: it is the contact stop")

    # The twin's cube is parked out of reach: the real cube is the one picked.
    cell.register_cube(at=cell.PARK_AT)
    task = cell.TASK.replace("(:max-steps 6000)", "(:max-steps 200000)")
    hw = Hardware(args.ur5vla, config=args.config, dry_run=not args.execute, use_ft=not args.no_ft).open()
    try:
        q = hw.q()
        print(
            f"arm at {np.round(np.degrees(q), 1).tolist()} deg; "
            f"gripper {'absent' if hw.gripper is None else f'{hw.gripper.position():.2f} closed'}; force {hw.force()} N"
        )
        if args.execute and input("The real arm will move. Type 'go' to start: ").strip() != "go":
            print("not started")
            return
        env = bench.Env(bench.task(task), renderer="auto")
        facade_cls = functools.partial(
            RealFacade,
            hw=hw,
            speed_deg_s=args.speed,
            contact_n=args.contact_n,
            confirm=None if args.yes else ask,
        )
        trace = Path("outputs/ur5_cell") / time.strftime("%Y%m%d-%H%M%S")
        policy = CellGraph(env, graph="pick_cube", privileged=False, facade_cls=facade_cls, trace_dir=str(trace))
        bench.rollout(env, policy)
        print(f"graph: {graph_outcome(trace)}\ntrace in {trace}")
    finally:
        hw.close()

    if not args.no_view:
        rss.view(env)


if __name__ == "__main__":
    main()
