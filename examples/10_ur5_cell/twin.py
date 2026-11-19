# SPDX-License-Identifier: Apache-2.0
"""How a graph's motions reach an arm, the same way in the sim run and the real one.

The graph runtime turns ``robot.go_to_pose`` into one IK solve and one
``move_to_joints_blocking`` on the facade when no Cartesian servo is attached
(``motion="plan"`` with no planner). That is a UR ``moveJ``: one joint target,
interpolated by the controller at a capped joint speed. :class:`MoveJFacade`
does exactly that in the twin, so the sim run executes the sequence of joint
targets the real arm will be sent, at the real arm's speed, and a graph that
passes here has been tried on the commands the hardware gets.

:class:`CellGraph` is :class:`~robosimstudio_policy_agent.SolutionGraph` with
that facade and that motion mode; ``real.py`` hands it a facade that drives the
hardware instead.
"""

from __future__ import annotations

import math

import cell
import numpy as np

from robosimstudio_policy_agent import SolutionGraph
from robosimstudio_policy_agent.facade import GapEnvFacade

MOVE_SPEED_DEG_S = 30.0
"""Fastest joint's speed in a move [deg/s]. ur5-VLA's ``robot.move_speed_deg_s`` is 20 for teleop
bring-up; the cube skill there runs at about this."""

MOVE_ACCEL_DEG_S2 = 60.0
"""Joint acceleration [deg/s^2] for the real ``moveJ``."""


def control_hz(env) -> float:
    """The environment's control rate [Hz]: one ``env.step`` is ``1 / control_hz`` s."""
    pipeline = getattr(env, "pipeline", None)
    hz = getattr(pipeline, "control_hz", None) or getattr(env, "control_hz", None)
    return float(hz) if hz else 50.0


def movej_waypoints(start, goal, hz: float, speed_deg_s: float = MOVE_SPEED_DEG_S) -> list[np.ndarray]:
    """A ``moveJ``'s joint path, one point per control step: every joint arrives together.

    Smoothstepped, so it starts and stops at rest, and timed so the joint that
    travels furthest peaks at ``1.5 * speed`` -- the mean speed of a smoothstep is
    two-thirds of its peak.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    travel = float(np.max(np.abs(goal - start))) if len(goal) else 0.0
    if travel < 1e-6:
        return []
    duration = travel / math.radians(speed_deg_s)
    n = max(2, int(math.ceil(duration * hz)))
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    s = s * s * (3.0 - 2.0 * s)
    return [start + (goal - start) * float(k) for k in s]


def check_joint_box(q) -> None:
    """Raise if ``q`` leaves the joint box the real cell's safety layer enforces.

    The same refusal the hardware gives, in the twin: without it the twin turned
    wrist 3 past the 63 degrees the wrist's cables allow, and passed.
    """
    deg = np.degrees(np.asarray(q, dtype=np.float64)[: len(cell.JOINTS)])
    low, high = np.asarray(cell.JOINT_MIN_DEG), np.asarray(cell.JOINT_MAX_DEG)
    out = (deg < low) | (deg > high)
    if out.any():
        j = int(np.argmax(out))
        raise RuntimeError(f"refused: {cell.JOINTS[j]} to {deg[j]:.1f} deg, outside [{low[j]:.0f}, {high[j]:.0f}]")


class MoveJFacade(GapEnvFacade):
    """The twin's facade: a joint target is reached by a ``moveJ``, not a step input.

    The runtime's own blocking move writes the goal once and waits, which on a
    2000 N m/rad servo is a lunge no UR controller would make. This walks the
    same straight line in joint space the controller interpolates, then settles.
    """

    speed_deg_s = MOVE_SPEED_DEG_S

    def measured_q(self, arm_id: int = 0) -> np.ndarray:
        """This arm's joints as the twin reads them [rad]."""
        joints = self.get_proprioception()[f"robot_joint_pos_{int(arm_id)}"]
        return np.asarray(joints[:-1], dtype=np.float64)

    def move_to_joints_blocking(self, target, *, tolerance: float = 0.01, max_steps: int = 120, arm_id: int = 0):
        goal = np.asarray(target, dtype=np.float64).reshape(-1)
        check_joint_box(goal)
        if max_steps == 0:  # the runtime's fire-and-forget contract
            return super().move_to_joints_blocking(goal, tolerance=tolerance, max_steps=0, arm_id=arm_id)
        for q in movej_waypoints(self.measured_q(arm_id), goal, control_hz(self.env), self.speed_deg_s):
            self.bus.call(self.apply_joint_action, q, arm_id)
        return super().move_to_joints_blocking(
            goal, tolerance=tolerance, max_steps=max(int(max_steps), 120), arm_id=arm_id
        )


class CellGraph(SolutionGraph):
    """A graph run whose motions go through ``facade_cls``, one ``moveJ`` per pose.

    Args:
        env: The environment.
        facade_cls: Built as ``facade_cls(env, bus)``; :class:`MoveJFacade` for the
            sim run, ``real.RealFacade`` (partially applied) for the hardware.
        **kwargs: As :class:`~robosimstudio_policy_agent.SolutionGraph`; ``motion``
            is fixed to ``"plan"`` so ``go_to_pose`` reaches the facade as one target.
    """

    def __init__(self, env, *, facade_cls=MoveJFacade, **kwargs) -> None:
        self.facade_cls = facade_cls
        self.facade = None
        kwargs["motion"] = "plan"
        super().__init__(env, **kwargs)

    def _new_connector(self):
        from robosimstudio_policy_agent.connector import make_connector

        self.facade = self.facade_cls(self.env, self.bus)
        return make_connector(
            self.env,
            bus=self.bus,
            privileged=self.privileged,
            task_tools=self.task_tools,
            suite_tools=self.suite_tools,
            roll=self.roll,
            motion=self.motion,
            facade=self.facade,
            capture_spans=self.tool_spans,
            skills=self.skills,
        )
