# SPDX-License-Identifier: Apache-2.0
"""The real cell behind the graph runtime: the UR5, the 2F-85, the Axia80 and the wrist D415.

:class:`RealFacade` is :class:`twin.MoveJFacade` with the hardware put where
the twin was:

* **Motion.** Each joint target the graph's IK produces becomes one UR
  ``moveJ``, checked first by ur5-VLA's ``SafetyLayer`` (the joint box the cube
  skill was proven in, and the workspace box swept along the path). While it
  runs, the Axia80 is read at 100 Hz and the arm is stopped (``stopJ``) on more
  than ``contact_n`` -- 25 N against the 11 to 13 N a normal scripted pick peaks
  at in the recorded runs -- or on any robot stop. The twin then mirrors the
  joints the arm measured.
* **Gripper.** Open and close go to the Robotiq over Modbus, and wait for it to
  stop. ``robot.get_gripper`` reads the hand, not the twin: the twin's cube is
  parked out of reach, so only the real jaws can say they hold something.
* **Seeing.** ``robot.get_observation``'s wrist block is the D415's latest frame
  taken after the arm came to rest, posed by the measured joints through UR's DH
  and the hand-eye solve -- not by the twin.

Everything device-side is ur5-VLA's own code (``RTDEArm``, ``RobotiqGripper``,
``AxiaFTSensor``, ``RealSenseCamera``, ``HandEye``, ``SafetyLayer``), imported
from its checkout, so this drives the cell through the plumbing that collected
its datasets.

``dry_run`` connects the arm receive-only and the gripper read-only: nothing
moves, every motion is safety-checked and printed, the twin carries it out, and
the camera is the real one at the real arm's pose.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cell
import numpy as np
from twin import MoveJFacade

log = logging.getLogger("ur5_cell.real")

CONTACT_N = 25.0
"""Stop a motion when the tared Axia80 reads more than this [N]. See the module docstring."""

SETTLE_S = 0.25
"""After a motion, a frame is used only if captured this long after the arm stopped [s]."""

GRIP_TIMEOUT_S = 3.0
"""Longest a 2F-85 open or close may take before it is called stuck [s]. A full stroke is 0.6 s at speed 1."""


def import_ur5vla(root: Path):
    """Put ur5-VLA's ``src`` on the path and return its modules this file uses."""
    src = Path(root).expanduser().resolve() / "src"
    if not (src / "ur5vla").is_dir():
        raise FileNotFoundError(f"no ur5-VLA checkout at {root} (expected {src / 'ur5vla'})")
    sys.path.insert(0, str(src))
    from ur5vla import config
    from ur5vla.cameras import build_camera
    from ur5vla.control.safety import SafetyLayer, describe
    from ur5vla.perception.handeye import HandEye
    from ur5vla.robot.arm import RTDEArm
    from ur5vla.robot.axia80 import AxiaFTSensor
    from ur5vla.robot.kinematics import fk
    from ur5vla.robot.robotiq import RobotiqGripper

    return {
        "config": config,
        "build_camera": build_camera,
        "SafetyLayer": SafetyLayer,
        "describe": describe,
        "HandEye": HandEye,
        "RTDEArm": RTDEArm,
        "AxiaFTSensor": AxiaFTSensor,
        "fk": fk,
        "RobotiqGripper": RobotiqGripper,
    }


@dataclass
class Hardware:
    """The cell's devices, opened from ur5-VLA's own config.

    Args:
        root: The ur5-VLA checkout.
        config: Its config file, relative to ``root``. ``engineered.yaml`` carries
            the joint and workspace boxes the cube skill runs inside.
        dry_run: Receive-only arm and read-only gripper: nothing can move.
        use_ft: Open the Axia80. Without it there is no contact stop, so a run
            that moves the arm requires it unless this is False on purpose.
    """

    root: Path
    config: str = "configs/engineered.yaml"
    dry_run: bool = True
    use_ft: bool = True

    def open(self) -> Hardware:
        m = self.m = import_ur5vla(self.root)
        cfg = self.cfg = m["config"].load(str(Path(self.root) / self.config))
        self.fk = m["fk"]
        self.describe = m["describe"]
        self.safety = m["SafetyLayer"](cfg.safety, cfg.robot.period_s)

        handeye = m["HandEye"].load(Path(self.root) / "configs/calibration/wrist_handeye.yaml")
        self.T_flange_cam = np.asarray(handeye.T_flange_cam, dtype=np.float64)
        moved = float(np.linalg.norm(self.T_flange_cam[:3, 3] - cell.T_FLANGE_CAM[:3, 3]))
        if moved > 0.002:
            log.warning(
                "the hand-eye solve has moved %.1f mm from cell.T_FLANGE_CAM: the twin's camera is stale", 1000 * moved
            )

        self.arm = m["RTDEArm"](cfg.robot)
        self.arm.connect(receive_only=self.dry_run)
        self.gripper = m["RobotiqGripper"](cfg.gripper)
        try:
            self.gripper.connect(read_only=self.dry_run)
        except RuntimeError as exc:
            if not self.dry_run:
                raise
            # A dry run never commands the hand, so it can go on without one; say so.
            log.warning("no gripper (%s); the dry run goes on without it", exc)
            self.gripper = None
        if not self.dry_run:
            self.gripper.activate()
        self.ft = None
        if self.use_ft:
            self.ft = m["AxiaFTSensor"](cfg.ft_sensor)
            self.ft.connect()
            if not self.dry_run:
                # The sensor's own bias, so the contact stop reads the load from here on.
                # Not on a dry run: the bias lives in the sensor, which ur5-VLA tares for itself.
                self.ft.tare(0.5)
        self.camera = m["build_camera"]("wrist", dict(cfg.cameras["wrist"]))
        self.camera.start()
        deadline = time.monotonic() + 10.0
        while self.camera.latest() is None:
            if time.monotonic() > deadline:
                raise RuntimeError("the wrist D415 delivered no frame in 10 s")
            time.sleep(0.05)
        return self

    def close(self) -> None:
        for name in ("camera", "ft", "gripper", "arm"):
            device = getattr(self, name, None)
            if device is None:
                continue
            try:
                (device.stop if name == "camera" else device.close)()
            except Exception as exc:
                log.warning("closing the %s: %s", name, exc)

    # ------------------------------------------------------------ readings
    def q(self) -> np.ndarray:
        return np.asarray(self.arm.read_state().q, dtype=np.float64)

    def force(self) -> float | None:
        """The tared Axia80's force magnitude [N], or None without one."""
        if self.ft is None:
            return None
        w = self.ft.wrench()
        return None if w is None else float(np.linalg.norm(np.asarray(w)[:3]))

    def frame_after(self, t: float, timeout_s: float = 2.0):
        """The wrist camera's first frame captured after monotonic time ``t``."""
        deadline = time.monotonic() + timeout_s
        while True:
            frame = self.camera.latest()
            if frame is not None and frame.t_capture > t and frame.depth is not None:
                return frame
            if time.monotonic() > deadline:
                raise RuntimeError("the wrist D415 delivered no fresh depth frame")
            time.sleep(0.01)

    # ------------------------------------------------------------ motion
    def check(self, q_to: np.ndarray) -> None:
        """Raise unless a ``moveJ`` from here to ``q_to`` stays inside every box."""
        state = self.arm.read_state()
        if self.ft is not None:
            # The sensor's wrench, as ur5-VLA's loop merges it: else the force limits read the
            # CB3's joint-current estimate.
            state = dataclasses.replace(state, ft=self.ft.wrench(), ft_range=self.ft.rated_range())
        # A dry run's arm is receive-only, so no control script runs: check the robot itself.
        condition = self.safety.robot_condition if self.dry_run else self.safety.check_state
        violation = condition(state) or self.safety.check_command(q_to, q_to)
        violation = violation or self.safety.check_path(state.q, q_to)
        if violation is not None:
            raise RuntimeError(f"refused: {self.describe(violation)}")

    def move_j(self, q_to: np.ndarray, speed_deg_s: float, accel_deg_s2: float, contact_n: float) -> np.ndarray:
        """One guarded ``moveJ``; returns the measured joints at rest."""
        rtde = self.arm._c  # ur_rtde's control interface: the async moveJ and stopJ live there
        if not rtde.moveJ([float(v) for v in q_to], math.radians(speed_deg_s), math.radians(accel_deg_s2), True):
            raise RuntimeError("moveJ refused (robot stopped, or the target is outside its limits)")
        try:
            while rtde.getAsyncOperationProgress() >= 0:
                state = self.arm.read_state()
                if state.protective_stop or state.emergency_stop:
                    raise RuntimeError(f"the robot stopped (safety mode {state.safety_mode})")
                force = self.force()
                if force is not None and force > contact_n:
                    raise RuntimeError(f"contact: {force:.1f} N on the Axia80, over {contact_n:.0f} N")
                time.sleep(0.01)
        except BaseException:
            rtde.stopJ(math.radians(2 * accel_deg_s2))
            raise
        return self.q()

    def grip(self, closed: float) -> float:
        """Command the 2F-85 (0 open .. 1 closed) and wait for it to stop; returns the closed reading."""
        self.gripper.command(closed)
        start = time.monotonic()
        time.sleep(0.1)
        while time.monotonic() - start < GRIP_TIMEOUT_S:
            reading = self.gripper.reading()
            if reading.fault:
                raise RuntimeError(f"gripper fault: {reading.fault}")
            if reading.motion in ("at target", "contact while closing", "contact while opening", "idle"):
                return float(reading.position)
            time.sleep(0.02)
        raise RuntimeError("the gripper did not stop within its timeout")


class RealFacade(MoveJFacade):
    """The graph's env surface over the real cell, with the twin following the arm.

    Args:
        env, bus: As :class:`~robosimstudio_policy_agent.facade.GapEnvFacade`.
        hw: The open :class:`Hardware`.
        speed_deg_s, accel_deg_s2: The real ``moveJ``'s joint speed and acceleration.
        contact_n: The Axia80 contact stop [N].
        confirm: Called with a description before every real motion; return False to refuse it.
    """

    def __init__(
        self, env, bus, *, hw: Hardware, speed_deg_s=20.0, accel_deg_s2=40.0, contact_n=CONTACT_N, confirm=None
    ):
        super().__init__(env, bus)
        self.hw = hw
        self.real_speed = float(speed_deg_s)
        self.real_accel = float(accel_deg_s2)
        self.contact_n = float(contact_n)
        self.confirm = confirm
        self.rested_at = time.monotonic()
        base = np.eye(4)
        base[:3, 3] = -cell.WORK_IN_BASE + [0.0, 0.0, cell.SURFACE_Z]
        self.T_world_base = base

    # ------------------------------------------------------------ motion
    def move_to_joints_blocking(self, target, *, tolerance: float = 0.01, max_steps: int = 120, arm_id: int = 0):
        goal = np.asarray(target, dtype=np.float64).reshape(-1)[:6]
        self.hw.check(goal)
        here = self.hw.q()
        where = cell.base_to_work(self.hw.fk(goal)[:3, 3])
        what = (f"moveJ {np.degrees(np.max(np.abs(goal - here))):.1f} deg (largest joint) to flange at "
                f"work ({where[0]:+.3f}, {where[1]:+.3f}, {where[2]:+.3f}) m")  # fmt: skip
        if self.hw.dry_run:
            print(f"[dry run] would {what}")
            return super().move_to_joints_blocking(goal, tolerance=tolerance, max_steps=max_steps, arm_id=arm_id)
        if self.confirm is not None and not self.confirm(what):
            raise RuntimeError(f"operator refused: {what}")
        print(what)
        measured = self.hw.move_j(goal, self.real_speed, self.real_accel, self.contact_n)
        self.rested_at = time.monotonic()
        # The twin follows the arm, to the joints it measured rather than the ones it was sent.
        super().move_to_joints_blocking(measured, tolerance=0.002, max_steps=max_steps, arm_id=arm_id)
        return bool(np.max(np.abs(measured - goal)) < max(tolerance, 0.005))

    # ------------------------------------------------------------ gripper
    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        super()._set_gripper(fraction, arm_id)  # the twin's jaws, which robot.* reads the command back from
        if not self.hw.dry_run:
            closed = self.hw.grip(1.0 - float(fraction))
            print(f"gripper -> {1.0 - fraction:.2f} closed; stopped at {closed:.3f}")

    def set_grip(self, fraction: float, *, ramp_steps: int = 1, settle_steps: int = 0, arm_id: int = 0) -> None:
        """One motion, not a ramp: the Robotiq ramps for itself, and each step would wait on the serial line."""
        self._set_gripper(fraction, arm_id)
        for _ in range(int(settle_steps)):
            self.bus.call(self._step_once)

    # ------------------------------------------------------------ readings
    def _proprioception(self, obs=None):
        out = super()._proprioception(obs)
        if not self.hw.dry_run:
            # The real jaws, as GaP's open fraction: only they know whether a cube is between them.
            joints = np.asarray(out["robot_joint_pos_0"], dtype=np.float64).copy()
            joints[-1] = 1.0 - float(self.hw.gripper.position())
            out["robot_joint_pos_0"] = joints
        return out

    def _observation(self, *, refresh_cameras: bool = False, obs=None, body_q=None):
        out = super()._observation(refresh_cameras=refresh_cameras, obs=obs, body_q=body_q)
        if "wrist" not in out:
            return out
        frame = self.hw.frame_after(self.rested_at + SETTLE_S)
        q = self.hw.q()
        intr = self.hw.camera.intrinsics()
        from robosimstudio.core.math import quat_xyzw_to_wxyz, xform_from_matrix

        pose = xform_from_matrix(self.T_world_base @ self.hw.fk(q) @ self.hw.T_flange_cam)
        out["wrist"] = {
            "images": {"rgb": frame.rgb, "depth": frame.depth},
            "depth_data": frame.depth,
            "intrinsics": np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]]),
            "pose": np.concatenate([pose[:3], quat_xyzw_to_wxyz(pose[3:7])]),
        }
        return out
