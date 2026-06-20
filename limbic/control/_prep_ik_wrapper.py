"""Day 1: ikpy-backed kinematics for bronny (SO-101), shaped like LeRobot's API.

Why this file exists
--------------------
LeRobot 0.5.1 ships a clean Cartesian solver, `lerobot.model.kinematics.RobotKinematics`,
but its `placo` backend has NO Windows ARM64 wheel and we are not compiling it
(same risk class as the rejected MuJoCo build). ikpy is pure Python and is the
only solver that runs on this Snapdragon/Windows machine. It has already been
validated on the real URDF (FK->IK round-trip = 0.00 mm).

`SO101Kinematics` below mirrors the installed LeRobot `RobotKinematics` interface
so placo can be dropped in later (e.g. on an x64 box) with a near one-line change:

    LeRobot 0.5.1 RobotKinematics:
      __init__(self, urdf_path, target_frame_name="gripper_frame_link", joint_names=None)
      forward_kinematics(self, joint_pos_deg)                         -> 4x4 (meters)
      inverse_kinematics(self, current_joint_pos, desired_ee_pose,
                         position_weight=1.0, orientation_weight=0.01) -> joint_deg

All public methods speak DEGREES for joints and METERS for Cartesian poses, just
like RobotKinematics. The two things ikpy needs that LeRobot hides internally are
also hidden here:
  1. ikpy works in RADIANS  -> converted inside every method.
  2. ikpy's joint frames differ from bronny's by a per-joint sign+offset
     (bronny_deg = sign * ikpy_deg + offset) -> applied inside every method,
     so callers always get values ready for SO101Follower.send_action.

This module does KINEMATICS ONLY. It never opens the serial port; joint commands
go out through the existing, working SO101Follower on COM7.
"""

from __future__ import annotations

import numpy as np
from ikpy.chain import Chain

# --------------------------------------------------------------------------- #
# Chain definition (validated — do not re-derive; see CLAUDE.md Section 2).
# --------------------------------------------------------------------------- #
from pathlib import Path as _P
URDF_PATH = str(_P(__file__).resolve().parents[2] / "assets" / "so101" / "so101_new_calib.urdf")
TARGET_FRAME_NAME = "gripper_frame_link"  # tool point used for grasping

# base + 5 revolute arm joints + fixed gripper_frame. base[0] and gripper_frame[6]
# are masked out; the 5 active joints (base->tip) are the arm joints below.
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, False]

# The 5 IK-active joints in chain order (base -> tip). The `gripper` jaw revolute
# is deliberately NOT part of the IK chain.
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
N_ARM_JOINTS = len(ARM_JOINT_NAMES)

# --------------------------------------------------------------------------- #
# ikpy <-> bronny frame mapping:  bronny_deg = sign * ikpy_deg + offset
# (per joint, same order as ARM_JOINT_NAMES = [pan, lift, elbow, wrist_flex, wrist_roll]).
#
# MEASURED on bronny (Day 1 Step C). Method: drove the arm to the centered pose
# (all bronny ~0), measured elbow/wrist/tip height + horizontal-from-pan-axis with
# a ruler, least-squares fit the ikpy flex angles to those positions, then resolved
# each joint's sign with one small single-joint move (tip up = +1 / tip down = -1;
# shoulder_pan swing right = +1).
#   * shoulder_pan offset = 0 BY DEFINITION (bronny 0 == ikpy 0 sets our azimuth
#     reference; note ikpy "forward" at pan 0 is the base -X axis).
#   * wrist_roll is NOT yet calibrated (provisional sign +1 / offset 0). It barely
#     affects tip POSITION but DOES set gripper roll ORIENTATION, so it must be
#     measured before relying on top-down grasp orientation.  TODO: calibrate wrist_roll.
# --------------------------------------------------------------------------- #
IKPY_TO_BRONNY_SIGN: np.ndarray | None = np.array([+1, -1, -1, -1, +1], dtype=float)
IKPY_TO_BRONNY_OFFSET: np.ndarray | None = np.array([0.0, -90.5, -81.5, -7.6, 0.0], dtype=float)

# Height of the base_link origin above the WORK TABLE (where objects rest), so the
# table plane is z = -BASE_HEIGHT_ABOVE_TABLE_MM/1000 in the base frame (meters).
# Originally 83.4 mm (Day 1). RE-MEASURED 2026-06-16 after the camera/mount redesign
# shifted the base height: commanded centerline z=50 mm, ruler-measured tip = 70 mm
# (a constant +20 mm too high everywhere) -> +20 mm. So table-frame z reads height
# above the WORK table; re-measure with z_check.py if the setup is reworked again.
BASE_HEIGHT_ABOVE_TABLE_MM: float = 103.4

# Default wrist_roll (bronny deg) held during top-down grasps. wrist_roll is not
# yet calibrated, and letting the solver drive it to extremes both perturbs the
# (offset-from-axis) tip AND amplifies the bad roll calibration. Pinning it to the
# middle (0) keeps grasps accurate; the 4 well-calibrated joints do the reaching.
DEFAULT_WRIST_ROLL_DEG: float = 0.0


class SO101Kinematics:
    """ikpy FK/IK for the SO-101, mirroring LeRobot 0.5.1 `RobotKinematics`.

    Args:
        urdf_path: path to so101_new_calib.urdf.
        target_frame_name: tip frame for FK/IK (default the grasp tool point).
        joint_names: arm joint names in chain order; defaults to ARM_JOINT_NAMES.

    Joint conventions for the public methods:
        * inputs/outputs are in DEGREES
        * if the frame mapping is calibrated, they are in BRONNY convention
          (ready for SO101Follower.send_action); if not, they are RAW IKPY
          degrees and must not drive the arm (see `is_calibrated`).
        * a trailing gripper value is allowed and passed through untouched,
          matching how RobotKinematics handles extra joints.
    """

    def __init__(
        self,
        urdf_path: str = URDF_PATH,
        target_frame_name: str = TARGET_FRAME_NAME,
        joint_names: list[str] | None = None,
    ):
        self.urdf_path = urdf_path
        self.target_frame_name = target_frame_name
        self.joint_names = list(joint_names) if joint_names is not None else list(ARM_JOINT_NAMES)

        self.chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=["base_link"],
            active_links_mask=ACTIVE_LINKS_MASK,
        )
        # ikpy operates on the full link vector (length = number of chain links).
        self._n_links = len(self.chain.links)
        # chain indices of the active arm joints, in order (1..5 for this URDF).
        self._active_idx = [i for i, a in enumerate(self.chain.active_links_mask) if a]

        # No-roll chain for top-down grasps: wrist_roll is masked out (held fixed),
        # so pan/lift/elbow/wrist_flex solve position + pointing-down and the
        # uncalibrated roll can't perturb the tip. See DEFAULT_WRIST_ROLL_DEG.
        noroll_mask = [False, True, True, True, True, False, False]
        self.chain_noroll = Chain.from_urdf_file(
            urdf_path, base_elements=["base_link"], active_links_mask=noroll_mask,
        )
        self._wroll_chain_idx = self._active_idx[4]  # chain link index of wrist_roll

        # Relax the solver's joint bounds to the motors' REAL reachable ranges
        # (bronny soft limits). The URDF's symmetric limits, combined with our
        # measured frame offset, otherwise cap the solver several degrees short of
        # what the arm can physically reach when stretching low (shoulder_lift /
        # elbow especially). Real commands are still clamped to the soft limits by
        # the controller, so this only unlocks reachable poses.
        self._apply_soft_limit_bounds(self.chain)
        self._apply_soft_limit_bounds(self.chain_noroll)

    def _apply_soft_limit_bounds(self, chain) -> None:
        from .safety import JOINT_SOFT_LIMITS
        if IKPY_TO_BRONNY_SIGN is None or IKPY_TO_BRONNY_OFFSET is None:
            return
        for slot, chain_i in enumerate(self._active_idx):
            lo_b, hi_b = JOINT_SOFT_LIMITS[ARM_JOINT_NAMES[slot]]
            s, o = IKPY_TO_BRONNY_SIGN[slot], IKPY_TO_BRONNY_OFFSET[slot]
            ik1, ik2 = s * (lo_b - o), s * (hi_b - o)
            chain.links[chain_i].bounds = (np.radians(min(ik1, ik2)), np.radians(max(ik1, ik2)))

    # ------------------------------------------------------------------ #
    # Frame-offset handling (bronny <-> ikpy), all in degrees.
    # ------------------------------------------------------------------ #
    @property
    def is_calibrated(self) -> bool:
        """True once the ikpy<->bronny sign/offset have been measured & set."""
        return IKPY_TO_BRONNY_SIGN is not None and IKPY_TO_BRONNY_OFFSET is not None

    @staticmethod
    def _ikpy_to_bronny_deg(ikpy_deg: np.ndarray) -> np.ndarray:
        """ikpy arm degrees -> bronny arm degrees. Identity if uncalibrated."""
        if IKPY_TO_BRONNY_SIGN is None or IKPY_TO_BRONNY_OFFSET is None:
            return np.asarray(ikpy_deg, dtype=float)
        return IKPY_TO_BRONNY_SIGN * np.asarray(ikpy_deg, dtype=float) + IKPY_TO_BRONNY_OFFSET

    @staticmethod
    def _bronny_to_ikpy_deg(bronny_deg: np.ndarray) -> np.ndarray:
        """bronny arm degrees -> ikpy arm degrees. Identity if uncalibrated."""
        if IKPY_TO_BRONNY_SIGN is None or IKPY_TO_BRONNY_OFFSET is None:
            return np.asarray(bronny_deg, dtype=float)
        return (np.asarray(bronny_deg, dtype=float) - IKPY_TO_BRONNY_OFFSET) / IKPY_TO_BRONNY_SIGN

    # ------------------------------------------------------------------ #
    # Internal: pack 5 arm joints (bronny deg) -> full ikpy link vector (rad).
    # ------------------------------------------------------------------ #
    def _bronny_deg_to_ikpy_vector(self, bronny_arm_deg: np.ndarray) -> np.ndarray:
        ikpy_arm_deg = self._bronny_to_ikpy_deg(bronny_arm_deg[:N_ARM_JOINTS])
        ikpy_arm_rad = np.deg2rad(ikpy_arm_deg)
        vec = np.zeros(self._n_links, dtype=float)
        for slot, chain_i in enumerate(self._active_idx):
            vec[chain_i] = ikpy_arm_rad[slot]
        return vec

    def _ikpy_vector_to_bronny_deg(self, ikpy_vector_rad: np.ndarray) -> np.ndarray:
        ikpy_arm_rad = np.array([ikpy_vector_rad[i] for i in self._active_idx], dtype=float)
        ikpy_arm_deg = np.rad2deg(ikpy_arm_rad)
        return self._ikpy_to_bronny_deg(ikpy_arm_deg)

    @staticmethod
    def _clip_seed_to_bounds(seed: np.ndarray, chain) -> np.ndarray:
        """Clip the IK seed into each link's bounds (scipy rejects out-of-bounds seeds)."""
        out = np.asarray(seed, dtype=float).copy()
        for i, link in enumerate(chain.links):
            b = getattr(link, "bounds", None)
            if b and b[0] is not None and b[1] is not None:
                out[i] = min(max(out[i], b[0] + 1e-4), b[1] - 1e-4)
        return out

    # ------------------------------------------------------------------ #
    # Public API — mirrors LeRobot 0.5.1 RobotKinematics.
    # ------------------------------------------------------------------ #
    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """FK: arm joint degrees (bronny convention) -> 4x4 tip pose in meters.

        Mirrors RobotKinematics.forward_kinematics. Accepts a trailing gripper
        value (uses only the first 5 arm joints), like the LeRobot version.
        """
        joint_pos_deg = np.asarray(joint_pos_deg, dtype=float)
        vec = self._bronny_deg_to_ikpy_vector(joint_pos_deg)
        return self.chain.forward_kinematics(vec)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        *,
        top_down: bool = False,
        fixed_roll_deg: float = DEFAULT_WRIST_ROLL_DEG,
    ) -> np.ndarray:
        """IK: 4x4 target pose (meters) -> arm joint degrees (bronny convention).

        Mirrors RobotKinematics.inverse_kinematics(current_joint_pos,
        desired_ee_pose, position_weight, orientation_weight). `current_joint_pos`
        seeds the solver (degrees) and any trailing gripper value is preserved in
        the output, matching the LeRobot version.

        ikpy uses an orientation MODE rather than position/orientation weights, so
        the two weight args are accepted for signature-compatibility and mapped to
        the closest ikpy behavior:
          * top_down=True  -> constrain the tip's Z axis to point straight down at
            the table (orientation_mode="Z", target [0,0,-1]); the grasp mode.
          * else, if orientation_weight > 0 -> full-pose match (orientation_mode
            "all") using the rotation in desired_ee_pose.
          * else (orientation_weight == 0) -> position-only IK.
        """
        current_joint_pos = np.asarray(current_joint_pos, dtype=float)
        desired_ee_pose = np.asarray(desired_ee_pose, dtype=float)

        seed = self._bronny_deg_to_ikpy_vector(current_joint_pos)
        target_xyz = desired_ee_pose[:3, 3]

        if top_down:
            # Pin wrist_roll at `fixed_roll_deg` (bronny) and solve the other 4
            # joints for position + tip-Z-down, using the no-roll chain.
            roll_ikpy_deg = self._bronny_to_ikpy_deg(
                np.array([0.0, 0.0, 0.0, 0.0, float(fixed_roll_deg)]))[4]
            seed[self._wroll_chain_idx] = np.deg2rad(roll_ikpy_deg)
            seed = self._clip_seed_to_bounds(seed, self.chain_noroll)
            ik_solution = self.chain_noroll.inverse_kinematics(
                target_position=target_xyz,
                target_orientation=[0, 0, -1],
                orientation_mode="Z",
                initial_position=seed,
            )
        elif orientation_weight > 0:
            seed = self._clip_seed_to_bounds(seed, self.chain)
            ik_solution = self.chain.inverse_kinematics(
                target_position=target_xyz,
                target_orientation=desired_ee_pose[:3, :3],
                orientation_mode="all",
                initial_position=seed,
            )
        else:
            seed = self._clip_seed_to_bounds(seed, self.chain)
            ik_solution = self.chain.inverse_kinematics(
                target_position=target_xyz,
                initial_position=seed,
            )

        arm_bronny_deg = self._ikpy_vector_to_bronny_deg(ik_solution)

        # Preserve trailing (gripper) joints, like RobotKinematics does.
        if len(current_joint_pos) > N_ARM_JOINTS:
            result = np.empty_like(current_joint_pos)
            result[:N_ARM_JOINTS] = arm_bronny_deg
            result[N_ARM_JOINTS:] = current_joint_pos[N_ARM_JOINTS:]
            return result
        return arm_bronny_deg


# --------------------------------------------------------------------------- #
# Software-only validation: FK -> IK -> FK round-trip (no arm, no offset needed).
# Runs in raw-ikpy mode, which is exactly what proves the chain + interface.
# --------------------------------------------------------------------------- #
def _roundtrip_test() -> float:
    kin = SO101Kinematics()

    print(f"URDF: {kin.urdf_path}")
    print(f"Tip frame: {kin.target_frame_name}")
    print(f"Chain links: {kin._n_links}, active arm-joint indices: {kin._active_idx}")
    print(f"is_calibrated (ikpy<->bronny offset measured?): {kin.is_calibrated}")
    print(f"Arm joints: {kin.joint_names}\n")

    # Arbitrary, well-within-limits arm pose in degrees (raw ikpy convention).
    arm_deg = np.array([10.0, 20.0, -25.0, 15.0, 30.0], dtype=float)

    pose = kin.forward_kinematics(arm_deg)
    target_xyz = pose[:3, 3]
    print("Test arm joints (deg):", arm_deg)
    print("FK tip position (m):  ", np.round(target_xyz, 5))

    ik_deg = kin.inverse_kinematics(
        current_joint_pos=np.zeros(N_ARM_JOINTS),
        desired_ee_pose=pose,
        orientation_weight=0.01,
    )
    pose_check = kin.forward_kinematics(ik_deg)
    xyz_check = pose_check[:3, 3]

    err_mm = float(np.linalg.norm(xyz_check - target_xyz) * 1000.0)
    print("IK solution (deg):    ", np.round(ik_deg, 4))
    print("FK of IK (m):         ", np.round(xyz_check, 5))
    print(f"\nFK->IK->FK round-trip position error: {err_mm:.4f} mm")
    return err_mm


if __name__ == "__main__":
    _roundtrip_test()
