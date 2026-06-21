"""mink (MuJoCo) reaching IK for the SO-101 — a drop-in for the closed-form solver.

Why this exists
---------------
The deterministic closed-form planar solver (``_prep_planar_ik``) is the original
reaching engine, but it was unreliable on this rig. ``mink`` is an open-source
differential-IK library (https://github.com/kevinzakka/mink) on top of MuJoCo;
both ship as prebuilt cp313 win_amd64 wheels (no compiler needed, unlike the
placo/box2ai route). It is verified on THIS machine: FK->IK round-trip 0.0008 mm
and MuJoCo-vs-ikpy frame agreement 0.0000 mm.

What it does
------------
``MinkSO101IK.solve(x, y, z, pitch_deg)`` takes a TABLE-frame target (mm) plus an
approach pitch and returns the 5 arm joints in this arm's degree convention
(``ACTIVE_JOINTS`` order), or ``None`` if it can't reach it. That is exactly the
contract of ``_prep_planar_ik.PlanarSO101IK.solve``, so it slots straight into
``kinematics.solve_ik`` with no other changes — the workspace clamp, soft-limit
clamp, and empirical accuracy correction all stay where they are.

Frames & units (shared, verified):
  * MuJoCo model frame == ikpy/URDF base frame, so ``calibration.table_to_ikpy_m``
    converts the table target and ``calibration.ikpy_to_arm_deg`` converts the
    solved joint radians to arm degrees — the SAME measured mappings ikpy uses.
  * The IK tip is a ``site`` re-added at the URDF ``gripper_frame_link`` tool
    point (MuJoCo fuses that fixed frame away on import, so we restore it).

Heavy deps (``mujoco``, ``mink``) are imported lazily inside the class, so
importing this module — or ``limbic.control`` — never requires them.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from . import calibration
from .ik_chain import ACTIVE_JOINTS, URDF_PATH

# Tool-point of the IK tip = the URDF ``gripper_frame_joint`` origin relative to
# ``gripper_link`` (xyz metres, rpy = (0, pi, 0) -> quat (w,x,y,z) below).
_TCP_POS = (-0.0079, -0.000218121, -0.0981274)
_TCP_QUAT = (0.0, 0.0, 1.0, 0.0)  # 180 deg about Y

# Deterministic seed pose (URDF radians, ACTIVE_JOINTS order): a forward top-down
# crouch so the iterative solve settles to the correct front-reaching branch every
# call (without it, mink flips the base ~180 deg and reaches backward). Found by a
# seed sweep that maximised in-soft-limit convergence across the workspace.
# ``shoulder_pan`` is overridden per-solve to the target azimuth (see ``solve``).
_SEED_RAD = {"shoulder_pan": 0.0, "shoulder_lift": -0.3, "elbow_flex": -1.2,
             "wrist_flex": -0.4, "wrist_roll": 0.0}

# wrist_roll is LOCKED at this fixed arm-degree angle (the rig's chosen gripper
# jaw orientation). With roll pinned, the remaining 4 joints can't also satisfy a
# full down-orientation, so the solve runs POSITION-ONLY -- which is what we want
# here, and as a bonus the gripper tilts naturally as it reaches farther out.
# Override with $LIMBIC_WRIST_ROLL_DEG. (wrist_flex / tilt stays free.)
LOCKED_WRIST_ROLL_DEG = float(os.environ.get("LIMBIC_WRIST_ROLL_DEG", "-6.46"))

# Accept tolerance for "reached the point". Mid-workspace solves land < 2 mm;
# the looser 8 mm bound only matters in the close-in fold zone (<~130 mm reach).
# Genuinely out-of-reach targets stay above this and the caller clamps inward.
_POS_TOL_MM = 8.0
_MAX_ITERS = 400
_DT = 1.0


def _mesh_free_urdf() -> str:
    """Write a copy of the SO-101 URDF with all mesh geoms stripped.

    MuJoCo needs only the joint tree for IK, not the STL meshes (which aren't
    bundled). Stripping ``<visual>``/``<collision>`` leaves valid bodies (each
    link keeps its ``<inertial>``), so the model compiles with zero extra files.
    """
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    out = os.path.join(tempfile.gettempdir(), "limbic_so101_nomesh.urdf")
    tree.write(out)
    return out


class MinkSO101IK:
    """Differential-IK reaching solver for the SO-101, mink/MuJoCo backed."""

    def __init__(self, orientation_cost: float = 0.0):
        import mujoco  # lazy: heavy, optional dep
        import mink

        self._mujoco = mujoco
        self._mink = mink

        spec = mujoco.MjSpec.from_file(_mesh_free_urdf())
        site = spec.body("gripper_link").add_site()
        site.name = "tcp"
        site.pos = list(_TCP_POS)
        site.quat = list(_TCP_QUAT)
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)

        names = [mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(self._model.njnt)]
        self._qadr = {n: int(self._model.jnt_qposadr[i]) for i, n in enumerate(names)}
        self._tcp_sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        # Enforce THIS rig's soft limits (safety.py) inside the solve, converted
        # arm-degrees -> URDF radians, so mink only returns in-envelope poses
        # (e.g. shoulder_pan stays within +-80 deg, the camera guard).
        from .safety import JOINT_SOFT_LIMITS
        los = calibration.arm_to_ikpy_rad({j: JOINT_SOFT_LIMITS[j][0] for j in ACTIVE_JOINTS})
        his = calibration.arm_to_ikpy_rad({j: JOINT_SOFT_LIMITS[j][1] for j in ACTIVE_JOINTS})
        # wrist_roll locked angle in URDF radians (pinned hard in the model below).
        zeros = {j: 0.0 for j in ACTIVE_JOINTS}
        self._roll_ik = calibration.arm_to_ikpy_rad(
            {**zeros, "wrist_roll": LOCKED_WRIST_ROLL_DEG})["wrist_roll"]
        for j in ACTIVE_JOINTS:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = sorted((los[j], his[j]))
            if j == "wrist_roll":
                # Pin roll at the locked angle -> the solve is position-only on the
                # other 4 joints (full down-orientation isn't simultaneously
                # satisfiable with roll fixed; the gripper tilts naturally instead).
                lo, hi = self._roll_ik - 1e-3, self._roll_ik + 1e-3
            self._model.jnt_range[jid] = [lo, hi]
            self._model.jnt_limited[jid] = 1
            if j == "shoulder_pan":
                self._pan_lo, self._pan_hi = lo, hi

        self._cfg = mink.Configuration(self._model)
        self._task = mink.FrameTask(
            frame_name="tcp", frame_type="site",
            position_cost=1.0, orientation_cost=float(orientation_cost), lm_damping=1.0,
        )
        self._limits = [mink.ConfigurationLimit(self._model)]
        self._seed = self._q_from_active(_SEED_RAD)
        for j in ACTIVE_JOINTS:  # keep the seed strictly inside the soft limits
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = self._model.jnt_range[jid]
            a = self._qadr[j]
            self._seed[a] = min(max(self._seed[a], lo + 1e-4), hi - 1e-4)

        # Self-calibrate the tool's APPROACH AXIS (which way the gripper points) from
        # the model, so achieved-pitch read-back needs no hardcoded axis convention.
        # At the downward seed crouch the approach axis is the site's local axis that
        # points most vertically; its sign is fixed so it points DOWN (z < 0) there
        # (=> pitch -90 reads "straight down"). Used by :meth:`tool_pitch_deg`.
        self._data.qpos[:] = self._seed
        mujoco.mj_kinematics(self._model, self._data)
        m = self._data.site_xmat[self._tcp_sid]  # 9, row-major 3x3; columns = local axes in world
        cols = [np.array([m[c], m[3 + c], m[6 + c]]) for c in range(3)]
        self._approach_col = int(np.argmax([abs(v[2]) for v in cols]))
        self._approach_sign = -1.0 if cols[self._approach_col][2] > 0 else 1.0

    def _q_from_active(self, active_rad: dict[str, float]) -> np.ndarray:
        q = np.zeros(self._model.nq)
        for j, v in active_rad.items():
            q[self._qadr[j]] = v
        return q

    def solve(self, x: float, y: float, z: float, pitch_deg: float,
              elbow_up: bool = True) -> np.ndarray | None:
        """Table-frame (x, y, z) mm + approach pitch (deg) -> arm degrees[5] or None.

        Return order is ``ACTIVE_JOINTS``. ``None`` means the iterative solve didn't
        reach the point within tolerance (caller clamps inward). ``elbow_up`` is
        accepted for signature parity with the closed-form solver.
        """
        mink = self._mink
        bx, by, bz = calibration.table_to_ikpy_m(x, y, z)
        # Position-only target (orientation_cost=0): with wrist_roll pinned we drive
        # the tip TO the point and let the approach angle fall out of the geometry.
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.identity(), np.array([bx, by, bz]))
        self._task.set_target(target)

        # Seed shoulder_pan at the target azimuth so the solve settles on the
        # front-reaching branch (and the +-80 deg pan limit blocks the 180-flip).
        seed = self._seed.copy()
        pan_rad = float(np.clip(-np.arctan2(y, x), self._pan_lo, self._pan_hi))
        seed[self._qadr["shoulder_pan"]] = pan_rad
        self._cfg.update(seed)
        for _ in range(_MAX_ITERS):
            vel = mink.solve_ik(self._cfg, [self._task], _DT, "daqp", 1e-3,
                                limits=self._limits)
            self._cfg.integrate_inplace(vel, _DT)
            err = np.linalg.norm(
                self._cfg.get_transform_frame_to_world("tcp", "site").translation()
                - np.array([bx, by, bz]))
            if err < _POS_TOL_MM / 1000.0:
                break
        else:
            return None  # never converged within tolerance

        # wrist_roll comes back at its pinned locked value; the other 4 joints
        # carry the reach.
        ikpy_rad = {j: float(self._cfg.q[self._qadr[j]]) for j in ACTIVE_JOINTS}
        arm_deg = calibration.ikpy_to_arm_deg(ikpy_rad)
        return np.array([arm_deg[j] for j in ACTIVE_JOINTS], dtype=float)

    def fk(self, arm_deg: dict[str, float]) -> tuple[float, float, float]:
        """Arm-degree joints -> tool-tip table-frame (x, y, z) mm, via the MuJoCo model.

        The read-back counterpart of :meth:`solve`, kept on the same mink/MuJoCo
        model so the live path never touches ikpy.
        """
        mujoco = self._mujoco
        rad = calibration.arm_to_ikpy_rad({j: arm_deg[j] for j in ACTIVE_JOINTS})
        self._data.qpos[:] = 0.0
        for j in ACTIVE_JOINTS:
            self._data.qpos[self._qadr[j]] = rad[j]
        mujoco.mj_kinematics(self._model, self._data)
        x_m, y_m, z_m = (float(v) for v in self._data.site_xpos[self._tcp_sid])
        return calibration.ikpy_to_table_mm(x_m, y_m, z_m)

    def tool_pitch_deg(self, arm_deg: dict[str, float]) -> float:
        """Achieved tool approach pitch (deg) for these joints: -90 = straight down.

        mink solves POSITION-ONLY, so the wrist's tilt is whatever the geometry
        produced — not the pitch that was asked for. This reads it back from the
        model (the self-calibrated approach axis) so the caller can compensate for
        how far the claw overhangs once the wrist is off vertical (the tilt-grasp
        position fix; see ``calibration.claw_overhang_offset``). z is shared between
        the base and table frames, so the pitch is frame-independent.
        """
        import numpy as np

        mujoco = self._mujoco
        rad = calibration.arm_to_ikpy_rad({j: arm_deg[j] for j in ACTIVE_JOINTS})
        self._data.qpos[:] = 0.0
        for j in ACTIVE_JOINTS:
            self._data.qpos[self._qadr[j]] = rad[j]
        mujoco.mj_kinematics(self._model, self._data)
        m = self._data.site_xmat[self._tcp_sid]
        c = self._approach_col
        a = self._approach_sign * np.array([m[c], m[3 + c], m[6 + c]])
        return float(np.degrees(np.arctan2(a[2], np.hypot(a[0], a[1]))))
