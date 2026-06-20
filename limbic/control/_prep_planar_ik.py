"""Analytical (closed-form) planar IK for bronny (SO-101).

Replaces the ikpy numerical solver for reaching. The SO-101 is effectively:
  * shoulder_pan  -> azimuth (which vertical slice the arm reaches in)
  * shoulder_lift + elbow_flex + wrist_flex -> a planar 2-link-plus-wrist problem
    in that slice, giving tip (radius, height) AND a chosen approach PITCH
  * wrist_roll    -> fixed (roll about the approach axis)

Because the arm moves in a clean vertical plane (verified: tip stays in-plane),
this is solved in closed form: deterministic, exact (0 model error), no optimizer
branch-jumping. You give position + an approach pitch; pan is set from the target
azimuth, the planar 2-link IK is analytic.

Pitch convention (approach axis = gripper pointing direction, in the reach plane):
   pitch = -90  -> straight DOWN (top-down grasp)
   pitch =   0  -> pointing horizontally outward (forward)
   pitch = -45  -> 45 deg down-and-out
Geometry (link lengths, frame offsets, angle constants) is extracted once from the
validated FK chain so it stays consistent with everything measured.
"""

import numpy as np

from ._prep_ik_wrapper import (
    SO101Kinematics,
    BASE_HEIGHT_ABOVE_TABLE_MM,
    IKPY_TO_BRONNY_SIGN,
    IKPY_TO_BRONNY_OFFSET,
)

PAN_AXIS_X_MM = 38.8


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class PlanarSO101IK:
    def __init__(self, kin: SO101Kinematics | None = None):
        self.kin = kin or SO101Kinematics()
        self._extract_geometry()

    # ------------------------------------------------------------------ #
    def _fk_full(self, lift, elbow, wflex):
        v = np.zeros(len(self.kin.chain.links))
        v[2], v[3], v[4] = np.deg2rad([lift, elbow, wflex])
        return self.kin.chain.forward_kinematics(v, full_kinematics=True)

    def _planar(self, p_m):
        """base xyz (m) -> (r, z) mm in the reach plane (r=forward from pan axis, z above table)."""
        p = np.asarray(p_m) * 1000.0
        return np.array([PAN_AXIS_X_MM - p[0], p[2] + BASE_HEIGHT_ABOVE_TABLE_MM])

    @staticmethod
    def _seg_angle(F, a, b):
        d = (F[b][:3, 3] - F[a][:3, 3]) * 1000.0
        return np.degrees(np.arctan2(d[2], -d[0]))  # angle from horizontal in (forward, up)

    def _extract_geometry(self):
        names = [l.name for l in self.kin.chain.links]
        self.iL = names.index("shoulder_lift")
        self.iE = names.index("elbow_flex")
        self.iW = names.index("wrist_flex")
        self.iT = names.index("gripper_frame_joint")
        F = self._fk_full(0, 0, 0)
        self.L1 = np.linalg.norm((F[self.iE][:3, 3] - F[self.iL][:3, 3]) * 1000.0)  # lift->elbow
        self.L2 = np.linalg.norm((F[self.iW][:3, 3] - F[self.iE][:3, 3]) * 1000.0)  # elbow->wrist
        self.L3 = np.linalg.norm((F[self.iT][:3, 3] - F[self.iW][:3, 3]) * 1000.0)  # wrist->tip
        self.P0 = self._planar(F[self.iL][:3, 3])      # lift axis (r0, z0)
        self.C_arm = self._seg_angle(F, self.iL, self.iE)   # upper-arm angle at ikpy_lift=0
        self.C_fore = self._seg_angle(F, self.iE, self.iW)  # forearm angle at zeros
        self.C_grip = self._seg_angle(F, self.iW, self.iT)  # gripper-segment angle at zeros
        zax = F[self.iT][:3, 2]
        self.C_app = np.degrees(np.arctan2(zax[2], -zax[0]))  # approach pitch at zeros

    # ------------------------------------------------------------------ #
    def solve(self, x, y, z, pitch_deg, elbow_up=True):
        """Target table-frame (x,y,z) mm + approach pitch (deg) -> bronny[5] (deg) or None.

        Returns None if the point is out of reach. wrist_roll is set to 0.
        """
        # 1) azimuth -> pan (ikpy). At pan 0 the arm reaches at table azimuth 0 (+x).
        pan_ikpy = -np.degrees(np.arctan2(y, x))
        r = float(np.hypot(x, y))

        # 2) gripper segment angle that yields the desired approach pitch
        gA = pitch_deg + (self.C_grip - self.C_app)
        # 3) wrist position (work back along the gripper segment from the tip)
        wr = r - self.L3 * np.cos(np.radians(gA))
        wz = z - self.L3 * np.sin(np.radians(gA))

        # 4) planar 2-link IK from lift axis P0 to wrist (wr,wz)
        dr, dz = wr - self.P0[0], wz - self.P0[1]
        D = float(np.hypot(dr, dz))
        if D > self.L1 + self.L2 or D < abs(self.L1 - self.L2):
            return None
        cos_off = np.clip((self.L1**2 + D**2 - self.L2**2) / (2 * self.L1 * D), -1, 1)
        off = np.degrees(np.arccos(cos_off))
        base = np.degrees(np.arctan2(dz, dr))
        arm_angle = base + off if elbow_up else base - off

        a1 = np.radians(arm_angle)
        P1 = self.P0 + self.L1 * np.array([np.cos(a1), np.sin(a1)])  # elbow position
        fore_angle = np.degrees(np.arctan2(wz - P1[1], wr - P1[0]))

        # 5) back out ikpy joint angles from the planar angles
        ik_lift = _wrap180(arm_angle - self.C_arm)
        ik_elbow = _wrap180((fore_angle - arm_angle) - (self.C_fore - self.C_arm))
        ik_wflex = _wrap180(pitch_deg - ik_lift - ik_elbow - self.C_app)

        ik = np.array([pan_ikpy, ik_lift, ik_elbow, ik_wflex, 0.0])
        bronny = IKPY_TO_BRONNY_SIGN * ik + IKPY_TO_BRONNY_OFFSET
        return bronny


# Natural-language -> approach pitch (deg). "you know what orientation is needed."
ORIENTATION_PITCH = {
    "down": -90.0, "straight down": -90.0, "top-down": -90.0, "vertical": -90.0,
    "forward": 0.0, "horizontal": 0.0, "out": 0.0,
    "45": -45.0, "diagonal": -45.0, "down-forward": -45.0,
}


def interpret_orientation(text) -> float:
    if isinstance(text, (int, float)):
        return float(text)
    return ORIENTATION_PITCH.get(str(text).strip().lower(), -90.0)


if __name__ == "__main__":
    ik = PlanarSO101IK()
    print(f"geometry: L1={ik.L1:.1f} L2={ik.L2:.1f} L3={ik.L3:.1f} P0=({ik.P0[0]:.1f},{ik.P0[1]:.1f})")
    print(f"  C_arm={ik.C_arm:.2f} C_fore={ik.C_fore:.2f} C_grip={ik.C_grip:.2f} C_app={ik.C_app:.2f}\n")
    # round-trip validation against the FK
    print("ROUND-TRIP (analytical solve -> FK -> compare):")
    for (x, y, z, pitch) in [(200, 0, 50, -90), (180, -50, 30, -90), (180, 90, 60, -90),
                             (220, 0, 0, -90), (200, 0, 100, -45), (170, 40, 80, 0)]:
        b = ik.solve(x, y, z, pitch)
        if b is None:
            print(f"  ({x},{y},{z},p{pitch}): UNREACHABLE"); continue
        fk = ik.kin.forward_kinematics(b)
        p = fk[:3, 3] * 1000
        tx, ty, tz = PAN_AXIS_X_MM - p[0], -p[1], p[2] + BASE_HEIGHT_ABOVE_TABLE_MM
        zax = fk[:3, 2]
        got_pitch = np.degrees(np.arctan2(zax[2], np.hypot(zax[0], zax[1]) * np.sign(-zax[0] if abs(zax[0]) > 1e-9 else 1)))
        err = np.hypot(np.hypot(tx - x, ty - y), tz - z)
        inlim = all(abs(b[i]) <= [80, 100, 95, 95, 165][i] + 0.5 for i in range(5))
        print(f"  ({x:3d},{y:+4d},{z:3d},p{pitch:+3d}) -> bronny=[{b[0]:+.0f},{b[1]:+.0f},{b[2]:+.0f},{b[3]:+.0f},{b[4]:+.0f}]"
              f"  pos_err={err:4.1f}mm  inlimits={inlim}")
