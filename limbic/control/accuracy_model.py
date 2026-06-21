"""Data-driven open-loop accuracy correction (§A.6 / §5).

This replaces hand-fitted constants with a model fit from a PERSISTENT
measurement dataset, with robust outlier rejection and leave-one-out
cross-validation so the EXPECTED real-world accuracy is known before the arm
ever moves. It is the cure for "tweak a constant, move, get surprised, repeat":
every measurement is kept, the fit is reproducible, and it reports how well it
actually generalises.

The physical samples are (AIM -> REAL) pairs, both in planar (forward-reach, z)
table-frame mm:

    aim  = where the IK put the MODEL tip = forward_kinematics(solve_ik(target))
    real = where the tip ACTUALLY landed (ruler, or camera triangulation)

These pairs are CORRECTION-VERSION-INDEPENDENT: 'aim' is the raw geometric
target the solver hit, independent of whatever correction produced it, so the
dataset accumulates cleanly across re-fits (you never have to re-measure because
the correction changed).

``command_for_real(real_fwd, real_z, pitch)`` returns the AIM to feed the IK so
the tip lands at the desired real position -- i.e. the fitted inverse map
real -> aim, faded toward identity as the approach tilts off vertical (we only
trust vertical-approach data; identity is the safe default elsewhere).

Pure numpy; no hardware. If the dataset is too thin to fit, callers fall back to
the affine constants in ``calibration``.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Pitch fade: correction full when vertical, off (aim = real) when horizontal.
_PITCH_FULL_DEG = -88.0
_PITCH_NONE_DEG = -82.0

# Candidate feature sets, simplest first. Cross-validation picks the LEAST
# complex one that generalises -- guards against the over-fitting that made the
# hand-tweaked plane oscillate on sparse, noisy points.
#   f = real_fwd (mm), z = real_z (mm), both centred+scaled before features.
_FEATURE_SETS: dict[str, list] = {
    "affine":  [lambda f, z: 1.0, lambda f, z: f, lambda f, z: z],
    "bilinear": [lambda f, z: 1.0, lambda f, z: f, lambda f, z: z, lambda f, z: f * z],
    "quadratic": [lambda f, z: 1.0, lambda f, z: f, lambda f, z: z,
                  lambda f, z: f * z, lambda f, z: f * f, lambda f, z: z * z],
}

# Minimum samples to even attempt a fit (need > params, with margin for CV).
_MIN_SAMPLES = 4
# Ridge strength (on standardised features) -- gentle; keeps the fit from blowing
# up when features are collinear without washing out real structure.
_RIDGE = 1e-3
# Robust rejection: drop a point whose residual exceeds this * MAD, then refit.
_ROBUST_K = 3.0
_ROBUST_ITERS = 2


def _z_blend(pitch_deg: float) -> float:
    if pitch_deg <= _PITCH_FULL_DEG:
        return 1.0
    if pitch_deg >= _PITCH_NONE_DEG:
        return 0.0
    return (pitch_deg - _PITCH_NONE_DEG) / (_PITCH_FULL_DEG - _PITCH_NONE_DEG)


@dataclass
class Sample:
    """One physical (aim -> real) measurement, planar mm, at a given pitch."""

    aim_fwd: float
    aim_z: float
    real_fwd: float
    real_z: float
    pitch: float = -90.0
    source: str = ""
    note: str = ""


# --------------------------------------------------------------------------- #
# Dataset I/O -- the single source of truth lives in a CSV.
# --------------------------------------------------------------------------- #
_CSV_FIELDS = [
    "ts", "pitch", "aim_fwd", "aim_z", "real_fwd", "real_z",
    "target_x", "target_y", "target_z", "meas_x", "meas_y", "meas_z",
    "source", "note",
]


def load_samples(csv_path: pathlib.Path) -> list[Sample]:
    """Read all (aim->real) samples from the dataset CSV (empty if absent)."""
    if not pathlib.Path(csv_path).exists():
        return []
    out: list[Sample] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out.append(Sample(
                    aim_fwd=float(row["aim_fwd"]), aim_z=float(row["aim_z"]),
                    real_fwd=float(row["real_fwd"]), real_z=float(row["real_z"]),
                    pitch=float(row.get("pitch", -90.0) or -90.0),
                    source=row.get("source", ""), note=row.get("note", ""),
                ))
            except (KeyError, ValueError):
                continue  # skip malformed / blank rows
    return out


def record_landing(csv_path: pathlib.Path, target_xyz: tuple[float, float, float],
                   measured_xyz: tuple[float, float, float], pitch: float = -90.0,
                   source: str = "", note: str = "") -> dict[str, Any]:
    """Append one (aim -> real) sample from a commanded target and its measured
    landing. The ONE seam every measurement source uses -- ruler (the CLI) or
    camera (closed-loop auto-measure) -- so the dataset stays consistent.

    'aim' is recomputed from the target via the IK (the model tip it hits), so the
    sample is correction-version-independent. Returns the row written.
    """
    import datetime as _dt

    # Lazy import to avoid a circular import (kinematics -> calibration -> here).
    from .kinematics import forward_kinematics, solve_ik

    tx, ty, tz = target_xyz
    mx, my, mz = measured_xyz
    sol = solve_ik(tx, ty, tz, pitch)
    ax, ay, az = forward_kinematics(sol.joints)
    row = {
        "ts": _dt.date.today().isoformat(), "pitch": pitch,
        "aim_fwd": round(float(math.hypot(ax, ay)), 2), "aim_z": round(float(az), 2),
        "real_fwd": round(math.hypot(mx, my), 2), "real_z": round(mz, 2),
        "target_x": tx, "target_y": ty, "target_z": tz,
        "meas_x": mx, "meas_y": my, "meas_z": mz, "source": source, "note": note,
    }
    append_sample_row(csv_path, row)
    return row


def append_sample_row(csv_path: pathlib.Path, row: dict[str, Any]) -> None:
    """Append one measurement row, writing a header if the file is new."""
    path = pathlib.Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _CSV_FIELDS})


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #
@dataclass
class AxisFit:
    """A fitted scalar map real(f,z) -> aim_axis, plus its feature/scaling spec."""

    feature_set: str
    coef: list[float]
    f_mean: float
    f_std: float
    z_mean: float
    z_std: float
    loo_rms: float                 # leave-one-out CV RMS, mm (expected error)
    train_rms: float

    def _design_row(self, f: float, z: float) -> np.ndarray:
        fs = (f - self.f_mean) / self.f_std
        zs = (z - self.z_mean) / self.z_std
        return np.array([fn(fs, zs) for fn in _FEATURE_SETS[self.feature_set]])

    def predict(self, f: float, z: float) -> float:
        return float(self._design_row(f, z) @ np.array(self.coef))


@dataclass
class AccuracyModel:
    """Fitted aim = f(real) correction for both planar axes, with metadata."""

    fwd: AxisFit
    z: AxisFit
    n_samples: int
    n_outliers: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    # --- use -------------------------------------------------------------- #
    def command_for_real(self, real_fwd_mm: float, real_z_mm: float,
                         pitch_deg: float = -90.0) -> tuple[float, float]:
        """Aim (fwd, z) to feed the IK so the tip lands at the desired real point."""
        blend = _z_blend(pitch_deg)
        aim_f = self.fwd.predict(real_fwd_mm, real_z_mm)
        aim_z = self.z.predict(real_fwd_mm, real_z_mm)
        # Fade toward identity (aim = real) off vertical.
        return (real_fwd_mm + blend * (aim_f - real_fwd_mm),
                real_z_mm + blend * (aim_z - real_z_mm))

    def real_for_command(self, aim_fwd_mm: float, aim_z_mm: float,
                         pitch_deg: float = -90.0) -> tuple[float, float]:
        """Inverse: which desired real lands at this aim. Diagnostic only.

        Newton solve with a numeric 2x2 Jacobian (the forward map can be steep, so
        a plain fixed-point diverges); converges quadratically in a few steps.
        """
        target = np.array([aim_fwd_mm, aim_z_mm])
        r = target.copy()
        h = 0.1
        for _ in range(50):
            c = np.array(self.command_for_real(r[0], r[1], pitch_deg))
            err = target - c
            if abs(err[0]) < 1e-5 and abs(err[1]) < 1e-5:
                break
            cf = np.array(self.command_for_real(r[0] + h, r[1], pitch_deg))
            cz = np.array(self.command_for_real(r[0], r[1] + h, pitch_deg))
            J = np.column_stack([(cf - c) / h, (cz - c) / h])
            try:
                r = r + np.linalg.solve(J, err)
            except np.linalg.LinAlgError:
                r = r + err  # singular -> fall back to a damped step
        return float(r[0]), float(r[1])

    @property
    def expected_accuracy_mm(self) -> tuple[float, float]:
        """LOO-CV expected (fwd, z) command error in mm -- the honest accuracy."""
        return (self.fwd.loo_rms, self.z.loo_rms)

    # --- persistence ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples, "n_outliers": self.n_outliers,
            "meta": self.meta,
            "fwd": self.fwd.__dict__, "z": self.z.__dict__,
        }

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: pathlib.Path) -> "AccuracyModel | None":
        path = pathlib.Path(path)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return cls(fwd=AxisFit(**d["fwd"]), z=AxisFit(**d["z"]),
                       n_samples=d["n_samples"], n_outliers=d.get("n_outliers", 0),
                       meta=d.get("meta", {}))
        except Exception:
            return None


def _ridge_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ridge least squares (no penalty on the intercept, col 0)."""
    n_feat = X.shape[1]
    P = _RIDGE * np.eye(n_feat)
    P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def _loo_rms(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out CV RMS via the ridge hat matrix (closed form, cheap)."""
    n, p = X.shape
    P = _RIDGE * np.eye(p)
    P[0, 0] = 0.0
    A = X.T @ X + P
    try:
        H = X @ np.linalg.solve(A, X.T)
    except np.linalg.LinAlgError:
        return float("inf")
    beta = _ridge_fit(X, y)
    resid = y - X @ beta
    denom = np.clip(1.0 - np.diag(H), 1e-6, None)   # press residuals
    return float(np.sqrt(np.mean((resid / denom) ** 2)))


def _fit_axis(f: np.ndarray, z: np.ndarray, target: np.ndarray) -> tuple[AxisFit, np.ndarray]:
    """Fit one axis: pick the feature set by LOO-CV, return (fit, residuals)."""
    f_mean, f_std = float(f.mean()), float(f.std() or 1.0)
    z_mean, z_std = float(z.mean()), float(z.std() or 1.0)
    fs, zs = (f - f_mean) / f_std, (z - z_mean) / z_std

    best = None
    for name, fns in _FEATURE_SETS.items():
        if len(fns) >= len(f):          # need more points than params for CV
            continue
        X = np.column_stack([np.vectorize(fn)(fs, zs) for fn in fns])
        loo = _loo_rms(X, target)
        if best is None or loo < best[0]:
            beta = _ridge_fit(X, target)
            train = float(np.sqrt(np.mean((target - X @ beta) ** 2)))
            best = (loo, name, beta, train, X)

    if best is None:   # too few points even for affine -> plain affine, no CV
        name = "affine"
        X = np.column_stack([np.vectorize(fn)(fs, zs) for fn in _FEATURE_SETS[name]])
        beta = _ridge_fit(X, target)
        train = float(np.sqrt(np.mean((target - X @ beta) ** 2)))
        best = (float("nan"), name, beta, train, X)

    loo, name, beta, train, X = best
    fit = AxisFit(feature_set=name, coef=[float(b) for b in beta],
                  f_mean=f_mean, f_std=f_std, z_mean=z_mean, z_std=z_std,
                  loo_rms=loo, train_rms=train)
    resid = target - X @ beta
    return fit, resid


def fit_model(samples: list[Sample]) -> tuple["AccuracyModel | None", dict[str, Any]]:
    """Fit an AccuracyModel from samples with robust outlier rejection + LOO-CV.

    Returns ``(model_or_None, report)``. ``report`` always describes what
    happened (n used, outliers dropped, per-axis CV accuracy, per-point
    residuals) so the caller can print an honest summary.
    """
    pts = [s for s in samples if abs(s.pitch + 90.0) < 1e-6 or s.pitch <= _PITCH_FULL_DEG]
    report: dict[str, Any] = {"n_total": len(samples), "n_vertical": len(pts)}
    if len(pts) < _MIN_SAMPLES:
        report["status"] = "too_few"
        report["message"] = (
            f"only {len(pts)} vertical-approach samples (need >= {_MIN_SAMPLES}); "
            "falling back to affine constants. Collect more with calibrate_accuracy."
        )
        return None, report

    f = np.array([s.real_fwd for s in pts])
    z = np.array([s.real_z for s in pts])
    aim_f = np.array([s.aim_fwd for s in pts])
    aim_z = np.array([s.aim_z for s in pts])

    keep = np.ones(len(pts), bool)
    outliers: list[dict] = []
    for _ in range(_ROBUST_ITERS):
        ff, fz = _fit_axis(f[keep], z[keep], aim_f[keep])
        zf, zz = _fit_axis(f[keep], z[keep], aim_z[keep])
        # Combined residual magnitude per kept point.
        rf = ff.predict  # noqa
        res = np.hypot(
            [aim_f[i] - ff.predict(f[i], z[i]) for i in range(len(pts))],
            [aim_z[i] - zf.predict(f[i], z[i]) for i in range(len(pts))],
        )
        kept_res = res[keep]
        mad = np.median(np.abs(kept_res - np.median(kept_res))) or 1.0
        thresh = _ROBUST_K * 1.4826 * mad
        new_keep = keep & (res <= max(thresh, 8.0))   # never drop within 8mm noise
        if new_keep.sum() < _MIN_SAMPLES or (new_keep == keep).all():
            keep = new_keep if new_keep.sum() >= _MIN_SAMPLES else keep
            break
        keep = new_keep

    for i in range(len(pts)):
        if not keep[i]:
            outliers.append({"i": i, "real": [round(float(f[i]), 1), round(float(z[i]), 1)],
                             "resid_mm": round(float(res[i]), 1),
                             "source": pts[i].source, "note": pts[i].note})

    ff, _ = _fit_axis(f[keep], z[keep], aim_f[keep])
    zf, _ = _fit_axis(f[keep], z[keep], aim_z[keep])
    model = AccuracyModel(fwd=ff, z=zf, n_samples=int(keep.sum()),
                          n_outliers=len(outliers),
                          meta={"feature_fwd": ff.feature_set, "feature_z": zf.feature_set})

    report.update({
        "status": "ok",
        "n_used": int(keep.sum()),
        "n_outliers": len(outliers),
        "outliers": outliers,
        "expected_accuracy_mm": {"fwd": round(ff.loo_rms, 1), "z": round(zf.loo_rms, 1)},
        "train_rms_mm": {"fwd": round(ff.train_rms, 1), "z": round(zf.train_rms, 1)},
        "feature_set": {"fwd": ff.feature_set, "z": zf.feature_set},
        "residuals": [
            {"real": [round(float(f[i]), 1), round(float(z[i]), 1)],
             "resid_fwd": round(float(aim_f[i] - ff.predict(f[i], z[i])), 1),
             "resid_z": round(float(aim_z[i] - zf.predict(f[i], z[i])), 1),
             "kept": bool(keep[i]), "source": pts[i].source}
            for i in range(len(pts))
        ],
    })
    return model, report
