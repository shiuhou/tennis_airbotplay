"""Low-spin ball parameters and quadratic drag in SI world coordinates.

No Magnus or spin-transfer model is claimed. Material values are uncalibrated.
This module has no Isaac dependency so the force law can be checked independently.
"""

import json
import math

import numpy as np

from .config import ROOT


def load_ball_config(path=ROOT / "configs/ball.json"):
    data = json.loads(path.read_text())
    positive = ("mass_kg", "radius_m", "air_density_kg_m3", "gravity_m_s2",
                "contact_offset_m", "table_top_m", "net_height_m", "net_thickness_m")
    for key in positive:
        if not math.isfinite(data[key]) or data[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    for key in ("drag_coefficient", "friction", "table_restitution", "paddle_restitution", "net_restitution"):
        if not math.isfinite(data[key]) or data[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
        if "restitution" in key and data[key] > 1:
            raise ValueError(f"{key} must not exceed one.")
    size = np.asarray(data["table_size_m"])
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("table_size_m must have three positive finite lengths.")
    return data


def drag_force(velocity_m_s, config):
    """Return F[N] = -rho Cd pi r² |v| v / 2 for (..., 3) velocities."""
    velocity = np.asarray(velocity_m_s, dtype=float)
    if velocity.ndim < 1 or velocity.shape[-1] != 3 or not np.isfinite(velocity).all():
        raise ValueError("Velocity must be finite with final dimension three.")
    scale = 0.5 * config["air_density_kg_m3"] * config["drag_coefficient"] * math.pi * config["radius_m"] ** 2
    return -scale * np.linalg.norm(velocity, axis=-1, keepdims=True) * velocity
