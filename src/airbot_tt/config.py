"""Fixed-base scene and low-speed simulation settings (SI units)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "assets/airbot_play/airbot_play.urdf"
DT = 1.0 / 240
BASE_POSITION = (0.0, 0.0, 0.02)
HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

COMMAND_SPEED = 0.15
COMMAND_ACCELERATION = 0.30
SIM_SPEED = 0.40
# Position-drive tuning for the bare arm under gravity.
DRIVE_STIFFNESS = (800.0, 800.0, 700.0, 450.0, 400.0, 350.0)
DRIVE_DAMPING = (16.0, 16.0, 12.0, 6.0, 5.0, 5.0)
CONTACT_TRIP_N = 5.0
# Reviewed link5/fixer nominal separation is 0.946981 mm; other pairs retain 1 mm.
RACKET_WRIST_CLEARANCE_M = 0.0005
RACKET_CONTACT_OFFSET_M = 0.0002
