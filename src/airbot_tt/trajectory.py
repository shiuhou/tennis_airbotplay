"""Bounded joint-space timing and discrete path preflight for AIRBOT Play."""

import math

import numpy as np

from .config import COMMAND_ACCELERATION, COMMAND_SPEED, DT

DEFAULT_SAMPLE_STEP_RAD = 0.01


def _positive_joint_limits(value, name):
    limits = np.asarray(value, dtype=float)
    if limits.shape not in ((), (6,)) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError(f"{name} must be a positive finite scalar or six joint values.")
    return np.broadcast_to(limits, (6,)).copy()


def _preflight(points, require_clear, max_joint_step):
    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError("Collision sample step must be positive and finite.")
    if require_clear is None:
        return 0
    count = 0
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        delta = end - start
        steps = max(1, int(np.ceil(np.max(np.abs(delta)) / max_joint_step)))
        for step in range(0 if i == 0 else 1, steps + 1):
            try:
                q = end.copy() if step == steps else start + delta * (step / steps)
                require_clear(q)
            except ValueError as error:
                raise ValueError(f"Path segment {i + 1}, sample {step}/{steps}: {error}") from error
            count += 1
    return count


def _time(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Time must be a finite number of seconds.") from error
    if not np.isfinite(seconds):
        raise ValueError("Time must be a finite number of seconds.")
    return seconds


def _check_timeline(duration):
    if not math.isfinite(duration) or math.ulp(duration) > DT:
        raise ValueError("Duration exceeds the resolution of the fixed-step clock.")


class Trajectory:
    """Rest-to-rest quintic segments visiting every waypoint."""

    def __init__(
        self,
        model,
        waypoints,
        *,
        vmax=COMMAND_SPEED,
        amax=COMMAND_ACCELERATION,
        durations=None,
        require_clear=None,
        max_joint_step=DEFAULT_SAMPLE_STEP_RAD,
    ):
        self.velocity_limit = np.minimum(_positive_joint_limits(vmax, "Velocity limit"), model.velocity)
        self.acceleration_limit = _positive_joint_limits(amax, "Acceleration limit")
        self.points = np.array([model.validate(q) for q in waypoints])
        if len(self.points) < 2:
            raise ValueError("A trajectory needs a start and at least one target.")
        deltas = np.abs(np.diff(self.points, axis=0))
        # For s(u)=10u^3-15u^4+6u^5, max(s')=1.875 and max(|s''|)=10/sqrt(3).
        with np.errstate(over="ignore"):
            velocity_time = np.max(1.875 * deltas / self.velocity_limit, axis=1)
        acceleration_time = np.max(np.sqrt((10 / np.sqrt(3)) * deltas) / np.sqrt(self.acceleration_limit), axis=1)
        minimum_times = np.maximum(velocity_time, acceleration_time)

        if durations is None:
            self.durations = np.maximum(0.1, minimum_times)
        else:
            self.durations = np.array(durations, dtype=float, copy=True)
            if self.durations.shape != minimum_times.shape:
                raise ValueError("Provide one duration per trajectory segment.")
            if not np.isfinite(self.durations).all() or np.any(self.durations <= 0):
                raise ValueError("Segment durations must be positive and finite.")
            too_short = np.flatnonzero(self.durations < minimum_times)
            if len(too_short):
                i = int(too_short[0])
                raise ValueError(f"Segment {i + 1} needs at least {minimum_times[i]:.6g} s for its joint limits.")

        with np.errstate(over="ignore"):
            self.ends = np.cumsum(self.durations)
        self.duration = float(self.ends[-1])
        _check_timeline(self.duration)
        if np.any(np.diff(self.ends) <= 0):
            raise ValueError("Segment durations do not form an increasing timeline.")
        self.checked_samples = _preflight(self.points, require_clear, max_joint_step)
        for value in (self.points, self.durations, self.ends, self.velocity_limit, self.acceleration_limit):
            value.setflags(write=False)

    def sample(self, t):
        """Return joint position, velocity and acceleration at time t."""
        t = _time(t)
        if t <= 0:
            return self.points[0].copy(), np.zeros(6), np.zeros(6)
        if t >= self.duration:
            return self.points[-1].copy(), np.zeros(6), np.zeros(6)
        i = int(np.searchsorted(self.ends, t, side="right"))
        start_t = 0.0 if i == 0 else self.ends[i - 1]
        duration = self.durations[i]
        u = np.clip((t - start_t) / duration, 0.0, 1.0)
        ds = 30 * u**2 * (1 - u)**2
        dds = 60 * u * (1 - u) * (1 - 2*u)
        delta = self.points[i + 1] - self.points[i]
        # Evaluate close to either endpoint without cancellation around u=1.
        nearest = min(u, 1 - u)
        s = nearest**3 * (10 - 15*nearest + 6*nearest**2)
        q = self.points[i] + s * delta if u <= 0.5 else self.points[i + 1] - s * delta
        v = ds * (delta / duration)
        a = dds * ((delta / duration) / duration)
        return q, v, a


class VelocityTrajectory:
    """One bounded velocity pulse with symmetric ramps and endpoint hold."""

    def __init__(
        self,
        model,
        start,
        velocity,
        duration,
        *,
        vmax=COMMAND_SPEED,
        amax=COMMAND_ACCELERATION,
        require_clear=None,
        max_joint_step=DEFAULT_SAMPLE_STEP_RAD,
    ):
        self.velocity_limit = np.minimum(_positive_joint_limits(vmax, "Velocity limit"), model.velocity)
        self.acceleration_limit = _positive_joint_limits(amax, "Acceleration limit")
        self.duration = _time(duration)
        if self.duration <= 0:
            raise ValueError("Pulse duration must be positive.")
        _check_timeline(self.duration)
        self.velocity = np.array(velocity, dtype=float, copy=True)
        if self.velocity.shape != (6,) or not np.isfinite(self.velocity).all():
            raise ValueError("Expected six finite joint velocity values in rad/s.")
        if np.any(np.abs(self.velocity) > self.velocity_limit):
            raise ValueError("Velocity pulse exceeds a joint speed limit.")
        with np.errstate(over="ignore"):
            self.ramp_time = float(np.max(np.abs(self.velocity) / self.acceleration_limit))
        if self.ramp_time > self.duration / 2:
            raise ValueError(f"Velocity ramps require at least {2*self.ramp_time:.6g} s.")
        start = model.validate(start)
        end = model.validate(start + self.velocity * (self.duration - self.ramp_time))
        self.points = np.array([start, end])
        self.durations = np.array([self.duration])
        self.ends = self.durations
        self.checked_samples = _preflight(self.points, require_clear, max_joint_step)
        for value in (self.points, self.durations, self.velocity, self.velocity_limit, self.acceleration_limit):
            value.setflags(write=False)

    def sample(self, t):
        t = _time(t)
        if t <= 0:
            return self.points[0].copy(), np.zeros(6), np.zeros(6)
        if t >= self.duration:
            return self.points[-1].copy(), np.zeros(6), np.zeros(6)
        remaining = self.duration - t
        if t < self.ramp_time:
            fraction = t / self.ramp_time
            q = self.points[0] + self.velocity * (0.5 * t * fraction)
            return q, self.velocity * fraction, self.velocity / self.ramp_time
        if remaining < self.ramp_time:
            fraction = remaining / self.ramp_time
            q = self.points[-1] - self.velocity * (0.5 * remaining * fraction)
            return q, self.velocity * fraction, -self.velocity / self.ramp_time
        q = self.points[0] + self.velocity * (t - 0.5 * self.ramp_time)
        return q, self.velocity.copy(), np.zeros(6)
