"""Check encoded demo metadata, decoded motion and synchronized feedback."""

import argparse
import json
import math
from pathlib import Path
import subprocess


RUNS = Path(".agents/runs/foundation")


def check(video, report_path):
    report = json.loads(report_path.read_text())
    metadata = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration,codec_name,pix_fmt",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True, timeout=20,
    )
    stream = json.loads(metadata.stdout)["streams"][0]
    frames = report["frames"]
    assert report["status"] == "PASS"
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (report["width"], report["height"])
    assert stream["r_frame_rate"] == f'{report["fps"]}/1'
    assert int(stream["nb_frames"]) == len(frames)
    assert math.isclose(float(stream["duration"]), len(frames) / report["fps"], abs_tol=1e-5)
    assert 1 < report["duration_s"] < 60 and report["reset_verified"]
    assert report["direct_state_writes_during_motion"] == 0
    for index, frame in enumerate(frames):
        assert frame["frame"] == index
        assert math.isclose(frame["time_s"], (index + 1) / report["fps"], abs_tol=1e-9)
        assert len(frame["q_rad"]) == len(frame["qd_rad_s"]) == len(frame["target_rad"]) == 6
        assert all(math.isfinite(value) for key in ("q_rad", "qd_rad_s", "target_rad") for value in frame[key])

    motion = [frame for frame in frames if frame["phase"] == "trajectory"]
    initial = frames[0]["q_rad"]

    def displacement(frame):
        return max(abs(angle - reference) for angle, reference in zip(frame["q_rad"], initial, strict=True))

    peak = max(motion, key=displacement)
    assert displacement(peak) > 0.2 and frames[-1]["phase"] == "reset"
    assert displacement(frames[-1]) < 0.015
    # Decode the initial and peak robot frames at reduced size for an independent pixel check.
    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video), "-vf",
            f'select=eq(n\\,0)+eq(n\\,{peak["frame"]}),scale=240:180',
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        check=True, capture_output=True, timeout=30,
    ).stdout
    nbytes = 240 * 180 * 3
    assert len(decoded) == 2 * nbytes
    baseline, moved = decoded[:nbytes], decoded[nbytes:]
    assert max(baseline) - min(baseline) > 40
    changed = sum(abs(a - b) > 20 for a, b in zip(baseline, moved, strict=True))
    assert changed > 120, f"Decoded robot movement changed only {changed} channel values."
    print(f"PASS H.264 decode, {len(frames)} frames at {report['fps']} fps, synchronized motion and reset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=RUNS / "airbot_play.mp4")
    parser.add_argument("--report", type=Path, default=RUNS / "demo.json")
    args = parser.parse_args()
    check(args.video, args.report)
