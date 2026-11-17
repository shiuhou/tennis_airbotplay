"""Record bounded Isaac Sim RGB and synchronize the measured robot state."""

import math
from pathlib import Path
from uuid import uuid4

import imageio.v2 as imageio
import numpy as np
from pxr import Gf, UsdGeom
from omni.syntheticdata import SyntheticData
import isaaclab.sim as sim_utils
import omni.replicator.core as rep
import omni.usd


FPS = 10
WIDTH = 960
HEIGHT = 720
CAMERA_PATH = "/World/DemoCamera"
EXPORT = "LdrColorExportRawArray"


class _Camera:
    def __init__(self, focus):
        config = sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)
        )
        config.func(CAMERA_PATH, config)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(CAMERA_PATH)
        camera = UsdGeom.Camera(prim)
        eye = Gf.Vec3d(*(focus + np.array([0.95, 1.03, 0.58])))
        transform = Gf.Matrix4d().SetLookAt(
            eye, Gf.Vec3d(*focus), Gf.Vec3d(0, 0, 1)
        ).GetInverse()
        camera.ClearXformOpOrder()
        camera.AddTransformOp().Set(transform)
        self.product = rep.create.render_product(CAMERA_PATH, (WIDTH, HEIGHT))
        self.synthetic = SyntheticData.Get()
        try:
            self.synthetic.activate_node_template(EXPORT, 0, [self.product.path])
        except Exception:
            self.product.destroy()
            raise

    def rgb(self):
        values = self.synthetic.get_node_attributes(
            EXPORT, ["outputs:data", "outputs:width", "outputs:height"], self.product.path
        )
        if not values:
            raise RuntimeError("Kit has not produced a camera frame.")
        width, height = int(values["outputs:width"]), int(values["outputs:height"])
        data = np.asarray(values["outputs:data"], dtype=np.uint8)
        if (width, height) != (WIDTH, HEIGHT) or data.size != width * height * 4:
            raise RuntimeError(f"Invalid Kit camera buffer: {width}x{height}, {data.size} bytes.")
        rgb = data.reshape(height, width, 4)[:, :, :3].copy()
        if float(rgb.std()) < 2.0:
            raise RuntimeError("Kit provided a blank camera frame.")
        return rgb

    def close(self):
        try:
            self.synthetic.deactivate_node_template(EXPORT, 0, [self.product.path])
        finally:
            self.product.destroy()


def record(control, video_path):
    """Render two multi-joint moves and a reset using the regular drive commands."""
    path = Path(video_path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("The video destination must have an .mp4 extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.stem}.{uuid4().hex}.partial.mp4")
    frame_steps = round(1 / (FPS * control.dt))
    timestep = frame_steps * control.dt
    if not math.isclose(timestep, 1 / FPS):
        raise RuntimeError("The recording frame interval must be a whole number of physics steps.")

    home = control.initial.copy()
    base = np.array(control.state()["base_world"]["position_m"])
    focus = base + 0.45 * control.model.fk(home)[:3, 3]
    camera = _Camera(focus)
    frames = []
    commands = []
    baseline_rgb = None
    change_pixels = 0
    initial_writes = control.state_writes
    points = (
        home + np.array([0.34, -0.32, 0.38, 0.28, 0.23, -0.23]),
        home + np.array([-0.31, -0.15, 0.22, -0.20, -0.20, 0.26]),
    )

    def advance(seconds, phase):
        nonlocal baseline_rgb, change_pixels
        for _ in range(math.ceil(seconds / timestep)):
            for _ in range(frame_steps):
                control.step()
                if control.fault:
                    raise RuntimeError(f"Recording stopped by the controller: {control.fault}")
            control.sim.render()
            rgb = camera.rgb()
            if baseline_rgb is None:
                baseline_rgb = rgb
            if phase == "trajectory":
                delta = np.abs(rgb.astype(np.int16) - baseline_rgb)
                change_pixels = max(change_pixels, int(np.count_nonzero(delta.max(axis=2) > 25)))
            writer.append_data(rgb)
            state = control.state()
            frames.append(
                {
                    "frame": len(frames),
                    "time_s": (len(frames) + 1) * timestep,
                    "phase": phase,
                    "q_rad": list(state["q_rad"]),
                    "qd_rad_s": list(state["qd_rad_s"]),
                    "target_rad": list(state["target_rad"]),
                    "tcp_base": state["tcp_base"],
                }
            )

    try:
        for _ in range(12):
            control.sim.render()
        with imageio.get_writer(
            str(partial_path), format="FFMPEG", fps=FPS, codec="libx264",
            pixelformat="yuv420p", macro_block_size=None, output_params=["-crf", "20"],
        ) as writer:
            advance(0.5, "initial")
            plan = control.start_trajectory(points)
            commands.append({"waypoints_rad": [point.tolist() for point in points], **plan})
            advance(plan["duration_s"] + 0.5, "trajectory")
            measured_motion = max(float(np.max(np.abs(np.array(frame["q_rad"]) - home))) for frame in frames)
            if measured_motion < 0.2 or change_pixels < 500:
                raise RuntimeError("The recording did not capture a measurable arm movement.")
            if control.state_writes != initial_writes:
                raise RuntimeError("The demonstration changed joint state outside reset.")
            np.testing.assert_allclose(control.q(), points[-1], atol=0.015)
            control.reset()
            np.testing.assert_allclose(control.q(), home, atol=1e-6)
            np.testing.assert_allclose(control.state()["qd_rad_s"], 0.0, atol=1e-6)
            advance(0.7, "reset")
        if partial_path.stat().st_size < 1024:
            raise RuntimeError("The encoder did not produce a usable MP4.")
        partial_path.replace(path)
    finally:
        try:
            camera.close()
        finally:
            partial_path.unlink(missing_ok=True)
    return {
        "status": "PASS",
        "renderer": "Isaac Sim Kit RTX, LdrColorExportRawArray",
        "video": str(path),
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "physics_dt_s": control.dt,
        "duration_s": len(frames) / FPS,
        "max_changed_pixels": change_pixels,
        "max_joint_displacement_rad": measured_motion,
        "reset_verified": True,
        "direct_state_writes_during_motion": 0,
        "commands": commands,
        "frames": frames,
    }


def record_returns(device, video_path):
    """Three real PhysX returns, captured at 60 Hz and played at quarter speed."""
    import hashlib
    from PIL import Image, ImageDraw, ImageFont
    from pxr import UsdShade
    from .config import ROOT
    from .shot_scene import SingleShot
    from .planning import load_strike_config, plan_return, serves
    cfg = load_strike_config()
    task = SingleShot(device, speed_limit=cfg['simulation_speed_limit_rad_s'],
                      physics_hz=cfg['physics_hz'], initial=cfg['ready_q_rad'])
    c = task.control
    path = Path(video_path)
    if path.suffix.lower() != '.mp4':
        raise ValueError('Video must use .mp4.')
    path.parent.mkdir(parents=True, exist_ok=True)
    camera = _Camera(np.array([-.15, 0, 1.05]))
    stage = c.sim.stage
    prim = stage.GetPrimAtPath(CAMERA_PATH)
    cam = UsdGeom.Camera(prim)
    cam.ClearXformOpOrder()
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(-3.3, -3.3, 3.15), Gf.Vec3d(-.05, 0, 1.0), Gf.Vec3d(0, 0, 1)).GetInverse())
    def color_object(prim_path, name, color):
        material_path = '/World/VideoMaterials/'+name
        material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=.7)
        material_cfg.func(material_path, material_cfg)
        binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path))
        for purpose in ('preview','full'):
            binding.Bind(UsdShade.Material.Get(stage,material_path),
                         bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose=purpose)
    # Keep all physical bodies' materials intact; only goal outlines receive display materials.
    colors = [(0.1, .85, .95), (1., .7, .1), (.95, .3, .75)]
    for index, (goal, color) in enumerate(zip(cfg['goals_xy_m'], colors)):
        x, y = goal; w = cfg['goal_half_width_m']; z = task.ball_config['table_top_m']+.003
        line = UsdGeom.BasisCurves.Define(stage, f'/World/GoalOutline{index}')
        line.CreateTypeAttr('linear'); line.CreateWrapAttr('nonperiodic')
        line.CreateCurveVertexCountsAttr([5])
        line.CreatePointsAttr([(x-w,y-w,z),(x+w,y-w,z),(x+w,y+w,z),(x-w,y+w,z),(x-w,y-w,z)])
        line.CreateWidthsAttr([.008]*5); line.CreateDisplayColorAttr([color])
        color_object(f'/World/GoalOutline{index}',f'Goal{index}',color)
    font = ImageFont.load_default()
    report = dict(status='STARTING', renderer='Isaac Sim Kit RTX', physics_hz=cfg['physics_hz'],
                  capture_hz=60, playback_fps=15, playback_speed=.25, trials=[], frames=0,
                  scope='three measured returns of validation_0; illustration, not a new 36-shot evaluation',
                  source_sha256={})
    for name in ['src/airbot_tt/recording.py','src/airbot_tt/runtime.py','src/airbot_tt/shot_scene.py',
                 'src/airbot_tt/planning.py','src/airbot_tt/trajectory.py','configs/strike.json']:
        report['source_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    step_count = round(1/(60*c.dt))
    assert math.isclose(step_count*c.dt, 1/60)
    partial = path.with_name('.'+path.stem+'.partial.mp4')
    serve = serves('validation', cfg)[0]
    try:
        with imageio.get_writer(str(partial), format='FFMPEG', fps=15, codec='libx264',
                               pixelformat='yuv420p', macro_block_size=None, output_params=['-crf','18']) as writer:
            for index, goal in enumerate(cfg['goals_xy_m']):
                plan = plan_return(c.model,c.checker,cfg['ready_q_rad'],
                    np.r_[cfg['serve_position_m'],cfg['serve_velocity_m_s']],goal,
                    np.array(task.config['base_position_m']),task.ball_config,cfg)
                if not plan['accepted']:raise RuntimeError(str(plan))
                task.reset(serve)
                for _ in range(12):c.sim.render()
                writes = c.state_writes; started = False; samples = []; step = 0
                def capture():
                    c.sim.render()
                    frame = Image.fromarray(camera.rgb())
                    # Screen-space marker follows the measured ball centre; no simulated state is modified.
                    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(-3.3,-3.3,3.15),Gf.Vec3d(-.05,0,1.),Gf.Vec3d(0,0,1))
                    point = view.Transform(Gf.Vec3d(*map(float,task.state[:3])))
                    if point[2] < 0:
                        focal_pixels=24/20.955*WIDTH
                        px=WIDTH/2+focal_pixels*point[0]/(-point[2])
                        py=HEIGHT/2-focal_pixels*point[1]/(-point[2])
                        ImageDraw.Draw(frame).ellipse((px-5,py-5,px+5,py+5),fill=(255,140,0),outline='white',width=1)
                    banner = Image.new('RGB',(WIDTH//2,38),(17,24,36))
                    draw = ImageDraw.Draw(banner)
                    label = ['LEFT','CENTRE','RIGHT'][index]
                    draw.text((9,4), f'M5 | {label} target | PhysX + ball marker | 0.25x', font=font, fill='white')
                    draw.text((9,21), f't = {task.time:.3f} s   {task.events.outcome or task.events.phase}', font=font,
                              fill=tuple(int(v*255) for v in colors[index]))
                    frame.paste(banner.resize((WIDTH,76),Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS),(0,0))
                    writer.append_data(np.asarray(frame)); report['frames'] += 1
                    return np.asarray(frame)
                capture()
                while task.events.outcome is None:
                    if not started and task.time+1e-10 >= cfg['reaction_delay_s']:
                        c.start_prepared_strike(plan['path']); started = True
                    task.step(); step += 1
                    if step % step_count == 0 or task.events.outcome is not None:
                        samples.append(dict(time_s=task.time, ball=task.state.tolist(),
                                            tcp=c.state()['tcp_world'], q_rad=c.q().tolist()))
                        frame = capture()
                if c.state_writes != writes:raise RuntimeError('Unexpected joint state write')
                if task.events.outcome != 'legal_return':
                    report['trials'].append(dict(goal=goal,outcome=task.events.outcome,events=task.events.events))
                    report['status']='FAIL'
                    error=RuntimeError(task.events.outcome);error.report=report
                    raise error
                for _ in range(12):writer.append_data(frame);report['frames']+=1
                report['trials'].append(dict(goal=goal,serve=serve,events=task.events.events,
                                              outcome=task.events.outcome,samples=samples))
        partial.replace(path)
    finally:
        camera.close(); partial.unlink(missing_ok=True)
    report.update(status='PASS', video=str(path), duration_s=report['frames']/15)
    return report
