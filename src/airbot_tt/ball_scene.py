"""Isolated PhysX ball testbed: table, solid net and fixed CAD paddle.

The paddle is held by a static fixture, not the driven AIRBOT articulation.
The net is a rigid slab; flexible mesh and rubber compliance are not modeled.
"""

import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_rotate_inverse

from .ball import drag_force, load_ball_config
from .collision import stl_triangles
from .config import ROOT
from .tool import racket_model


class BallTestbed:
    def __init__(self, dt, device="cpu", *, sim=None, static_paddle=True, contact_filters=(), root="/World"):
        self.config = c = load_ball_config()
        self.sim = sim or sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=dt, device=device, gravity=(0, 0, -c["gravity_m_s2"]),
            physx=sim_utils.PhysxCfg(bounce_threshold_velocity=0.01, enable_ccd=True),
        ))
        self.dt = dt
        self.device = device
        stage = self.sim.stage
        collision = sim_utils.CollisionPropertiesCfg(contact_offset=c["contact_offset_m"], rest_offset=0.0)

        def material(restitution):
            return sim_utils.RigidBodyMaterialCfg(
                static_friction=c["friction"], dynamic_friction=c["friction"],
                restitution=restitution, restitution_combine_mode="min", friction_combine_mode="min",
            )

        table = sim_utils.CuboidCfg(size=tuple(c["table_size_m"]), collision_props=collision,
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                                   physics_material=material(c["table_restitution"]))
        table.func(f"{root}/Table", table, translation=(0, 0, c["table_top_m"] - c["table_size_m"][2] / 2))
        net = sim_utils.CuboidCfg(size=(c["net_thickness_m"], c["table_size_m"][1], c["net_height_m"]),
                                 rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                                 collision_props=collision, physics_material=material(c["net_restitution"]))
        net.func(f"{root}/Net", net, translation=(0, 0, c["table_top_m"] + c["net_height_m"] / 2))

        if static_paddle:
            # Reuse the reviewed CAD mesh and exact named TCP transform. In this
            # fixture TCP X is world +Z and the face normal TCP Z is world +X.
            model = racket_model(ROOT / "configs/racket.json")
            points = stl_triangles(ROOT / model.tool_geometry["components"]["paddle"]["mesh"]).reshape(-1, 3)
            points_tcp = (points - model.link6_tcp[:3, 3]) @ model.link6_tcp[:3, :3]
            self.paddle_center = np.array([-1.0, 0.0, 1.1])
            fixture_rotation = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
            points_world = points_tcp @ fixture_rotation.T + self.paddle_center
            vertices, indices = np.unique(points_world, axis=0, return_inverse=True)
            stage = self.sim.stage
            mesh = UsdGeom.Mesh.Define(stage, f"{root}/Paddle")
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
            mesh.CreateFaceVertexCountsAttr([3] * (len(points_world) // 3))
            mesh.CreateFaceVertexIndicesAttr(indices.tolist())
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            # Static triangles preserve the actual flat striking face. A single
            # convex hull bridges the thicker handle to the blade and tilts it.
            # This is a reference fixture, not a collider for the moving arm.
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
            offset = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            offset.CreateContactOffsetAttr(c["contact_offset_m"])
            offset.CreateRestOffsetAttr(0.0)
            mat = material(c["paddle_restitution"])
            mat.func(f"{root}/PaddleMaterial", mat)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                UsdShade.Material.Get(stage, f"{root}/PaddleMaterial"), materialPurpose="physics")
            if static_paddle == 'compound':
                UsdPhysics.CollisionAPI(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
                for index, part in enumerate(model.tool_geometry['components']['paddle']['collision_parts']):
                    pts = stl_triangles(ROOT/part['mesh']).reshape(-1,3)
                    pts = ((pts-model.link6_tcp[:3,3]) @ model.link6_tcp[:3,:3]) @ fixture_rotation.T + self.paddle_center
                    pts, ids = np.unique(pts, axis=0, return_inverse=True)
                    child = UsdGeom.Mesh.Define(stage, f'{root}/Paddle/Collision{index}')
                    child.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
                    child.CreateFaceVertexCountsAttr([3]*(len(ids)//3))
                    child.CreateFaceVertexIndicesAttr(ids.tolist())
                    UsdPhysics.CollisionAPI.Apply(child.GetPrim())
                    UsdPhysics.MeshCollisionAPI.Apply(child.GetPrim()).CreateApproximationAttr('convexHull')
                    offset = PhysxSchema.PhysxCollisionAPI.Apply(child.GetPrim())
                    offset.CreateContactOffsetAttr(c['contact_offset_m'])
                    offset.CreateRestOffsetAttr(0.)
            self.paddle_mesh_sha256 = model.tool_geometry["components"]["paddle"]["sha256"]

        self.ball = RigidObject(RigidObjectCfg(
            prim_path=f"{root}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=c["radius_m"], mass_props=sim_utils.MassPropertiesCfg(mass=c["mass_kg"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0, angular_damping=0, sleep_threshold=0,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                    max_depenetration_velocity=10,
                ),
                collision_props=collision, physics_material=material(1.0), activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.3, 2.0)),
        ))
        # Lab 2.1 exposes scene CCD, but body CCD is a native PhysX USD attribute.
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(f"{root.replace('.*', '0')}/Ball")).CreateEnableCCDAttr(True)
        self.sensor = ContactSensor(ContactSensorCfg(prim_path=f"{root}/Ball", update_period=0,
                                                       filter_prim_paths_expr=list(contact_filters)))
        if sim is None:
            self.sim.reset()
        self.reset_count = 0

    def reset(self, position, velocity, quaternion=(1, 0, 0, 0)):
        """The only method that writes ball state; each call starts a new trial."""
        state = self.ball.data.default_root_state.clone()
        state[0, :3] = torch.as_tensor(position, device=self.device)
        state[0, 3:7] = torch.as_tensor(quaternion, device=self.device)
        state[0, 7:10] = torch.as_tensor(velocity, device=self.device)
        state[0, 10:13] = 0
        self.ball.write_root_state_to_sim(state)
        self.ball.reset()
        self.sensor.reset()
        self.reset_count += 1

    def prepare_step(self, *, drag=True):
        force = drag_force(self.ball.data.root_lin_vel_w[0].cpu().numpy(), self.config) if drag else np.zeros(3)
        # Lab 2.1 accepts body-local forces only. Convert the world drag using
        # the live ball orientation, including any rotation from contact.
        force_world = torch.as_tensor(force, dtype=torch.float32, device=self.device).reshape(1, 3)
        force_body = quat_rotate_inverse(self.ball.data.root_quat_w, force_world)
        self.ball.set_external_force_and_torque(
            force_body.reshape(1, 1, 3), torch.zeros((1, 1, 3), device=self.device),
        )
        self.ball.write_data_to_sim()

    def read_step(self):
        self.ball.update(self.dt)
        self.sensor.update(self.dt)
        state = self.ball.data.root_state_w[0].cpu().numpy().copy()
        contact = self.sensor.data.net_forces_w[0, 0].cpu().numpy().copy()
        if not np.isfinite(state).all() or not np.isfinite(contact).all():
            raise RuntimeError("Non-finite ball state or contact force.")
        return np.r_[state[:3], state[7:10]], contact

    def step(self, *, drag=True):
        self.prepare_step(drag=drag)
        self.sim.step(render=False)
        return self.read_step()
