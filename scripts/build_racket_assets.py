"""Derive SI meshes, unit-mass inertia and TCP from the reviewed CAD assembly.

Run with CadQuery 2.8 / OCP 7.9. No density or measured mass is inferred from STEP.
The generated metadata records a homogeneous-solid distribution per component.
"""

import hashlib
import json
from pathlib import Path
import struct

import cadquery as cq
import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'assets/derived/fixer_fit/airbot_j6_fixer_paddle_fit.STEP'
OUTPUT = ROOT / 'assets/derived/racket'
PARTS = {'fixer': 'Fixer_Airbot_2_Flat_Mount', 'paddle': 'Paddle_Centred_To_Stop'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_mesh(shape, path, angular_tolerance=0.16):
    vertices, faces = shape.tessellate(0.08, angular_tolerance)
    points = np.array([p.toTuple() for p in vertices]) / 1000
    triangles = points[np.array(faces)]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    keep = lengths > 1e-16
    triangles, cross, lengths = triangles[keep], cross[keep], lengths[keep]
    signed_volume = np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6
    assert abs(signed_volume/(shape.Volume()*1e-9)-1) < .005
    with path.open('wb') as stream:
        stream.write(b'AIRBOT tool CAD mesh; metres'.ljust(80, b'\0'))
        stream.write(struct.pack('<I', len(triangles)))
        for triangle, normal in zip(triangles, cross/lengths[:, None]):
            stream.write(struct.pack('<12fH', *normal, *triangle.ravel(), 0))
    return {'mesh': str(path.relative_to(ROOT)), 'sha256': digest(path),
            'bounds_link6_m': [points.min(0).tolist(), points.max(0).tolist()]}


def main():
    assembly = cq.Assembly.importStep(str(SOURCE), unit='MM')
    shapes = {c.name: c.obj.moved(assembly.loc*c.loc) for c in assembly.children}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': digest(SOURCE),
                'length_unit': 'm', 'frame': 'link6',
                'distribution': 'homogeneous CAD solid per component; requires user confirmation',
                'components': {}}
    for name, part in PARTS.items():
        shape = shapes[part]
        assert len(shape.Solids()) == 1 and shape.isValid()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape.wrapped, props)
        tensor = props.MatrixOfInertia()
        # OCP tensor about the volume centroid: mm^5 / mm^3 -> mm^2 -> m^2.
        unit_tensor = np.array([[tensor.Value(i+1, j+1) for j in range(3)] for i in range(3)]) / props.Mass()*1e-6
        assert np.linalg.eigvalsh(unit_tensor).min() > 0
        data = export_mesh(shape, OUTPUT/f'{name}.STL')
        data.update(volume_m3=props.Mass()*1e-9,
                    center_of_mass_link6_m=(np.array(props.CentreOfMass().Coord())/1000).tolist(),
                    inertia_per_kg_m2=unit_tensor.tolist())
        if name == 'paddle':
            # The thick handle ends below link6 Z=205 mm. Separate it from
            # the blade so convex cooking cannot bridge across the flat face.
            data['collision_split_link6_z_m'] = 0.205
            data['collision_parts'] = []
            for label, z0, z1 in [('handle', 0, 205), ('blade', 205, 400)]:
                box = cq.Solid.makeBox(400, 400, z1-z0, cq.Vector(-200, -200, z0))
                piece = shape.intersect(box)
                assert piece.isValid() and len(piece.Solids()) == 1
                data['collision_parts'].append(export_mesh(piece, OUTPUT/f'paddle_{label}_collision.STL', 0.08))
        metadata['components'][name] = data
    paddle = shapes[PARTS['paddle']]
    planes = [f for f in paddle.Faces() if f.geomType() == 'PLANE']
    face = max(planes, key=lambda f: f.Area())
    normal = np.array(face.normalAt().toTuple())
    # Choose the side whose outward normal has positive link6 Y.
    if normal[1] < 0:
        face = max((f for f in planes if np.array(f.normalAt().toTuple()) @ normal < -.999), key=lambda f: f.Area())
        normal = np.array(face.normalAt().toTuple())
    butt = min(planes, key=lambda f: f.Center().z)
    forward = np.array((face.Center()-butt.Center()).toTuple())
    forward -= normal*(normal @ forward)
    forward /= np.linalg.norm(forward)
    rotation = np.column_stack((forward, np.cross(normal, forward), normal))
    assert np.linalg.det(rotation) > .999999
    metadata['racket_tcp_candidate'] = {
        'definition': 'centroid of positive-link6-Y planar striking face; X toward blade, Z outward face normal, Y=Z cross X',
        'position_link6_m': (np.array(face.Center().toTuple())/1000).tolist(),
        'quaternion_link6_wxyz': Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist(),
        'face_area_m2': face.Area()*1e-6,
    }
    target = OUTPUT/'geometry.json'
    target.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
