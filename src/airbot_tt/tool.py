"""Build a fixed, inertial tool URDF from reviewed CAD and explicit load data."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .config import ROOT, URDF
from .kinematics import Kinematics, pose_matrix


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def racket_model(profile_path):
    """Create a model only when masses, distribution and TCP are confirmed."""
    path = Path(profile_path).resolve()
    profile = json.loads(path.read_text())
    if profile.get('confirmed') is not True:
        raise ValueError('Tool masses, homogeneous CAD distribution and TCP require confirmation.')
    geometry_path = ROOT/'assets/derived/racket/geometry.json'
    geometry = json.loads(geometry_path.read_text())
    if _hash(ROOT/geometry['source']) != geometry['source_sha256']:
        raise ValueError('Racket assembly differs from the reviewed geometry metadata.')
    tcp = geometry['racket_tcp_candidate']
    link6_tcp = pose_matrix(tcp['position_link6_m'] + tcp['quaternion_link6_wxyz'])
    xml = ET.parse(URDF).getroot()
    # Generated URDF lives outside the original asset directory.
    for element in xml.findall('.//mesh'):
        element.set('filename', str((URDF.parent/element.get('filename')).resolve()))
    for name, data in geometry['components'].items():
        mass = float(profile['mass_kg'][name])
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f'{name} mass must be finite and positive.')
        mesh = ROOT/data['mesh']
        if _hash(mesh) != data['sha256']:
            raise ValueError(f'{name} mesh does not match the CAD metadata.')
        link = ET.SubElement(xml, 'link', name=name)
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, data['center_of_mass_link6_m'])), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=str(mass))
        tensor = mass*np.array(data['inertia_per_kg_m2'])
        ET.SubElement(inertial, 'inertia', **{key: str(tensor[i, j]) for key, i, j in
                      [('ixx', 0, 0), ('ixy', 0, 1), ('ixz', 0, 2), ('iyy', 1, 1), ('iyz', 1, 2), ('izz', 2, 2)]})
        for kind in ('visual', 'collision'):
            parts = data.get('collision_parts', [data]) if kind == 'collision' else [data]
            for part in parts:
                part_mesh = ROOT/part['mesh']
                if _hash(part_mesh) != part['sha256']:
                    raise ValueError(f'{name} collision mesh does not match metadata.')
                element = ET.SubElement(link, kind)
                ET.SubElement(ET.SubElement(element, 'geometry'), 'mesh', filename=str(part_mesh))
                if kind == 'visual':
                    material = ET.SubElement(element, 'material', name=name+'_material')
                    ET.SubElement(material, 'color', rgba='0.1 0.55 0.6 1' if name == 'fixer' else '0.95 0.5 0.1 1')
        joint = ET.SubElement(xml, 'joint', name=f'{name}_mount', type='fixed')
        ET.SubElement(joint, 'parent', link='link6' if name == 'fixer' else 'fixer')
        ET.SubElement(joint, 'child', link=name)
        ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
    output = ROOT/'.agents/runs/foundation/racket'
    output.mkdir(parents=True, exist_ok=True)
    generated = output/'airbot_racket.urdf'
    ET.indent(xml)
    ET.ElementTree(xml).write(generated, encoding='utf-8', xml_declaration=True)
    flange = np.eye(4)
    flange[2, 3] = .0865
    model = Kinematics(generated, tcp_transform=np.linalg.solve(flange, link6_tcp))
    model.tool_profile = profile
    model.tool_geometry = geometry
    model.link6_tcp = link6_tcp
    return model
