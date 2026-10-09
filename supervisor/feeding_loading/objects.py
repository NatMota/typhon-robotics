"""Versioned block appearance; legacy collisions and dynamics remain unchanged."""
import xml.etree.ElementTree as ET
from pathlib import Path


def adapt_block(tree, config):
    block = next(b for b in tree.iter('body') if b.get('name') == 'cube_main')
    obj = config['object']
    half = [obj[k] / 2 for k in ('width_m', 'depth_m', 'height_m')]
    geoms = list(block.iter('geom'))
    for i, geom in enumerate(geoms):
        for key in ('mesh', 'material', 'density', 'fromto'):
            geom.attrib.pop(key, None)
        geom.set('type', 'box')
        geom.set('size', ' '.join(map(str, half)))
        geom.set('pos', '0 0 0')
        geom.set('quat', '1 0 0 0')
        geom.set('rgba', '1 .34 .025 1')
        geom.set('mass', str(obj['mass_kg'] if i == 0 else 0.))
        geom.set('friction', '1 .005 .0001')
        if i:
            geom.set('contype', '0')
            geom.set('conaffinity', '0')
    version = obj.get('visual_version', 'm1_seven_segment_v1')
    if version == 'duplo_31110pb074_v2':
        return _duplo_visual_v2(block, obj, geoms)
    if version != 'm1_seven_segment_v1':
        raise ValueError('Unknown block visual version: ' + version)
    # Frozen legacy appearance: this seven-segment 2 IS 180-degree symmetric.
    sx, sy = half[0] * .52, half[1] * .63
    stroke = min(half[:2]) * .10
    z = half[2] + .00035
    for i, (x, y, dx, dy) in enumerate([
        (0, sy, sx, stroke), (sx, sy / 2, stroke, sy / 2),
        (0, 0, sx, stroke), (-sx, -sy / 2, stroke, sy / 2),
        (0, -sy, sx, stroke),
    ]):
        ET.SubElement(block, 'geom', name=f'block_mark_2_{i}', type='box',
                      pos=f'{x} {y} {z}', size=f'{dx} {dy} .0002',
                      rgba='.02 .12 .9 1', contype='0', conaffinity='0',
                      group='1', mass='0')
    return block


def _duplo_visual_v2(block, obj, legacy_geoms):
    """Photographic numeral silhouette and studs on the brick's physical top end.

    The brick lies on its side: printed face is local +Z, studs point +Y.
    This is an appearance update; the full envelope remains the collision proxy.
    """
    import numpy as np
    from PIL import Image
    width, depth, height = (obj[k] for k in ('width_m', 'depth_m', 'height_m'))
    # Approximate stud proportions; no new contact or mass is introduced.
    projection = depth * (4.8 / 43.)
    body_depth, body_y = depth - projection, -projection / 2
    for geom in legacy_geoms:
        geom.set('rgba', '0 0 0 0')

    def visual(name, shape, pos, size, color, **kwargs):
        return ET.SubElement(block, 'geom', name=name, type=shape,
            pos=' '.join(map(str, pos)), size=' '.join(map(str, size)), rgba=color,
            contype='0', conaffinity='0', mass='0', group='1', **kwargs)

    visual('duplo_v2_body', 'box', [0, body_y, 0],
           [width / 2, body_depth / 2, height / 2], '1 .34 .025 1')
    for i, (x, z) in enumerate(( (x, z) for x in [-width / 4, width / 4]
                                for z in [-height / 4, height / 4] )):
        visual('duplo_v2_stud_' + str(i), 'cylinder', [x, depth / 2 - projection / 2, z],
               [min(width, height) * .15, projection / 2], '1 .34 .025 1',
               quat='.7071067811865476 -.7071067811865476 0 0')
    path = Path(__file__).resolve().parents[2] / 'assets/duplo_31110pb074_v2/numeral_2_mask.png'
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L')) >= 128
    # Uniform scale preserves the observed numeral aspect ratio. Scanline boxes
    # avoid texture mirroring/filtering and inherit the same rigid body rotation.
    canvas = min(width * .92, body_depth * .78)
    pitch = canvas / max(mask.shape)
    n = 0
    for row, pixels in enumerate(mask):
        edges = np.diff(np.r_[False, pixels, False].astype(int))
        for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            x = ((start + end) / 2 - mask.shape[1] / 2) * pitch
            y = body_y + (mask.shape[0] / 2 - row - .5) * pitch
            visual('duplo_v2_numeral_' + str(n), 'box', [x, y, height / 2 + .00004],
                   [(end - start) * pitch / 2, pitch / 2, .00004], '.02 .12 .9 1')
            n += 1
    return block


def add_workstations(tree, config):
    world = tree.find('worldbody')
    ws, task = config['workspace'], config['task']
    x, y = ws['staging_position_xy_m']
    stage = ET.SubElement(world, 'body', name='m1_stage', mocap='true',
                          pos=f'{x} {y} {ws["table_z_m"]}')
    half = ws['staging_size_m'] / 2
    for i, (px, py, dx, dy) in enumerate([
        (0, -half, half, .002), (0, half, half, .002),
        (-half, 0, .002, half), (half, 0, .002, half),
    ]):
        ET.SubElement(stage, 'geom', name=f'staging_border_{i}', type='box',
                      pos=f'{px} {py} .0003', size=f'{dx} {dy} .00025',
                      rgba='.05 .35 .12 1', contype='0', conaffinity='0', group='1')
    # Arrow specifies staging yaw, independent of the object's blue numeral.
    ET.SubElement(stage, 'geom', name='stage_orientation', type='box',
                  pos=f'0 {half + .008} .0003', size='.013 .002 .00025',
                  rgba='.05 .35 .12 1', contype='0', conaffinity='0', group='1')
    if task['type'] == 'loading':
        x, y = task['target_position_xy_m']
        fixture = ET.SubElement(world, 'body', name='m1_fixture', mocap='true',
                                pos=f'{x} {y} {ws["table_z_m"]}')
        inner = ws['fixture_inner_size_m'] / 2
        thick, height, floor = (ws[k] for k in
            ('fixture_wall_thickness_m', 'fixture_wall_height_m', 'fixture_floor_height_m'))
        if floor > 0:
            ET.SubElement(fixture, 'geom', name='fixture_floor', type='box',
                          pos=f'0 0 {floor / 2}', size=f'{inner + thick} {inner + thick} {floor / 2}',
                          rgba='.32 .36 .40 1', friction='1 .005 .0001', group='1')
        for i, (px, py, dx, dy) in enumerate([
            (0, -inner - thick / 2, inner + thick, thick / 2),
            (0, inner + thick / 2, inner + thick, thick / 2),
            (-inner - thick / 2, 0, thick / 2, inner),
            (inner + thick / 2, 0, thick / 2, inner),
        ]):
            ET.SubElement(fixture, 'geom', name=f'fixture_wall_{i}', type='box',
                          pos=f'{px} {py} {floor + height / 2}', size=f'{dx} {dy} {height / 2}',
                          rgba='.42 .46 .50 1', friction='1 .005 .0001', group='1')
