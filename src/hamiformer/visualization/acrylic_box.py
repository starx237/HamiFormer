from hamiformer.utils.paths import project_root
import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

def material(name, color, metallic=0, roughness=0.5, texture=0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Metallic'].default_value = metallic
    p.inputs['Roughness'].default_value = roughness
    if texture:
        noise = m.node_tree.nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 70
        bump = m.node_tree.nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = texture
        bump.inputs['Distance'].default_value = 0.006
        m.node_tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
        m.node_tree.links.new(bump.outputs['Normal'], p.inputs['Normal'])
    return m

def tube(name, points, radius, mat):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    spline = curve.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for p, q in zip(spline.points, points):
        p.co = (*q, 1)
    ob = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob

def aim(obj, at):
    obj.rotation_euler = (Vector(at) - obj.location).to_track_quat('-Z', 'Y').to_euler()

def acrylic_material(reflection=0.025):
    m = bpy.data.materials.new('Clear acrylic / visibility-first')
    m.use_nodes = True
    nodes = m.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    clear = nodes.new('ShaderNodeBsdfTransparent')
    glossy = nodes.new('ShaderNodeBsdfGlossy')
    glossy.inputs['Color'].default_value = (0.83, 0.94, 1, 1)
    glossy.inputs['Roughness'].default_value = 0.06
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = reflection
    m.node_tree.links.new(clear.outputs[0], mix.inputs[1])
    m.node_tree.links.new(glossy.outputs[0], mix.inputs[2])
    m.node_tree.links.new(mix.outputs[0], out.inputs['Surface'])
    return m

def scene(data, method, output):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    if data.get('device') == 'GPU':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = dev.type == 'CUDA'
        if not any((dev.use for dev in prefs.devices)):
            raise RuntimeError('CUDA render device unavailable')
        sc.cycles.device = 'GPU'
    sc.cycles.samples = data.get('render_samples', 96)
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.use_denoising = True
    sc.render.threads_mode = 'FIXED'
    sc.render.threads = 8
    sc.render.resolution_x = data.get('resolution', 900)
    sc.render.resolution_y = data.get('resolution', 900)
    sc.render.resolution_percentage = 100
    sc.world.use_nodes = True
    sc.world.node_tree.nodes['Background'].inputs[0].default_value = (0.72, 0.79, 0.9, 1)
    sc.world.node_tree.nodes['Background'].inputs[1].default_value = 0.35
    sc.render.film_transparent = True
    sc.use_nodes = False
    if not bpy.app.build_options.opencolorio:
        raise RuntimeError('Use a full Blender distribution with OpenColorIO enabled.')
    sc.view_settings.view_transform = 'Filmic'
    sc.view_settings.look = 'Medium High Contrast'
    floor = material('Porcelain ground', (0.83, 0.88, 0.91), roughness=0.6)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.035))
    ob = bpy.context.object
    ob.name = 'Physical floor z=0'
    ob.dimensions = (2, 2, 0.07)
    ob.data.materials.append(floor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = ob.modifiers.new('Soft stage edges', 'BEVEL')
    bevel.width = 0.012
    bevel.segments = 3
    style = data.get('display_style', {})
    acrylic = acrylic_material(style.get('acrylic_reflection', 0.025))
    for name, location, dimensions in [('Left', (-1.003, 0, 1), (0.006, 2, 2)), ('Right', (1.003, 0, 1), (0.006, 2, 2)), ('Back', (0, 1.003, 1), (2, 0.006, 2)), ('Front', (0, -1.003, 1), (2, 0.006, 2))]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=location)
        wall = bpy.context.object
        wall.name = name + ' acrylic wall'
        wall.dimensions = dimensions
        wall.data.materials.append(acrylic)
        wall.visible_shadow = False
    rail = material('Polished acrylic edge tint', style.get('rail_color', (0.4, 0.64, 0.72)), 0.25, 0.22)
    for x in (-1, 1):
        for y in (-1, 1):
            tube('Boundary upright', [(x, y, 0), (x, y, 2)], 0.005, rail)
    for z in (0, 2):
        for v in (-1, 1):
            tube('Boundary rail', [(-1, v, z), (1, v, z)], 0.005, rail)
            tube('Boundary rail', [(v, -1, z), (v, 1, z)], 0.005, rail)
    grid = material('Floor reference grid', (0.43, 0.5, 0.56), roughness=0.9)
    for k in range(-4, 5):
        v = k / 5
        tube('Grid', [(-1, v, 0.0003), (1, v, 0.0003)], 0.001, grid)
        tube('Grid', [(v, -1, 0.0003), (v, 1, 0.0003)], 0.001, grid)
    colors = data['colors']
    q = data[method]
    for i, (pos, attrs, color) in enumerate(zip(q, data['attrs'], colors)):
        e = float(attrs[2])
        lo, hi = data.get('restitution_range', (0.9, 0.98))
        if hi <= lo:
            raise ValueError('Invalid restitution display range')
        t = min(1, max(0, (e - lo) / (hi - lo)))
        t = t * t * (3 - 2 * t)
        if style.get('palette_srgb', False):
            color = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in color]
        mat = material('Object %d restitution %.4f' % (i, e), color, metallic=t, roughness=0.95 - 0.83 * t, texture=0.65 * (1 - t) ** 2)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=attrs[1], location=pos)
        ball = bpy.context.object
        ball.name = 'Ball_%02d' % i
        ball.data.materials.append(mat)
        ball.visible_glossy = False
        for poly in ball.data.polygons:
            poly.use_smooth = True
    steel = material('Spring stainless steel', style.get('spring_color', (0.48, 0.55, 0.61)), style.get('spring_metallic', 0.9), style.get('spring_roughness', 0.23))
    for i, j in data['edges']:
        a, b = (Vector(q[i]), Vector(q[j]))
        length = (b - a).length
        if length < 1e-08:
            continue
        d = (b - a).normalized()
        start = a + d * data['attrs'][i][1]
        end = b - d * data['attrs'][j][1]
        if (end - start).dot(d) <= 0:
            continue
        ref = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((0, 1, 0))
        u = d.cross(ref).normalized()
        v = d.cross(u).normalized()
        rest = Vector(data['initial'][j]) - Vector(data['initial'][i])
        turns = max(5, min(18, round(rest.length / 0.07)))
        points = []
        for k in range(turns * 24 + 1):
            s = k / (turns * 24)
            ang = s * turns * math.tau
            taper = min(1, s / 0.06, (1 - s) / 0.06)
            p = start + (end - start) * s + 0.016 * taper * (u * math.cos(ang) + v * math.sin(ang))
            points.append(p)
        tube('Helical spring %d-%d' % (i, j), points, 0.0028, steel)
    for loc, power, size in [((2, -3, 5), 450, 3), ((-3, -1, 3), 300, 2.5), ((1, 3, 4), 500, 2)]:
        bpy.ops.object.light_add(type='AREA', location=loc)
        light = bpy.context.object
        light.data.energy = power
        light.data.shape = 'DISK'
        light.data.size = size
        aim(light, (0, 0, 0.8))
    camera_view = data.get('camera_view')
    if camera_view:
        target = Vector(camera_view.get('target', (0, 0, 1)))
        distance = float(camera_view.get('distance', 7.0))
        elev = math.radians(float(camera_view['elev_degrees']))
        azim = math.radians(float(camera_view['azim_degrees']))
        horizontal = distance * math.cos(elev)
        location = target + Vector((horizontal * math.cos(azim), horizontal * math.sin(azim), distance * math.sin(elev)))
    else:
        target = Vector((0, 0, 0.95))
        location = Vector((3.7, -5.6, 3.1))
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    aim(cam, target)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = data.get('ortho_scale', 3.35)
    sc.camera = cam
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_depth = '8'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.filepath = str(output / (method + '.png'))
    if data.get('save_blend', True):
        bpy.ops.wm.save_as_mainfile(filepath=str(output / (method + '.blend')))
    bpy.ops.render.render(write_still=True)
if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:]
    data = json.loads(Path(args[0]).read_text())
    output = Path(args[1])
    output.mkdir(parents=True, exist_ok=True)
    for method, positions in data['positions'].items():
        values = positions if isinstance(positions[0][0], list) else [positions]
        for index, q in enumerate(values):
            name = f'{method}_{index:06d}'
            data[name] = q
            scene(data, name, output)
            del data[name]
