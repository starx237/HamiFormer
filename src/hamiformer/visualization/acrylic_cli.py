from hamiformer.utils.paths import project_root
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--fps', type=float, default=30)
    a = p.parse_args()
    if not 0 < a.fps <= 100:
        p.error('fps must be in (0,100]')
    blender = os.environ.get('BLENDER') or shutil.which('blender')
    if not blender:
        raise RuntimeError('Set BLENDER to a full Blender 3.6 executable')
    data = json.loads(a.input.read_text())
    defaults = {'ortho_scale': 3.15, 'render_samples': 128, 'resolution': 900, 'save_blend': False, 'camera_view': {'elev_degrees': 23, 'azim_degrees': 38, 'target': [0, 0, 1], 'distance': 7}, 'display_style': {'acrylic_reflection': 0.017, 'rail_color': [0.4, 0.64, 0.72], 'palette_srgb': True, 'spring_color': [0.1, 0.13, 0.16], 'spring_metallic': 0.65, 'spring_roughness': 0.38}}
    data = {**defaults, **data}
    names = list(data['positions'])
    shapes = [np.asarray(data['positions'][name]).shape for name in names]
    if any((len(s) not in (2, 3) or s[-1] != 3 for s in shapes)):
        raise ValueError('positions must contain [objects,3] or [frames,objects,3] arrays')
    lengths = [s[0] if len(s) == 3 else 1 for s in shapes]
    if len(set(lengths)) != 1:
        raise ValueError('Methods must have equal frame counts')
    data['positions'] = {f'm{i}': data['positions'][name] for i, name in enumerate(names)}
    a.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hamiformer-render-') as temporary:
        work = Path(temporary)
        scene = work / 'scene.json'
        scene.write_text(json.dumps(data))
        subprocess.run([blender, '--background', '--python-exit-code', '1', '--python', str(Path(__file__).with_name('acrylic_box.py')), '--', str(scene), str(work)], check=True)
        frames = []
        for index in range(lengths[0]):
            panels = []
            for method in range(len(names)):
                rgba = Image.open(work / f'm{method}_{index:06d}.png').convert('RGBA')
                bg = Image.new('RGBA', rgba.size, 'white')
                bg.alpha_composite(rgba)
                panels.append(bg.convert('RGB'))
            width, height = panels[0].size
            frame = Image.new('RGB', (width * len(names), height + 40), 'white')
            draw = ImageDraw.Draw(frame)
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', 24)
            except OSError:
                font = ImageFont.load_default()
            for i, panel in enumerate(panels):
                frame.paste(panel, (i * width, 40))
                draw.text((i * width + width / 2, 20), f'{names[i]} | {index / a.fps:.2f}s', font=font, fill='#253343', anchor='mm')
            frames.append(frame)
        if len(frames) == 1:
            frames[0].save(a.output / 'comparison.png')
        else:
            indices = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
            sheet = Image.new('RGB', (300 * len(indices), 314), 'white')
            for i, k in enumerate(indices):
                sheet.paste(frames[k].resize((300, 314)), (300 * i, 0))
            palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
            durations = [10 * (round((i + 1) * 100 / a.fps) - round(i * 100 / a.fps)) for i in range(len(frames))]
            quantized[0].save(a.output / 'comparison.gif', save_all=True, append_images=quantized[1:], duration=durations, loop=0, disposal=2, optimize=False)
if __name__ == '__main__':
    main()
