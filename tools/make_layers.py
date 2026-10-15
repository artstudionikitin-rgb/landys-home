"""Вырезает из картинок работ только изменившуюся область относительно расчищенного участка.

Результат — прозрачные WebP-слои, которые в прототипе накладываются поверх
plot-v2-cleared.jpg в любом сочетании. Запуск:
    python make_layers.py <папка img>
Нужны Pillow и NumPy.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SIZE = 1600          # как у JPEG-картинок в прототипе
THRESHOLD = 30       # насколько пиксель должен отличаться, чтобы попасть в слой (0–255)
JOBS = {"pond": "plot-v2-pond.png", "fence": "plot-v2-fence.png", "barn": "plot-v2-barn.png"}


def load(path):
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


def mask_for(base, job):
    # сглаживаем перед сравнением, чтобы мелкий шум текстуры не попадал в маску
    a = np.asarray(base.filter(ImageFilter.GaussianBlur(2)), dtype=np.int16)
    b = np.asarray(job.filter(ImageFilter.GaussianBlur(2)), dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    m = Image.fromarray(((diff > THRESHOLD) * 255).astype(np.uint8))
    # убираем одиночные точки, закрываем дырки, делаем мягкий край
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    m = m.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.MinFilter(9))
    return m.filter(ImageFilter.GaussianBlur(4))


def main(img_dir):
    img_dir = Path(img_dir)
    base = load(img_dir / "plot-v2-cleared.png")
    for key, name in JOBS.items():
        job = load(img_dir / name)
        m = mask_for(base, job)
        layer = job.copy()
        layer.putalpha(m)
        layer.save(img_dir / f"layer-{key}.webp", quality=86, method=6)
        m.save(img_dir / f"mask-{key}.png")
        share = (np.asarray(m) > 128).mean() * 100
        print(f"{key}: {share:.1f}% of image in layer")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
