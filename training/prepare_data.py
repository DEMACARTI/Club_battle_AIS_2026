"""Rasterizes the downloaded drawings into train/validation arrays.

Each word gets up to MAX_PER_WORD drawings, mixed from its Quick, Draw!
categories as listed in sources.py (e.g. 40% ceiling fans for "fan"). On top
of the complete drawings, a share is also added with only their first
strokes, so the classifier learns to guess while the player is still drawing.
Line width is jittered so the model doesn't depend on the exact pixel
rendering of PIL versus the browser canvas, and some drawings get a small
random wobble on every point, like a shakier hand.

Output: data/dataset/ with x.npy (uint8 28x28 images), y.npy (word index),
partial.npy (unfinished drawing?) and source.npy (index into categories.json),
already shuffled; the first VAL_COUNT rows are the validation set.
"""

import json
import os
import random
from multiprocessing import Pool

import numpy as np

import raster
import sources

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, 'data', 'raw')
OUT_DIR = os.path.join(HERE, 'data', 'dataset')
MAX_PER_WORD = 12000
PARTIAL_SHARE = 0.3
WOBBLE_SHARE = 0.3  # drawings rendered with a few units of random point noise
VAL_SHARE = 0.04
LINE_WIDTHS = (12, 14, 16, 18, 20)


def read_words():
  with open(os.path.join(HERE, 'words.txt')) as f:
    return [w.strip() for w in f if w.strip()]


def wobble(drawing, rng):
  return [[[x + rng.gauss(0, 1.5) for x in xs], [y + rng.gauss(0, 1.5) for y in ys]]
          for xs, ys in drawing]


def render_word(args):
  label, word, categories = args
  rng = random.Random(label)
  picked = []  # (drawing, source index)
  for category, share in sources.sources_for(word):
    with open(os.path.join(RAW_DIR, sources.raw_name(category))) as f:
      drawings = json.load(f)
    rng.shuffle(drawings)
    source = categories.index(category)
    picked += [(d, source) for d in drawings[:int(MAX_PER_WORD * share)]]
  rng.shuffle(picked)

  images, partial, source_ids = [], [], []
  def add(drawing, source, is_partial):
    if rng.random() < WOBBLE_SHARE:
      drawing = wobble(drawing, rng)
    images.append(raster.rasterize(drawing, rng.choice(LINE_WIDTHS)))
    partial.append(is_partial)
    source_ids.append(source)

  for d, source in picked:
    add(d, source, False)
  for d, source in picked[:int(len(picked) * PARTIAL_SHARE)]:
    if len(d) >= 3:
      keep = max(1, int(round(len(d) * rng.uniform(0.4, 0.85))))
      add(d[:keep], source, True)
  images = (np.stack(images) * 255).round().astype(np.uint8)
  return label, images, np.array(partial), np.array(source_ids, np.int16)


def main():
  words = read_words()
  categories = sources.all_categories(words)
  with Pool() as pool:
    results = pool.map(render_word, [(i, w, categories) for i, w in enumerate(words)])

  x = np.concatenate([r[1] for r in results])
  y = np.concatenate([np.full(len(r[1]), r[0], np.int16) for r in results])
  partial = np.concatenate([r[2] for r in results])
  source = np.concatenate([r[3] for r in results])
  for label, images, _, _ in results:
    print(f'{words[label]}: {len(images)}')
  del results

  # Shuffle in place (same seed => same permutation for every array), which
  # avoids holding a second copy of the ~1 GB image array.
  for array in (x, y, partial, source):
    np.random.default_rng(0).shuffle(array)
  os.makedirs(OUT_DIR, exist_ok=True)
  for name, array in (('x', x), ('y', y), ('partial', partial), ('source', source)):
    np.save(os.path.join(OUT_DIR, name + '.npy'), array)
  with open(os.path.join(OUT_DIR, 'categories.json'), 'w') as f:
    json.dump({'categories': categories, 'val_count': int(len(x) * VAL_SHARE)}, f)
  print(f'{len(x)} images ({x.nbytes / 1e9:.2f} GB), validation {int(len(x) * VAL_SHARE)}')


if __name__ == '__main__':
  main()
