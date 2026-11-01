"""Turns a stroke drawing into the 28x28 image the classifier sees.

web/js/classifier.js (rasterize) mirrors this function; if you change one,
change the other, or the model will see different images in the browser than
it was trained on.
"""

import numpy as np
from PIL import Image, ImageDraw

CANVAS = 224  # render size; 224 = 28 * 8, so downsampling is an exact 8x8 box
PADDING = 16  # empty border around the drawing, in render pixels
LINE_WIDTH = 16  # in render pixels, i.e. 2 px at 28x28
OUT = 28


def rasterize(strokes, line_width=LINE_WIDTH):
  """strokes: list of [[x...], [y...]] in any units. Returns float32 28x28."""
  xs = np.concatenate([np.asarray(s[0], np.float32) for s in strokes])
  ys = np.concatenate([np.asarray(s[1], np.float32) for s in strokes])
  min_x, min_y = xs.min(), ys.min()
  w, h = xs.max() - min_x, ys.max() - min_y
  scale = (CANVAS - 2 * PADDING) / max(w, h, 1.0)
  off_x = (CANVAS - w * scale) / 2 - min_x * scale
  off_y = (CANVAS - h * scale) / 2 - min_y * scale

  img = Image.new('L', (CANVAS, CANVAS), 0)
  draw = ImageDraw.Draw(img)
  r = line_width / 2
  for sx, sy in strokes:
    pts = [(x * scale + off_x, y * scale + off_y) for x, y in zip(sx, sy)]
    if len(pts) > 1:
      draw.line(pts, fill=255, width=line_width)
    # Round caps and joins, like canvas lineCap/lineJoin = 'round'.
    for x, y in pts:
      draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

  a = np.asarray(img, np.float32) / 255.0
  return a.reshape(OUT, CANVAS // OUT, OUT, CANVAS // OUT).mean(axis=(1, 3))
