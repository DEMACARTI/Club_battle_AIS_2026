"""Download a sample of Quick, Draw! simplified drawings for every game word.

Covers each word's own category plus the variant categories listed in
sources.py (e.g. "ceiling fan" for "fan"). Each category's ndjson file is
50-200 MB, so instead of downloading it whole we fetch byte ranges from five
different parts of the file with HTTP Range requests and keep only the
complete lines. That's ~13,000 recognized drawings per category.

Output: data/raw/<category>.json, a list of drawings, each a list of strokes
[[x0, x1, ...], [y0, y1, ...]].
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import sources

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'data', 'raw')
BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/'
CHUNK_BYTES = 1_600_000
CHUNK_POSITIONS = (0.0, 0.2, 0.4, 0.6, 0.8)  # fractions of the file to sample


def read_words():
  with open(os.path.join(HERE, 'words.txt')) as f:
    return [w.strip() for w in f if w.strip()]


def fetch_range(url, start, end):
  req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
  with urllib.request.urlopen(req, timeout=60) as resp:
    return resp.read()


def file_size(url):
  req = urllib.request.Request(url, method='HEAD')
  with urllib.request.urlopen(req, timeout=60) as resp:
    return int(resp.headers['Content-Length'])


def download_category(category):
  out_path = os.path.join(OUT_DIR, sources.raw_name(category))
  if os.path.exists(out_path):
    return category, 'cached'
  url = BASE_URL + urllib.parse.quote(category) + '.ndjson'
  size = file_size(url)
  drawings = []
  for pos in CHUNK_POSITIONS:
    start = int(size * pos)
    end = min(size - 1, start + CHUNK_BYTES)
    lines = fetch_range(url, start, end).split(b'\n')
    # The first and last lines of a range are usually cut in half.
    if start > 0:
      lines = lines[1:]
    lines = lines[:-1]
    for line in lines:
      try:
        record = json.loads(line)
      except ValueError:
        continue
      # Keep drawings that Quick, Draw!'s own model recognized; the rest are
      # often scribbles or unfinished.
      if record.get('recognized'):
        drawings.append(record['drawing'])
  with open(out_path, 'w') as f:
    json.dump(drawings, f, separators=(',', ':'))
  return category, len(drawings)


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  categories = sources.all_categories(read_words())
  with ThreadPoolExecutor(max_workers=12) as pool:
    results = pool.map(download_category, categories)
    for i, (category, result) in enumerate(results, 1):
      print(f'[{i}/{len(categories)}] {category}: {result}', flush=True)
  sys.exit(0)


if __name__ == '__main__':
  main()
