"""Exports the trained classifier for the browser.

Folds BatchNorm into the conv weights, quantizes every weight matrix to int8
(one scale per output channel) and writes web/js/classifier-weights.js as an
ordered list of layers that web/js/classifier.js runs:
  conv     3x3 convolution, padding 1, followed by ReLU
  maxpool  2x2 max pooling
  gap      global average pooling
  dense    fully connected, ReLU if relu is set
The quantized model is re-checked with a NumPy forward pass that does the same
arithmetic as the JS code, and a few reference outputs are saved for
test_classifier.js.
"""

import base64
import json
import os
import sys

import numpy as np
import torch
from torch import nn

import train

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_JS = os.path.join(HERE, '..', 'web', 'js')


def quantize(kind, w, b, **extra):
  flat = w.reshape(w.shape[0], -1)
  scale = np.abs(flat).max(axis=1) / 127.0
  scale[scale == 0] = 1.0
  q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8)
  return {'kind': kind, 'shape': list(w.shape), 'q': q,
          'scale': scale.astype(np.float32), 'bias': b.astype(np.float32), **extra}


def to_layers(model):
  """Flattens SketchNet into the layer list described above."""
  layers = []
  modules = list(model.features) + list(model.classifier)
  for i, m in enumerate(modules):
    if isinstance(m, nn.Conv2d):
      bn = modules[i + 1]
      w = m.weight.detach().double().numpy()
      s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
      b = bn.bias.detach().double().numpy() - bn.running_mean.detach().double().numpy() * s
      layers.append(quantize('conv', w * s[:, None, None, None], b))
    elif isinstance(m, nn.MaxPool2d):
      layers.append({'kind': 'maxpool'})
    elif isinstance(m, nn.AdaptiveAvgPool2d):
      layers.append({'kind': 'gap'})
    elif isinstance(m, nn.Linear):
      layers.append(quantize('dense', m.weight.detach().double().numpy(),
                             m.bias.detach().double().numpy(), relu=False))
    elif isinstance(m, nn.ReLU) and layers and layers[-1]['kind'] == 'dense':
      layers[-1]['relu'] = True
  return layers


def numpy_forward(layers, images):
  """Same arithmetic as classifier.js, batched. images: (N, 28, 28) in 0..1."""
  x = images[:, None].astype(np.float32)
  for layer in layers:
    kind = layer['kind']
    if kind == 'conv':
      w = (layer['q'].astype(np.float32) * layer['scale'][:, None]).reshape(layer['shape'])
      n, _, h, wd = x.shape
      padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
      out = np.zeros((n, w.shape[0], h, wd), np.float32)
      for ky in range(3):
        for kx in range(3):
          out += np.einsum('nchw,oc->nohw', padded[:, :, ky:ky + h, kx:kx + wd], w[:, :, ky, kx])
      x = np.maximum(out + layer['bias'][None, :, None, None], 0)
    elif kind == 'maxpool':
      n, c, h, wd = x.shape
      x = x.reshape(n, c, h // 2, 2, wd // 2, 2).max(axis=(3, 5))
    elif kind == 'gap':
      x = x.mean(axis=(2, 3))
    else:
      w = layer['q'].astype(np.float32) * layer['scale'][:, None]
      x = x.reshape(len(x), -1) @ w.T + layer['bias']
      if layer['relu']:
        x = np.maximum(x, 0)
  e = np.exp(x - x.max(axis=1, keepdims=True))
  return e / e.sum(axis=1, keepdims=True)


def b64(array):
  return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def main():
  words = open(os.path.join(HERE, 'words.txt')).read().split()
  model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'data', 'model.pt')
  model = train.SketchNet(len(words))
  model.load_state_dict(torch.load(model_path, map_location='cpu'))
  model.eval()
  layers = to_layers(model)

  meta = json.load(open(os.path.join(train.DATA, 'categories.json')))
  n = min(8000, meta['val_count'])
  x_val = np.load(os.path.join(train.DATA, 'x.npy'), mmap_mode='r')[:n].astype(np.float32) / 255
  y_val = np.load(os.path.join(train.DATA, 'y.npy'))[:n]
  partial = np.load(os.path.join(train.DATA, 'partial.npy'))[:n]
  with torch.no_grad():
    torch_pred = model(torch.from_numpy(x_val)[:, None]).argmax(1).numpy()
  probs = np.concatenate([numpy_forward(layers, x_val[i:i + 500]) for i in range(0, n, 500)])
  q_pred = probs.argmax(1)
  top3 = (np.argsort(-probs, axis=1)[:, :3] == y_val[:, None]).any(axis=1)
  print(f'float32 accuracy {np.mean(torch_pred == y_val):.4f}, '
        f'int8 accuracy {np.mean(q_pred == y_val):.4f} '
        f'(finished {np.mean(q_pred[~partial] == y_val[~partial]):.4f}, '
        f'partial {np.mean(q_pred[partial] == y_val[partial]):.4f}), '
        f'int8 top-3 {top3.mean():.4f}')

  payload = {'words': words, 'layers': []}
  for layer in layers:
    entry = {k: v for k, v in layer.items() if k not in ('q', 'scale', 'bias')}
    if 'q' in layer:
      entry.update(weights=b64(layer['q']), scales=b64(layer['scale']), bias=b64(layer['bias']))
    payload['layers'].append(entry)
  out_path = os.path.join(WEB_JS, 'classifier-weights.js')
  with open(out_path, 'w') as f:
    f.write('// Generated by training/export.py. Do not edit.\n')
    f.write('window.SKETCH_CLASSIFIER_WEIGHTS = ')
    json.dump(payload, f, separators=(',', ':'))
    f.write(';\n')
  print(f'wrote {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)')

  reference = {'images': x_val[:8].reshape(8, -1).round(6).tolist(),
               'probs': probs[:8].tolist()}
  with open(os.path.join(HERE, 'data', 'reference.json'), 'w') as f:
    json.dump(reference, f)


if __name__ == '__main__':
  main()
