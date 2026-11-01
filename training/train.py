"""Trains the sketch classifier that guesses the player's drawings.

Input: data/dataset/ from prepare_data.py. Output: data/model.pt.
"""

import json
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'dataset')
EPOCHS = 10
BATCH = 512
LR = 3e-3


class SketchNet(nn.Module):
  """Three conv stages, global average pooling, two dense layers.

  Mirrored in web/js/classifier.js via the layer list export.py writes.
  """

  def __init__(self, num_classes):
    super().__init__()
    def block(cin, cout):
      return [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
              nn.BatchNorm2d(cout), nn.ReLU()]
    self.features = nn.Sequential(
        *block(1, 32), *block(32, 32), nn.MaxPool2d(2),
        *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
        *block(64, 128), *block(128, 128), nn.AdaptiveAvgPool2d(1))
    self.classifier = nn.Sequential(
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, num_classes))

  def forward(self, x):
    return self.classifier(self.features(x))


def augment(x):
  """Small random rotation, scale and shift, applied on the GPU."""
  n = x.shape[0]
  angle = (torch.rand(n, device=x.device) - 0.5) * 2 * np.radians(12)
  scale = 1 + (torch.rand(n, 2, device=x.device) - 0.5) * 0.2
  shift = (torch.rand(n, 2, device=x.device) - 0.5) * 0.12
  cos, sin = torch.cos(angle), torch.sin(angle)
  theta = torch.stack([
      torch.stack([cos * scale[:, 0], -sin * scale[:, 0], shift[:, 0]], 1),
      torch.stack([sin * scale[:, 1], cos * scale[:, 1], shift[:, 1]], 1),
  ], 1)
  grid = F.affine_grid(theta, x.shape, align_corners=False)
  return F.grid_sample(x, grid, align_corners=False)


def predict(model, x, device):
  model.eval()
  preds = []
  with torch.no_grad():
    for i in range(0, len(x), 2048):
      xb = torch.from_numpy(x[i:i + 2048]).to(device).float().div(255)
      preds.append(model(xb.unsqueeze(1)).argmax(1).cpu().numpy())
  return np.concatenate(preds)


def main():
  device = 'mps' if torch.backends.mps.is_available() else 'cpu'
  meta = json.load(open(os.path.join(DATA, 'categories.json')))
  n_val = meta['val_count']
  x = np.load(os.path.join(DATA, 'x.npy'))
  y = np.load(os.path.join(DATA, 'y.npy')).astype(np.int64)
  partial = np.load(os.path.join(DATA, 'partial.npy'))
  source = np.load(os.path.join(DATA, 'source.npy'))
  x_val, y_val = x[:n_val], y[:n_val]
  partial_val, source_val = partial[:n_val], source[:n_val]
  x_train = torch.from_numpy(x[n_val:])  # stays in RAM; batches go to the GPU
  y_train = torch.from_numpy(y[n_val:])
  num_classes = int(y.max()) + 1
  fan = meta['categories'].index('ceiling fan')
  print(f'{len(x_train)} train, {n_val} val, {num_classes} classes, device {device}',
        flush=True)

  torch.manual_seed(0)
  model = SketchNet(num_classes).to(device)
  steps_per_epoch = len(x_train) // BATCH
  opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * steps_per_epoch)

  for epoch in range(EPOCHS):
    model.train()
    start = time.time()
    order = torch.randperm(len(x_train))
    total_loss = 0.0
    for b in range(steps_per_epoch):
      idx = order[b * BATCH:(b + 1) * BATCH]
      xb = x_train[idx].to(device).float().div(255).unsqueeze(1)
      yb = y_train[idx].to(device)
      loss = F.cross_entropy(model(augment(xb)), yb, label_smoothing=0.05)
      opt.zero_grad()
      loss.backward()
      opt.step()
      sched.step()
      total_loss += loss.item()
    correct = predict(model, x_val, device) == y_val
    print(f'epoch {epoch + 1}/{EPOCHS} loss {total_loss / steps_per_epoch:.3f} '
          f'val acc {correct.mean():.4f} (finished drawings '
          f'{correct[~partial_val].mean():.4f}, partial {correct[partial_val].mean():.4f}, '
          f'ceiling fans as "fan" {correct[source_val == fan].mean():.4f}) '
          f'{time.time() - start:.0f}s', flush=True)
    torch.save(model.state_dict(), os.path.join(HERE, 'data', 'model.pt'))

  print('saved data/model.pt')


if __name__ == '__main__':
  main()
