// Guesses what a stroke drawing shows, with a confidence for each word.
//
// The network is trained by training/train.py and exported by
// training/export.py into classifier-weights.js. rasterize() mirrors
// training/raster.py: if you change one, change the other.
//
// Written as one self-contained function so ai-workers.js can also run it
// inside a background thread (only predictImage() is used there).
function SketchClassifierModule(root) {
  'use strict';

  const CANVAS = 224;
  const PADDING = 16;
  const LINE_WIDTH = 16;
  const OUT = 28;

  function decodeBase64(text) {
    if (typeof atob === 'function') {
      const bin = atob(text);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes.buffer;
    }
    const buf = Buffer.from(text, 'base64');
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
  }

  // Layers run in order: conv (3x3 + ReLU), maxpool (2x2), gap (global
  // average pool) and dense (ReLU if layer.relu). See training/export.py.
  function loadLayers(payload) {
    return payload.layers.map((layer) => {
      if (!layer.weights) return { kind: layer.kind };
      const q = new Int8Array(decodeBase64(layer.weights));
      const scales = new Float32Array(decodeBase64(layer.scales));
      const bias = new Float32Array(decodeBase64(layer.bias));
      const outCount = layer.shape[0];
      const perOut = q.length / outCount;
      const weights = new Float32Array(q.length);
      for (let o = 0; o < outCount; o++) {
        for (let j = 0; j < perOut; j++) {
          weights[o * perOut + j] = q[o * perOut + j] * scales[o];
        }
      }
      return { kind: layer.kind, shape: layer.shape, relu: layer.relu, weights, bias };
    });
  }

  // 3x3 convolution with zero padding, followed by ReLU. Layout is CHW.
  function conv3x3Relu(input, channels, h, w, layer) {
    const outChannels = layer.shape[0];
    const hw = h * w;
    const out = new Float32Array(outChannels * hw);
    const weights = layer.weights;
    for (let o = 0; o < outChannels; o++) {
      const outBase = o * hw;
      out.fill(layer.bias[o], outBase, outBase + hw);
      for (let c = 0; c < channels; c++) {
        const inBase = c * hw;
        for (let ky = 0; ky < 3; ky++) {
          const dy = ky - 1;
          const y0 = Math.max(0, -dy);
          const y1 = Math.min(h, h - dy);
          for (let kx = 0; kx < 3; kx++) {
            const k = weights[((o * channels + c) * 3 + ky) * 3 + kx];
            if (k === 0) continue;
            const dx = kx - 1;
            const x0 = Math.max(0, -dx);
            const x1 = Math.min(w, w - dx);
            for (let y = y0; y < y1; y++) {
              const outRow = outBase + y * w;
              const inRow = inBase + (y + dy) * w + dx;
              for (let x = x0; x < x1; x++) out[outRow + x] += k * input[inRow + x];
            }
          }
        }
      }
    }
    for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
    return out;
  }

  function maxPool2(input, channels, h, w) {
    const oh = h >> 1;
    const ow = w >> 1;
    const out = new Float32Array(channels * oh * ow);
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < oh; y++) {
        for (let x = 0; x < ow; x++) {
          const i = c * h * w + 2 * y * w + 2 * x;
          out[(c * oh + y) * ow + x] = Math.max(
              input[i], input[i + 1], input[i + w], input[i + w + 1]);
        }
      }
    }
    return out;
  }

  function globalAveragePool(input, channels, h, w) {
    const hw = h * w;
    const out = new Float32Array(channels);
    for (let c = 0; c < channels; c++) {
      let sum = 0;
      for (let i = 0; i < hw; i++) sum += input[c * hw + i];
      out[c] = sum / hw;
    }
    return out;
  }

  function dense(input, layer, relu) {
    const [outCount, inCount] = layer.shape;
    const out = new Float32Array(outCount);
    for (let o = 0; o < outCount; o++) {
      let sum = layer.bias[o];
      const base = o * inCount;
      for (let i = 0; i < inCount; i++) sum += layer.weights[base + i] * input[i];
      out[o] = relu && sum < 0 ? 0 : sum;
    }
    return out;
  }

  function softmax(logits) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const out = new Float32Array(logits.length);
    let total = 0;
    for (let i = 0; i < logits.length; i++) {
      out[i] = Math.exp(logits[i] - max);
      total += out[i];
    }
    for (let i = 0; i < out.length; i++) out[i] /= total;
    return out;
  }

  // Draws strokes ([[xs], [ys]] in any units) into a 28x28 Float32Array.
  function rasterize(strokes, scratchCanvas) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [xs, ys] of strokes) {
      for (let i = 0; i < xs.length; i++) {
        minX = Math.min(minX, xs[i]); maxX = Math.max(maxX, xs[i]);
        minY = Math.min(minY, ys[i]); maxY = Math.max(maxY, ys[i]);
      }
    }
    const w = maxX - minX;
    const h = maxY - minY;
    const scale = (CANVAS - 2 * PADDING) / Math.max(w, h, 1);
    const offX = (CANVAS - w * scale) / 2 - minX * scale;
    const offY = (CANVAS - h * scale) / 2 - minY * scale;

    const canvas = scratchCanvas || document.createElement('canvas');
    canvas.width = CANVAS;
    canvas.height = CANVAS;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, CANVAS, CANVAS);
    ctx.strokeStyle = '#fff';
    ctx.fillStyle = '#fff';
    ctx.lineWidth = LINE_WIDTH;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const [xs, ys] of strokes) {
      if (xs.length === 1) {
        ctx.beginPath();
        ctx.arc(xs[0] * scale + offX, ys[0] * scale + offY, LINE_WIDTH / 2, 0,
                2 * Math.PI);
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(xs[0] * scale + offX, ys[0] * scale + offY);
      for (let i = 1; i < xs.length; i++) {
        ctx.lineTo(xs[i] * scale + offX, ys[i] * scale + offY);
      }
      ctx.stroke();
    }

    const pixels = ctx.getImageData(0, 0, CANVAS, CANVAS).data;
    const cell = CANVAS / OUT;
    const out = new Float32Array(OUT * OUT);
    for (let y = 0; y < CANVAS; y++) {
      for (let x = 0; x < CANVAS; x++) {
        out[((y / cell) | 0) * OUT + ((x / cell) | 0)] += pixels[(y * CANVAS + x) * 4];
      }
    }
    const norm = 255 * cell * cell;
    for (let i = 0; i < out.length; i++) out[i] /= norm;
    return out;
  }

  class SketchClassifier {
    constructor(payload) {
      this.words = payload.words;
      this.layers = loadLayers(payload);
      this.scratchCanvas = null;
    }

    // image: Float32Array(784), 0 = empty, 1 = ink. Returns probabilities.
    predictImage(image) {
      let x = image, c = 1, h = OUT, w = OUT;
      for (const layer of this.layers) {
        if (layer.kind === 'conv') {
          x = conv3x3Relu(x, c, h, w, layer);
          c = layer.shape[0];
        } else if (layer.kind === 'maxpool') {
          x = maxPool2(x, c, h, w);
          h >>= 1;
          w >>= 1;
        } else if (layer.kind === 'gap') {
          x = globalAveragePool(x, c, h, w);
          h = w = 1;
        } else {
          x = dense(x, layer, layer.relu);
        }
      }
      return softmax(x);
    }

    // Returns [{word, confidence}] sorted by confidence, best first.
    classify(strokes, topK = 5) {
      if (!this.scratchCanvas) this.scratchCanvas = document.createElement('canvas');
      const probs = this.predictImage(rasterize(strokes, this.scratchCanvas));
      return this.words
          .map((word, i) => ({ word, confidence: probs[i] }))
          .sort((a, b) => b.confidence - a.confidence)
          .slice(0, topK);
    }
  }

  root.SketchClassifier = SketchClassifier;
  root.rasterizeSketch = rasterize;
  if (typeof module !== 'undefined') module.exports = { SketchClassifier, rasterize };
}
SketchClassifierModule(typeof window !== 'undefined' ? window : globalThis);
