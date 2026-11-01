// Sketch-RNN sampling in plain JavaScript, for the AI's drawings.
//
// Loads the pretrained per-word models that magenta-js uses
// (quickdraw-models/sketchRNN/*/<word>.gen.json) and samples from them with
// the same math as SketchRNN.update(), getPDF() and sample() in
// @magenta/sketch. Running it without TensorFlow.js avoids one GPU round trip
// per pen step, which made the library take seconds per drawing.
// dev/check-model.html compares this file against the library.
//
// Written as one self-contained function so ai-workers.js can also run it
// inside a background thread.
function SketchRNNModule(root) {
  'use strict';

  const MODEL_BASE = 'https://storage.googleapis.com/quickdraw-models/sketchRNN/';
  const NMIX = 20;
  const FORGET_BIAS = 1.0;
  const MAX_WEIGHT = 10.0; // weights are stored as int16 scaled to +-10
  const MAX_STEPS = 250;

  function modelUrl(modelSet, word) {
    return `${MODEL_BASE}${modelSet}/${word}.gen.json`;
  }

  function decodeWeights(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const raw = new Int16Array(bytes.buffer);
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = MAX_WEIGHT * raw[i] / 32767;
    return out;
  }

  // json: the parsed .gen.json file, [info, dims, weightStrings].
  function parseModel(json) {
    const [info, dims, strings] = json;
    const w = strings.map(decodeWeights);
    const units = dims[0][0];
    // magenta-js stacks the input->gates and hidden->gates kernels.
    const lstmKernel = new Float32Array(w[2].length + w[3].length);
    lstmKernel.set(w[2]);
    lstmKernel.set(w[3], w[2].length);
    return { info, units, outKernel: w[0], outBias: w[1], lstmKernel, lstmBias: w[4] };
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  let spareGauss = null;
  function gauss() { // Marsaglia polar method, as in magenta-js
    if (spareGauss !== null) {
      const v = spareGauss;
      spareGauss = null;
      return v;
    }
    let u, v, r;
    do {
      u = 2 * Math.random() - 1;
      v = 2 * Math.random() - 1;
      r = u * u + v * v;
    } while (r === 0 || r > 1);
    const c = Math.sqrt(-2 * Math.log(r) / r);
    spareGauss = v * c;
    return u * c;
  }

  function sampleIndex(probs) {
    const x = Math.random();
    let total = 0;
    for (let i = 0; i < probs.length; i++) {
      total += probs[i];
      if (total >= x) return i;
    }
    return probs.length - 1;
  }

  function softmax(logits, offset, n, temp) {
    const out = new Float32Array(n);
    let max = -Infinity;
    for (let i = 0; i < n; i++) max = Math.max(max, logits[offset + i] / temp);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      out[i] = Math.exp(logits[offset + i] / temp - max);
      sum += out[i];
    }
    for (let i = 0; i < n; i++) out[i] /= sum;
    return out;
  }

  class SketchRNN {
    // model: the result of parseModel().
    constructor(model) {
      Object.assign(this, model);
      this.outSize = model.outBias.length;
      this.gates = new Float32Array(4 * model.units);
      this.raw = new Float32Array(this.outSize);
    }

    zeroState() {
      return { c: new Float32Array(this.units), h: new Float32Array(this.units) };
    }

    // stroke: [dx, dy, p1, p2, p3] in model units. Same as tf.basicLSTMCell
    // with input concat([stroke, h]) and gate order i, j, f, o.
    update(stroke, state) {
      const n = this.units, n4 = 4 * n, k = this.lstmKernel, g = this.gates;
      g.set(this.lstmBias);
      for (let i = 0; i < 5 + n; i++) {
        const v = i < 5 ? stroke[i] : state.h[i - 5];
        if (v === 0) continue;
        const row = i * n4;
        for (let j = 0; j < n4; j++) g[j] += v * k[row + j];
      }
      const c = new Float32Array(n), h = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        c[j] = state.c[j] * sigmoid(g[2 * n + j] + FORGET_BIAS) +
               sigmoid(g[j]) * Math.tanh(g[n + j]);
        h[j] = Math.tanh(c[j]) * sigmoid(g[3 * n + j]);
      }
      return { c, h };
    }

    // Same parameters as magenta-js SketchRNN.getPDF(state, temperature).
    getPDF(state, temperature) {
      const raw = this.raw, n = this.units, m = this.outSize, k = this.outKernel;
      raw.set(this.outBias);
      for (let i = 0; i < n; i++) {
        const v = state.h[i];
        if (v === 0) continue;
        const row = i * m;
        for (let j = 0; j < m; j++) raw[j] += v * k[row + j];
      }
      const discreteTemp = 0.5 + temperature * 0.5;
      const sqrtTemp = Math.sqrt(temperature);
      const pdf = {
        pen: softmax(raw, 0, 3, discreteTemp),
        pi: softmax(raw, 3, NMIX, discreteTemp),
        muX: raw.slice(3 + NMIX, 3 + 2 * NMIX),
        muY: raw.slice(3 + 2 * NMIX, 3 + 3 * NMIX),
        sigmaX: new Float32Array(NMIX),
        sigmaY: new Float32Array(NMIX),
        corr: new Float32Array(NMIX),
      };
      for (let i = 0; i < NMIX; i++) {
        pdf.sigmaX[i] = Math.exp(raw[3 + 3 * NMIX + i]) * sqrtTemp;
        pdf.sigmaY[i] = Math.exp(raw[3 + 4 * NMIX + i]) * sqrtTemp;
        pdf.corr[i] = Math.tanh(raw[3 + 5 * NMIX + i]);
      }
      return pdf;
    }

    // Same as magenta-js SketchRNN.sample(), but in model units.
    sample(pdf) {
      const i = sampleIndex(pdf.pi);
      const pen = [0, 0, 0];
      pen[sampleIndex(pdf.pen)] = 1;
      const rho = pdf.corr[i];
      const z1 = gauss(), z2 = gauss();
      const dx = Math.sqrt(1 - rho * rho) * pdf.sigmaX[i] * z1 +
                 rho * pdf.sigmaX[i] * z2 + pdf.muX[i];
      const dy = pdf.sigmaY[i] * z2 + pdf.muY[i];
      return [dx, dy, pen[0], pen[1], pen[2]];
    }

    // Samples one drawing. Returns {strokes, segments}: strokes in the
    // [[xs], [ys]] format the classifier takes, segments as [x0, y0, x1, y1]
    // in drawing order for animation.
    draw(temperature) {
      let state = this.zeroState();
      let input = [0, 0, 1, 0, 0]; // start token, pen down
      let x = 0, y = 0;
      let penWasDown = true;
      const strokes = [];
      const segments = [];
      let current = null;
      for (let step = 0; step < MAX_STEPS; step++) {
        state = this.update(input, state);
        input = this.sample(this.getPDF(state, temperature));
        const [dx, dy, penDown, , penEnd] = input;
        if (penEnd === 1) break;
        const nx = x + dx, ny = y + dy;
        if (penWasDown) {
          if (!current) {
            current = [[x], [y]];
            strokes.push(current);
          }
          current[0].push(nx);
          current[1].push(ny);
          segments.push([x, y, nx, ny]);
        } else {
          current = null;
        }
        x = nx;
        y = ny;
        penWasDown = penDown === 1;
      }
      return { strokes, segments };
    }
  }

  async function loadSketchRNN(modelSet, word) {
    const response = await fetch(modelUrl(modelSet, word));
    if (!response.ok) throw new Error(`model for ${word}: HTTP ${response.status}`);
    return new SketchRNN(parseModel(await response.json()));
  }

  root.SketchRNNLite = { SketchRNN, loadSketchRNN, parseModel, modelUrl };
}
SketchRNNModule(typeof window !== 'undefined' ? window : self);
