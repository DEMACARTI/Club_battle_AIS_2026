// Background threads for the two neural networks, so neither ever blocks the
// page while the player draws or watches:
//   SketchArtist  - Sketch-RNN drawing (sketch-rnn.js): downloads models and
//                   samples the AI's drawings.
//   SketchGuesser - the classifier (classifier.js): turns strokes into
//                   guesses with confidences.
// They are separate threads so that preparing the next AI drawing never
// delays the AI's guesses at the player's drawing.
//
// Each worker is built from in-memory source (a Blob URL) rather than a
// separate file, which also works for pages opened straight from disk. If a
// browser refuses workers anyway, the same handlers run on the page.
(function (root) {
  'use strict';

  // --- Handlers: run inside the worker (or on the page as a fallback) ----
  // These are stringified into the worker, so they may only use their
  // `scope` argument, not anything else from this file.

  function artistHandlers(scope) {
    const models = new Map(); // word -> Promise<SketchRNN>
    function getModel(modelSet, word) {
      if (!models.has(word)) {
        const ready = scope.SketchRNNLite.loadSketchRNN(modelSet, word);
        ready.catch(() => models.delete(word));
        models.set(word, ready);
      }
      return models.get(word);
    }
    return {
      async load({ modelSet, word }) {
        await getModel(modelSet, word);
      },
      async draw({ modelSet, word, temperature, count }) {
        const model = await getModel(modelSet, word);
        const drawings = [];
        for (let i = 0; i < count; i++) drawings.push(model.draw(temperature));
        return drawings;
      },
      drop({ keep }) {
        for (const word of models.keys()) if (!keep.includes(word)) models.delete(word);
      },
    };
  }

  function guesserHandlers(scope) {
    let classifier = null;
    return {
      init({ weights }) {
        classifier = new scope.SketchClassifier(weights);
      },
      predict({ image }) {
        return classifier.predictImage(image);
      },
    };
  }

  // Worker entry point: answers {id, type, ...} with {id, result} or {id, error}.
  function serve(handlers) {
    self.onmessage = async ({ data }) => {
      try {
        const result = await handlers[data.type](data);
        self.postMessage({ id: data.id, result });
      } catch (err) {
        self.postMessage({ id: data.id, error: String((err && err.message) || err) });
      }
    };
  }

  // --- Page side ---------------------------------------------------------

  class BackgroundWorker {
    // modules: self-contained functions(scope) to run first, e.g.
    // SketchRNNModule. createHandlers: artistHandlers or guesserHandlers.
    constructor(modules, createHandlers) {
      this.pending = new Map(); // id -> {resolve, reject, message}
      this.nextId = 1;
      this.createHandlers = createHandlers;
      try {
        const source = modules.map((fn) => `(${fn})(self);`).join('\n') +
            `\n(${serve})((${createHandlers})(self));`;
        const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
        this.worker = new Worker(url);
        this.worker.onmessage = (e) => this.onReply(e.data);
        this.worker.onerror = (e) => {
          e.preventDefault();
          this.fallBackToPage();
        };
      } catch (err) {
        this.fallBackToPage();
      }
    }

    // Last resort: run the handlers on the page, and resend anything waiting.
    fallBackToPage() {
      if (this.handlers) return;
      console.warn('Web workers unavailable; running the AI on the page (may stutter).');
      if (this.worker) this.worker.terminate();
      this.worker = null;
      this.handlers = this.createHandlers(root);
      for (const { message } of this.pending.values()) this.runOnPage(message);
    }

    async runOnPage(message) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      try {
        const result = await this.handlers[message.type](message);
        this.onReply({ id: message.id, result });
      } catch (err) {
        this.onReply({ id: message.id, error: String((err && err.message) || err) });
      }
    }

    onReply({ id, result, error }) {
      const request = this.pending.get(id);
      if (!request) return;
      this.pending.delete(id);
      if (error) request.reject(new Error(error));
      else request.resolve(result);
    }

    call(type, data = {}) {
      const message = { ...data, type, id: this.nextId++ };
      return new Promise((resolve, reject) => {
        this.pending.set(message.id, { resolve, reject, message });
        if (this.handlers) this.runOnPage(message);
        else this.worker.postMessage(message);
      });
    }
  }

  class SketchArtist {
    // modelSet: 'large_models' (1024-unit LSTMs, ~11 MB each) or 'models'
    // (512 units, ~3 MB).
    constructor(modelSet = 'large_models') {
      this.modelSet = modelSet;
      this.worker = new BackgroundWorker([root.SketchRNNModule], artistHandlers);
    }

    // Downloads a word's model ahead of time.
    preload(word) {
      this.worker.call('load', { modelSet: this.modelSet, word }).catch(() => {});
    }

    // Samples `count` drawings of `word`; see SketchRNN.draw() for the format.
    drawMany(word, temperature, count) {
      return this.worker.call('draw', { modelSet: this.modelSet, word, temperature, count });
    }

    // Frees memory for every model except the given words.
    keepOnly(words) {
      this.worker.call('drop', { keep: words.filter(Boolean) }).catch(() => {});
    }
  }

  class SketchGuesser {
    constructor(weights) {
      this.words = weights.words;
      this.worker = new BackgroundWorker([root.SketchClassifierModule], guesserHandlers);
      this.worker.call('init', { weights });
      this.scratchCanvas = document.createElement('canvas');
    }

    // Probability per word (in this.words order). Rasterizing needs a canvas,
    // so it happens here; it takes well under a millisecond.
    probabilities(strokes) {
      const image = root.rasterizeSketch(strokes, this.scratchCanvas);
      return this.worker.call('predict', { image });
    }

    // Resolves to [{word, confidence}] sorted best first.
    async classify(strokes, topK = 5) {
      const probs = await this.probabilities(strokes);
      return this.words
          .map((word, i) => ({ word, confidence: probs[i] }))
          .sort((a, b) => b.confidence - a.confidence)
          .slice(0, topK);
    }
  }

  root.SketchArtist = SketchArtist;
  root.SketchGuesser = SketchGuesser;
})(window);
