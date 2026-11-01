// Checks that web/js/classifier.js computes the same probabilities as the
// NumPy reference in export.py. Run after export.py:  node test_classifier.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const web = path.join(__dirname, '..', 'web', 'js');
const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(web, 'classifier-weights.js'), 'utf8'), sandbox);
const { SketchClassifier } = require(path.join(web, 'classifier.js'));

const classifier = new SketchClassifier(sandbox.window.SKETCH_CLASSIFIER_WEIGHTS);
const reference = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'data', 'reference.json'), 'utf8'));

let worst = 0;
const start = Date.now();
reference.images.forEach((image, n) => {
  const probs = classifier.predictImage(Float32Array.from(image));
  probs.forEach((p, i) => { worst = Math.max(worst, Math.abs(p - reference.probs[n][i])); });
});
const ms = (Date.now() - start) / reference.images.length;
console.log(`max probability difference ${worst.toExponential(2)}, ${ms.toFixed(1)} ms per image`);
process.exit(worst < 1e-4 ? 0 : 1);
