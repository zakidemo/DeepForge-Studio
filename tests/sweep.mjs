// Exhaustively sweeps the option combinations the ML panels expose and writes
// each generated script out. Catches invalid pairings (e.g. a penalty a solver
// does not support) that a single default-value test would miss.
import fs from 'fs';
import path from 'path';
const ROOT = '..';
const OUT = './sweep';

globalThis.localStorage = { _d: {}, getItem: () => null, setItem() {}, removeItem() {} };
globalThis.document = { getElementById: (id) => (id === 'numClasses' ? { value: '3' } : null), querySelector: () => null, querySelectorAll: () => [] };
globalThis.window = {};

const { codeGenerator } = await import(path.join(ROOT, 'js/code-generator.js'));
const { state } = await import(path.join(ROOT, 'js/state.js'));
const { mlConfigurations } = await import(path.join(ROOT, 'js/config/ml-config.js'));

fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

// For each model: cross-product of every select/checkbox option, with range
// params pinned at min, default and max.
function combos(params) {
  const axes = Object.entries(params).map(([key, p]) => {
    if (p.type === 'select') return p.options.map(v => [key, v]);
    if (p.type === 'checkbox') return [[key, true], [key, false]];
    return [...new Set([p.min, p.default, p.max])].map(v => [key, v]);
  });
  return axes.reduce((acc, axis) => acc.flatMap(c => axis.map(v => [...c, v])), [[]])
             .map(pairs => Object.fromEntries(pairs));
}

let n = 0, written = 0;
const failures = [];
for (const [model, cfg] of Object.entries(mlConfigurations)) {
  for (const [i, params] of combos(cfg.params).entries()) {
    for (const scaleFeatures of [true, false]) {
      n++;
      state.model = model;
      state.currentMode = 'prebuilt';
      state.mlConfig = { params, preprocessing: { scaleFeatures, testSize: 0.2, randomState: 42 } };
      try {
        const code = codeGenerator.generatePythonScript();
        fs.writeFileSync(path.join(OUT, `${model}__${i}__${scaleFeatures ? 'scaled' : 'raw'}.py`), code);
        written++;
      } catch (e) {
        failures.push(`${model} #${i} scale=${scaleFeatures}: ${e.message}`);
      }
    }
  }
}
console.log(`swept ${n} configurations, wrote ${written}, ${failures.length} generation errors`);
failures.slice(0, 10).forEach(f => console.log('  ' + f));
