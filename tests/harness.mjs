// Headless driver for Layernaut Studio's code generator.
// Stubs the browser globals the generator touches, then enumerates every
// model type the interface exposes and writes each export to disk.
import fs from 'fs';
import path from 'path';

const ROOT = '..';
const OUT = process.env.OUTDIR || './generated';

// --- browser stubs --------------------------------------------------------
const FIELDS = {
  numClasses: process.env.NUMCLASSES || '10', epochs: process.env.EPOCHS || '3', batchSize: '32', lr: '0.001',
  optimizer: 'adam', lossFunction: 'categorical_crossentropy',
  freezeLayers: 'base', inputSize: process.env.INPUTSIZE || '224', customTop: 'default'
};
globalThis.localStorage = {
  _d: {}, getItem(k) { return this._d[k] ?? null; },
  setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; }
};
globalThis.document = {
  getElementById(id) { return id in FIELDS ? { value: FIELDS[id] } : null; },
  querySelector() { return null; }, querySelectorAll() { return []; },
  addEventListener() {}
};
globalThis.window = {};

const { codeGenerator } = await import(path.join(ROOT, 'js/code-generator.js'));
const { state } = await import(path.join(ROOT, 'js/state.js'));
const { models } = await import(path.join(ROOT, 'js/config/models.js'));
const { layerTypes } = await import(path.join(ROOT, 'js/config/layers.js'));
const { mlConfigurations } = await import(path.join(ROOT, 'js/config/ml-config.js'));

fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

function reset() {
  state.model = null; state.customLayers = []; state.customLayerConfigs = [];
  state.currentMode = 'prebuilt'; state.modelMode = 'scratch'; state.mlConfig = null;
}

const cases = [];

// 1. every model in the model grid, in every mode the UI offers for it
for (const [key, m] of Object.entries(models)) {
  if (m.type === 'ml') {
    cases.push({ id: `ml_${key}`, kind: 'ml', label: m.name, setup: () => { reset(); state.model = key; } });
  } else {
    if (m.fromScratch !== false) {
      cases.push({ id: `dl_${key}_scratch`, kind: 'dl', label: `${m.name} (scratch)`,
        setup: () => { reset(); state.model = key; state.modelMode = 'scratch'; } });
    }
    if (m.supportsPretrained) {
      cases.push({ id: `dl_${key}_pretrained`, kind: 'dl', label: `${m.name} (pretrained)`,
        setup: () => { reset(); state.model = key; state.modelMode = 'pretrained'; } });
    }
  }
}

// 2. every freezing option the interface offers, on a pretrained backbone.
// These were previously untested: 'all_but_last' was offered but unimplemented.
for (const freeze of ['none', 'base', 'partial', 'all_but_last']) {
  cases.push({ id: `freeze_${freeze}`, kind: 'freeze', label: `Pretrained VGG16, freeze=${freeze}`,
    setup: () => { reset(); state.model = 'vgg16'; state.modelMode = 'pretrained'; FIELDS.freezeLayers = freeze; } });
}

// 3. every layer type the Custom Builder offers, used on its own
for (const { type } of layerTypes) {
  cases.push({ id: `custom_${type}`, kind: 'custom', label: `Custom Builder: ${type}`,
    setup: () => { reset(); state.currentMode = 'custom'; state.customLayers = [type]; state.customLayerConfigs = [{}]; } });
}

// 3. an empty custom stack (reachable: switch to Custom Builder, export immediately)
cases.push({ id: 'custom_empty', kind: 'custom', label: 'Custom Builder: no layers',
  setup: () => { reset(); state.currentMode = 'custom'; } });

// 4. a realistic multi-layer custom CNN
cases.push({ id: 'custom_cnn', kind: 'custom', label: 'Custom Builder: CNN stack',
  setup: () => { reset(); state.currentMode = 'custom';
    state.customLayers = ['Conv2D', 'BatchNorm', 'MaxPool', 'Conv2D', 'GlobalAvgPool', 'Dense'];
    state.customLayerConfigs = [{ filters: 32 }, {}, {}, { filters: 64 }, {}, { units: 128 }]; } });

const report = [];
for (const c of cases) {
  FIELDS.freezeLayers = 'base';
  c.setup();
  let py = null, nb = null, err = null;
  // cases that SHOULD refuse to generate
  try { py = codeGenerator.generatePythonScript(); } catch (e) { err = `script: ${e.message}`; }
  try { nb = codeGenerator.generateColabNotebook(); } catch (e) { err = (err ? err + ' | ' : '') + `notebook: ${e.message}`; }
  if (py !== null) fs.writeFileSync(path.join(OUT, `${c.id}.py`), py);
  if (nb !== null) fs.writeFileSync(path.join(OUT, `${c.id}.ipynb`), nb);

  const flags = [];
  if (c.kind === 'freeze' && py) {
    const wants = c.id.replace('freeze_', '');
    const emitted = /trainable = False/.test(py);
    if (wants !== 'none' && !emitted) flags.push('NO_FREEZE: option produced no freezing code');
  }
  const shouldRefuse = (c.id === 'custom_empty');
  if (shouldRefuse) { if (!err) flags.push('SHOULD_REFUSE: exported an empty model'); }
  else if (err) flags.push(`THREW: ${err}`);
  if (py && py.includes('# Unsupported ML model type')) flags.push('EMPTY_BODY: generator emitted placeholder comment');
  if (c.kind === 'custom' && py) {
    // did the requested layer actually appear in the Sequential block?
    const layerName = c.id.replace('custom_', '');
    // Functional API: count the layer applications between the input and head.
    const emitted = (py.split('x = inputs')[1] || '').split('outputs =')[0];
    const nLayers = (emitted.match(/= layers\./g) || []).length;
    if (layerTypes.some(l => l.type === layerName) && nLayers < 1) flags.push('SILENT_DROP: layer produced no code');
  }
  if (c.id === 'custom_empty') err = null;
  report.push({ ...c, err, flags, hasMlConfig: c.kind === 'ml' ? !!mlConfigurations[c.id.slice(3)] : null });
}

fs.writeFileSync('./report.json', JSON.stringify(report.map(({ setup, ...r }) => r), null, 2));

console.log(`generated ${cases.length} cases into ${OUT}`);
for (const r of report) {
  if (r.err || r.flags.length) console.log(`  ✗ ${r.id.padEnd(28)} ${r.err || r.flags.join('; ')}`);
}
const { codeGenerator: cg } = await import(path.join(ROOT, 'js/code-generator.js'));
const mlKeys = Object.entries(models).filter(([, m]) => m.type === 'ml').map(([k]) => k);
const noBuilder = mlKeys.filter(k => !cg.mlBuilders[k]);
console.log(`\nCoverage: ${mlKeys.length - noBuilder.length}/${mlKeys.length} ML models have a code generator` + (noBuilder.length ? ` — MISSING: ${noBuilder.join(', ')}` : ''));
const noLayerCase = layerTypes.map(l => l.type).filter(t => report.find(r => r.id === `custom_${t}`)?.flags.length);
console.log(`Coverage: ${layerTypes.length - noLayerCase.length}/${layerTypes.length} builder layer types emit code` + (noLayerCase.length ? ` — BROKEN: ${noLayerCase.join(', ')}` : ''));

console.log('\nML models with no parameter panel in mlConfigurations:');
console.log('  ' + Object.entries(models).filter(([k, m]) => m.type === 'ml' && !mlConfigurations[k]).map(([k]) => k).join(', '));
