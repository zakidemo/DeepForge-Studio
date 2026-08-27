// Exercises the layer-config modal against a minimal DOM, since the modal now
// builds every layer type's form from one spec.
import path from 'path';
const els = {};
function makeEl(id) {
  return {
    id, value: '', checked: false, innerHTML: '', dataset: {},
    classList: { add() {}, remove() {} }, setAttribute() {}, appendChild() {}, remove() {}
  };
}
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.window = { addEventListener() {} };
globalThis.document = {
  addEventListener() {},
  body: { appendChild() {} },
  getElementById: (id) => (els[id] ||= makeEl(id)),
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => makeEl('tmp')
};

const { handlers } = await import(path.join(process.cwd(), 'js/handlers.js'));
const { state } = await import(path.join(process.cwd(), 'js/state.js'));
const { layerTypes } = await import(path.join(process.cwd(), 'js/config/layers.js'));

let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL ' + msg); } };

// 1. every layer type opens a form
state.customLayers = []; state.customLayerConfigs = [];
for (const { type } of layerTypes) {
  state.editingLayerIndex = null;
  state.pendingLayerType = type;
  document.getElementById('layerConfigForm').innerHTML = '';
  handlers.showLayerConfig();
  const html = document.getElementById('layerConfigForm').innerHTML;
  check(html.includes('applyLayerConfigBtn'), `${type}: modal has no apply button`);
  check(html.length > 50, `${type}: modal body is empty`);
}
console.log(`opened config for all ${layerTypes.length} layer types`);

// 2. skip control is absent for the first layer, present once a layer exists
state.customLayers = []; state.pendingLayerType = 'Conv2D'; state.editingLayerIndex = null;
handlers.showLayerConfig();
check(!document.getElementById('layerConfigForm').innerHTML.includes('layercfg_skipFrom'), 'skip offered with no earlier layer');

state.customLayers = ['Conv2D', 'BatchNorm'];
state.customLayerConfigs = [{}, {}];
state.pendingLayerType = 'Conv2D'; state.editingLayerIndex = null;
handlers.showLayerConfig();
const html = document.getElementById('layerConfigForm').innerHTML;
check(html.includes('layercfg_skipFrom'), 'skip control missing when earlier layers exist');
check(html.includes('From layer 1 (Conv2D)'), 'earlier layer not listed as a skip source');
check(html.includes('From layer 2 (BatchNorm)'), 'second earlier layer not listed');
check(!html.includes('From layer 3'), 'a later layer was offered as a skip source');
console.log('skip-connection control behaves correctly');

console.log(failures ? `\n${failures} failure(s)` : '\nall UI config checks passed');


// --- the API key must never reach an exported artefact ---------------------
{
  const { codeGenerator } = await import(path.join(process.cwd(), 'js/code-generator.js'));
  const { geminiOptimizer } = await import(path.join(process.cwd(), 'js/gemini.js'));
  globalThis.sessionStorage = { _d: {}, getItem(k){return this._d[k]??null;}, setItem(k,v){this._d[k]=v;}, removeItem(k){delete this._d[k];} };
  const SECRET = 'AQ.Ab_TEST_KEY_SHOULD_NEVER_APPEAR_IN_EXPORTS';
  geminiOptimizer.setApiKey(SECRET);
  state.currentMode = 'custom';
  state.customLayers = ['Conv2D', 'GlobalAvgPool', 'Dense'];
  state.customLayerConfigs = [{}, {}, { units: 16 }];
  let leaked = 0;
  for (const [label, produce] of [
    ['python script', () => codeGenerator.generatePythonScript()],
    ['colab notebook', () => codeGenerator.generateColabNotebook()],
    ['model preview', () => codeGenerator.generateModelCode()],
    ['config json', () => JSON.stringify(handlers.buildConfigObject())]
  ]) {
    if (produce().includes(SECRET)) { leaked++; console.log(`  FAIL API key leaked into ${label}`); }
  }
  console.log(leaked ? `${leaked} leak(s)` : 'no export contains the API key');
  if (leaked) process.exit(1);
}
