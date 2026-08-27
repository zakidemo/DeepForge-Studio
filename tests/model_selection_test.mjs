// The provider retires models on its own schedule. These tests assert the
// client discovers what is available rather than depending on a fixed name.
import path from 'path';
const els = {};
const mk = (id) => ({ id, value:'', innerHTML:'', dataset:{}, classList:{add(){},remove(){}},
                      setAttribute(){}, removeAttribute(){}, closest:()=>null });
globalThis.localStorage = { getItem:()=>null, setItem(){}, removeItem(){} };
globalThis.sessionStorage = { _d:{}, getItem(k){return this._d[k]??null;}, setItem(k,v){this._d[k]=v;}, removeItem(k){delete this._d[k];} };
globalThis.window = { addEventListener(){} };
globalThis.document = { addEventListener(){}, body:{appendChild(){}}, createElement:()=>mk('t'),
                        getElementById:(id)=>(els[id] ||= mk(id)), querySelector:()=>null, querySelectorAll:()=>[] };

const { geminiOptimizer } = await import(path.join(process.cwd(), 'js/gemini.js'));
let fail = 0;
const check = (c,m) => { if(!c){ fail++; console.log('  FAIL '+m); } };

// no model name is hardcoded into the URL
geminiOptimizer.setApiKey('test-key');
check(!geminiOptimizer.baseURL.includes('gemini-2.0-flash'), 'retired model still in the endpoint');

// discovery picks a preferred model from what the account actually offers
globalThis.fetch = async () => ({ ok:true, json: async () => ({ models: [
  { name:'models/gemini-3.5-flash', supportedGenerationMethods:['generateContent'] },
  { name:'models/gemini-3.6-flash', supportedGenerationMethods:['generateContent'] },
  { name:'models/text-embedding-004', supportedGenerationMethods:['embedContent'] },
  { name:'models/imagen-4', supportedGenerationMethods:['generateContent'] }
]})});
const available = await geminiOptimizer.listModels();
check(!available.includes('text-embedding-004'), 'embedding model offered for text generation');
check(!available.includes('imagen-4'), 'image model offered for text generation');
check(geminiOptimizer.getModel() === 'gemini-3.5-flash', `preferred model not chosen (got ${geminiOptimizer.getModel()})`);
check(geminiOptimizer.baseURL.includes('gemini-3.5-flash'), 'endpoint not rebuilt for the chosen model');

// when none of the preferences exist, it still finds something usable
geminiOptimizer.setModel('');
globalThis.fetch = async () => ({ ok:true, json: async () => ({ models: [
  { name:'models/gemini-9.9-flash-future', supportedGenerationMethods:['generateContent'] }
]})});
await geminiOptimizer.listModels();
check(geminiOptimizer.getModel() === 'gemini-9.9-flash-future', 'no fallback when preferences are unavailable');

// a retirement produces an actionable message, not a raw provider string
geminiOptimizer.setModel('gemini-2.0-flash');
globalThis.fetch = async () => ({ ok:false, status:404, json: async () => ({
  error: { message: 'This model models/gemini-2.0-flash is no longer available.' } }) });
try {
  await geminiOptimizer.makeRequest('hi', 10);
  check(false, 'retired model did not raise');
} catch (e) {
  check(/no longer available from the provider/.test(e.message), 'error message not rewritten');
  check(/Reconnect/.test(e.message), 'error does not tell the user what to do');
}

console.log(fail ? `\n${fail} failure(s)` : '\nmodel selection: all checks passed');
