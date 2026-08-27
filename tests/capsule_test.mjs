// Verifies the capsule is a real, readable ZIP with the expected members.
import fs from 'fs';
import path from 'path';
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
const { makeZip, requirementsTxt, environmentMd, capsuleReadme } = await import(path.join(process.cwd(), 'js/capsule.js'));

const config = { toolVersion: '2.1.0', schemaVersion: 2, exportedAt: new Date().toISOString(),
                 model: 'simple_cnn', modelMode: 'scratch', seed: 42 };
const files = [
  { name: 'train.py', content: 'print("hello")\n' },
  { name: 'notebook.ipynb', content: '{"cells": []}' },
  { name: 'config.json', content: JSON.stringify(config, null, 2) },
  { name: 'requirements.txt', content: requirementsTxt(false, false) },
  { name: 'ENVIRONMENT.md', content: environmentMd(config, 42) },
  { name: 'README.md', content: capsuleReadme(config) }
];
const blob = makeZip(files);
const buf = Buffer.from(await blob.arrayBuffer());
fs.writeFileSync('capsule_test.zip', buf);
console.log(`capsule written: ${buf.length} bytes, ${files.length} members`);
