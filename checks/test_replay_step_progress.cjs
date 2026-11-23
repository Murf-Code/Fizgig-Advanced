// Verify the real browser function, not a duplicate implementation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '..', 'src', 'fizgig', 'web', 'static', 'app.js'), 'utf8');
const start = source.indexOf('function epochStepsLeft(p) {');
const end = source.indexOf('// the run\'s figures', start);
assert(start >= 0 && end > start, 'epochStepsLeft source block must exist');
const calculate = vm.runInNewContext(source.slice(start, end) + '\nepochStepsLeft;', {});

assert.equal(calculate({step: 5, total_steps: 10, total_epochs: 2, epoch_step: 0, steps_per_epoch: 5}), '5 / 5');
assert.equal(calculate({step: 6, total_steps: 10, total_epochs: 2, epoch_step: 1, steps_per_epoch: 5}), '4 / 5');
assert.equal(calculate({step: 10, total_steps: 10, total_epochs: 2, epoch_step: 5, steps_per_epoch: 5}), '0 / 5');
assert.equal(calculate({step: 12, total_steps: 20, total_epochs: 4, epoch_step: 2, steps_per_epoch: 5}), '3 / 5');
assert.equal(calculate({step: 2, total_steps: 20, total_epochs: 4}), '3 / 5');
assert.equal(calculate({step: 5, total_steps: 20, total_epochs: 4}), '0 / 5');
console.log('Global Replay Steps left: 6 browser calculation assertions passed');
