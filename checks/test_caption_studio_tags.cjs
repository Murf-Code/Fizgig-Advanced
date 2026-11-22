const assert = require("node:assert/strict");
const T = require("../src/fizgig/web/static/caption_studio_tags.js");

assert.equal(T.kind("ohwx, portrait, long hair"), "tags");
assert.equal(T.kind("A curious lion poses for a photograph at sunset."), "prose");
assert.equal(T.kind("A very long sentence without punctuation that describes a complex scene"), "prose");
assert.deepEqual(T.tags("A person smiles, while sitting in a café."), []);
assert.equal(T.compose("First sentence, second sentence.", ["wrong"]), "First sentence, second sentence.");
assert.deepEqual(T.tags("lion; cosmic mane; blue", "tags", ";"), ["lion", "cosmic mane", "blue"]);
assert.equal(T.compose("lion; blue\nBright sunset", ["blue", "lion"], "tags", ";"), "blue; lion\nBright sunset");
assert.equal(T.search(["red hair", "blue eyes"], '"red hair" AND NOT "green eyes"'), true);
// Compiled filter must behave exactly like one-off evaluation while reusing the
// parsed query across thousands of images in the same dataset.
const compiled=T.compileSearch('("red hair" OR lion) AND NOT "green eyes"');
assert.equal(compiled(["red hair","blue eyes"]),true);
assert.equal(compiled(["lion","green eyes"]),false);
assert.equal(compiled(["Lion","blue eyes"]),true);
assert.equal(compiled(["pink hair"]),false);
assert.throws(()=>T.compileSearch("wolf AND ("),/Expected a tag|Missing closing parenthesis/);
assert.equal(T.search(["red hair", "blue eyes"], '("red hair" OR "black hair") AND blue'), false);
assert.equal(T.search(["red hair", "blue eyes"], '("red hair" OR "black hair") AND "blue eyes"'), true);
assert.equal(T.search(["red hair"], 'NOT missing AND red'), false);
assert.equal(T.search(["red hair"], 'NOT missing AND "red hair"'), true);
assert.equal(T.search(["red hair"], '"red hair" -green'), true);
assert.equal(T.search(["red hair"], '"red hair" AND (green OR blue)'), false);
assert.equal(T.search(["red hair"], 'RED'), false); // exact tags, not substrings
assert.throws(() => T.search([], 'red AND (blue'), /Missing closing parenthesis/);
assert.throws(() => T.search([], 'red OR'), /Expected a tag/);
assert.deepEqual(T.unique(["cat", "Cat", "lion"]), ["cat", "lion"]);
assert.equal(T.wildcards(["lion", "cat", "cat"]), "cat\nlion\n");
assert.deepEqual(T.order(["z", "a", "b"], {rule:"alpha"}), ["a", "b", "z"]);
assert.deepEqual(T.order(["z", "a", "b"], {rule:"reverse"}), ["b", "a", "z"]);
assert.deepEqual(T.order(["z", "a", "b"], {rule:"frequency",
  counts:new Map([["a",10],["z",1],["b",5]])}), ["a", "b", "z"]);
assert.deepEqual(T.order(["z", "a", "b"], {rule:"token-asc",
  tokens:new Map([["a",4],["z",1],["b",2]])}), ["z", "b", "a"]);
assert.deepEqual(T.order(["z", "a", "b"], {rule:"alpha",front:["b"],back:["z"]}),
  ["b", "a", "z"]);
assert.deepEqual(T.order(["z", "a", "b", "A"], {rule:"alpha"}), ["a", "b", "z"]);

assert.equal(T.captionWildcards(["lion, blue mane\nAt sunset", "lion, blue mane", ""]),
  "lion, blue mane At sunset\nlion, blue mane\n");
assert.equal(T.captionWildcards([]), "");

console.log("Caption Studio tag parsing and Boolean filter checks passed");
