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


// Search index rows normalize once, then stop after the first eight ranked
// suggestions; this must preserve frequency ordering and exclusion of tags
// already assigned to the active image.
const rows=[
  ["Lion",42,"lion"],["fur",38,"fur"],["Blue fur",31,"blue fur"],
  ["feline",25,"feline"],["lioness",13,"lioness"],["lion cub",9,"lion cub"]
];
assert.deepEqual(T.findTagMatches(rows,"Li"),[
  rows[0],rows[3],rows[4],rows[5]]);
assert.deepEqual(T.findTagMatches(rows,"",null),rows);
assert.deepEqual(T.findTagMatches(rows,"fur",new Set(["fur"])),[rows[2]]);
assert.deepEqual(T.findTagMatches(rows,"lion",new Set(["lion"]),1),[rows[4]]);
const large=Array.from({length:10000},(_,i)=>["tag"+i,i,"tag"+i]);
assert.equal(T.findTagMatches(large,"tag",null,8).length,8);
assert.equal(T.findTagMatches(large,"tag9999",null,8)[0][0],"tag9999");
console.log("Caption Studio tag parsing and Boolean filter checks passed");

assert.equal(T.addFilter("", "red hair", "and"), '"red hair"');
assert.equal(T.addFilter("", "red hair", "not"), 'NOT "red hair"');
assert.equal(T.addFilter("", "red hair", "or"), '"red hair"');
const conjunct=T.addFilter('blue OR black', "red hair", "and");
assert.equal(T.search(["blue"],conjunct),false);
assert.equal(T.search(["blue","red hair"],conjunct),true);
assert.equal(T.search(["black","red hair"],conjunct),true);
const excluded=T.addFilter('blue OR black', "red hair", "not");
assert.equal(T.search(["blue"],excluded),true);
assert.equal(T.search(["blue","red hair"],excluded),false);
const disjunct=T.addFilter('blue AND black', "red hair", "or");
assert.equal(T.search(["red hair"],disjunct),true);
assert.equal(T.search(["blue"],disjunct),false);
assert.equal(T.search(["blue","black"],disjunct),true);
assert.throws(()=>T.addFilter("blue","red","invalid"),/Unknown filter mode/);


// Click-away and cross-list focus must never leave two tag sets armed for
// a later Delete action. The menu is exempt until its own action executes.
const mockTarget=klass=>({
  closest(selector) {
    return selector.split(",").some(s=>s.trim()==="."+klass)?this:null;
  },
});
assert.equal(T.tagSelectionScope(mockTarget("studio-image-tag")),"image");
assert.equal(T.tagSelectionScope(mockTarget("studio-inline-image-tag")),"image");
assert.equal(T.tagSelectionScope(mockTarget("studio-index-tag")),"all");
assert.equal(T.tagSelectionScope(mockTarget("studio-inline-tag-edit")),"all");
assert.equal(T.tagSelectionScope(mockTarget("studio-tag-context-menu")),"menu");
assert.equal(T.tagSelectionScope(mockTarget("studio-image-context-menu")),"menu");
assert.equal(T.tagSelectionScope(mockTarget("studio-tag-list")) ,null);
assert.equal(T.tagSelectionScope(null),null);
const imagePicked=new Set(["image tag"]);
const allPicked=new Set(["dataset tag"]);
assert.deepEqual(T.scopeTagSelections(imagePicked,allPicked,"menu"),
  {image:false,all:false});
assert.equal(imagePicked.size,1);
assert.equal(allPicked.size,1);
assert.deepEqual(T.scopeTagSelections(imagePicked,allPicked,"image"),
  {image:false,all:true});
assert.deepEqual([...imagePicked],["image tag"]);
assert.equal(allPicked.size,0,"selecting an image tag disarms All Tags Delete");
allPicked.add("dataset tag");
assert.deepEqual(T.scopeTagSelections(imagePicked,allPicked,"all"),
  {image:true,all:false});
assert.equal(imagePicked.size,0,"selecting All Tags disarms Image Tags Delete");
assert.deepEqual([...allPicked],["dataset tag"]);
imagePicked.add("another image tag");
assert.deepEqual(T.scopeTagSelections(imagePicked,allPicked,null),
  {image:true,all:true});
assert.equal(imagePicked.size,0,"clicking outside clears Image Tags");
assert.equal(allPicked.size,0,"clicking outside clears All Tags");
