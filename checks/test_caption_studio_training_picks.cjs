"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const Picks = require("../src/fizgig/web/static/caption_studio_training_picks.js");

assert.deepEqual(Picks.names(["sub/a.png", "sub/a.png", "b.png"]), ["sub/a.png", "b.png"]);
assert.deepEqual(Picks.names("a.png; a.png\nb.png"), ["a.png", "b.png"]);
assert.equal(Picks.present(["sub/A.png"], "sub/a.png"), true);

const validation = Picks.toggleValidation(["sub/a.png", "b.png"], ["sub/a.png", "c.png"]);
assert.equal(validation.remove, false);
assert.deepEqual(validation.picked, ["sub/a.png", "b.png", "c.png"]);
const removedValidation = Picks.toggleValidation(validation.picked, ["sub/a.png", "c.png"]);
assert.equal(removedValidation.remove, true);
assert.deepEqual(removedValidation.picked, ["b.png"]);

const lion = {name:"lion.png", caption:"lion, cosmic mane\nA colorful male lion"};
const fox = {name:"sub/fox.webp", caption:"fox, red fur"};
const empty = {name:"empty.png", caption:""};
const manual = "My carefully edited sample prompt";

const first = Picks.toggleSamples(manual, [], [lion, empty]);
assert.equal(first.removing, false);
assert.deepEqual(first.changed, ["lion.png"]);
assert.deepEqual(first.skipped, ["empty.png"]);
assert.equal(first.prompts, "My carefully edited sample prompt\nlion, cosmic mane A colorful male lion");
assert.deepEqual(first.pairs, [{
  index:1, hash:Picks.hash("lion, cosmic mane A colorful male lion"),
  image:"lion.png",fromWorkbench:true,
}]);

const second = Picks.toggleSamples(first.prompts, first.pairs, [lion, fox]);
assert.equal(second.removing, false);
assert.deepEqual(second.changed, ["sub/fox.webp"]);
assert.equal(second.pairs[1].index, 2);
assert.equal(second.prompts.split("\n").length, 3);
const removeLion = Picks.toggleSamples(second.prompts, second.pairs, [lion]);
assert.equal(removeLion.removing, true);
assert.equal(removeLion.prompts, "My carefully edited sample prompt\nfox, red fur");
assert.deepEqual(removeLion.pairs.map(x=>[x.image,x.index]), [["sub/fox.webp",1]]);
const removeFox = Picks.toggleSamples(removeLion.prompts, removeLion.pairs, [fox]);
assert.equal(removeFox.prompts, manual);
assert.deepEqual(removeFox.pairs, []);

const manualPair = {index:0,hash:Picks.hash(manual),image:"lion.png"};
const removeManual = Picks.toggleSamples(manual, [manualPair], [lion]);
assert.equal(removeManual.prompts,manual);
assert.deepEqual(removeManual.pairs,[]);

const edited = Picks.toggleSamples("My carefully edited sample prompt\nUser modified this line",
  first.pairs, [lion]);
assert.equal(edited.prompts,"My carefully edited sample prompt\nUser modified this line");
assert.deepEqual(edited.pairs,[]);

const multiple = Picks.toggleSamples("custom\n# ignore me", [], [lion, fox]);
const removeBoth = Picks.toggleSamples(multiple.prompts, multiple.pairs, [fox, lion]);
assert.equal(removeBoth.prompts, "custom\n# ignore me");
assert.deepEqual(removeBoth.pairs, []);

// The same context-menu function is wired to both the thumbnail rail and Waterfall.
const ui = fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const html = fs.readFileSync("src/fizgig/web/static/index.html","utf8");
assert.match(ui,/oncontextmenu:e=>openImageMenu\(e,it\)/);
assert.match(ui,/tagbenchWaterfall\(items,chooseImage,openImageMenu,/);
assert.match(ui,/toggleValidationImages=path=>/);
assert.match(ui,/toggleSampleImages=path=>/);
assert.match(ui,/VALIDATION_PICKED/);
assert.match(ui,/SAMPLE_PAIRS/);
assert.match(ui,/Add "\+plural\+" to Validation Images/);
assert.match(ui,/Add "\+plural\+" to Sample Images/);
assert.match(html,/caption_studio_training_picks\.js/);
assert.doesNotMatch(ui,/const stageNav=el\("div",\{class:"studio-stage-nav"/);
assert.doesNotMatch(ui,/el\("div",\{class:"caption-card-title",text:mediaLabel\(current\)\}\)/);

console.log("Caption Workbench training-image context menu and uncluttered editor checks passed");
