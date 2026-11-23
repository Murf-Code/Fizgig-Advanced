// Keep the TagBench duplicate-review UI separate from Fizgig's original LaMa inpainting.
"use strict";
const fs=require("fs");
const assert=require("node:assert/strict");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const server=fs.readFileSync("src/fizgig/web/server.py","utf8");
const image=fs.readFileSync("src/fizgig/web/image_edit.py","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const duplicates=fs.readFileSync("src/fizgig/web/caption_duplicates.py","utf8");
assert.match(app,/function captionDuplicatePanel\(ext\)/);
const duplicatePanel=app.slice(app.indexOf("function captionDuplicatePanel(ext)"),app.indexOf("function captionDuplicatePanel(ext)")+16000);
assert.doesNotMatch(duplicatePanel,/text:"View caption"/);
assert.match(app,/\.studio-duplicate-card > img/);
assert.match(app,/\.st-preview, \.studio-duplicate-cards/);
assert.match(app,/title:"View full image"/);
assert.match(app,/thumb\.addEventListener\("keydown"/);
assert.match(css,/\.studio-duplicate-card>img:focus-visible/);

assert.match(duplicatePanel,/el\("label",\{class:"field-row"\},tick," Keep this image"\)/);
assert.match(duplicatePanel,/text:"Remove image\/s"/);
assert.match(duplicatePanel,/captionConfirmRemoval\(/);
assert.match(duplicatePanel,/text:"Compare images"/);
assert.match(duplicatePanel,/text:"Side by side"/);
assert.match(duplicatePanel,/text:"Overlay"/);
assert.match(duplicatePanel,/choice\.kept\.size\+" kept/);
assert.match(duplicatePanel,/text:" Merge tags into kept images"/);
assert.doesNotMatch(duplicatePanel,/text:"Difference"/);


assert.match(app,/min:"0",max:"16"/);
assert.match(app,/state\.strictness/);
assert.match(app,/choice\.kept\.add\(pic\.path\)/);
assert.match(app,/function captionDuplicatePanel[\s\S]*"\/api\/captions\/duplicates\/merge_group"/);
assert.match(app,/\/api\/captions\/remove/);
assert.match(app,/Move unkept images to removed\//);
assert.match(app,/Preview copied tags/);
assert.match(app,/Apply copied tags to keepers/);
assert.match(server,/path == "\/api\/captions\/duplicates\/scan"/);
assert.match(server,/path == "\/api\/captions\/duplicates\/merge_group"/);
assert.match(server,/path == "\/api\/masks\/edit\/remove"/);
assert.match(server,/image_edit\.remove\(folder, name/);
assert.match(image,/def lama_fill\(rgb, hole, gpu=True\):/);
assert.match(image,/return lama\.inpaint\(rgb, hole, gpu=gpu\)/);
assert.match(duplicates,/def fingerprint\(path\):/);
assert.match(duplicates,/def apart\(a, b\):/);
assert.match(duplicates,/def merge_group\(/);
assert.match(css,/\.studio-duplicate-cards/);
console.log("TagBench duplicate finder and existing local LaMa removal are wired separately");
