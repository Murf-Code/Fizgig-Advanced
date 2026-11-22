// Contract checks for the shared Image Prep editor's in-tab Caption Studio integration.
// Browser layout/drag/drop needs an interactive pass; these keep critical wiring from regressing.
"use strict";
const fs = require("fs");
const assert = require("node:assert/strict");
const js = fs.readFileSync("src/fizgig/web/static/app.js", "utf8");
const html = fs.readFileSync("src/fizgig/web/static/index.html", "utf8");
const css = fs.readFileSync("src/fizgig/web/static/style.css", "utf8");

for (const id of ["captionImageWorkspace", "captionImageEditorSlot",
                  "captionImageWorkspaceNote", "captionImageClose", "maskEdit"]) {
  assert.match(html, new RegExp('id="' + id + '"'), "missing HTML mount: " + id);
}
assert.equal((html.match(/id="maskEdit"/g) || []).length, 1,
             "must reuse exactly one original image editor with its existing event handlers");
assert.match(js, /slot\.append\(editor\)/, "move the existing editor into Caption Studio");
assert.match(js, /home\.parent\.insertBefore\(editor/, "return the editor to Image Prep when closed");
assert.match(js, /meState\(\)\.studioFolder=capItemFolder\(it\)/,
             "subfolder selection must reach the native editor");
assert.match(js, /folder=m\.studioFolder \|\| S\.imageFolder/,
             "meOpen must use the selected dataset folder");
assert.match(js, /folder=\$\{encodeURIComponent\(m\.folder \|\| S\.imageFolder\)\}/,
             "meGo must not force the top-level folder");
assert.match(js, /captionImageChanged\(\);\s*return r;/,
             "crop and remove must refresh Caption Studio metadata");
// The separate "removed pictures" panel was intentionally retired; removal is
// accessible from the image context menu and restored through shared history.
assert.match(js, /const removeSelectedImages=async\(targetPath=null\)=>/);
assert.match(js, /studioRecord\("Remove image"/);
assert.match(js, /api\("\/api\/captions\/remove"/);
assert.match(js, /api\("\/api\/captions\/restore"/);
assert.match(js, /function captionImageFinishPending\(\)/);
assert.match(js, /m\.dirty && !\(await meSave\(\)\)/);
assert.match(js, /captionImageClose\(\)\.then\(ok=>\{if\(ok\)showTab\(name\);\}\)/,
             "tab navigation must wait for the editor to save before hiding it");
assert.match(css, /#tab-captions \.caption-image-workspace/);
const start = js.indexOf('btn("Edit image / crop"');
const end = js.indexOf('const orderKey=', start);
assert(start !== -1 && end > start);
const handler = js.slice(start, end);
assert.match(handler, /await captionImageOpen\(current\)/);
assert.doesNotMatch(handler, /showTab\("prep"\)/,
                    "editing should never force-switch the user to Image Prep");
console.log("Caption Studio shared image editor wiring checks passed");
