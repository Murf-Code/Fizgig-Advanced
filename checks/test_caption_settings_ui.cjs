"use strict";
// The global checkbox switch has flex-basis:32px. A column-form label in
// Settings must override that basis or the switch becomes tall and sideways.
const assert=require("node:assert/strict");
const fs=require("node:fs");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const html=fs.readFileSync("src/fizgig/web/static/index.html","utf8");
const checkbox=css.match(/\.caption-settings-field input\[type=checkbox\]\{([^}]*)\}/);
assert(checkbox,"missing Settings switch override");
assert.match(checkbox[1],/flex\s*:\s*0 0 auto/);
assert.match(checkbox[1],/width\s*:\s*34px\s*;\s*height\s*:\s*20px/);
assert.match(checkbox[1],/align-self\s*:\s*flex-start/);
const caret=css.match(/\.caption-settings-advanced>summary::after\{([^}]*)\}/);
assert(caret,"missing Advanced disclosure arrow");
assert.match(caret[1],/border-right:\s*2px solid/);
assert.match(caret[1],/transform:rotate\(45deg\)/);
assert.match(app,/Caption generation & batch tools/);
assert.match(app,/AI captioning models, trigger words, and batch caption actions/);
assert.match(html,/id="meDialogWindow"/);
assert.match(html,/class="me-dialog-quick-note"/);
assert.doesNotMatch(html,/class="me-dialog-guide"/);
assert.match(html,/4\. Open Image Editor/);
assert.equal((html.match(/id="meCanvas"/g)||[]).length,1);
console.log("Settings switch orientation, advanced label, and glass editor guidance passed");
