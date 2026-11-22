// Contract tests for the one shared Image Prep editor, now mounted inline in Tags.
// Run: node checks/test_caption_studio_image.cjs
"use strict";
const fs=require("node:fs"),assert=require("node:assert/strict");
const js=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const html=fs.readFileSync("src/fizgig/web/static/index.html","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");

for(const id of ["captionImageWorkspace","captionImageWorkspaceNote","maskEdit","meStage",
                 "meCanvas","meSize","meApply","meUndo","meRedo","captionImageClose"])
  assert.match(html,new RegExp('id="'+id+'"'));
assert.equal((html.match(/id="maskEdit"/g)||[]).length,1);
assert.equal((html.match(/id="meStage"/g)||[]).length,1);
assert.match(js,/function captionInlineStageMount\(preview\)\{/);
assert.match(js,/preview\.replaceChildren\(stage\)/);
assert.match(js,/function captionInlineStageDetach\(\)\{/);
assert.match(js,/home\.parent\.insertBefore\(stage/);
assert.match(js,/if\(S\.cap\.imageEditing\?\.inline\)captionInlineStageDetach\(\)/,
  "the live canvas must survive Tags thumbnail/filter updates");
assert.match(js,/captionInlineSyncStage\(preview\)/);
assert.match(js,/S\.cap\.imageEditing=\{folder,name:it\.name,path:it\.path,inline:true\}/);
assert.match(js,/m\.studioFolder=folder/);
assert.match(js,/folder=m\.studioFolder \|\| S\.imageFolder/);
assert.match(js,/folder=\$\{encodeURIComponent\(m\.folder \|\| S\.imageFolder\)\}/);
assert.match(js,/captionImageChanged\(\);\s*return r;/);
assert.match(js,/m\.dirty&&\!\(await meSave\(\)\)/);
assert.match(js,/if\(!\(await meRmSave\(false\)\)\)return false;/);
assert.match(js,/captionImageClose\(\)\.then\(ok=>\{if\(ok\)showTab\(name\);\}\)/);

const start=js.indexOf('const pictureActions=el("div",{class:"studio-image-toolbar"');
const end=js.indexOf('const orderKey=',start);
assert(start>0&&end>start,"Tags toolbar missing");
const toolbar=js.slice(start,end);
for(const word of ['btn("Remove"','btn("Crop"','btn("Undo"','btn("Redo"','btn("Apply"'])
  assert(toolbar.includes(word),"missing tool "+word);
assert.match(toolbar,/sizeLabel\.hidden=tool!=="remove"/);
assert.match(toolbar,/applyImage\.hidden=!active/);
assert.match(toolbar,/await captionImageOpen\(current,tool\)/);
assert.match(toolbar,/await captionImageClose\(\)/);
assert.match(toolbar,/await meApply\(\)/);
assert.doesNotMatch(toolbar,/Edit image \/ crop/);
assert.doesNotMatch(toolbar,/showTab\("prep"\)/);
assert.match(js,/if\(m\.tool!==tool\)meTool\(tool\)/,
  "switching thumbnails while cropping must not toggle Crop off");
assert.match(js,/S\.cap\.studioStepImage=stepImage/);
assert.match(js,/document\.querySelector\("#captionsGrid \.studio-image-apply"\)\?\.click\(\)/);
assert.match(css,/#tab-captions \.studio-image-stage \.studio-preview-editing #meStage\{/);
assert.match(css,/#tab-captions \.studio-image-toolbar\{/);
assert.match(css,/#tab-captions \.studio-image-toolbar \[hidden\]\{display:none!important\}/);

// Keep the removed-image workflow and shared undo history separate from
// in-image Remove; "Remove" on the toolbar must not delete dataset pictures.
assert.match(js,/const removeSelectedImages=async\(targetPath=null\)=>/);
assert.match(js,/studioRecord\("Remove image"/);
assert.match(js,/api\("\/api\/captions\/remove"/);
assert.match(js,/api\("\/api\/captions\/restore"/);
console.log("Tags inline Remove/Crop editor lifecycle and shared-history checks passed");
