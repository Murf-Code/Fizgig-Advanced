"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");

const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");

// The TagBench-inspired sidebar has TWO visible lists instead of tabbing
// between Image Tags and All Tags. The bottom half may show AI tools.
assert.match(app,/const editorPane=el\("section",\{class:"studio-tag-editor-pane"\}\)/);
assert.match(app,/const tagIndexPane=el\("section",\{class:"studio-tag-index-pane"\}\)/);
assert.match(app,/right\.querySelector\("\.studio-image-tags-heading"\)/);
assert.match(app,/right\.querySelector\("\.studio-add-tag"\),tags,word,naturalEditor,suggestions/);
assert.match(app,/while\(right\.firstChild\)tagIndexPane\.append\(right\.firstChild\)/);
assert.match(app,/rightDock\.append\(showAll,showAi\)/);
assert.match(app,/right\.append\(editorPane,rightDock,tagIndexPane,aiPane\)/);
assert.match(app,/const showAll=btn\("All Tags",\(\)=>showTabs\("all"\)\)/);
assert.match(app,/const showAi=btn\("Auto-Captioner",\(\)=>showTabs\("ai"\)\)/);
assert.doesNotMatch(app,/editorPane\.hidden\s*=/);
assert.doesNotMatch(app,/showImage=btn\("Image Tags"/);
assert.match(app,/tagIndexPane\.hidden=S\.cap\.studioDock!=="all"/);
assert.match(app,/aiPane\.hidden=S\.cap\.studioDock!=="ai"/);

// Search, tag frequency sort with an independent ascending/descending
// control, auto-completion, and keyboard-accessible natural-language toggle.
assert.match(app,/const sortDirection=el\("select"/);
assert.match(app,/sortDirection\.addEventListener\("change",updateTagSort\)/);
assert.match(app,/indexSearch\.addEventListener\("input"/);
assert.match(app,/studioTagAutocomplete/);
assert.match(app,/class:"studio-natural-toggle"/);
assert.match(css,/#tab-captions \.studio-host \.studio-image-tag-list\s*\{/);
assert.match(css,/#tab-captions \.studio-host \.studio-tag-index-pane \.studio-explore>/);
assert.match(css,/min-height:0;\s*height:100%;\s*overscroll-behavior:contain/);
assert.match(css,/#tab-captions \.studio-tag-dock\[data-active="all"\]/);

// The user explicitly asked to leave the existing global selector alone.
// Waterfall and Constellation still replace the full workbench stage.
assert.match(app,/const viewButtons=el\("div",\{class:"studio-view-rail"/);
assert.match(app,/\[\["list","Tags"\],\["waterfall","Waterfall"\],\["constellation","Constellation"\]\]/);
assert.match(app,/const openView=\(mode\)=>/);
assert.match(app,/fullStage\.append\(flow\)/);
assert.match(app,/fullStage\.append\(network\)/);
assert.match(app,/const flow=CaptionStudioViz\.tagbenchWaterfall\(/);
assert.match(css,/#tab-captions \.studio-tb-explore\s*\{/);

// The left splitter starts at the smallest allowed 15% (rather than the old
// 19%) without discarding custom widths. Thumbnail cards render 15% smaller.
assert.match(app,/sizes=\{left:15,right:36\}/);
assert.match(app,/sizes\.left===19&&sizes\.right===36/);
assert.match(app,/Math\.max\(15,Math\.min\(30,sizes\.left\)\)/);
assert.match(css,/#tab-captions \.studio-thumb-rail \.studio-browser-row\s*\{[^}]*width:85%;/);
assert.match(css,/max-height:calc\(var\(--caption-settings-preview-height,360px\) \* \.85\)/);

// The center image fits available vertical space, preserving aspect ratio.
// The old 68vh/750px image ceiling prevented this on tall workbench layouts.
assert.match(css,/#tab-captions \.studio-host \.studio-editor\{[^}]*height:min\(82dvh,950px\)/);
assert.match(css,/#tab-captions \.studio-image-stage \.studio-preview\{[^}]*flex:1 1 0;[^}]*height:100%;/);
assert.match(css,/#tab-captions \.studio-image-stage \.studio-preview img,[\s\S]*?max-height:100%;[\s\S]*?object-fit:contain/);
assert.doesNotMatch(css,/\.studio-image-stage \.studio-preview img[^\n]*max-height:min\(68vh,750px\)/);

console.log("TagBench stacked sidebar and unchanged Tags/Waterfall/Constellation selector checks passed");
