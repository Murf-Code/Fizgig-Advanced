"use strict";
// Regression suite for image-first Tags UX and a 10k+ image dataset.
// Run: node checks/test_caption_workbench_focus.cjs
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const get=(start,end)=>{
  const a=app.indexOf(start),b=app.indexOf(end,a+start.length);
  assert(a>=0&&b>a,"Missing implementation "+start);
  return app.slice(a,b);
};
// Geometry is derived from source-image aspect ratios, not network image load
// events. This must produce a stable full-scroll range for real datasets.
const geometryContext={Float64Array,Math};
vm.runInNewContext(get("function studioThumbGeometry(","function renderCaptionStudio("),geometryContext);
const sizes=[[496,496],[1300,600],[600,1300],null,[496,496]];
const items=Array.from({length:11400},(_,i)=>({size:sizes[i%sizes.length]}));
const g=geometryContext.studioThumbGeometry(items,240);
assert.equal(g.heights.length,items.length);
assert.equal(g.offsets.length,items.length+1);
assert(g.total>1000000);
for(let i=1;i<g.offsets.length;i++)assert(g.offsets[i]>g.offsets[i-1]);
for(const i of [0,1,10,600,5722,11399])
  assert.equal(geometryContext.studioThumbIndex(g.offsets,g.offsets[i]+1),i);
assert.equal(geometryContext.studioThumbIndex(g.offsets,1e12),items.length-1);
const last=geometryContext.studioThumbIndex(g.offsets,g.total);
assert.equal(last,items.length-1);
const narrow=geometryContext.studioThumbGeometry(items,98);
assert(narrow.total<g.total);
assert(narrow.heights.every(n=>n>=64));
const upper=geometryContext.studioThumbIndex(g.offsets,g.offsets[6500]+840)+10;
const lower=geometryContext.studioThumbIndex(g.offsets,g.offsets[6500])-7;
assert(upper-lower<80,"a viewport should mount dozens, not thousands");
assert.match(app,/const start=Math.max\(0,studioThumbIndex\(offsets,top\)-7\)/);
assert.match(app,/const end=Math.min\(items.length,studioThumbIndex\(offsets,bottom\)\+10\)/);
assert.match(app,/thumbs\.revealPath=path=>revealIndex\(/);
assert.match(app,/S\.cap\.studioThumbScrollTop=priorRail\.scrollTop/);
assert.match(app,/thumbs\.addEventListener\("scroll",queueWindow,\{passive:true\}\)/);
assert.match(app,/thumbs\.dataset\.rendered=String\(end-start\)/);
assert.doesNotMatch(app,/const moreThumbs=el\(/);
assert.match(css,/\.studio-thumb-rail\.is-windowed \.studio-thumb-window\{/);

// The compact status must never imply success while a draft is unsaved.
assert.match(app,/id:"studioLiveStatus"/);
assert.match(app,/if\(failedTask\)\{state="attention";label="Save failed";\}/);
assert.match(app,/const savingTask=currentTasks\.some\(task=>task\.busy\|\|task\.pending\)/);
assert.match(app,/else if\(status\?\.attention\)\{state="attention";label="Needs attention";\}/);
assert.match(app,/state="saving";label="Saving…"/);
assert.match(app,/state="pending";label="Pending next epoch"/);
assert.match(app,/S\.cap\.studioStatusRefresh\?\.\(\)/);
assert.match(app,/actionBar\.append\(learningToggle,learningFilter,viewButtons,liveStatus\)/);
assert.match(css,/\.studio-workspace-status\[data-state="attention"\]/);

// Image remains a full-height stage; Remove/Crop have no large detached UI.
assert.match(app,/const pictureActions=el\("div",\{class:"studio-image-toolbar"/);
assert.match(app,/sizeLabel\.hidden=tool!=="remove"/);
assert.match(app,/applyImage\.hidden=!active/);
assert.match(css,/\.studio-image-stage \.studio-preview-editing #meStage/);
console.log("Focus-first Tags status and 11,400-image virtual rail checks passed");
