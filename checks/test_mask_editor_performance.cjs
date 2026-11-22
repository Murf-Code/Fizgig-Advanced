"use strict";
// Regression checks for Fizgig's shared Image Prep / Caption Workbench editor.
// Run: node checks/test_mask_editor_performance.cjs
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const source=(start,end)=>{
  const a=app.indexOf(start),b=app.indexOf(end,a+start.length);
  assert(a>=0&&b>a,"Missing editor source section: "+start);
  return app.slice(a,b);
};
assert.match(app,/mePaintFrame=requestAnimationFrame\(\(\)=>\{mePaintFrame=0;mePaint\(\);\}\)/);
assert.match(app,/if\(pan\|\|m\.cropDrag\|\|last\)meDraw\(\)/);
assert.match(app,/function meRemoveLayer\(\)/);
assert.match(app,/meMarkMaskDirty\(x,y,r\)/);
assert.match(app,/meMarkMaskDirty\(\);\s*m\.dirty = true; m\.samLoaded = true/);
assert.match(css,/\.me-stage \.me-brush-cursor\{/);
assert.match(css,/\.me-stage \.me-brush-cursor\[hidden\]/);
assert.match(css,/#tab-captions\.caption-theme-glass \.caption-image-workspace\{[^}]*backdrop-filter:none/s);

// Rapid invalidations get exactly one paint in each requestAnimationFrame.
let queued=[],paints=0;
const frame={requestAnimationFrame:fn=>(queued.push(fn),queued.length),mePaint:()=>paints++};
vm.runInNewContext(source("let mePaintFrame=0;","// the mask's own pixels"),frame);
frame.meDraw();frame.meDraw();frame.meDraw();
assert.equal(queued.length,1);
queued.shift()();assert.equal(paints,1);
frame.meDraw();assert.equal(queued.length,1);
queued.shift()();assert.equal(paints,2);

// Track the bounding rectangle of dabs; full operations request full refresh.
const state={mask:{width:2000,height:1000},rm:{width:2000,height:1000}};
const bounds={meState:()=>state,Number,Math};
vm.runInNewContext(source("function meDirtyRect(","function meMaskThemeKey("),bounds);
bounds.meMarkMaskDirty(100,100,10);
assert.deepEqual(JSON.parse(JSON.stringify(state.maskDirtyRect)),{l:87,t:87,r:113,b:113});
bounds.meMarkMaskDirty(140,130,5);
assert.deepEqual(JSON.parse(JSON.stringify(state.maskDirtyRect)),{l:87,t:87,r:148,b:138});
bounds.meMarkMaskDirty();
assert.equal(state.maskDirtyFull,true);assert.equal(state.maskDirtyRect,null);

// An unchanged mask reuses both 4MP overlays; subsequent brush touches
// composite only the changed rectangle and never the full image.
const calls=[];
const layers={meState:()=>state,meMaskThemeKey:()=> "same-theme",
  themeColor:name=>name,meCompositeRegion:(...args)=>calls.push(args),
  el:(tag,opts)=>({width:opts.width,height:opts.height}),Number,Math};
vm.runInNewContext(source("function meLayers(){","function meRemoveLayer(){"),layers);
state.maskDirtyFull=false;
layers.meLayers();
assert.equal(calls.length,2);
assert.equal(calls[0][4].r,2000);
layers.meLayers();assert.equal(calls.length,2);
state.maskDirtyRect={l:30,t:40,r:70,b:90};
layers.meLayers();assert.equal(calls.length,4);
assert.equal(calls[2][4].r-calls[2][4].l,40);
assert.equal(calls[3][4].b-calls[3][4].t,50);
state.maskDirtyFull=true;
layers.meLayers();assert.equal(calls.length,6);
assert.equal(calls[5][4].r,2000);
state.mask={width:2000,height:1000};
layers.meLayers();assert.equal(calls.length,8);

// Remove tool has an independent cached red overlay and dirty region.
vm.runInNewContext(source("function meRemoveLayer(){","// Batch pointer, zoom"),layers);
layers.meRemoveLayer();assert.equal(calls.length,9);
layers.meRemoveLayer();assert.equal(calls.length,9);
state.rmDirtyRect={l:12,t:15,r:55,b:70};
layers.meRemoveLayer();assert.equal(calls.length,10);
assert.equal(calls[9][4].l,12);
console.log("Image editor cached overlays, dirty regions and frame scheduling passed.");
