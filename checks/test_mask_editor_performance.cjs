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
// A soft brush reuses one gradient raster instead of recomputing it for
// every interpolated dab; editing brush settings invalidates the stamp.
const brushCalls={gradients:0,offscreen:0,maskContexts:0,blits:[],dirty:0};
const brushControls={meSize:{value:"48"},meSoft:{value:"60"},meStrength:{value:"80"}};
const makeGradient=()=>({addColorStop(){}});
const maskContext={
  globalCompositeOperation:"source-over",
  createRadialGradient(){brushCalls.gradients++;return makeGradient();},
  drawImage(){brushCalls.blits.push(this.globalCompositeOperation);},
  beginPath(){},arc(){},fill(){}
};
const brushState={mask:{getContext(){brushCalls.maskContexts++;return maskContext;}},
  zoom:1,tool:"brush",flip:false};
const brushVm={
  meState:()=>brushState,
  meMarkMaskDirty:()=>{brushCalls.dirty++;},
  $:name=>brushControls[name],Number,Math,
  el:(tag,opts)=>{
    assert.equal(tag,"canvas");brushCalls.offscreen++;
    return {...opts,getContext:()=>({
      createRadialGradient(){brushCalls.gradients++;return makeGradient();},
      beginPath(){},arc(){},fill(){}
    })};
  }
};
vm.runInNewContext(source("function meBrushStamp(","function meStateLine()"),brushVm);
for(let i=0;i<30;i++)brushVm.meDab(i,50);
assert.equal(brushCalls.offscreen,1);
assert.equal(brushCalls.gradients,1);
assert.equal(brushCalls.blits.length,30);
assert.equal(brushCalls.dirty,30);
brushVm.meStroke({x:0,y:10},{x:100,y:10});
assert.equal(brushCalls.gradients,1);
assert(brushCalls.blits.length>40);
brushControls.meSoft.value="75";
brushVm.meDab(20,20);
assert.equal(brushCalls.gradients,2);
brushState.tool="erase";
brushVm.meDab(22,20);
assert.equal(brushCalls.blits.at(-1),"destination-out");
// Very large stamps should not allocate huge temporary canvases.
const offscreenBefore=brushCalls.offscreen;
brushControls.meSize.value="400";brushState.zoom=.1;
brushVm.meDab(20,20);
assert.equal(brushCalls.offscreen,offscreenBefore);
assert.equal(brushCalls.gradients,3);

// Repeated resize/fit notifications must not zero and reallocate an
// unchanged canvas backing store (especially with inline Tags mounted).
let reallocations=0,fitDraws=0;
const stage={clientWidth:800,clientHeight:500};
const canvas={_width:800,_height:500,
  get width(){return this._width;},set width(n){reallocations++;this._width=n;},
  get height(){return this._height;},set height(n){reallocations++;this._height=n;}
};
const fitState={mask:{width:400,height:200},zoom:1,ox:0,oy:0};
const fitVm={
  meState:()=>fitState,$:name=>name==="meCanvas"?canvas:stage,
  S:{cap:{imageEditing:{inline:true}}},window:{innerHeight:900},
  Math,meDraw:()=>{fitDraws++;}
};
vm.runInNewContext(source("function meFit() {","// Recompose only"),fitVm);
fitVm.meFit();fitVm.meFit();
assert.equal(reallocations,0);
assert.equal(fitDraws,2);
stage.clientWidth=950;
fitVm.meFit();
assert.equal(reallocations,1);
assert.equal(canvas.width,950);

console.log("Image editor cached overlays, brush stamps, resize safety and frame scheduling passed.");
