"use strict";
// Regression tests for the native cursor and independently painted Remove tint.
// Run: node checks/test_remove_preview_latency.cjs
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const source=(begin,end)=>{
  const a=app.indexOf(begin),b=app.indexOf(end,a+begin.length);
  assert(a>=0&&b>a,"Missing source "+begin);
  return app.slice(a,b);
};
// A Remove stroke changes only the red overlay; it must never schedule
// another multi-megapixel photograph repaint on every pointer movement.
assert.match(app,/if\(last&&m\.tool==="remove"\)meDrawRemove\(\)/);
assert.match(app,/meRmLine\(p, p\); last = p; m\.rmHandEdited = true; meImgBar\(\); meDrawRemove\(\)/);
assert.match(app,/if\(overlay\.parentNode!==stage\)stage\.append\(overlay\)/);
assert.match(css,/\.me-stage canvas\.me-remove-overlay\{/);
assert.match(css,/\.me-stage canvas\.me-remove-overlay\[hidden\]/);

const tint={parentNode:null,style:{transform:""},classList:{add(name){this.latest=name;}},hidden:true};
const stage={append(child){child.parentNode=this;}};
const mainCanvas={clientWidth:800,clientHeight:500,width:800,height:500,
  offsetLeft:5,offsetTop:7,isConnected:true};
const state={rm:{},rmDirty:true,showBefore:false,tool:"remove",mask:{},
  zoom:2,ox:10,oy:15,rmTint:tint};
const frames=[];
let paints=0;
const overlayVM={
  meState:()=>state,$:id=>id==="meCanvas"?mainCanvas:stage,
  meRemoveLayer:()=>{paints++;return tint;},
  requestAnimationFrame:fn=>{frames.push(fn);return frames.length;}
};
vm.runInNewContext(source("let meRemoveFrame=0;","// Batch pointer, zoom"),overlayVM);
overlayVM.meDrawRemove();overlayVM.meDrawRemove();overlayVM.meDrawRemove();
assert.equal(frames.length,1,"pointer events should coalesce into one preview frame");
frames.shift()();
assert.equal(paints,1);
assert.equal(tint.hidden,false);
assert.equal(tint.classList.latest,"me-remove-overlay");
assert.equal(tint.parentNode,stage);
assert.equal(tint.style.transform,"matrix(2,0,0,2,15,22)");
state.zoom=1.5;state.ox=0;state.oy=0;
overlayVM.meSyncRemoveOverlay();
assert.equal(tint.style.transform,"matrix(1.5,0,0,1.5,5,7)");
state.showBefore=true;
assert.equal(overlayVM.meSyncRemoveOverlay(),false);
assert.equal(tint.hidden,true);
state.showBefore=false;state.tool="crop";
assert.equal(overlayVM.meSyncRemoveOverlay(),false);
assert.equal(tint.hidden,true);
state.tool="remove";state.rmDirty=false;
assert.equal(overlayVM.meSyncRemoveOverlay(),false);
assert.equal(tint.hidden,true);

// The native cursor bitmap is generated once per displayed size and then
// handed to the OS. Large brushes fall back to the CSS ring.
let cursorCanvases=0;
const cursorState={};
const nativeVM={meState:()=>cursorState,Math,Map,
  el:(name,options)=>{
    assert.equal(name,"canvas");cursorCanvases++;
    return {...options,
      getContext:()=>({beginPath(){},arc(){},stroke(){}}),
      toDataURL:()=>"data:image/png;base64,test"};
  }};
vm.runInNewContext(source("function meNativeBrushCursor(","function meWire()"),nativeVM);
const small=nativeVM.meNativeBrushCursor(24);
assert(small.includes('url("data:image/png'));
assert.equal(cursorCanvases,1);
assert.equal(nativeVM.meNativeBrushCursor(24),small);
assert.equal(cursorCanvases,1);
assert.equal(nativeVM.meNativeBrushCursor(180),null);
assert.equal(cursorCanvases,1);
assert.notEqual(nativeVM.meNativeBrushCursor(32),null);
assert.equal(cursorCanvases,2);
console.log("Native brush cursor and independent red preview regressions passed.");
