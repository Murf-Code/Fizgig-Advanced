"use strict";
// No Qt dependency: exercise the exact desktop-only JS source contract.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const code = fs.readFileSync("src/fizgig/web/static/qt_native_image_bridge.js", "utf8");
let current = "captions",renders=0,removals=0,invalid=0;
const items = [
  {path:"C:\\train\\a.png",name:"a.png",time:1},
  {path:"C:\\train\\b.png",name:"b.png",time:2},
  {path:"C:\\train\\c.png",name:"c.png",time:3}
];
const S = {
  imageFolder:"C:\\train",cap:{
    items, nativeVisibleItems:items.slice(1),
    studioSelection:new Set([items[1].path]),
    studioPath:items[1].path,
    nativeVisibleKey:"query:b",
    studioRevision:0,
    nativeImageActions:{
      remove:(path)=>{assert.equal(path,items[1].path);removals++;},
      review:()=>{invalid++;}
    }
  },
  gal:{items:[{path:"C:\\outputs\\s1.png",name:"s1.png",time:5},
              {path:"C:\\outputs\\v1.mp4",name:"v1.mp4",time:5}],run:"myRun"},
  prep:{look:{state:"done",folder:"C:\\train",rows:[
    {path:"C:\\train\\a.png",name:"a.png"},
    {path:"C:\\train\\c.png",name:"c.png"}]},marked:new Set()}
};
const context={window:{},S,document:{querySelector:()=>({id:"tab-"+current})},
  renderCaptions:()=>{renders++;},console};
vm.runInNewContext(code,context);
const api=context.window.FizgigNativeImages;
assert(api&&context.window.__FizgigQtNativeRail);
assert.equal(api.snapshot().items.length,2,"Qt sees the filtered image source");
assert.equal(api.snapshot().items[0].path,items[1].path);
assert.equal(api.snapshot().items[0].version,"2");
assert.equal(api.snapshot().tab,"captions");
const before=api.status().selection;
assert(api.select([items[1].path,items[2].path],items[2].path));
assert.equal(S.cap.studioSelection.size,2);
assert.equal(S.cap.studioPath,items[2].path);
assert.equal(renders,1);
assert.notEqual(api.status().selection,before,"multi-selection invalidates fingerprint");
assert(api.action("remove",items[1].path));
assert.equal(removals,1,"native removes delegate to existing workbench handler");
assert.equal(invalid,0);
assert.equal(api.action("unrecognized",items[1].path),false);
assert(api.select([items[0].path,items[1].path],items[0].path),
  "hidden paths do not get selected");
assert.equal(S.cap.studioSelection.size,1);
assert(S.cap.studioSelection.has(items[1].path));
current="samples";
assert.equal(api.snapshot().items.length,1,"non-image outputs are not decoded by Qt");
assert.equal(api.snapshot().items[0].name,"s1.png");
assert.equal(api.action("remove",items[1].path),false,
  "only Captions may call destructive image actions");
assert(api.select(["C:\\outputs\\s1.png"],"C:\\outputs\\s1.png"));
assert.equal(api.selection().selected.length,1);
current="prep";
assert.equal(api.snapshot().items.length,2);
assert(api.select(["C:\\train\\a.png"],"C:\\train\\a.png"));
assert.equal(api.selection().selected.length,1);
assert.equal(S.prep.marked.size,0,"preview selection must not mark dataset images for deletion");
current="training";
assert.equal(api.snapshot().items.length,0);
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
assert.match(app,/if\(!window\.__FizgigQtNativeRail\)\{/);
assert.match(app,/let prefetchTimer=null;\s*if\(!window\.__FizgigQtNativeRail\)/);
assert.match(app,/S\.cap\.nativeImageActions=\{remove:removeSelectedImages/);
assert.match(app,/S\.cap\.nativeDatasetGeneration=\(S\.cap\.nativeDatasetGeneration\|\|0\)\+1/);
assert.match(app,/S\.cap\.nativeVisibleItems=items/);
console.log("Native Qt image bridge: filtered Captions, multi-select, actions, Samples and Prep OK");
