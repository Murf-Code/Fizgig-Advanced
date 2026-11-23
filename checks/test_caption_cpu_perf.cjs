"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const rail=require("../src/fizgig/web/static/caption_thumb_rail.js");
const viz=require("../src/fizgig/web/static/caption_studio_viz.js");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const server=fs.readFileSync("src/fizgig/web/server.py","utf8");
const worker=fs.readFileSync("src/fizgig/web/static/caption_studio_graph_worker.js","utf8");

// The saved preview-height preference determines thumbnail resolution,
// including a portrait's tall display, while landscape images stay small.
const portrait=[400,1400],landscape=[1800,750];
assert.equal(rail.imageSizeForRow(160,landscape,1,360),160);
assert(rail.imageSizeForRow(160,portrait,1,360)>rail.imageSizeForRow(160,landscape,1,360));
assert(rail.imageSizeForRow(180,portrait,1,720)>=rail.imageSizeForRow(180,portrait,1,160));
assert([160,224,320,384].includes(rail.imageSizeForRow(600,portrait,2,720)));
for(const w of [60,140,300,700])for(const dpr of [1,1.5,2])
  assert([160,224,320,384].includes(rail.imageSizeForRow(w,[700,700],dpr,360)));
assert.match(app,/function studioThumbGeometry\(items,railWidth,gap=14,previewHeight=360\)/);
assert.match(app,/geometry=studioThumbGeometry\(items,width,14,previewHeight\)/);
assert.match(app,/imageSizeForRow\(railWidthForThumb,it\.size/);
assert.match(app,/row\("Image preview max height \(px\)",preview,/);
assert.doesNotMatch(app,/Math\.min\(306,width\*ratio\)/);
assert.match(app,/function captionStageMediaFor\(it\)/);
assert.match(app,/\/api\/preview\?path=/);
assert.match(app,/class:"studio-preview"\},captionStageMediaFor\(current\)/);
assert.match(server,/if path == "\/api\/preview":/);
assert.match(server,/size=1600/);
assert.match(server,/if path == "\/api\/thumb":/);
assert.match(server,/min\(384, requested\)/);
assert.match(server,/def _thumb_disk_path\(key\):/);
assert.match(server,/st\.st_mtime_ns/);
assert.match(server,/threading\.BoundedSemaphore\(3\)/);
assert.match(app,/const redraw=\(\)=>\{/);
assert.match(app,/if\(previousTagKeys&&keys\.length===previousTagKeys\.length/);
assert.match(app,/tags\.scrollTop=previousScroll;/);
assert.match(app,/S\.cap\.studioStageDispose\?\.\(\);/);
assert.match(app,/clearExploreStage\(\);/);
assert.match(app,/S\.cap\.studioGraphPending\?\.cancel\?\.\(\);/);
assert.match(app,/const generation=exploreGeneration;/);
assert.match(app,/if\(mountedExploreView===view\)return;/);
assert.match(app,/new Worker\("\/caption_studio_graph_worker\.js"\)/);
assert.match(app,/cancel:\(\)=>work\.cancel\?\.\(\)/);
assert.match(worker,/importScripts\("\/caption_studio_viz\.js"\)/);
assert.match(worker,/self\.onmessage=event=>/);
assert(viz.tagbenchModel([{tags:["lion","blue"]}]).nodes.length>0);

// Verify worker produces the *same* model as the existing TagBench algorithm
// when invoked with multiple tags and category-color mappings.
const items=[{tags:["lion","blue","smiling"]},{tags:["lion","red"]},
             {tags:["blue","smiling","lion"]},{tags:["red","tree"]}];
let last;
const sandbox={self:{postMessage(msg){last=msg;}},importScripts(path){
  assert.equal(path,"/caption_studio_viz.js");
  sandbox.self.CaptionStudioViz=viz;
}};
vm.runInNewContext(worker,sandbox);
sandbox.self.onmessage({data:{items,categories:{tags:{}}}});
assert(!last.error, String(last.error));
const expected=viz.tagbenchModel(items,{tags:{}});
assert.equal(last.model.nodes.length,expected.nodes.length);
assert.deepEqual([...last.model.colors], [...expected.colors]);
assert.deepEqual(last.model.edges,expected.edges);
assert.deepEqual(last.model.nodes.map(n=>[n.tag,n.x,n.y]),
  expected.nodes.map(n=>[n.tag,n.x,n.y]));
console.log("Caption CPU perf: adaptive thumbnails, preview cache, tag-button reuse and view cancellation OK");
