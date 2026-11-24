"use strict";
const assert=require("node:assert/strict");
const Viz=require("../src/fizgig/web/static/caption_studio_viz.js");
const dense=Array.from({length:15722},(_,i)=>({tags:["common","group_"+(i%8),"tag_"+(i%119),"tag_"+((i+7)%119)]}));
const model=Viz.tagbenchModel(dense);
assert.equal(model.nodes.length,120);
assert.ok(model.chosen.length<=4*model.nodes.length);
assert.ok(model.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));
assert.deepEqual(model,Viz.tagbenchModel(dense));
assert.equal(Viz.tagbenchModel([]).nodes.length,0);
assert.equal(model.nodes.find(n=>n.tag==="common").count,15722);
const communities=[
  ...Array.from({length:24},()=>({tags:["lion","shirt","forest"]})),
  ...Array.from({length:24},()=>({tags:["fox","jacket","city"]})),
  {tags:["shirt","jacket"]},
  {tags:["solitary"]}
];
const clustered=Viz.tagbenchModel(communities);
const groups=new Map(clustered.nodes.map(n=>[n.tag,n.group]));
assert.equal(groups.get("lion"),groups.get("shirt"));
assert.equal(groups.get("fox"),groups.get("jacket"));
assert.notEqual(groups.get("lion"),groups.get("fox"));
assert.notEqual(groups.get("solitary"),groups.get("lion"));
assert.ok(clustered.chosen.some(e=>e.a==="jacket"&&e.b==="shirt"&&e.w===1));
assert.ok(clustered.chosen.every(e=>clustered.edges.includes(e)));
console.log("Caption communities, real bridge links, counts and deterministic layout passed");

const atlas=clustered.atlas;
assert.ok(atlas.clusters.length>=3);
assert.ok(atlas.rivers.length<=14);
assert.ok(atlas.clusters.every(c=>c.unmapped||c.contours.length===3));
assert.equal(atlas.clusters.reduce((sum,c)=>sum+c.count,0),clustered.all.reduce((sum,n)=>sum+n.count,0));
for(const [i,c] of atlas.clusters.entries())for(const d of atlas.clusters.slice(i+1))
  assert.ok(Math.hypot(c.px-d.px,c.py-d.py)+1e-7>=c.r+d.r);
assert.equal(model.atlas.unmapped,model.all.length-model.nodes.length);
assert.equal(Viz.tagbenchModel([]).atlas.clusters.length,0);
console.log("Packed community atlas, density contours and bounded rivers passed");

const nested=Viz.tagbenchModel([
  ...Array.from({length:140},(_,i)=>({tags:["lion","mane","fur","lion-detail-"+i]})),
  ...Array.from({length:140},(_,i)=>({tags:["fox","tail","jacket","fox-detail-"+i]})),
  ...Array.from({length:15},(_,i)=>({tags:["unconnected-"+i]}))
]);
const atlasGroups=nested.atlas.clusters;
assert.equal(atlasGroups.reduce((sum,c)=>sum+c.memberCount,0),nested.all.length);
assert.equal(new Set(atlasGroups.flatMap(c=>c.memberTags)).size,nested.all.length);
assert.ok(atlasGroups.some(c=>!c.unmapped&&c.memberTags.some(t=>t.startsWith("lion-detail-"))));
assert.ok(atlasGroups.some(c=>c.unmapped&&c.memberTags.some(t=>t.startsWith("unconnected-"))));
for(const c of atlasGroups){
  assert.ok(c.children.length<=36);
  assert.equal(c.children.reduce((sum,n)=>sum+n.count,0),c.count);
  for(const [i,n] of c.children.entries()){
    assert.ok(Math.hypot(n.x,n.y)+n.r<=c.r+1e-7);
    for(const peer of c.children.slice(i+1))
      assert.ok(Math.hypot(n.x-peer.x,n.y-peer.y)+1e-7>=n.r+peer.r);
  }
}
console.log("Nested circle pack, long-tail assignments and containment passed");
