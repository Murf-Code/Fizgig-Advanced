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

const auraNodes=[{tag:"text",count:4,x:100,y:100},{tag:"watermark",count:4,x:160,y:100}];
const baseAura=Viz.auraDensity(auraNodes,4);
assert.equal(baseAura.x,130);
assert.ok(baseAura.contours.every(level=>level.length>0));
const cached=Viz.auraCache(baseAura,auraNodes,4,500);
assert.equal(cached(auraNodes,0,false),baseAura);
const moving=auraNodes.map(n=>({...n,x:n.x+60}));
const firstAura=cached(moving,0,true);
assert.equal(firstAura.x,190);
assert.ok(firstAura!==baseAura);
const movedAgain=moving.map(n=>({...n,y:n.y+40}));
assert.equal(cached(movedAgain,499,true),firstAura);
const secondAura=cached(movedAgain,500,true);
assert.equal(secondAura.y,140);
const finalNodes=movedAgain.map((n,i)=>({...n,x:n.x+i*80}));
assert.equal(cached(finalNodes,501,true),secondAura);
const finalAura=cached(finalNodes,502,false);
assert.equal(finalAura.x,230);
assert.ok(finalAura.spread>secondAura.spread);
assert.ok(JSON.stringify(finalAura.contours)!==JSON.stringify(secondAura.contours));
assert.equal(cached(finalNodes,9000,false),finalAura);
console.log("Dynamic aura geometry, 500ms throttle, final refresh and idle cache passed");

const paintField=Viz.auraDensity([{tag:"a",count:4,x:0,y:0},{tag:"b",count:4,x:30,y:0}],4);
assert.ok(paintField.density.grid instanceof Float32Array);
assert.ok(paintField.density.nx<=100&&paintField.density.ny<=100);
assert.equal(Viz.auraPaintAlpha(0,1),0);
assert.equal(Viz.auraPaintAlpha(1,1),255);
assert.equal(Viz.auraPaintAlpha(1,0),0);
assert.ok(Viz.auraPaintAlpha(.1,1)>0);
assert.ok(Viz.auraPaintAlpha(.1,1)<Viz.auraPaintAlpha(.5,1));
assert.ok(Viz.auraPaintAlpha(.5,1)<255);
assert.ok(clustered.atlas.clusters.every(c=>c.unmapped||c.density.grid.length===c.density.nx*c.density.ny));
console.log("Bounded paint textures and smooth density alpha passed");

const red=Viz.auraDensity([{tag:"red",count:4,x:0,y:0}],4).density;
const blue=Viz.auraDensity([{tag:"blue",count:4,x:45,y:0}],4).density;
const mix=Viz.mergeAuraFields([{density:red,rgb:[255,0,0]},{density:blue,rgb:[0,0,255]}]);
assert.equal(mix.length,1);
const surface=mix[0];
assert.ok(surface.nx<=160&&surface.ny<=160);
const at=(x,y)=>4*(Math.round((y-surface.top)/surface.step)*surface.nx+Math.round((x-surface.left)/surface.step));
const mid=at(22.5,0),left=at(0,0),right=at(45,0);
assert.ok(surface.rgba[mid]>70&&surface.rgba[mid+2]>70);
assert.ok(surface.rgba[left]>surface.rgba[left+2]);
assert.ok(surface.rgba[right+2]>surface.rgba[right]);
assert.equal(surface.rgba[mid+3],255);
assert.deepEqual(mix,Viz.mergeAuraFields([{density:blue,rgb:[0,0,255]},{density:red,rgb:[255,0,0]}]));
const far=Viz.auraDensity([{tag:"far",count:4,x:1000,y:0}],4).density;
assert.equal(Viz.mergeAuraFields([{density:red,rgb:[255,0,0]},{density:far,rgb:[0,0,255]}]).length,2);
const centerX=22.5;
assert.ok(surface.contours[0].every(([x1,y1,x2,y2])=>!(Math.min(x1,x2)<=centerX&&Math.max(x1,x2)>=centerX&&Math.abs((y1+y2)/2)<10)));
console.log("Shared liquid contours, reciprocal pigment mixing and separated fields passed");
