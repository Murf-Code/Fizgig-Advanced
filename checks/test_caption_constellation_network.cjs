"use strict";
const assert=require("node:assert/strict");
const Viz=require("../src/fizgig/web/static/caption_studio_viz.js");
const dense=Array.from({length:15722},(_,i)=>({tags:["common","group_"+(i%8),"tag_"+(i%119),"tag_"+((i+7)%119)]}));
const model=Viz.tagbenchModel(dense);
assert.equal(model.nodes.length,128);
assert.ok(model.chosen.length<=6*model.nodes.length);
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
