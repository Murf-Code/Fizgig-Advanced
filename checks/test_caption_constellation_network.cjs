"use strict";
const assert=require("node:assert/strict");
const Viz=require("../src/fizgig/web/static/caption_studio_viz.js");
const items=[
  ...Array.from({length:60},()=>({tags:["lion","mane","feline","solo"]})),
  ...Array.from({length:60},()=>({tags:["jacket","buttons","fabric","solo"]})),
  ...Array.from({length:8},()=>({tags:["lion","jacket","solo"]}))
];
const model=Viz.tagbenchModel(items);
const byTag=new Map(model.nodes.map(n=>[n.tag,n]));
assert.equal(byTag.get("lion").group,byTag.get("mane").group);
assert.equal(byTag.get("jacket").group,byTag.get("fabric").group);
assert.notEqual(byTag.get("lion").group,byTag.get("jacket").group);
// Eight lion/jacket captions are fewer than expected under independence.
assert.ok(!model.chosen.some(e=>e.a==="jacket"&&e.b==="lion"));
assert.ok(model.edges.some(e=>e.a==="jacket"&&e.b==="lion"&&e.w===8));
assert.ok(!model.chosen.some(e=>e.a==="solo"||e.b==="solo"));
assert.ok(model.anchorEdges.length>0&&model.anchorEdges.length<=6);
assert.ok(model.anchorEdges.every(e=>(e.a==="solo"||e.b==="solo")&&model.edges.some(r=>r.a===e.a&&r.b===e.b&&r.w===e.w)));
assert.ok(model.anchorEdges.every(e=>!model.chosen.some(r=>r.a===e.a&&r.b===e.b)));
const withTrigger=Viz.tagbenchModel(items.map(it=>({tags:[...it.tags,"second universal"]})),{},"SOLO");
assert.equal(withTrigger.nodes.find(n=>n.tag==="solo").anchor,1);
assert.ok(withTrigger.nodes.find(n=>n.tag==="second universal").anchor<1);
assert.ok(!withTrigger.chosen.some(e=>e.a==="second universal"||e.b==="second universal"));
assert.deepEqual(withTrigger.chosen,model.chosen);
const many=Array.from({length:125},(_,i)=>({tags:["common","label "+i]}));
const rare=Viz.tagbenchModel(many,{},"label 99");
assert.equal(rare.nodes.length,120);
assert.equal(rare.nodes.find(n=>n.tag==="label 99").anchor,1);
assert.ok(rare.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));
assert.ok(!Viz.tagbenchModel(items,{},"missing").nodes.some(n=>n.tag==="missing"));
assert.ok(model.chosen.every(e=>model.edges.some(x=>x.a===e.a&&x.b===e.b&&x.w===e.w)));
assert.equal(model.alike.length,0);
assert.deepEqual(model,Viz.tagbenchModel(items));
assert.equal(Viz.tagbenchModel([]).nodes.length,0);
const dense=Array.from({length:15722},(_,i)=>({tags:["common","group_"+(i%8),"tag_"+(i%119),"tag_"+((i+7)%119)]}));
const large=Viz.tagbenchModel(dense);
assert.equal(large.nodes.length,120);
assert.ok(large.chosen.length<=840);
assert.ok(large.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));
assert.ok(large.edges.length<=7140);

function fake(tag){
  const attrs={},children=[],handlers={},classes=new Set();
  const n={tag,attrs,children,handlers,style:{},hidden:false,textContent:"",
    classList:{toggle(k,v){if(v)classes.add(k);else classes.delete(k);},contains(k){return classes.has(k);}},
    setAttribute(k,v){attrs[k]=String(v);},append(...c){children.push(...c);},
    replaceChildren(...c){children.splice(0,children.length,...c);},
    addEventListener(k,fn){handlers[k]=fn;},getScreenCTM(){return {a:1};},
    querySelector(){return null;},focus(){},remove(){},contains(){return false;}
  };
  return n;
}
const prevDoc=global.document,prevRAF=global.requestAnimationFrame,prevCancel=global.cancelAnimationFrame;
try{
  global.document={createElement:fake,createElementNS:(_,tag)=>fake(tag)};
  global.requestAnimationFrame=()=>1;global.cancelAnimationFrame=()=>{};
  const picked=[],navigated=[];
  const stage=Viz.tagbenchConstellation(model,t=>picked.push(t),null,{
    preferences:{elasticity:0},onNavigate:(mode,tag)=>navigated.push([mode,tag])});
  const walk=n=>[n,...n.children.flatMap(walk)];
  let all=walk(stage),lion=all.find(n=>n.attrs["data-tag"]==="lion");
  assert.ok(lion);
  assert.equal(lion.handlers.pointerenter,undefined);
  assert.equal(lion.handlers.pointerleave,undefined);
  assert.ok(all.some(n=>n.tag==="path"&&/ Q /.test(n.attrs.d||"")));
  lion.handlers.keydown({key:"Enter",preventDefault(){},stopPropagation(){}});
  assert.deepEqual(picked,["lion"]);
  assert.equal(lion.attrs["aria-pressed"],"true");
  all=walk(stage);
  all.find(n=>n.textContent==="View in Tags"&&n.tag==="button").handlers.click();
  assert.deepEqual(navigated,[["list","lion"]]);
  all.find(n=>n.textContent==="Waterfall"&&n.tag==="button").handlers.click();
  assert.deepEqual(navigated,[["list","lion"],["waterfall","lion"]]);
  lion.handlers.keydown({key:"Enter",preventDefault(){},stopPropagation(){}});
  assert.deepEqual(picked,["lion",null]);
  stage.dispose();
  const marked=Viz.tagbenchModel(items,{},"solo");
  let saved;
  const viewport={zoom:2,pan:{x:43,y:-27}};
  const inspect=Viz.tagbenchConstellation(marked,()=>{}, "lion",{
    savedView:viewport,saveView:state=>{saved=state;},preferences:{elasticity:0}});
  const elements=walk(inspect),orb=tag=>elements.find(n=>n.attrs["data-tag"]===tag);
  assert.equal(elements.find(n=>n.tag==="svg").attrs.viewBox,"293 148 500 350");
  assert.ok(orb("solo").children.some(n=>n.tag==="path"));
  assert.ok(elements.some(n=>n.tag==="path"&&n.attrs.class?.includes("is-anchor")));
  assert.ok(!orb("lion").children.some(n=>n.tag==="path"));
  assert.equal(orb("lion").attrs["aria-pressed"],"true");
  const label=elements.find(n=>n.tag==="text"&&n.textContent==="lion");
  assert.equal(label.attrs.fill,"var(--text-primary)");
  assert.equal(label.attrs.filter,"url(#tb-label-backing)");
  const related=elements.filter(n=>n.tag==="li").map(n=>walk(n).find(c=>c.tag==="button").textContent);
  assert.ok(related.indexOf("mane")>=0&&related.indexOf("mane")<related.indexOf("solo"));
  assert.ok(elements.some(n=>n.textContent==="60 of 68 images"));
  inspect.dispose();
  assert.deepEqual(saved,{...viewport,picked:"lion"});
  const returned=Viz.tagbenchConstellation(marked,()=>{},saved.picked,{savedView:saved});
  assert.equal(walk(returned).find(n=>n.tag==="svg").attrs.viewBox,"293 148 500 350");
  assert.equal(walk(returned).find(n=>n.attrs["data-tag"]==="lion").attrs["aria-pressed"],"true");
  returned.dispose();
  assert.ok(!model.nodes.some(n=>n.isTrigger));
}finally{global.document=prevDoc;global.requestAnimationFrame=prevRAF;global.cancelAnimationFrame=prevCancel;}
console.log("Constellation network communities, large-dataset bounds, click selection, navigation and passive hover passed");
