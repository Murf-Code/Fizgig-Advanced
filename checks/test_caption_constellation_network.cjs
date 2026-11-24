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
assert.ok(model.chosen.some(e=>e.bridge));
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
}finally{global.document=prevDoc;global.requestAnimationFrame=prevRAF;global.cancelAnimationFrame=prevCancel;}
console.log("Constellation network communities, large-dataset bounds, click selection, navigation and passive hover passed");
