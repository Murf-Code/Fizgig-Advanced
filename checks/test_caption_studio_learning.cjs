const assert=require("node:assert/strict");
const L=require("../src/fizgig/web/static/caption_studio_learning.js");
const records=L.index([
  {path:"H:\\Dataset\\a.jpg",recent:.7,verdict:"stuck"},
  {path:"H:/Dataset/b.jpg",recent:-.3,verdict:"easy"},
]);
assert.equal(L.forPath(records,"h:/dataset/a.jpg").verdict,"stuck");
assert.equal(L.forPath(records,"h:\\dataset\\b.jpg").verdict,"easy");
assert.deepEqual(L.sort([{path:"h:/dataset/b.jpg",name:"b.jpg"},
  {path:"h:/dataset/missing.jpg",name:"missing.jpg"},
  {path:"h:/dataset/a.jpg",name:"a.jpg"}],records,"loss-high").map(x=>x.name),
  ["a.jpg","b.jpg","missing.jpg"]);
assert.deepEqual(L.sort([{path:"h:/dataset/b.jpg",name:"b.jpg"},
  {path:"h:/dataset/missing.jpg",name:"missing.jpg"},
  {path:"h:/dataset/a.jpg",name:"a.jpg"}],records,"loss-low").map(x=>x.name),
  ["b.jpg","a.jpg","missing.jpg"]);
const fake=()=>{
  const attrs=new Map();
  const tile={dataset:{},title:"test.jpg",
    getAttribute:n=>attrs.get(n)||null,
    setAttribute:(n,v)=>attrs.set(n,v),
    removeAttribute:n=>{ attrs.delete(n); if(n==="data-learning-status")delete tile.dataset.learningStatus; }}; return tile;
};
const tile=fake();tile.setAttribute("aria-label","test.jpg");
L.decorate(tile,{verdict:"stuck",recent:.7},true);
assert.equal(tile.dataset.learningStatus,"stuck");
L.decorate(tile,{verdict:"stuck",recent:.7},false);
assert.equal(tile.dataset.learningStatus,undefined); // verified below through attribute API in browser
console.log("caption learning helpers OK");
