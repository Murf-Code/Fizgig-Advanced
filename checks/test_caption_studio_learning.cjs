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
// Only a live, tracked run enables learning colors automatically.
// A manual opt-out for that exact run must be honored across refreshes.
const live={auto_enable:true,run:{id:"current"}};
assert.equal(L.shouldAutoEnable(live,""),true);
assert.equal(L.shouldAutoEnable(live,"current"),false);
assert.equal(L.shouldAutoEnable({auto_enable:false,run:{id:"old"}},""),false);
assert.equal(L.shouldAutoEnable({auto_enable:true,run:{id:"new"}},"current"),true);
assert.equal(L.shouldAutoEnable({auto_enable:true,run:null},""),false);
console.log("caption learning helpers OK");

// Explicit per-image verdicts, including absent/not-yet-evaluated images.
const sample=[
  {path:"h:/dataset/a.jpg"},{path:"h:/dataset/b.jpg"},
  {path:"h:/dataset/missing.jpg"}
];
assert.deepEqual(L.filterByVerdict(sample,records,"all"),sample);
assert.deepEqual(L.filterByVerdict(sample,records,"stuck"),[sample[0]]);
assert.deepEqual(L.filterByVerdict(sample,records,"easy"),[sample[1]]);
assert.deepEqual(L.filterByVerdict(sample,records,"none"),[sample[2]]);
assert.deepEqual(L.filterByVerdict(sample,records,"watch"),[]);
assert.ok(L.statusOptions.some(([value])=>value==="exhausted"));

let callback,observed=new Set();
class Observer{
  constructor(fn){callback=fn;}
  observe(t){observed.add(t);}
  unobserve(t){observed.delete(t);}
  disconnect(){observed.clear();}
}
const visible=L.visibility(Observer);
const stuck={dataset:{learningStatus:"stuck"}},easy={dataset:{learningStatus:"easy"}};
visible.update(stuck);visible.update(easy);
assert.equal(observed.size,1);
callback([{target:stuck,isIntersecting:true}]);
assert.equal(stuck.dataset.learningVisible,"true");
callback([{target:stuck,isIntersecting:false}]);
assert.equal(stuck.dataset.learningVisible,undefined);
stuck.dataset.learningStatus="easy";visible.update(stuck);
assert.equal(observed.size,0);
visible.dispose();
