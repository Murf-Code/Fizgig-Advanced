"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const rail=require("../src/fizgig/web/static/caption_thumb_rail.js");

function fakeHost(){
  return {
    children:[],
    insertBefore(node,before){
      const index=before?this.children.indexOf(before):this.children.length;
      assert(index>=0,"reference node must already be mounted");
      assert(!this.children.includes(node),"existing thumbnail must not be inserted again");
      this.children.splice(index,0,node);
    }
  };
}
const host=fakeHost(),mounted=new Map(),created=new Map();
let allocations=0,removals=0;
const create=index=>{
  allocations++;
  const row={
    index,
    remove(){
      const at=host.children.indexOf(row);
      assert(at>=0,"removed row must belong to the rail");
      host.children.splice(at,1);
      removals++;
    }
  };
  created.set(index,row);
  return row;
};
const paint=(start,end)=>{
  rail.reconcile(host,mounted,start,end,create);
  assert.deepEqual(host.children.map(node=>node.index),
    Array.from({length:end-start},(_,i)=>start+i),
    "rail must retain the precise ordered visible range");
  assert.equal(mounted.size,end-start);
};
paint(0,12);
assert.equal(allocations,12);
const stable=mounted.get(7);
paint(2,14);
assert.strictEqual(mounted.get(7),stable,
  "overlapping rows must keep their original node and decoded image");
assert.equal(allocations,14,"small scroll creates only newly exposed rows");
assert.equal(removals,2);
paint(1,13);
assert.strictEqual(mounted.get(7),stable);
assert.equal(allocations,15,"reverse scroll only adds one new row");
paint(1200,1211);
assert.equal(mounted.size,11,"large scrollbar jumps must not mount intervening rows");
assert.equal(removals,15,"large jumps release old DOM rows");
paint(1197,1208);
paint(1197,1208);
assert.equal(mounted.size,11,"same-window repaints are idempotent");
assert.deepEqual(rail.prefetchRange(20,31,100,1),[31,39]);
assert.deepEqual(rail.prefetchRange(20,31,100,-1),[12,20]);
assert.deepEqual(rail.prefetchRange(0,5,100,-1),[0,0]);
assert.deepEqual(rail.prefetchRange(95,100,100,1),[100,100]);

const oldImage=global.Image;
const loaded=[];
global.Image=class{
  constructor(){this.decoding="";this.src="";}
  decode(){loaded.push(this.src);return Promise.resolve();}
};
async function main(){
  try{
    const cache=new rail.ThumbPreloader(3);
    cache.reset("dataset-A");
    cache.prefetch("a","/a");
    cache.prefetch("b","/b");
    cache.prefetch("c","/c");
    cache.prefetch("a","/a"); // reusing must not decode again
    assert.equal(loaded.length,3);
    cache.prefetch("d","/d");
    assert.deepEqual([...cache.images.keys()],["c","a","d"],
      "least-recently-used prefetched images should be released");
    cache.forget("a");assert(!cache.images.has("a"));
    cache.reset("dataset-B");assert.equal(cache.images.size,0);
    cache.prefetch("next","/next");
    assert.equal(cache.images.size,1);
    cache.clear();assert.equal(cache.images.size,0);
  }finally{global.Image=oldImage;}
  const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
  const html=fs.readFileSync("src/fizgig/web/static/index.html","utf8");
  const server=fs.readFileSync("src/fizgig/web/server.py","utf8");
  assert.match(app,/CaptionThumbRail\.reconcile\(windowRows,mountedThumbs/);
  assert.doesNotMatch(app,/windowRows\.replaceChildren\(fragment\)/);
  assert.doesNotMatch(app,/thumbs\.addEventListener\("wheel"/);
  assert.match(app,/thumbs\.addEventListener\("scroll",queueWindow,\{passive:true\}\)/);
  assert.match(app,/S\.cap\.studioThumbPreloader\?\.forget\(path\)/);
  assert.match(app,/\&v="\+\s*\(S\.cap\.studioThumbRevisions\?\.get\(path\)\|\|0\)/);
  assert.match(html,/src="\/caption_thumb_rail\.js"/);
  assert.match(server,/"\/caption_thumb_rail\.js"/);

assert.match(app,/imageSizeForRow\(railWidthForThumb,it\.size/);
assert.match(app,/geometry=studioThumbGeometry\(items,width,14,previewHeight\)/);
  console.log("Native thumbnail scrolling, retained DOM rows, lookahead cache and image invalidation checks passed");
}
main().catch(error=>{console.error(error);process.exitCode=1;});
