"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const perf=require("../src/fizgig/web/static/caption_image_perf.js");

function fakeMask(w,h){
  const data=new Uint8ClampedArray(w*h*4);
  const context={
    getImageData(x,y,width,height){
      const pixels=new Uint8ClampedArray(width*height*4);
      for(let iy=0;iy<height;iy++)for(let ix=0;ix<width;ix++){
        const start=((y+iy)*w+x+ix)*4,at=(iy*width+ix)*4;
        pixels.set(data.subarray(start,start+4),at);
      }
      return {width,height,data:pixels};
    },
    putImageData(tile,x,y){
      for(let iy=0;iy<tile.height;iy++)for(let ix=0;ix<tile.width;ix++){
        const at=((y+iy)*w+x+ix)*4,start=(iy*tile.width+ix)*4;
        data.set(tile.data.subarray(start,start+4),at);
      }
    }
  };
  return {width:w,height:h,data,getContext:()=>context};
}
const mask=fakeMask(300,270);
const history=perf.tiles(mask);
perf.capture(history,mask,100,100,16);
perf.capture(history,mask,100,100,16);
assert.equal(history.items.size,1,"revisiting one tile must not duplicate history");
perf.capture(history,mask,135,135,15);
assert(history.items.size>1,"crossing tile boundary must record every touched tile");
const changed=[{x:100,y:100},{x:140,y:140}];
for(const {x,y} of changed)mask.data[(y*mask.width+x)*4+3]=220;
const backward=perf.restore(history,mask);
for(const {x,y} of changed)assert.equal(mask.data[(y*mask.width+x)*4+3],0);
assert(backward.rect);
const forward=perf.restore(backward.inverse,mask);
for(const {x,y} of changed)assert.equal(mask.data[(y*mask.width+x)*4+3],220);
assert(forward.inverse.items.size===backward.inverse.items.size);
assert.throws(()=>perf.restore(backward.inverse,fakeMask(20,20)),/dimensions changed/);

const closed=[];
function bitmap(width,height,id){return {width,height,close(){closed.push(id);}};}
const cache=new perf.PreviewCache(130);
cache.put("a",bitmap(4,4,"a"));cache.put("b",bitmap(4,4,"b"));
assert.strictEqual(cache.get("a"),cache.get("a"));
cache.put("c",bitmap(4,4,"c"));
assert(closed.includes("b"),"least-recently-used bitmap should be closed");
cache.clear();
assert.equal(cache.bytes,0);
assert.equal(cache.map.size,0);
assert(closed.includes("a")&&closed.includes("c"));
assert.equal(perf.previewSize(640,640),null);
const size=perf.previewSize(4000,2500);
assert(size.resizeWidth*size.resizeHeight<=1_500_000);
assert(size.resizeWidth>0&&size.resizeHeight>0);

async function main(){
  const old=global.FileReader;
  let blobCalls=0;
  global.FileReader=class {
    readAsDataURL(blob){this.result="data:image/png;base64,"+blob.payload;this.onload();}
  };
  try{
    const result=await perf.pngDataURL({
      toBlob(cb){blobCalls++;cb({payload:"AA=="});},
      toDataURL(){throw Error("Synchronous encoding must not be used");}
    });
    assert.equal(result,"data:image/png;base64,AA==");
    assert.equal(blobCalls,1);
    global.FileReader=undefined;
    assert.equal(await perf.pngDataURL({toDataURL:()=> "fallback"}),"fallback");
  }finally{global.FileReader=old;}
  const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
  assert.match(app,/mePush\(true\); meDab/);
  assert.match(app,/MaskEditorPerf\.capture\(m\.strokeHistory,m\.mask,x,y,r\)/);
  assert.match(app,/MaskEditorPerf\.restore\(top,m\.mask\)/);
  assert.match(app,/MaskEditorPerf\.pngDataURL\(m\.rm\)/);
  assert.match(app,/MaskEditorPerf\.pngDataURL\(out\)/);
  assert.match(app,/MaskEditorPerf\.previewSize\(m\.mask\.width,m\.mask\.height\)/);
  console.log("Tile-region undo, bounded image proxies and asynchronous PNG regression checks passed");
}
main().catch(error=>{console.error(error);process.exitCode=1;});
