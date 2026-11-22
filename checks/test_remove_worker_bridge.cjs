"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const bridgeSource=fs.readFileSync("src/fizgig/web/static/caption_remove_bridge.js","utf8");
const workerSource=fs.readFileSync("src/fizgig/web/static/mask_stroke_worker.js","utf8");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const server=fs.readFileSync("src/fizgig/web/server.py","utf8");
const html=fs.readFileSync("src/fizgig/web/static/index.html","utf8");
assert.match(server,/\/mask_stroke_worker\.js/);
assert.match(server,/\/caption_remove_bridge\.js/);
assert(html.indexOf("caption_remove_bridge.js")<html.indexOf('src="/app.js"'));
assert.match(app,/await meDrainRemoveWorker\(\)/);
assert.match(app,/mePrepareRemoveWorker\(\)/);
assert.match(app,/meRawRemoveSegment\(segment\)/);
assert.match(app,/meStopRemoveWorker\(\)/);
assert.match(app,/pointerRect=cv\.getBoundingClientRect\(\)/);

async function main(){
  const posted=[],instances=[],frames=[];
  let bitmapClosed=0;
  class MockWorker{
    constructor(url){assert.equal(url,"/mask_stroke_worker.js");instances.push(this);this.messages=[];}
    postMessage(message){this.messages.push(message);posted.push(message);}
    terminate(){this.terminated=true;}
    reply(data){this.onmessage({data});}
  }
  const sandbox={
    module:{exports:{}},Worker:MockWorker,OffscreenCanvas:class {},
    createImageBitmap:async()=>({close(){bitmapClosed++;}}),
    requestAnimationFrame:fn=>{frames.push(fn);return frames.length;},
    cancelAnimationFrame(){},
    Promise,Array,Math,console,
  };
  vm.runInNewContext(bridgeSource,sandbox);
  const Bridge=sandbox.module.exports;
  let patches=[],fallback=[];
  const original={width:900,height:550};
  const bridge=new Bridge(original,(rect,bitmap)=>patches.push([rect,bitmap]),seg=>fallback.push(seg));
  assert.equal(bridge.active,false);
  await Promise.resolve();await Promise.resolve();
  assert.equal(instances.length,1);
  assert.equal(instances[0].messages[0].type,"init");
  assert.equal(instances[0].messages[0].width,900);
  assert.equal(bridge.enqueue({ax:1}),false,"not-ready bridge must not swallow strokes");
  instances[0].reply({type:"ready",session:1});
  assert(bridge.active);
  const segment={ax:5,ay:5,bx:10,by:10,width:12,erase:false};
  const second={ax:9,ay:9,bx:14,by:14,width:12,erase:true};
  assert.equal(bridge.enqueue(segment),true);
  assert.equal(bridge.enqueue(second),true);
  const drained=bridge.drain();
  assert.equal(instances[0].messages.at(-1).type,"stroke");
  assert.equal(instances[0].messages.at(-1).segments.length,2);
  let closed=false;
  const patch={close(){closed=true;}};
  instances[0].reply({type:"patch",session:1,seq:1,
    rect:{l:0,t:0,r:20,b:20},bitmap:patch});
  await drained;
  assert.equal(patches.length,1);
  assert(closed);
  assert.equal(fallback.length,0);
  // A failed worker must synchronously replay every not-yet-acknowledged
  // segment; saved masks must never lose the end of a brush stroke.
  bridge.enqueue(segment);bridge.pump();
  bridge.enqueue(second);
  instances[0].reply({type:"error",session:1,seq:2,message:"GPU context lost"});
  assert.equal(fallback.length,2);
  assert.equal(fallback[0],segment);
  assert.equal(fallback[1],second);
  assert.equal(bridge.closed,true);
  assert.equal(instances[0].terminated,true);
  assert.equal(bridge.enqueue(segment),false);
  assert(bitmapClosed>=0);

  // Simulate worker stroke application: erase mode and dirty rectangles
  // must be sent back as transferable *small patches*, not entire images.
  let fills=0,strokes=0,transferred=0;
  class FakeCanvas{
    constructor(w,h){this.width=w;this.height=h;}
    getContext(){return {
      set globalCompositeOperation(_) {},
      set strokeStyle(_) {},set fillStyle(_) {},
      set lineCap(_) {},set lineJoin(_) {},set lineWidth(_) {},
      beginPath(){},arc(){},fill(){fills++;},
      moveTo(){},lineTo(){},stroke(){strokes++;},drawImage(){}}
    }
    transferToImageBitmap(){return {close(){}};}
  }
  const workerCtx={
    self:{postMessage(message,transfer){
      if(message.type==="patch"){
        assert.equal(transfer.length,1);transferred++;
        assert(message.rect.r-message.rect.l<=60);
      }
    }},
    OffscreenCanvas:FakeCanvas,Math
  };
  vm.runInNewContext(workerSource,workerCtx);
  workerCtx.self.onmessage({data:{type:"init",session:7,width:1000,height:1000}});
  workerCtx.self.onmessage({data:{type:"stroke",session:7,seq:3,
    segments:[{ax:10,ay:10,bx:10,by:10,width:10,erase:false},
      {ax:10,ay:10,bx:20,by:20,width:10,erase:true}]}});
  assert.equal(fills,1);assert.equal(strokes,1);assert.equal(transferred,1);
  console.log("Worker stroke batching, full-drain, error fallback and patch protocol passed");
}
main().catch(e=>{console.error(e);process.exitCode=1;});
