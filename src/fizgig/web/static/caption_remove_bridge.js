"use strict";
// Optional asynchronous Remove painter. Failure falls back to the unchanged
// synchronous canvas path, replaying every unacknowledged segment in order.
class RemoveStrokeBridge{
  constructor(mask,onPatch,onFallback){
    this.mask=mask;this.onPatch=onPatch;this.onFallback=onFallback;
    this.queue=[];this.inFlight=null;this.waiters=[];
    this.ready=false;this.failed=false;this.closed=false;
    this.frame=0;this.seq=0;this.session=1;
    try{
      if(typeof Worker!=="function"||typeof OffscreenCanvas==="undefined"||
          typeof createImageBitmap!=="function")throw new Error("No OffscreenCanvas worker");
      this.worker=new Worker("/mask_stroke_worker.js");
      this.worker.onmessage=e=>this.receive(e.data);
      this.worker.onerror=()=>this.fail();
      // A bitmap copy initializes the worker without transferring ownership
      // of the canonical main-thread mask. Keep editing possible meanwhile.
      createImageBitmap(mask).then(bitmap=>{
        if(this.closed){bitmap.close?.();return;}
        this.worker.postMessage({type:"init",session:this.session,
          width:mask.width,height:mask.height,bitmap},[bitmap]);
      }).catch(()=>this.fail());
    }catch(_){this.failed=true;this.closed=true;this.worker?.terminate();}
  }
  get active(){return this.ready&&!this.failed&&!this.closed;}
  enqueue(segment){
    if(!this.active)return false;
    this.queue.push(segment);
    if(!this.frame&&!this.inFlight){
      this.frame=requestAnimationFrame(()=>{
        this.frame=0;this.pump();
      });
    }
    return true;
  }
  pump(){
    if(!this.active||this.inFlight||!this.queue.length)return;
    if(this.frame){cancelAnimationFrame(this.frame);this.frame=0;}
    const batch={seq:++this.seq,segments:this.queue.splice(0)};
    this.inFlight=batch;
    try{this.worker.postMessage({type:"stroke",session:this.session,
      seq:batch.seq,segments:batch.segments});}
    catch(_){this.fail();}
  }
  receive(message){
    if(this.closed||message?.session!==this.session){
      message?.bitmap?.close?.();
      return;
    }
    if(message.type==="ready"){
      this.ready=true;this.pump();return;
    }
    if(message.type==="error"){this.fail();return;}
    if(message.type!=="patch"||!this.inFlight||message.seq!==this.inFlight.seq){
      message?.bitmap?.close?.();
      return;
    }
    try{
      if(message.rect&&message.bitmap)this.onPatch(message.rect,message.bitmap);
    }catch(_){
      message.bitmap?.close?.();this.fail();return;
    }
    message.bitmap?.close?.();
    this.inFlight=null;
    if(this.queue.length)this.pump();
    else this.resolveWaiters();
  }
  resolveWaiters(){
    if(this.inFlight||this.queue.length||this.frame)return;
    for(const resolve of this.waiters.splice(0))resolve();
  }
  drain(){
    if(this.closed||this.failed)return Promise.resolve();
    if(!this.inFlight&&!this.queue.length)return Promise.resolve();
    return new Promise(resolve=>{this.waiters.push(resolve);this.pump();});
  }
  fail(){
    if(this.closed)return;
    const remaining=[...(this.inFlight?.segments||[]),...this.queue];
    this.close();
    for(const segment of remaining)this.onFallback(segment);
  }
  close(){
    if(this.closed)return;
    this.closed=true;this.ready=false;
    if(this.frame){cancelAnimationFrame(this.frame);this.frame=0;}
    this.worker?.terminate();
    this.queue=[];this.inFlight=null;
    this.resolveWaiters();
  }
}
if(typeof module!=="undefined"&&module.exports)module.exports=RemoveStrokeBridge;
