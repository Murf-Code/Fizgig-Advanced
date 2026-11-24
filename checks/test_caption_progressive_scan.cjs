"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const source=app.slice(app.indexOf("async function captionProgressiveScan("),app.indexOf("async function captionIndexForLoad("));
const encode=new TextEncoder();
const payload=events=>events.map(e=>JSON.stringify(e)+"\n").join("");
const manifest={type:"manifest",total:2};
const first={type:"batch",items:[{name:"café.png",caption:"watercolor"}],scanned:1,total:2};
const second={type:"batch",items:[{name:"other.png",caption:null}],scanned:2,total:2};
const done={type:"done",total:2};
async function collect(events){
  const bytes=encode.encode(payload(events));
  let offset=0,canceled=false;
  const reader={async read(){
    if(offset>=bytes.length)return {done:true};
    const value=bytes.slice(offset,offset+7);offset+=7;return {value,done:false};
  },async cancel(){canceled=true;}};
  const response={ok:true,headers:{get:()=>"application/x-ndjson"},body:{getReader:()=>reader}};
  const scan={id:"test-scan",controller:new AbortController(),serverDone:false};
  const seen=[];
  const run=new Function("fetch","TOKEN","TextDecoder","setTimeout","clearTimeout",source+"\nreturn captionProgressiveScan;")(
    async()=>response,"token",TextDecoder,setTimeout,clearTimeout);
  try{await run("/api/captions/scan",scan,event=>seen.push(event));return {seen,scan};}
  finally{assert.equal(canceled,true,"reader must be released on completion or failure");}
}
async function main(){
  const r=await collect([{type:"discovering"},manifest,first,second,done]);
  assert.deepEqual(r.seen,[{type:"discovering"},manifest,first,second,done]);
  assert.equal(r.scan.serverDone,true);
  await assert.rejects(()=>collect([manifest,first]),/before completion/);
  await assert.rejects(()=>collect([manifest,first,done]),/missing images/);
  await assert.rejects(()=>collect([manifest,{...first,scanned:2},done]),/invalid dataset scan batch/);
  await assert.rejects(()=>collect([manifest,{type:"error",error:"drive disconnected"}]),/drive disconnected/);
  const cancelSource=app.slice(app.indexOf("function cancelCaptionScan("),app.indexOf("function captionScanBrowser("));
  const requests=[],scan={id:"cancel-test",controller:new AbortController(),serverDone:false};
  const S={cap:{studioScan:scan}};
  const cancel=new Function("S","api",cancelSource+"\nreturn cancelCaptionScan;")(S,async(...args)=>requests.push(args));
  cancel();
  assert.equal(scan.controller.signal.aborted,true);
  assert.equal(S.cap.studioScan,null);
  assert.deepEqual(requests,[["/api/captions/scan_cancel",{id:"cancel-test"}]]);
  const complete={id:"finished-test",controller:new AbortController(),serverDone:true};
  S.cap.studioScan=complete;cancel();
  assert.equal(complete.controller.signal.aborted,true);
  assert.equal(requests.length,1);
  console.log("Progressive stream framing, UTF-8, exact counts, incomplete scans and explicit cancellation passed");
}
main().catch(error=>{console.error(error);process.exitCode=1;});
