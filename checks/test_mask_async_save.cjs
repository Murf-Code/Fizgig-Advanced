"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const start=app.indexOf("async function meSave()");
const end=app.indexOf("async function meRemoveAllWm()",start);
assert(start>=0&&end>start);
assert.match(app,/m\.applyPreparing=true/);
assert.match(app,/if\(m\.stepBusy\|\|m\.samBusy\|\|m\.wmBusy\|\|m\.applyPreparing\|\|m\.saveTask\)/);
assert.match(app,/if\(m\.dirty&&\(!\(await meSave\(\)\)\|\|m\.dirty\)\)return false/);
const gate=[],changes=[];
const mask={width:20,height:20,getContext(){return {drawImage(){}};}};
const state={name:"picture.png",folder:"dataset",mask,dirty:true,
  maskRevision:1,list:[{name:"picture.png"}]};
const vmContext={
  meState:()=>state,
  MaskEditorPerf:{pngDataURL:()=>new Promise(resolve=>gate.push(resolve))},
  el:(tag,attrs)=>({width:attrs.width,height:attrs.height,
    getContext:()=>({fillRect(){},drawImage(){}})}),
  api:async(path,body)=>{changes.push({path,body});return {removed:false};},
  S:{imageFolder:"dataset"},
  captionImageChanged(){},meStateLine(){},maskInfo(){},
  $:()=>({textContent:""}),Promise,Number
};
vm.runInNewContext(app.slice(start,end),vmContext);

async function main(){
  const p1=vmContext.meSave();
  const p2=vmContext.meSave();
  assert.equal(gate.length,1,"simultaneous saves must reuse the same task");
  gate.shift()("data:image/png;base64,AQ==");
  assert.equal(await p1,true);assert.equal(await p2,true);
  assert.equal(changes.length,1);
  assert.equal(state.dirty,false);

  state.dirty=true;state.maskRevision++;
  const p3=vmContext.meSave();
  assert.equal(gate.length,1);
  // User keeps painting while previous mask PNG is being encoded.
  state.maskRevision++;state.dirty=true;
  gate.shift()("data:image/png;base64,Ag==");
  assert.equal(await p3,true);
  assert.equal(state.dirty,true,"a late response must not mark newer edits saved");
  assert.equal(changes.length,2);
  console.log("Asynchronous mask saves serialize and preserve later unsaved strokes");
}
main().catch(error=>{console.error(error);process.exitCode=1;});
