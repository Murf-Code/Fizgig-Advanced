"use strict";
// Exercise the actual shared history functions with stubbed browser and API
// boundaries. No GPU, browser, or image-model download is needed.
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const section=(first,last)=>{
  const a=app.indexOf(first),b=app.indexOf(last,a+first.length);
  assert(a>=0&&b>a,first);
  return app.slice(a,b);
};
const historySource=section("function studioHistory(){","// Use the Start tab's existing resumable");
const changeSource=section("function captionImageChanged(","function studioCategories(){");
const pendingSource=section("async function mePendingTravel(","function meRatio() {");

(async()=>{
  const calls=[],forgot=[],refreshes=[];
  const S={imageFolder:"dataset",cap:{items:[],studioThumbPreloader:{forget:p=>forgot.push(p)}}};
  const m={folder:"dataset",name:"a.png",pic:{},mask:{},undo:[],redo:[],list:[{name:"a.png"}]};
  let fail=false;
  const ctx=vm.createContext({S,capExt:()=>".txt",meState:()=>m,
    api:async(path,body)=>{calls.push({path,body});if(fail)throw Error("disk failure");},
    loadCaptions:async()=>refreshes.push("render"),meStopRemoveWorker:()=>{},
    meRefreshFlags:async()=>{},meReload:async()=>true,meImgBar:()=>{},meDraw:()=>{},console});
  vm.runInContext(historySource+changeSource,ctx);
  const entry={kind:"image",step:7,folder:"dataset",name:"a.png",path:"dataset/a.png"};
  ctx.studioRecord("Crop image",entry);
  // Undo after Crop closed its inline editor: update disk first, invalidate
  // the cached editor and the exact thumbnail, then render.
  fail=true;
  await assert.rejects(ctx.studioTravel("undo"),/disk failure/);
  assert.equal(ctx.studioHistory().past.length,1);
  assert.equal(forgot.length,0);
  fail=false;
  await ctx.studioTravel("undo");
  assert.equal(calls.at(-1).body.step,7);
  assert.equal(calls.at(-1).body.redo,false);
  assert.equal(m.pic,null);
  assert.equal(m.list.length,0);
  assert.equal(S.cap.studioThumbRevisions.get(entry.path),1);
  assert.equal(forgot.at(-1),entry.path);
  assert.equal(ctx.studioHistory().future.length,1);
  // Redo while that file's canvas is open.
  S.cap.imageEditing={path:entry.path,inline:true};
  Object.assign(m,{pic:{},mask:{},redo:[{step:7}],undo:[],rmDirty:false});
  await ctx.studioTravel("redo");
  assert.equal(calls.at(-1).body.redo,true);
  assert.equal(m.undo[0].step,7);
  assert.equal(S.cap.studioThumbRevisions.get(entry.path),2);
  assert.equal(ctx.studioHistory().past.length,1);
  assert.equal(refreshes.length,2);
  // Unapplied Erase gestures travel separately and never call the file API.
  const base={width:10,height:10,marks:[]};
  Object.assign(m,{tool:"remove",rm:base,rmDirty:true,rmStrokeHistory:{
    base,baseDirty:false,baseHandEdited:false,baseWatermark:false,current:null,
    past:[["first"],["second"]],future:[]}});
  const order=[];
  Object.assign(ctx,{meDrainRemoveWorker:async()=>order.push("drain"),
    meStopRemoveWorker:()=>order.push("stop"),
    meCopy:c=>({...c,marks:[...c.marks]}),
    mePaintRemoveSegment:(c,segment)=>c.marks.push(segment),
    meDrawRemove:()=>{},mePrepareRemoveWorker:()=>{},meDropPending:()=>false});
  vm.runInContext(pendingSource,ctx);
  const beforeCalls=calls.length;
  assert.equal(await ctx.mePendingTravel("undo"),true);
  assert.deepEqual(m.rm.marks,["first"]);
  assert.deepEqual(order.slice(0,2),["drain","stop"]);
  assert.equal(await ctx.mePendingTravel("undo"),true);
  assert.equal(m.rmDirty,false);
  assert.equal(await ctx.mePendingTravel("redo"),true);
  assert.deepEqual(m.rm.marks,["first"]);
  assert.equal(await ctx.mePendingTravel("redo"),true);
  assert.deepEqual(m.rm.marks,["first","second"]);
  assert.equal(calls.length,beforeCalls);
  assert.match(app,/goBack=btn\("Undo",\(\)=>void travelPictureHistory\("undo"\)\)/);
  assert.match(app,/goForward=btn\("Redo",\(\)=>void travelPictureHistory\("redo"\)\)/);
  assert.match(app,/const button=document\.querySelector\("#captionsGrid \.studio-image-"\+direction\)/);
  console.log("Workbench disk history, targeted preview invalidation and erase gesture undo/redo passed");
})().catch(e=>{console.error(e);process.exitCode=1;});
