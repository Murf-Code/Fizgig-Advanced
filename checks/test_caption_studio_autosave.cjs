"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");

// Execute the real, extracted autosave and history routines with a fake API.
// This catches accidental reintroduction of delayed tag saves, draft loss,
// unsafe cache restores and broken undo/redo without a live GPU session.
function between(start,end){
  const a=app.indexOf(start),b=app.indexOf(end,a);
  assert.ok(a>=0&&b>a,"Cannot locate autosave section: "+start);
  return app.slice(a,b);
}
const autosave=between(
  "  const autoTasks=S.cap.studioAutosaves",
  "  const ta=el(\"textarea\",{class:\"studio-caption\"");
const build=new Function("S","current","ext","capItemFolder","drafts","histories",
  "note","studioRecord","api","localStorage","setTimeout","clearTimeout","locked",
  autosave+"\nreturn {autoTask,queueAutosave,cacheKey};");
const historyBody=between(
  "  const hist=histories.get(current.path)",
  "  const word=el(\"div\",{class:\"field-hint\"}");
const buildHistory=new Function("histories","current","ta","autoTask","drafts",
  "cacheKey","queueAutosave","localStorage","locked","say","redraw",
  "clearTimeout","setTimeout",
  historyBody+"\nreturn {hist,remember,historyStep,persistHistory};");
const store=new Map();
const localStorage={
  getItem:k=>store.has(k)?store.get(k):null,
  setItem:(k,v)=>store.set(k,String(v)),
  removeItem:k=>store.delete(k)
};
const timers=[];
const schedule=(fn,ms)=>{
  timers.push(ms);
  if(ms===0)queueMicrotask(fn);
  return timers.length;
};
const tick=()=>new Promise(resolve=>global.setTimeout(resolve,0));
function setup(api,caption=null,extra={}){
  const current={name:"a.png",path:"/dataset/a.png",caption,...extra};
  const S={cap:{items:[current],studioAutosaves:new Map(),studioReviewed:new Set(),
    studioPath:current.path,studioRevision:0}};
  const drafts=new Map(),histories=new Map(),saved=[];
  const env=build(S,current,".txt",()=>"/dataset",drafts,histories,
    {isConnected:true,textContent:""},(...args)=>saved.push(args),
    api,localStorage,schedule,()=>{},false);
  return {...env,current,S,drafts,histories,saved};
}
async function test(){
  store.clear();timers.length=0;
  let calls=[];
  let env=setup(async(url,body)=>{
    calls.push({url,body});
    return {queued:false};
  });
  env.drafts.set(env.current.path,"one, two");
  env.queueAutosave(true);
  assert.equal(calls.length,1,"tag save must start synchronously");
  await tick();
  assert.equal(env.current.caption,"one, two");
  assert.equal(env.drafts.has(env.current.path),false);
  assert.equal(env.S.cap.studioRevision,1);

  store.clear();timers.length=0;
  const pending=[];
  env=setup((url,body)=>new Promise(resolve=>pending.push({body,resolve})));
  env.drafts.set(env.current.path,"first");env.queueAutosave(true);
  env.drafts.set(env.current.path,"second");env.queueAutosave(true);
  assert.equal(pending.length,1,"never race two saves of one image");
  pending[0].resolve({queued:false});
  await tick();await tick();
  assert.equal(pending.length,2,"newer edit must save after older request");
  assert.equal(pending[1].body.text,"second");
  pending[1].resolve({queued:false});
  await tick();
  assert.equal(env.current.caption,"second");

  store.clear();timers.length=0;
  let attempts=0;
  env=setup(async()=>{
    attempts++;
    if(attempts===1)throw new Error("disk busy");
    return {queued:false};
  });
  env.drafts.set(env.current.path,"recover me");env.queueAutosave(true);
  await tick();
  assert.equal(env.drafts.get(env.current.path),"recover me",
    "failed saves must not discard the draft");
  assert.equal(env.autoTask.pending,true);
  assert.ok(env.autoTask.retryMs>=2500);
  await env.autoTask.flush();
  assert.equal(env.current.caption,"recover me");

  store.clear();timers.length=0;
  const cacheKey="fizgig_caption_editor_history_v1:"+
    JSON.stringify(["/dataset","/dataset/a.png",".txt"]);
  store.set(cacheKey,JSON.stringify({lastSaved:"old",value:"old, improved",
    draft:"old, improved",past:["old"],future:[]}));
  env=setup(async()=>({queued:false}),"old");
  assert.equal(env.histories.get(env.current.path).value,"old, improved",
    "matching cached draft should recover after refresh");
  assert.equal(env.drafts.get(env.current.path),"old, improved");
  // A different on-disk caption must not be overwritten from a stale cache.
  env=setup(async()=>({queued:false}),"externally changed");
  assert.equal(env.histories.has(env.current.path),false);
  assert.equal(env.drafts.has(env.current.path),false);

  store.clear();timers.length=0;
  env=setup(async()=>({queued:false}),"");
  const ta={value:""};
  const history=buildHistory(env.histories,env.current,ta,env.autoTask,
    env.drafts,env.cacheKey,()=>{},localStorage,false,()=>{},()=>{},
    ()=>{},schedule);
  ta.value="animal";history.remember(true);
  ta.value="animal, blue";history.remember(true);
  assert.deepEqual(history.hist.past,["","animal"]);
  assert.equal(JSON.parse(store.get(env.cacheKey)).value,"animal, blue");
  history.historyStep("undo");
  assert.equal(ta.value,"animal");
  assert.deepEqual(history.hist.future,["animal, blue"]);
  history.historyStep("redo");
  assert.equal(ta.value,"animal, blue");
  assert.equal(JSON.parse(store.get(env.cacheKey)).value,"animal, blue");

  // Each discrete tag action requests an immediate save; prose typing stays
  // debounced to avoid network writes on every character.
  assert.match(app,/imageTagPicked\.clear\(\);remember\(true\);redraw\(\)/);
  assert.match(app,/ta\.value=studioCompose\(ta\.value,arr,"tags"\);remember\(true\);redraw\(\)/);
  assert.match(app,/newTag\.value="";\s*remember\(true\);redraw\(\)/);
  assert.match(app,/captionText\.addEventListener\("input",[\s\S]*?remember\(\);/);
  console.log("Caption editor immediate saves, recovery cache, and undo/redo passed");
}
test().catch(error=>{console.error(error);process.exitCode=1;});