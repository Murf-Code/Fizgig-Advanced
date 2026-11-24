const fs=require("node:fs");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const start=app.indexOf("async function captionLoadRequest"),end=app.indexOf("\nconst CAP_LEARNING_REFRESH_MS",start);
if(start<0||end<0)throw Error("Caption loader not found");
const src=app.slice(start,end);

function check(ok,message){if(!ok)throw Error(message);}
function harness(apiImpl,optionsImpl=async()=>{},initialOpts=true){
  const tab={dataset:{captionLoad:"initial"},classList:{contains:()=>true}};
  const elements=new Map([["tab-captions",tab]]);
  const $=id=>{if(!elements.has(id))elements.set(id,{style:{},replaceChildren(){}});return elements.get(id);};
  const S={imageFolder:"A",family:"anima",cap:{opts:initialOpts?{}:null,optsFamily:"anima"}};
  const states=[],renders=[];
  class Controller{constructor(){this.signal={aborted:false};}abort(){this.signal.aborted=true;}}
  const deps={S,$,TOKEN:"token",capExt:()=>".txt",captionInitialState:(state,detail)=>{tab.dataset.captionLoad=state;states.push([state,detail]);},
    capItemFolder:it=>it.path.split("/").slice(0,-1).join("/"),subfoldersHint:()=>"",pollCaptions:()=>{},
    loadCaptionOptions:optionsImpl,api:apiImpl,captionAiControlsRestore:()=>{},
    CaptionStudioSelection:{prune:()=>({selected:[],active:null,anchor:null})},
    renderCaptions:()=>renders.push(S.cap.items.map(it=>it.path)),
    AbortController:Controller,setTimeout:(fn,ms)=>setTimeout(fn,Math.min(ms,25)),clearTimeout,
    console:{warn:()=>{}}};
  const load=new Function(...Object.keys(deps),src+"\nreturn loadCaptions;")(...Object.values(deps));
  return {load,S,states,renders};
}
const item=folder=>({path:folder+"/img.png",name:"img.png"});
const never=()=>new Promise(()=>{});
async function run(){
  let liveStarted=false;
  const x=harness(async path=>{
    if(path.includes("live_status")){liveStarted=true;return never();}
    if(path.includes("reviews"))return {reviewed:["img.png"]};
    return {items:[item("A")]};
  });
  await x.load();
  check(x.states.at(-1)[0]==="ready","live status blocked mount");
  check(x.S.cap.studioReviewed.has("A/img.png"),"review lock lost");
  check(liveStarted,"live diagnostics not requested");
  let signal;
  const y=harness((path,body,s)=>{signal=s;return never();});
  await y.load();
  check(y.states.at(-1)[0]==="error"&&y.states.at(-1)[1].includes("Dataset scan timed out"),"scan timeout missing");
  check(signal.aborted,"hung scan not aborted");
  const z=harness(async path=>path.includes("reviews")?never():{items:[item("A")]});
  await z.load();
  check(z.states.at(-1)[0]==="error"&&z.renders.length===0,"review timeout must not unlock edits");
  const q=harness(async path=>path.includes("reviews")?{reviewed:[]}:path.includes("live_status")?{}:{items:[item("A")]},never,false);
  await q.load();
  check(q.states.at(-1)[0]==="ready","optional model options timeout blocked mount");
  let release;
  const stale=harness(async path=>{
    if(path.includes("reviews"))return {reviewed:[]};
    if(path.includes("live_status"))return {};
    if(path.includes("folder=A"))return new Promise(resolve=>{release=resolve;});
    return {items:[item("B")]};
  });
  const first=stale.load();
  await Promise.resolve();await Promise.resolve();
  check(typeof release==="function","first scan did not start");
  stale.S.imageFolder="B";release({items:[item("A")]});await first;
  await new Promise(resolve=>setTimeout(resolve,15));
  check(stale.states.at(-1)[0]==="ready"&&stale.renders.length===1&&stale.renders[0][0]==="B/img.png","stale scan did not restart correctly");
  return "5 loader scenarios passed";
}

run().then(console.log).catch(error=>{console.error(error);process.exitCode=1;});
