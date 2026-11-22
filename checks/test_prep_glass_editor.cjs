"use strict";
// Regression contract for the floating Image Prep editor. Captions continues
// to mount the same canvas inline without presenting the glass dialog.
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const html=fs.readFileSync("src/fizgig/web/static/index.html","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const js=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
for(const id of ["maskEdit","meDialogBackdrop","meDialogWindow","meDialogTitle",
  "meDialogClose","meStage","meCanvas"]){
  assert.equal((html.match(new RegExp('id="'+id+'"',"g"))||[]).length,1,id+" must have one live instance");
}
assert.match(html,/<details class="mask-edit" id="maskEdit">[\s\S]*?<div class="me-dialog-backdrop" id="meDialogBackdrop">/);
assert.match(html,/<section class="me-dialog-window" id="meDialogWindow" role="dialog" aria-modal="true"/);
assert.match(html,/<div class="me-dialog-content">[\s\S]*?id="meStage"/);
assert.match(css,/#tab-prep #maskEdit \.me-dialog-backdrop\{display:none\}/);
assert.match(css,/#tab-prep\.active #maskEdit\[open\] > \.me-dialog-backdrop\{/);
assert.match(css,/#tab-prep \.me-dialog-content>\.me-stage\{/);
assert.match(js,/const height=\(S\.cap\?\.imageEditing\?\.inline\|\|meDialogVisible\(\)\) \?/);
assert.match(js,/if\(name!=="prep"&&meDialogVisible\(\)\)/);
assert.match(js,/if\(S\.cap\?\.imageEditing\?\.inline\|\|!\$\("tab-prep"\)\.classList\.contains\("active"\)\)return/);
assert.match(js,/if\(event\.target===event\.currentTarget\)void meDialogDismiss\(\)/);
assert.match(js,/if\(m\.rmDirty\|\|\(m\.tool==="crop"&&m\.crop&&!meCropWhole\(\)\)\)return/);
assert.match(js,/captionInlineStageMount\(preview\)/);
assert.match(js,/captionInlineStageDetach\(\)/);
assert.match(js,/new RemoveStrokeBridge\(/);
assert.match(js,/MaskEditorPerf\.trimHistory\(m\.undo,ME_UNDO\)/);
assert.match(js,/MaskEditorPerf\.pngDataURL\(out\)/);

const start=js.indexOf("let meDialogPreviousFocus=null,meDialogDismissing=false;");
const end=js.indexOf("function meWire() {",start);
assert(start>=0&&end>start,"Modal lifecycle not found");
const snippet=js.slice(start,end);
const classState={active:true};
const prevFocus={isConnected:true,focused:false,focus(){this.focused=true;}};
const close={disabled:false,focus(){this.focused=true;}};
const details={open:false,querySelector:()=>prevFocus};
const tab={classList:{contains:()=>classState.active}};
const state={pic:null,mask:null,folder:"",studioFolder:null,dirty:false,tool:"brush",
  stepBusy:false,samBusy:false,wmBusy:false,applyPreparing:false};
let loaded=0,fitted=0,stopped=0,saves=0,wmSaves=0;
const classes=new Set();
const document={activeElement:prevFocus,body:{classList:{
  add(x){classes.add(x);},remove(x){classes.delete(x);}
}}};
const vmContext={
  $:id=>({
    maskEdit:details,"tab-prep":tab,meDialogClose:close,
    meState:{textContent:""}
  })[id],
  S:{cap:{imageEditing:null},imageFolder:"demo"},
  meState:()=>state,
  document,Promise,
  meOpen:async()=>{loaded++;state.pic={};state.mask={};state.folder="demo";},
  requestAnimationFrame:fn=>{fn();return 1;},
  meFit:()=>fitted++,
  meImgBar:()=>{},
  mePrepareRemoveWorker:()=>{},
  meSave:async()=>{saves++;state.dirty=false;return true;},
  meRmSave:async()=>{wmSaves++;return true;},
  meDrainRemoveWorker:async()=>{},
  meStopRemoveWorker:()=>stopped++
};
vm.runInNewContext(snippet,vmContext);
async function test(){
  assert.equal(vmContext.meDialogVisible(),false);
  await vmContext.meDialogShow();
  assert.equal(vmContext.meDialogVisible(),true);
  assert.equal(details.open,true);
  assert.equal(loaded,1);
  assert.equal(fitted,1);
  assert(classes.has("me-dialog-open"));
  state.stepBusy=true;
  assert.equal(await vmContext.meDialogDismiss(),false,
    "must block close during active destructive operations");
  state.stepBusy=false;
  state.dirty=true;
  assert.equal(await vmContext.meDialogDismiss(),true);
  assert.equal(saves,1);
  assert.equal(wmSaves,1);
  assert.equal(stopped,1);
  assert.equal(details.open,false);
  assert.equal(classes.size,0);
  assert(prevFocus.focused);
  state.pic={};state.mask={};state.folder="demo";
  await vmContext.meDialogShow();
  assert.equal(loaded,1,"reuse image instead of resetting history every open");
  assert.equal(await vmContext.meDialogDismiss(),true);
  classState.active=false;
  details.open=true;
  assert.equal(vmContext.meDialogVisible(),false,
    "Caption tab inline editor must not trigger the Image Prep modal");
  details.open=false;
  classState.active=true;
  vmContext.S.cap.imageEditing={inline:true};
  await vmContext.meDialogShow();
  assert.equal(details.open,false,
    "opening inline editor must not also open Image Prep overlay");
  console.log("Glass Image Prep overlay open/close, save guards, focus, and Caption separation passed");
}
test().catch(error=>{console.error(error);process.exitCode=1;});
