"use strict";
// Caption Workbench removal confirmation regression: real dialog interaction,
// safe cancellation, focus restoration, multi-item details and route coverage.
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const begin=app.indexOf("function captionConfirmRemoval(");
const end=app.indexOf("function renderCaptionStudio(",begin);
assert(begin>0&&end>begin,"Shared confirmation helper must exist before Studio");
const source=app.slice(begin,end);
class Element {
  constructor(tag,props={},children=[]){
    this.tag=tag;this.props=props;this.children=[];
    this.listeners={};this.parentNode=null;this.isConnected=false;
    this.textContent=props.text||"";
    this.onclick=props.onclick;this.id=props.id;
    this.append(...children);
  }
  append(...children){
    for(const x of children.flat()){
      if(x==null)continue;
      this.children.push(x);
      if(x instanceof Element){x.parentNode=this;this.connectChild(x,this.isConnected);}
    }
  }
  connectChild(x,connected){x.isConnected=connected;for(const c of x.children)if(c instanceof Element)this.connectChild(c,connected);}
  remove(){
    if(this.parentNode){this.parentNode.children=this.parentNode.children.filter(c=>c!==this);this.parentNode=null;}
    this.connectChild(this,false);
  }
  focus(){document.activeElement=this;}
  addEventListener(type,fn){this.listeners[type]=fn;}
}
const el=(tag,props,...children)=>new Element(tag,props,children);
const document={body:new Element("body"),activeElement:null,
  querySelector(selector){
    assert.equal(selector,".caption-removal-backdrop");
    return this.body.children.find(x=>x.props.class==="caption-removal-backdrop")||null;
  }};
document.body.isConnected=true;
const opener=new Element("button");opener.isConnected=true;opener.focus();
const context={el,document,Promise};
vm.runInNewContext(source+"\nthis.confirmRemoval=captionConfirmRemoval;",context);
const confirmRemoval=context.confirmRemoval;
const event=(key,target=null)=>({key,target,preventDefault(){this.prevented=true;},stopPropagation(){this.stopped=true;}});
const overlay=()=>document.querySelector(".caption-removal-backdrop");
const buttons=()=>{
  const modal=overlay().children[0];
  const footer=modal.children.at(-1);
  return {modal,cancel:footer.children[0],confirm:footer.children[1]};
};
(async()=>{
  let acted=false;
  let reply=confirmRemoval({title:"Remove 2 images?",detail:"Moves to removed/",items:["cat.png","dog.png"],
    confirmLabel:"Move to removed/"}).then(ok=>{acted=ok;return ok;});
  let o=overlay();
  assert(o,"Dialog must render before any mutation");
  assert.equal(o.children[0].props.role,"dialog");
  assert.equal(o.children[0].props["aria-modal"],"true");
  const info=o.children[0].children[1].children[1];
  assert.deepEqual(info.children.map(x=>x.textContent),["cat.png","dog.png"]);
  assert.equal(document.activeElement,buttons().cancel,"Cancel must receive initial focus");
  assert.equal(await confirmRemoval({title:"Second operation"}),false,"duplicate pending confirmation must not override first");
  buttons().cancel.onclick();
  assert.equal(await reply,false);
  assert.equal(acted,false);
  assert.equal(overlay(),null);
  assert.equal(document.activeElement,opener);

  reply=confirmRemoval({title:"Remove one tag?",detail:"Undoable",items:["crying"]});
  o=overlay();const esc=event("Escape");
  o.listeners.keydown(esc);
  assert(esc.prevented&&esc.stopped);
  assert.equal(await reply,false);

  reply=confirmRemoval({title:"Remove image?",detail:"Undoable",items:["one.png"]});
  o=overlay();o.listeners.pointerdown(event("",o));
  assert.equal(await reply,false);

  reply=confirmRemoval({title:"Delete tags?",detail:"Selected images",items:["tag1","tag2"]});
  assert.equal(document.activeElement,buttons().cancel);
  buttons().confirm.onclick();
  assert.equal(await reply,true);
  assert.equal(overlay(),null);
  assert.equal(document.activeElement,opener);

  assert.match(css,/\.caption-removal-backdrop\{/);
  assert.match(css,/\.caption-removal-modal\{/);
  assert.match(app,/if\(!\(await captionConfirmRemoval\(/);
  assert.match(app,/const removePickedTags=async\(onlyTag=null\)=>/);
  assert.match(app,/menuAction\('Remove "'\+tag\+'"',\(\)=>void removePickedTags\(tag\)/);
  assert.match(app,/const deletePickedTags=async\(contextTag=null\)=>/);
  assert.match(app,/async function capRemove\(it, card, status\)/);
  assert.match(app,/title:"Remove "\+donors\.length\+" duplicate image"/);
  console.log("Caption removal confirmation interaction and route checks passed");
})().catch(e=>{console.error(e);process.exitCode=1;});
