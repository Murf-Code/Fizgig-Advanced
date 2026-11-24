"use strict";
// Caption Workbench removal confirmation regression: real dialog interaction,
// safe cancellation, focus restoration, multi-item details and route coverage.
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const begin=app.indexOf("function captionTagRemovalTitle(");
const end=app.indexOf("function renderCaptionStudio(",begin);
assert(begin>0&&end>begin,"Shared confirmation helper must exist before Studio");
const source=app.slice(begin,end);
class Element {
  constructor(tag,props={},children=[]){
    this.tag=tag;this.props=props;this.children=[];
    this.listeners={};this.parentNode=null;this.isConnected=false;
    this.textContent=props.text||"";
    this.onclick=props.onclick;this.id=props.id;
    this.classList={contains:name=>(props.class||"").split(/\s+/).includes(name)};
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
  setAttribute(key,value){this.props[key]=value;}
  querySelectorAll(selector){const matches=c=>selector.split(",").some(s=>{s=s.trim();return s.startsWith(".")?c.classList.contains(s.slice(1)):c.tag===s;});return this.children.flatMap(c=>c instanceof Element?[...(matches(c)?[c]:[]),...c.querySelectorAll(selector)]:[]);}
  click(){this.onclick?.();}
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
const context={el,document,Promise,studioTokens:caption=>String(caption||"").split("\n")[0].split(",").map(s=>s.trim()).filter(Boolean),
  studioCaptionMode:it=>it.mode||"tags",studioUnique:tags=>[...new Map(tags.map(t=>[t.toLowerCase(),t])).values()],
  studioCompose:(caption,tags)=>tags.join(", ")+(String(caption||"").match(/\n[\s\S]*$/)?.[0]||""),
  capItemFolder:it=>"dataset"};
vm.runInNewContext(source+"\nthis.confirmRemoval=captionConfirmRemoval;this.tagConfirmation=captionTagRemovalConfirmation;",context);
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
  assert.equal(document.activeElement,buttons().modal,"Dialog must receive initial focus");
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

  reply=confirmRemoval({title:"Remove tags?",detail:"Selected images",items:["tag1","tag2"]});
  assert.equal(document.activeElement,buttons().modal);
  buttons().confirm.onclick();
  assert.equal(await reply,true);
  assert.equal(overlay(),null);
  assert.equal(document.activeElement,opener);

  reply=confirmRemoval({title:"Remove 100 images?",detail:"Undoable",items:Array.from({length:100},(_,i)=>"image"+i+".png")});
  const bulk=buttons().modal.children[1],names=bulk.children[1],toggle=bulk.children[2];
  assert.equal(names.children.length,3);
  assert.match(toggle.textContent,/97 more/);
  toggle.onclick();
  assert.equal(names.children.length,100);
  assert.equal(toggle.props["aria-expanded"],"true");
  toggle.onclick();
  assert.equal(names.children.length,3);
  buttons().cancel.onclick();
  assert.equal(await reply,false);

  reply=confirmRemoval({title:"Remove tags?"});
  o=overlay();o.listeners.keydown({...event("Enter"),repeat:true});
  assert.equal(overlay(),o,"Held Enter must not confirm");
  o.listeners.keydown({...event("Enter"),isComposing:true});
  assert.equal(overlay(),o,"IME Enter must not confirm");
  o.listeners.keydown(event("Enter"));
  assert.equal(await reply,true,"Default Enter confirms");

  reply=confirmRemoval({title:"Remove tags?"});
  o=overlay();o.listeners.keydown(event("Tab"));
  assert.equal(document.activeElement,buttons().cancel);
  o.listeners.keydown(event("Enter"));
  assert.equal(await reply,false,"Focused Cancel must cancel");

  reply=confirmRemoval({title:"Remove tags?"});
  o=overlay();o.listeners.keydown({...event("Tab"),shiftKey:true});
  assert.equal(document.activeElement,buttons().confirm);
  o.listeners.keydown(event("Enter"));
  assert.equal(await reply,true);

  reply=confirmRemoval({title:"Remove images?",items:Array.from({length:6},(_,i)=>"image"+i)});
  o=overlay();const listToggle=buttons().modal.children[1].children[2];
  listToggle.focus();listToggle.click();
  assert.equal(listToggle.props["aria-expanded"],"true");
  assert.equal(overlay(),o);
  o.listeners.keydown(event("Enter"));
  assert.equal(await reply,true,"Enter confirms even after clicking View All");

  const one=context.tagConfirmation(["cat"],["cat, CAT","dog","Cat"]);
  assert.equal(one.title,"Remove “cat” from 2 images?");
  const five=context.tagConfirmation(["cat","dog","bird","fish","fox"],["cat, cat, dog","CAT","dog"]);
  assert.equal(five.tagCounts.length,5);
  assert.equal(five.tagCounts[0].count,2);
  assert.equal(five.tagCounts[1].count,2);
  assert.equal(five.tagCounts[2].count,0);
  for(const n of [6,100]){
    const result=context.tagConfirmation(Array.from({length:n},(_,i)=>"tag"+i),[]);
    assert.equal(result.title,"Remove all selected tags?");
    assert.match(result.detail,new RegExp("^"+n+" tags selected"));
    assert.equal(result.tagCounts,undefined);
  }

  // Rename uses the same modal, but reports only affected images and a
  // non-destructive primary action rather than a generic system dialog.
  const files=[
    {name:"a.png",path:"a",caption:"cat, dog"},
    {name:"b.png",path:"b",caption:"dog,dog"},
    {name:"c.png",path:"c",caption:"CAT, lion"},
    {name:"d.png",path:"d",caption:"cat",mode:"prose"},
    {name:"e.png",path:"e",caption:"cat"},
    {name:"f.png",path:"f",caption:"cat, bird\nKeep this description."}
  ];
  const plan=context.captionBulkTagChanges(files,"rename","cat","lion",new Map(),new Set(["e"]));
  assert.equal(plan.changes.length,3);
  assert.equal(plan.locked,1);
  assert.equal(plan.blockedDraft,false);
  assert.equal(plan.changes.map(c=>c.path).join(","),"a,c,f");
  assert.equal(plan.changes[1].after,"lion");
  assert.equal(plan.changes[2].after,"lion, bird\nKeep this description.");
  assert.equal(context.captionBulkTagChanges(files,"rename","cat","lion",
    new Map([["b","dog, bird"]]),new Set(["e"])).blockedDraft,false);
  assert.equal(context.captionBulkTagChanges(files,"rename","cat","lion",
    new Map([["b","dog, cat"]]),new Set(["e"])).blockedDraft,true);
  const title=context.captionTagRenameTitle("cat","lion",plan.changes.length);
  assert.equal(title,"Rename “cat” to “lion” in 3 images?");
  assert.equal(context.captionTagRenameTitle("cat","lion",1),"Rename “cat” to “lion” in 1 image?");
  reply=confirmRemoval({title,detail:"This can be undone.",confirmLabel:"Rename",destructive:false,compact:true});
  o=overlay();
  assert.equal(buttons().confirm.textContent,"Rename");
  assert.equal(buttons().confirm.props.class,"btn btn-primary");
  o.listeners.keydown(event("Enter"));
  assert.equal(await reply,true);
  assert.doesNotMatch(app,/confirm\("Apply "\+mode\+" tag to "/);
  assert.match(app,/captionTagRenameTitle\(old\.trim\(\),newTag\.trim\(\),changes\.length\)/);
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
