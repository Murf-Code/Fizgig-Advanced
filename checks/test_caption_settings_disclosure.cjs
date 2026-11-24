"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
const source=app.slice(app.indexOf("let captionSettingsDialog=null;"),app.indexOf("// The original AI caption controls"));
function node(tag,attrs={},...children){
  return {tag,children,open:false,hidden:!!attrs.hidden,isConnected:true,value:attrs.value||"",style:{setProperty(){}},
    classList:{add(){},remove(){}},append(...nodes){this.children.push(...nodes)},setAttribute(){},addEventListener(){},focus(){}};
}
const original=node("details"),tab=node("section");
tab.querySelector=()=>original;
const doc={body:node("body"),activeElement:node("button"),querySelector:()=>null};
const local={getItem:()=>null,setItem(){}};
const S={cap:{}};
const init=new Function("S","$","document","localStorage","el",source+"\nreturn {open:captionWorkbenchSettingsOpen,dialog:()=>captionSettingsDialog};")(S,()=>tab,doc,local,node);
init.open(false,true);
const d=init.dialog();
assert.equal(d.original.open,true,"nested details must start open even during hidden initialization");
init.open(false);
assert.equal(d.advancedPanel.open,false);
d.advancedPanel.open=true; // native details disclosure
assert.equal(d.original.open,true,"opening visible disclosure must expose tools");
init.open(true);
assert.equal(d.advancedPanel.open,true);
assert.equal(d.original.open,true);
init.open(false);
assert.equal(d.original.open,true,"reopening settings cannot collapse the hidden drawer");
assert.match(css,/\.caption-settings-body\{[^}]*overflow-y:auto/);
assert.match(css,/\.caption-settings-footer\{flex:none/);
assert.match(source,/dialog.append\(heading,settingsBody,actions\)/);
console.log("Caption Settings nested disclosure and scroll layout passed");
