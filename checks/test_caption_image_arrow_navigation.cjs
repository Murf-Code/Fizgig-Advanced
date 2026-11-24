"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const start=app.indexOf("function captionImageArrowDelta(");
const end=app.indexOf("function captionInlineStageDetach(",start);
assert(start>=0&&end>start);
const context=vm.createContext({});
vm.runInContext(app.slice(start,end),context);
for(const [key,expected] of [["ArrowUp",-1],["ArrowLeft",-1],["ArrowDown",1],["ArrowRight",1],["PageDown",0]]){
  assert.equal(context.captionImageArrowDelta({key}),expected);
  for(const opts of [{ctrlKey:true},{metaKey:true},{shiftKey:true},{altKey:true},{isComposing:true},
    {defaultPrevented:true},{target:{closest:()=>true}}])
    assert.equal(context.captionImageArrowDelta({key,...opts}),0);
}
assert.match(app,/thumbs\.addEventListener\("keydown",event=>\{/);
assert.match(app,/event\.preventDefault\(\);event\.stopPropagation\(\);\s*stepImage\(delta\);\s*focusImageRail\(\)/);
assert.match(app,/else \{renderCaptions\(\);focusImageRail\(\);\}/);
assert.match(app,/S\.cap\?\.imageEditing\?\.inline&&captionImageArrowDelta\(e\)/);
assert.match(app,/S\.cap\.studioStepImage\?\.\(delta\)/);
assert.match(app,/S\.cap\.studioThumbRevealPath=next\.path;\s*renderCaptions\(\)/);
console.log("Workbench four-way arrow navigation, native input guards and rail focus checks passed");
