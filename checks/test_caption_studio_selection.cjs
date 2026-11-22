const assert = require("node:assert/strict");
const Selection = require("../src/fizgig/web/static/caption_studio_selection.js");

const items = Array.from({length:125}, (_,i)=>({path:"/set/image_"+i+".png"}));
const key = i => items[i].path;
const pick = (selected,active,anchor,index,modifiers={}) =>
  Selection.pick(items,selected,active,anchor,key(index),modifiers,24);

let next=pick([],null,null,70);
assert.deepEqual(next.selected,[key(70)],"Waterfall selection is carried into Tags");
assert.equal(next.active,key(70));
assert.equal(next.page,2,"selected image's page follows the click");
assert.equal(next.anchor,key(70));

next=pick(next.selected,next.active,next.anchor,72,{ctrlKey:true});
assert.deepEqual(next.selected,[key(70),key(72)],"Ctrl adds a second image");
assert.equal(next.active,key(72));
assert.equal(next.page,3);

next=pick(next.selected,next.active,next.anchor,70,{metaKey:true});
assert.deepEqual(next.selected,[key(72)],"Cmd toggles an image off");
assert.equal(next.active,key(70),"active focus can remain on a deselected image");

next=pick(next.selected,next.active,next.anchor,75,{shiftKey:true});
assert.deepEqual(next.selected,[key(70),key(71),key(72),key(73),key(74),key(75)],
  "Shift selects across rows and pages");
assert.equal(next.anchor,key(70),"Shift range retains original anchor");
assert.equal(next.active,key(75));
assert.equal(next.page,3);

next=pick(next.selected,next.active,next.anchor,10);
assert.deepEqual(next.selected,[key(10)],"plain click replaces multiselection");

const filtered=[items[2],items[4],items[70]];
next=Selection.pick(filtered,[key(10),key(2)],key(10),key(10),key(70),{ctrlKey:true},24);
assert.deepEqual(next.selected,[key(10),key(2),key(70)],
  "filtering doesn't discard selected images outside the current view");
assert.equal(next.page,0,"page calculation uses visible ordering");
assert.deepEqual(Selection.pick(filtered,next.selected,next.active,next.anchor,"/outside.png"),
  {selected:next.selected,active:next.active,anchor:next.anchor,page:null},
  "invalid targets cannot mutate selection");

const restored=Selection.prune([items[2],items[70]],next.selected,next.active,next.anchor);
assert.deepEqual(restored.selected,[key(2),key(70)],"refresh clears removed selections");
assert.equal(restored.active,key(70));
assert.equal(restored.anchor,key(70));
assert.deepEqual(Selection.prune([],next.selected,next.active,next.anchor),
  {selected:[],active:null,anchor:null},"empty dataset clears stale state");

console.log("Caption Workbench shared view-selection checks passed");
