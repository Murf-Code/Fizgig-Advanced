"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const Tags=require("../src/fizgig/web/static/caption_studio_tags.js");
const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");
// Both views consume the canonical Boolean tag query, never a separate
// Waterfall filter or the All Tags vocabulary-search field.
const captions=[["lion","golden fur"],["lion","blue eyes"],["tiger","blue eyes"]];
for(const [query,expected] of [
  ['"lion"',[0,1]],['"golden fur"',[0]],
  ['("lion") AND "blue eyes"',[1]],['"tiger" OR "golden fur"',[0,2]],
  ['"lion" AND NOT "blue eyes"',[0]],['',[0,1,2]]
]){
  const predicate=Tags.compileSearch(query);
  assert.deepEqual(captions.map((tags,i)=>predicate(tags)?i:-1).filter(i=>i>=0),expected);
}
assert.ok(app.includes('const match=CaptionStudioTags.compileSearch(S.cap.studioQuery);'));
assert.ok(app.includes('const flow=CaptionStudioViz.tagbenchWaterfall(items,chooseImage,openImageMenu,'));
assert.ok(app.includes('if(S.cap.studioQuery){\n          const header=el("header",{class:"studio-waterfall-filter"'));
assert.ok(app.includes('createTagFilterBar(),el("span",{class:"studio-waterfall-filter-count"'));
assert.ok(app.includes('text:items.length+" "+(items.length===1?"image":"images")'));
assert.ok(app.includes('".studio-waterfall-filter .studio-tag-query":".studio-tag-query"'));
// Exercise the actual delayed-click commit helper using a tiny state fixture.
const from=app.indexOf("  const commitTagFilter=()=>{");
const to=app.indexOf("  let tagEditing=null;",from);
assert.ok(from>=0&&to>from);
const cap={studioQuery:"",page:2,studioThumbScrollTop:50};
const commit=new Function("S","clearTimeout",
  'let pendingTagClick=1,pendingTagFilter="golden fur";\n'+app.slice(from,to)+'\nreturn commitTagFilter;')({cap},()=>{});
assert.equal(commit(),true);
assert.equal(cap.studioQuery,'"golden fur"');
assert.equal(cap.page,0);
assert.equal(cap.studioThumbScrollTop,0);
assert.equal(commit(),false);
assert.ok(app.includes('if(view==="waterfall"&&commitTagFilter())'));
assert.ok(css.includes('.studio-tb-explore.is-filtered-waterfall > .studio-tb-waterfall'));
console.log("Waterfall shared tag filters, Boolean matching, count and fast-switch checks passed");
