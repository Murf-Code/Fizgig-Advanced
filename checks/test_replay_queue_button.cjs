const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const html = fs.readFileSync("src/fizgig/web/static/index.html", "utf8");
assert.match(html, /id="streamSyncMode" checked/);
assert.match(html, /Global Replay Experiment 5B · 8 runs × 5 epochs/);
assert.doesNotMatch(html, /id="streamSyncQueue"|id="replayCompareQueue"|id="replayCompareToggle"/);
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]).find(s=>s.includes('fetch("/api/queue/replay-compare"'));
assert.ok(script);
let handler, requests = [];
const elements = {
  streamSyncMode: {checked:true, addEventListener(){}},
  btnQueue: {contains(target){return target === this;}},
  streamSyncNote: {}, replayComparePrompt: {value:"fixed"},
  imageFolder: {value:"C:/dataset"},
};
const document = {
  getElementById: id => elements[id],
  querySelector: () => ({content:"token"}),
  addEventListener(name, callback, capture) {
    assert.equal(name, "click"); assert.equal(capture, true); handler = callback;
  }
};
vm.runInNewContext(script, {document, fetch:async (url, opts)=>{
  requests.push([url, JSON.parse(opts.body)]);
  return {json: async()=>({ok:true, output:"out"})};
}});
(async()=>{
  const event = {target:elements.btnQueue, prevented:false, stopped:false,
    preventDefault(){this.prevented=true;}, stopImmediatePropagation(){this.stopped=true;}};
  await handler(event);
  assert.ok(event.prevented && event.stopped);
  assert.deepEqual(requests, [["/api/queue/replay-compare", {image_folder:"C:/dataset"}]]);
  assert.match(elements.streamSyncNote.textContent, /8 deterministic runs queued \(145 updates each\)/);
  elements.streamSyncMode.checked=false;
  event.prevented=event.stopped=false;
  await handler(event);
  assert.equal(requests.length,1);
  assert.ok(!event.prevented && !event.stopped);
  elements.streamSyncMode.checked=true;
  elements.imageFolder.value="";
  await handler(event);
  assert.equal(requests.length,1);
  console.log("Replay queue button regression passed");
})().catch(err=>{console.error(err);process.exitCode=1;});
