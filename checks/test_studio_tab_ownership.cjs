"use strict";
// The global Studio GPU worker is shared. A foreign load must never make this
// tab repaint its outputs or present the other tab's controls as its own.
const fs = require("node:fs");
const assert = require("node:assert/strict");
const js = fs.readFileSync("src/fizgig/web/static/app.js", "utf8");
const start = js.indexOf("async function studioPoll() {");
const end = js.indexOf("// Download original / Download tweaked:", start);
assert(start >= 0 && end > start, "Studio polling implementation exists");
const fn = js.slice(start, end);
assert(fn.includes('st.load_session !== studioSession()'));
const els = {};
const el = () => ({
  style: {width: ""},
  textContent: "", src: "my-existing-picture", disabled: false,
  dataset: {}, parentElement: {classList: {toggle() {}}},
  classList: {contains() {return false;}}
});
const $ = id => els[id] || (els[id] = el());
const S = {st: {ready: true, shown: 7, loadedKey: "mine", pendingFirst: false,
  poll: null, foreign: false}};
const status = {load_id: 19, load_session: "another-tab", state: "ready",
  shown: 18, rendering: true, status: "Loading other person's model"};
const api = async path => {
  assert.equal(path, "/api/studio/status");
  return status;
};
const poll = new Function("S", "$", "api", "studioSession", "clearTimeout", "setTimeout",
  fn + "\nreturn studioPoll;")(S, $, api, () => "my-tab", () => {}, () => {});
(async () => {
  $("stBase").src = "my-original.png";
  $("stTweak").src = "my-tweaked.png";
  await poll();
  assert.equal(S.st.foreign, true);
  assert.equal(S.st.ready, false);
  assert.equal(S.st.loadedKey, "");
  assert.equal($("stStart").textContent, "Start");
  assert.equal($("stUnload").disabled, true);
  assert.match($("stStatus").textContent, /Another Studio tab/);
  assert.equal($("stBase").src, "my-original.png");
  assert.equal($("stTweak").src, "my-tweaked.png");
  console.log("Studio other-tab image and control isolation passed");
})().catch(err => {console.error(err);process.exitCode=1;});

for (const name of ["load", "render", "settings", "donor", "reset", "save", "frank_export"]) {
  const pattern = new RegExp('api\\("/api/studio/' + name + '", \\{(?!studio_session:)','g');
  assert.doesNotMatch(js, pattern, "unscoped Studio mutation request: " + name);
}
