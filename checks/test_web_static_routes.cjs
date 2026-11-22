"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

// Regression: index.html loaded CaptionStudioSelection before its script was
// in the server's static route allowlist. The workbench appeared completely
// blank even though all source files passed node --check.
const root = path.resolve("src/fizgig/web/static");
const index = fs.readFileSync(path.join(root, "index.html"), "utf8");
const server = fs.readFileSync("src/fizgig/web/server.py", "utf8");
const route = server.match(/if path in \(([\s\S]*?)\):\s*\n\s*return self\._file\(os\.path\.join\(STATIC_DIR, path\[1:\]\)/);
assert.ok(route, "static server route was not found");
const allowed = new Set([...route[1].matchAll(/"\/([^"]+)"/g)].map(m => m[1]));
const scriptFiles = [...index.matchAll(/<script\s+src="\/([^"?]+)(?:\?[^"]*)?"/g)].map(m => m[1]);
assert.ok(scriptFiles.length >= 5, "expected webapp scripts in index.html");
for (const name of scriptFiles) {
  assert.ok(fs.existsSync(path.join(root, name)), "missing script on disk: " + name);
  assert.ok(allowed.has(name), "server will return HTTP 404 for script: /" + name);
}
assert.ok(scriptFiles.indexOf("caption_studio_selection.js") >= 0);
assert.ok(scriptFiles.indexOf("caption_studio_training_picks.js") >= 0);
assert.ok(scriptFiles.indexOf("caption_studio_selection.js") < scriptFiles.indexOf("app.js"));
assert.ok(scriptFiles.indexOf("caption_studio_training_picks.js") < scriptFiles.indexOf("app.js"));
console.log("All " + scriptFiles.length + " webapp scripts exist and are routed by Fizgig's HTTP server");
