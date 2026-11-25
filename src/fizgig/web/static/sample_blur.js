/* Generated samples only: explicit callers in app.js; no dataset DOM scans. */
const SampleBlur = (() => {
  let enabled = false, wired = false, save = null, serial = Promise.resolve();
  const revisions = new WeakMap(), bound = new WeakSet(), results = new Map(), revealed = new Set();
  const token = () => document.querySelector('meta[name="fizgig-token"]').content;
  const key = img => JSON.stringify([img.dataset.sampleBlurPath, img.dataset.sampleBlurTime]);
  const live = (img, rev) => enabled && revisions.get(img) === rev && !!img.dataset.sampleBlurPath;
  function label(img, text) {
    let node = img._sampleBlurLabel;
    if (!text) { if (node) node.hidden = true; return; }
    if (!node) {
      node = document.createElement("span");
      node.className = "sample-blur-label";
      node.setAttribute("role", "status");
      img._sampleBlurLabel = node;
    }
    if (img.parentNode && node.parentNode !== img.parentNode) img.after(node);
    node.textContent = text; node.hidden = false;
  }
  function display(img, state, reason) {
    img.classList.toggle("sample-blur-pending", state === "pending");
    img.classList.toggle("sample-blur-obscured", state === "flagged" || state === "unknown");
    const blocked = ["pending","flagged","unknown"].includes(state);
    img.dataset.sampleBlurState = state;
    if (blocked) {
      if (!img.hasAttribute("data-sample-blur-tab")) {
        img.dataset.sampleBlurTab = img.getAttribute("tabindex") ?? "";
        img.dataset.sampleBlurTitle = img.getAttribute("title") ?? "";
        img.dataset.sampleBlurRole = img.getAttribute("role") ?? "";
      }
      img.tabIndex = 0; img.setAttribute("role","button");
      img.title = (reason || "Checking sample…") + " Click to reveal.";
    } else if (img.hasAttribute("data-sample-blur-tab")) {
      for (const [attr, prop] of [["tabindex","sampleBlurTab"],["title","sampleBlurTitle"],["role","sampleBlurRole"]]) {
        if (img.dataset[prop]) img.setAttribute(attr,img.dataset[prop]); else img.removeAttribute(attr);
        delete img.dataset[prop];
      }
    }
    label(img, state === "pending" ? "Checking sample…" : state === "flagged" ? "NSFW · click to reveal" :
      state === "unknown" ? (reason || "Not scanned") + " · click to reveal" : "");
  }
  function reveal(img, event) {
    if (!["pending","flagged","unknown"].includes(img.dataset.sampleBlurState)) return;
    event.preventDefault(); event.stopImmediatePropagation();
    revealed.add(key(img));
    for (const other of document.querySelectorAll("img[data-sample-blur-path]")) {
      if (key(other) === key(img)) { revisions.set(other, (revisions.get(other)||0)+1); display(other,"clear"); }
    }
    display(img,"clear");
  }
  async function scan(img, rev) {
    if (!live(img,rev) || !img.isConnected || revealed.has(key(img))) return;
    const k = key(img);
    let result = results.get(k);
    if (!result) {
      try {
        const q = new URLSearchParams({path:img.dataset.sampleBlurPath,t:token()});
        const r = await fetch("/api/sample_safety?" + q, {cache:"no-store"});
        if (!r.ok) throw Error("Scan unavailable");
        result = await r.json();
        if (["clear","flagged"].includes(result.state)) {
          results.set(k,result);
          if (results.size > 2048) results.delete(results.keys().next().value);
        }
      } catch (_) { result = {state:"unknown",reason:"Sample scan unavailable"}; }
    }
    if (live(img,rev) && !revealed.has(k)) display(img, ["clear","flagged"].includes(result.state) ? result.state : "unknown",result.reason);
  }
  const queue = img => {
    const rev = revisions.get(img);
    serial = serial.then(() => scan(img,rev)).catch(() => {});
  };
  const observer = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    for (const e of entries) if (e.isIntersecting) { observer.unobserve(e.target); queue(e.target); }
  }, {rootMargin:"150px"}) : null;
  function refresh(img) {
    observer?.unobserve(img);
    revisions.set(img,(revisions.get(img)||0)+1);
    if (!enabled || !img.dataset.sampleBlurPath || revealed.has(key(img))) { display(img,"clear"); return; }
    const cached = results.get(key(img));
    if (cached) { display(img,cached.state,cached.reason); return; }
    display(img,"pending");
    if (observer) observer.observe(img); else queue(img);
  }
  function mark(img,path,time) {
    if (!img) return;
    if (!bound.has(img)) {
      bound.add(img);
      img.addEventListener("click", e => reveal(img,e),true);
      img.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") reveal(img,e); },true);
    }
    if (path) { img.dataset.sampleBlurPath = path; img.dataset.sampleBlurTime = String(time ?? ""); }
    else { delete img.dataset.sampleBlurPath; delete img.dataset.sampleBlurTime; }
    refresh(img);
  }
  function setEnabled(value) {
    const changed = enabled !== Boolean(value); enabled = Boolean(value);
    const box = document.getElementById("blurNsfwSamples");
    if (box) box.checked = enabled;
    if (!changed) return;
    revealed.clear(); results.clear();
    for (const img of document.querySelectorAll("img[data-sample-blur-path]")) refresh(img);
  }
  function wire(onSave) {
    if (wired) return;
    wired = true; save = onSave;
    const box = document.getElementById("blurNsfwSamples"), status = document.getElementById("blurNsfwStatus");
    box.disabled = false; box.checked = enabled;
    box.addEventListener("change", async () => {
      const value = box.checked, previous = enabled;
      box.disabled = true;
      if (value) setEnabled(true);
      try { await save(value); setEnabled(value); status.textContent = "Local image detection · click a blurred sample to reveal it."; }
      catch(e) { setEnabled(previous); status.textContent = e.message || "Could not save preference."; }
      finally { box.disabled = false; }
    });
  }
  // Labels of off-DOM figures are appended once they enter the document.
  new MutationObserver(entries => {
    for (const entry of entries) for (const node of entry.addedNodes) {
      if (node.nodeType !== 1) continue;
      const imgs = [...(node.matches("img[data-sample-blur-path]") ? [node] : []), ...node.querySelectorAll("img[data-sample-blur-path]")];
      for (const img of imgs) if (img._sampleBlurLabel && !img._sampleBlurLabel.hidden && !img._sampleBlurLabel.isConnected) img.after(img._sampleBlurLabel);
    }
  }).observe(document.documentElement,{childList:true,subtree:true});
  return {mark,setEnabled,wire};
})();
