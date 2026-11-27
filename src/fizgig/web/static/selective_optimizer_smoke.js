(() => {
  const button = document.getElementById("btnSelectiveOptimizerSmoke");
  if (!button) return;
  const note = document.createElement("span");
  note.className = "field-hint";
  note.setAttribute("role", "status");
  button.parentNode.insertAdjacentElement("afterend", note);
  const match = document.createElement("button");
  match.type = "button";
  match.className = button.className;
  match.textContent = "Test Mode · AdamW Match";
  button.insertAdjacentElement("afterend", match);
  const soap = document.createElement("button");
  soap.type = "button";
  soap.className = button.className;
  soap.textContent = "Test Mode · SOAP 5 epochs";
  match.insertAdjacentElement("afterend", soap);
  const compat = document.createElement("button");
  compat.type = "button";
  compat.className = button.className;
  compat.textContent = "Test Mode · Selective optimizer compatibility";
  soap.insertAdjacentElement("afterend", compat);
  const effective = make("button", button.className, "Test Mode · Effective Selective");
  effective.type = "button"; compat.insertAdjacentElement("afterend", effective);
  const optimizerChoices = [
    ["ademamix8bit", "AdEMAMix 8-bit", "0.00025"],
    ["pagedademamix8bit", "Paged AdEMAMix 8-bit", "0.00025"],
    ["lion8bit", "Lion 8-bit", "0.000025"],
    ["polora", "PoLoRA", "0.02 · magnitude 3"],
    ["lora_tsd", "LoRA-TSD", "0.005 · magnitude 3"],
    ["prodigy", "Prodigy", "Native · multiplier 1"],
    ["automagic3", "Automagic v3", "Native · starts at 0.000001"],
    ["automagic4", "Automagic v4", "Native · starts at 0.000001"]
  ];
  function make(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  }
  function currentPrompt() {
    // app.js owns the training form. Read its current text, never modify it.
    try {
      if (typeof S !== "undefined" && typeof S.values?.SAMPLE_PROMPT === "string")
        return S.values.SAMPLE_PROMPT;
    } catch (_) {}
    return "";
  }
  let researchDialog = null;
  function openResearch(mode, trigger) {
    const folder = document.getElementById("imageFolder")?.value.trim();
    if (!folder) { note.textContent = "Select the original 15-image Krea 2 dataset."; return; }
    if (researchDialog) return;
    const selective = mode === "selective";
    const geometry = mode === "effective";
    const dlg = make("dialog", "modal-window");
    researchDialog = dlg;
    dlg.setAttribute("aria-labelledby", "researchTestTitle");
    Object.assign(dlg.style, {width:"620px", margin:"auto", padding:"0",
      borderRadius:"12px", color:"var(--text-primary)", maxHeight:"85vh"});
    const sheet = make("style");
    sheet.textContent = "dialog.research-test::backdrop {background:color-mix(in srgb,var(--shadow,#000) 75%,transparent)}";
    dlg.classList.add("research-test");
    dlg.append(sheet);
    const header = make("div", "modal-header");
    const title = make("span", "", geometry ? "Effective-weight Selective comparison" : selective ? "Selective optimizer comparison" : "SOAP learning-rate comparison");
    title.id = "researchTestTitle";
    const close = make("button", "btn", "✕"); close.type = "button"; close.setAttribute("aria-label","Close");
    header.append(title,close);
    const body = make("div", "modal-body");
    body.append(make("p", "", geometry ? "Compare unrestricted, existing Selective and effective-weight Selective for PoLoRA and LoRA-TSD. No LR compensation." : selective
      ? "Each selected optimizer runs twice: Global Replay, then Selective Global Replay. External LR compensation stays off."
      : "Compare SOAP-512 and Compact SOAP-32 at two learning rates each."));
    const checks = [];
    if (selective) {
      const controls = make("div");
      Object.assign(controls.style,{display:"flex",gap:"8px",margin:"12px 0"});
      const all = make("button","btn","Select all"), none = make("button","btn","Clear all");
      all.type = none.type = "button";
      controls.append(all,none); body.append(controls);
      const list = make("div");
      Object.assign(list.style,{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))",gap:"8px"});
      for (const [id,name,rate] of optimizerChoices) {
        const item = make("label");
        Object.assign(item.style,{display:"flex",alignItems:"center",gap:"10px",padding:"10px",
          border:"1px solid var(--border)",borderRadius:"8px",background:"var(--bg-surface)",cursor:"pointer"});
        const check = make("input");
        check.type = "checkbox"; check.checked = true; check.value = id;
        Object.assign(check.style,{width:"18px",height:"18px",flex:"none",accentColor:"var(--accent)"});
        const desc = make("div");
        desc.append(make("strong","",name),make("div","field-hint",rate));
        item.append(check,desc); list.append(item); checks.push(check);
      }
      body.append(list);
      all.onclick = () => { checks.forEach(c => c.checked = true); update(); };
      none.onclick = () => { checks.forEach(c => c.checked = false); update(); };
      checks.forEach(c => c.addEventListener("change",update));
    } else if (geometry) {
      body.append(make("p","field-hint","PoLoRA 0.02 · LoRA-TSD 0.005 · three matched arms each. Unreliable effective scores remain active."));
    } else {
      body.append(make("p","field-hint","SOAP-512: 0.00025 and 0.00040 · Compact SOAP-32: 0.00025 and 0.00050"));
    }
    const label = make("label","","Sample prompts");
    label.htmlFor = "researchSamplePrompt";
    label.style.display = "block"; label.style.marginTop = "16px";
    const prompt = make("textarea"); prompt.id = "researchSamplePrompt"; prompt.rows = 3;
    prompt.value = currentPrompt();
    prompt.placeholder = "Enter a representative prompt (one per line)";
    Object.assign(prompt.style,{width:"100%",boxSizing:"border-box",marginTop:"6px",resize:"vertical",
      background:"var(--bg-inset)",color:"var(--text-primary)",border:"1px solid var(--border)",borderRadius:"8px",padding:"10px"});
    const help = make("div","field-hint","Same prompts and seed for every run. Samples before training and after every epoch.");
    const summary = make("p",""); summary.setAttribute("aria-live","polite");
    const error = make("div","field-hint"); error.setAttribute("role","alert");
    error.style.color = "var(--danger)"; error.style.whiteSpace = "pre-wrap";
    body.append(label,prompt,help,summary,error);
    const footer = make("div","modal-footer");
    const cancel = make("button","btn","Cancel"), submit = make("button","btn btn-primary","Queue tests");
    cancel.type = submit.type = "button"; footer.append(cancel,submit);
    dlg.append(header,body,footer);
    let busy = false;
    function update() {
      const count = geometry ? 6 : selective ? checks.filter(c => c.checked).length * 2 : 4;
      const epochs = geometry ? 6 : 5;
      summary.textContent = count + " runs · " + epochs + " epochs each · " + count * epochs * 13 + " training updates";
      submit.textContent = busy ? "Queueing…" : "Queue " + count + " tests";
      submit.disabled = busy || count === 0 || !prompt.value.trim() || prompt.value.trim().length > 4000;
    }
    function dismiss() { if (!busy) dlg.close(); }
    close.onclick = cancel.onclick = dismiss;
    dlg.addEventListener("cancel",e => { if (busy) e.preventDefault(); });
    dlg.addEventListener("close",() => { dlg.remove(); researchDialog = null; trigger.focus(); });
    prompt.addEventListener("input",update);
    submit.onclick = async () => {
      if (submit.disabled) return;
      busy = true; close.disabled = cancel.disabled = true; update(); error.textContent = "";
      try {
        const payload = {image_folder:folder,sample_prompt:prompt.value.trim()};
        if (selective) payload.optimizers = checks.filter(c => c.checked).map(c => c.value);
        const res = await fetch(geometry ? "/api/queue/selective_effective" : selective ? "/api/queue/selective_compat" : "/api/queue/soap_sweep", {
          method:"POST",headers:{"Content-Type":"application/json",
            "X-Fizgig-Token":document.querySelector('meta[name="fizgig-token"]').content},
          body:JSON.stringify(payload)});
        const data = await res.json();
        if (!res.ok || !data.ok) throw Error(data.error || (data.problems || []).join("; ") || "Could not queue tests");
        note.textContent = data.count + (geometry ? " six-epoch" : " five-epoch") + " runs queued. Press Start queue. Report: " + data.output_dir;
        busy = false; dlg.close();
      } catch (e) {
        busy = false; close.disabled = cancel.disabled = false; error.textContent = e.message; update();
      }
    };
    document.body.append(dlg);
    update();
    dlg.showModal();
    prompt.focus();
  }
  soap.addEventListener("click",() => openResearch("soap",soap));
  compat.addEventListener("click",() => openResearch("selective",compat));
  effective.addEventListener("click",() => openResearch("effective",effective));
  const skip = document.createElement("button");
  skip.type = "button";
  skip.className = button.className;
  skip.style.display = "none";
  match.insertAdjacentElement("afterend", skip);
  const cancelSkip = document.createElement("button");
  cancelSkip.type = "button";
  cancelSkip.className = button.className;
  cancelSkip.textContent = "Cancel";
  skip.insertAdjacentElement("afterend", cancelSkip);
  let activeId = null, armedId = null, armTimer = null, skipBusy = false;
  function drawSkip() {
    skip.textContent = armedId ? "Confirm termination" : "Terminate test & continue";
    skip.style.display = activeId ? "" : "none";
    cancelSkip.style.display = armedId ? "" : "none";
    skip.disabled = skipBusy || !activeId;
  }
  function disarm() {
    clearTimeout(armTimer);
    armedId = null;
    drawSkip();
  }
  cancelSkip.addEventListener("click", disarm);
  document.addEventListener("keydown", e => { if (e.key === "Escape") disarm(); });
  async function updateSkip() {
    let id = null;
    try {
      const res = await fetch("/api/run", {cache: "no-store"});
      if (!res.ok) throw Error("status unavailable");
      const run = await res.json();
      const v = run.values || {};
      id = run.running && ["starting", "running"].includes(run.state) &&
        (v.__optimizer_match || v.__selective_optimizer_smoke) ? run.history_id : null;
    } catch (_) {}
    if (id !== activeId) disarm();
    activeId = id;
    drawSkip();
  }
  skip.addEventListener("click", async () => {
    const id = activeId;
    if (!id || skipBusy) return;
    if (armedId !== id) {
      armedId = id;
      armTimer = setTimeout(disarm, 5000);
      drawSkip();
      return;
    }
    disarm();
    skipBusy = true;
    drawSkip();
    try {
      const res = await fetch("/api/queue/skip_optimizer_test", {
        method: "POST", headers: {"Content-Type": "application/json",
          "X-Fizgig-Token": document.querySelector('meta[name="fizgig-token"]').content},
        body: JSON.stringify({history_id: id})});
      const data = await res.json();
      if (!res.ok || !data.ok) throw Error("Test already ended or could not be terminated.");
      note.textContent = "Test termination requested. The queue advances after the process exits.";
    } catch (e) { note.textContent = e.message; }
    finally { skipBusy = false; await updateSkip(); }
  });
  drawSkip();
  updateSkip();
  setInterval(updateSkip, 3000);
  let previousGroup = "";
  match.addEventListener("click", async () => {
    const folder = document.getElementById("imageFolder")?.value.trim();
    if (!folder) { note.textContent = "Select the original 15-image Krea 2 dataset."; return; }
    const previous = window.prompt("Previous AdamW match output folder (blank for initial measurement):", previousGroup);
    if (previous === null) return;
    match.disabled = true;
    try {
      const res = await fetch("/api/queue/optimizer_match", {
        method: "POST", headers: {"Content-Type": "application/json",
          "X-Fizgig-Token": document.querySelector('meta[name="fizgig-token"]').content},
        body: JSON.stringify({image_folder: folder, previous_group: previous.trim()})});
      const data = await res.json();
      if (!res.ok || !data.ok) throw Error(data.error || (data.problems || []).join("; ") || "Could not queue calibration");
      previousGroup = data.output_dir;
      note.textContent = data.count + " calibration runs queued. Press Start queue. Report: " + data.output_dir;
    } catch (e) { note.textContent = e.message; }
    finally { match.disabled = false; }
  });
  button.addEventListener("click", async () => {
    const folder = document.getElementById("imageFolder")?.value.trim();
    if (!folder) { note.textContent = "Select the small 15-image Krea 2 dataset first."; return; }
    button.disabled = true;
    try {
      const res = await fetch("/api/queue/selective_optimizer_smoke", {
        method: "POST", headers: {"Content-Type": "application/json",
          "X-Fizgig-Token": document.querySelector('meta[name="fizgig-token"]').content},
        body: JSON.stringify({image_folder: folder})});
      const data = await res.json();
      if (!res.ok || !data.ok) throw Error(data.error || (data.problems || []).join("; ") || "Could not queue tests");
      note.textContent = data.count + " one-epoch tests queued. Press Start queue. Report: " + data.output_dir;
    } catch (e) { note.textContent = e.message; }
    finally { button.disabled = false; }
  });
})();
