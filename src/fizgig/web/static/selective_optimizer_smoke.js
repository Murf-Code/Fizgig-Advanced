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
  soap.addEventListener("click", async () => {
    const folder = document.getElementById("imageFolder")?.value.trim();
    if (!folder) { note.textContent = "Select the original 15-image Krea 2 dataset."; return; }
    const prompt = window.prompt("Fixed sample prompt (same for all four SOAP runs):", "");
    if (!prompt?.trim()) return;
    soap.disabled = true;
    try {
      const res = await fetch("/api/queue/soap_sweep", {
        method: "POST", headers: {"Content-Type": "application/json",
          "X-Fizgig-Token": document.querySelector('meta[name="fizgig-token"]').content},
        body: JSON.stringify({image_folder: folder, sample_prompt: prompt.trim()})});
      const data = await res.json();
      if (!res.ok || !data.ok) throw Error(data.error || (data.problems || []).join("; ") || "Could not queue SOAP");
      note.textContent = data.count + " five-epoch SOAP runs queued. Press Start queue. Report: " + data.output_dir;
    } catch (e) { note.textContent = e.message; }
    finally { soap.disabled = false; }
  });
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
