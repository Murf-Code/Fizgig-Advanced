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
