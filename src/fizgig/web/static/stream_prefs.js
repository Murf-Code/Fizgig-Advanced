(() => {
  const card = document.getElementById("streamPinCard"), box = document.getElementById("streamPin"),
        status = document.getElementById("streamPinStatus");
  if (!card || !box) return;
  const headers = {"X-Fizgig-Token": document.querySelector('meta[name="fizgig-token"]').content,
                   "Content-Type": "application/json"};
  let busy = false;
  async function load() {
    if (busy) return;
    busy = true;
    try {
      const response = await fetch("/api/prefs", {headers});
      if (!response.ok) throw Error("Log in to load the setting.");
      const data = await response.json();
      box.checked = String(data.settings.pin_block_swap) === "1";
      box.disabled = false;
      status.textContent = "";
    } catch (e) { box.disabled = true; status.textContent = e.message; }
    finally { busy = false; }
  }
  box.addEventListener("change", async () => {
    busy = true; box.disabled = true;
    try {
      const response = await fetch("/api/prefs", {method:"POST", headers,
        body:JSON.stringify({changes:{pin_block_swap:box.checked ? "1" : "0"}})});
      const data = await response.json();
      if (!response.ok || data.error) throw Error(data.error || "Save failed");
      status.textContent = "Saved for the next run.";
    } catch(e) { box.checked = !box.checked; status.textContent = e.message; }
    finally { busy = false; box.disabled = false; }
  });
  if ("IntersectionObserver" in window) new IntersectionObserver(entries => {
    if (entries.some(e => e.isIntersecting)) load();
  }).observe(card);
  else load();
})();
