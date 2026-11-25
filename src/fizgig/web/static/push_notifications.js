(() => {
  "use strict";
  const el = id => document.getElementById(id);
  const card = el("pushCard");
  if (!card) return;
  const status = el("pushStatus"), inputs = ["pushEnabled", "pushTopic", "pushToken", "pushNames", "pushForget"];
  let loaded = false, busy = false;
  async function api(method, suffix = "", data) {
    const options = {method, headers: {"X-Fizgig-Token": document.querySelector('meta[name="fizgig-token"]').content}};
    if (data !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(data);
    }
    const response = await fetch("/api/notifications" + suffix, options);
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || "Notification request failed.");
    return result;
  }
  function render(cfg) {
    el("pushEnabled").checked = cfg.enabled;
    el("pushTopic").value = cfg.topic;
    el("pushToken").value = "";
    el("pushToken").placeholder = cfg.token_saved ? "Token saved (leave blank to keep)" : "Optional access token";
    el("pushNames").checked = cfg.names;
    el("pushForget").checked = false;
    const list = el("pushEvents");
    list.replaceChildren();
    for (const [key, label] of Object.entries(cfg.labels)) {
      const row = document.createElement("label");
      row.className = "field-row";
      const check = document.createElement("input");
      check.type = "checkbox";
      check.dataset.pushEvent = key;
      check.checked = cfg.events[key];
      row.append(check, document.createTextNode(label));
      list.append(row);
    }
    loaded = true;
  }
  function lock(value) {
    busy = value;
    for (const id of [...inputs, "pushSave", "pushTest", "pushGenerate", "pushReload"]) el(id).disabled = value;
    for (const check of card.querySelectorAll("[data-push-event]")) check.disabled = value;
    if (!value && !loaded) {
      el("pushSave").disabled = true;
      el("pushTest").disabled = true;
    }
  }
  async function load() {
    if (busy) return;
    lock(true);
    try { render(await api("GET")); status.textContent = ""; }
    catch (e) { status.textContent = e.message + " Try Reload."; }
    finally { lock(false); }
  }
  el("pushReload").addEventListener("click", load);
  el("pushGenerate").addEventListener("click", () => {
    const bytes = crypto.getRandomValues(new Uint8Array(18));
    el("pushTopic").value = "fizgig-" + Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
  });
  el("pushSave").addEventListener("click", async () => {
    if (!loaded || busy) return;
    const data = {enabled: el("pushEnabled").checked, topic: el("pushTopic").value,
                  names: el("pushNames").checked, events: {}};
    if (el("pushForget").checked) data.token = "";
    else if (el("pushToken").value.trim()) data.token = el("pushToken").value.trim();
    for (const check of card.querySelectorAll("[data-push-event]")) data.events[check.dataset.pushEvent] = check.checked;
    lock(true);
    try { render(await api("POST", "", data)); status.textContent = "Saved."; }
    catch (e) { status.textContent = e.message; }
    finally { lock(false); }
  });
  el("pushTest").addEventListener("click", async () => {
    if (!loaded || busy) return;
    lock(true);
    try { await api("POST", "/test", {}); status.textContent = "Test accepted by ntfy. Check your subscribed device."; }
    catch (e) { status.textContent = e.message; }
    finally { lock(false); }
  });
  lock(false);
  // Fetch only when Preferences is actually visible (and after login).
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting) && !loaded && !busy) load();
    });
    observer.observe(card);
  } else {
    load();
  }
})();
