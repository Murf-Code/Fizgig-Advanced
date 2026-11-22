/* TagBench-inspired Caption Workbench appearance.
 * This module changes presentation only: no dataset, caption, or training state.
 * "Classic" uses the existing Fizgig stylesheet unmodified.
 */
(() => {
  "use strict";
  const root = document.getElementById("tab-captions");
  const controls = [...document.querySelectorAll("[data-caption-theme]")];
  if (!root || controls.length !== 2) return;

  const storageKey = "fizgig.caption-workbench.appearance";
  const choose = (name, save = true) => {
    const appearance = name === "classic" ? "classic" : "glass";
    root.classList.toggle("caption-theme-glass", appearance === "glass");
    document.body.classList.toggle("caption-glass-ui", appearance === "glass");
    for (const control of controls) {
      const active = control.dataset.captionTheme === appearance;
      control.classList.toggle("is-active", active);
      control.setAttribute("aria-pressed", String(active));
    }
    if (save) {
      try { localStorage.setItem(storageKey, appearance); }
      catch (_) { /* Private-browser storage can be unavailable. */ }
    }
  };
  let saved = null;
  try { saved = localStorage.getItem(storageKey); }
  catch (_) { /* The switch still works without storage. */ }
  // New installs showcase Glass; existing users can restore Classic at any time.
  choose(saved === "classic" ? "classic" : "glass", false);
  for (const control of controls) {
    control.addEventListener("click", () => choose(control.dataset.captionTheme));
  }
})();
