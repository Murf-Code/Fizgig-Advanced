/* Shared image-selection rules for the Tags and Waterfall views.
 * Keep selection independent of rendering/pagination so switching views never
 * replaces the active image with whichever thumbnail happens to be on page 1.
 */
const CaptionStudioSelection = (() => {
  const pathsOf = items => new Set(items.map(item => item.path));

  function pick(items, selectedPaths, activePath, anchorPath, targetPath, modifiers = {}, pageSize = 50) {
    const order = items.map(item => item.path);
    const target = order.indexOf(targetPath);
    if (target < 0)
      return {selected: [...selectedPaths], active: activePath || null, anchor: anchorPath || null, page: null};
    const selected = new Set(selectedPaths);
    const anchor = order.indexOf(anchorPath || activePath);
    let nextAnchor = targetPath;
    if (modifiers?.shiftKey && anchor >= 0) {
      selected.clear();
      for (const path of order.slice(Math.min(anchor, target), Math.max(anchor, target) + 1))
        selected.add(path);
      nextAnchor = order[anchor];
    } else if (modifiers?.ctrlKey || modifiers?.metaKey) {
      selected.has(targetPath) ? selected.delete(targetPath) : selected.add(targetPath);
    } else {
      selected.clear();
      selected.add(targetPath);
    }
    return {
      selected: [...selected],
      active: targetPath,
      anchor: nextAnchor,
      page: Math.floor(target / Math.max(1, pageSize)),
    };
  }

  function prune(items, selectedPaths, activePath, anchorPath) {
    const available = pathsOf(items);
    return {
      selected: [...selectedPaths].filter(path => available.has(path)),
      active: available.has(activePath) ? activePath : null,
      anchor: available.has(anchorPath) ? anchorPath : null,
    };
  }

  return Object.freeze({pick, prune});
})();
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioSelection;
