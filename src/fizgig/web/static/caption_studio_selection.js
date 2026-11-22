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

  // The Tags rail loads thumbnails progressively and does not share the
  // Cards/Table page size. Keep a far-away active image in the mounted range.
  function initialMountCount(total, activeIndex, remembered=0, batch=64) {
    const size=Math.max(1,Math.trunc(batch)||64);
    const active=Math.max(0,Math.trunc(activeIndex)||0);
    const keep=Math.max(0,Math.trunc(remembered)||0);
    return Math.min(Math.max(0,Math.trunc(total)||0),
      Math.max(size,keep,Math.ceil((active+1)/size)*size));
  }
  function nextMountCount(mounted,total,batch=64) {
    return Math.min(Math.max(0,Math.trunc(total)||0),
      Math.max(0,Math.trunc(mounted)||0)+Math.max(1,Math.trunc(batch)||64));
  }

  function captionTargets(items, selectedPaths, reviewedPaths, overwrite=false) {
    const selected=new Set(selectedPaths||[]);
    const reviewed=new Set(reviewedPaths||[]);
    const eligible=(items||[]).filter(it=>selected.has(it.path)&&it.size&&
      !reviewed.has(it.path)&&/\.(png|jpe?g|webp|bmp|avif|jxl|mp4)$/i.test(it.name));
    return overwrite?eligible:eligible.filter(it=>it.caption===null);
  }
  return Object.freeze({pick, prune, initialMountCount, nextMountCount, captionTargets});
})();
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioSelection;
