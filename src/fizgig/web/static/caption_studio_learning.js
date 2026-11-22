/* Small, dependency-free shared learning ring + loss sorting helpers.
 * No ring without a loss-watch verdict. Never infer "easy" from a low raw loss.
 */
const CaptionStudioLearning = (() => {
  const tones = Object.freeze({
    stuck: "Stuck", suspect: "Suspect", watch: "Watching",
    learning: "Learning", easy: "Easy", exhausted: "Fully mined",
    warmup: "Warming up", excluded: "Excluded"
  });
  const pathKey = path => String(path || "").replace(/\\/g, "/").replace(/\/+/g, "/").toLowerCase();
  const index = images => new Map((images || []).filter(im=>im&&im.path)
    .map(im=>[pathKey(im.path), im]));
  const forPath = (map, path) => map?.get(pathKey(path)) || null;
  function decorate(node, im, enabled) {
    if (!node.dataset.learningBaseTitle) node.dataset.learningBaseTitle=node.title || node.getAttribute("aria-label") || "";
    if (!node.dataset.learningBaseLabel) node.dataset.learningBaseLabel=node.getAttribute("aria-label") || "";
    node.removeAttribute("data-learning-status");
    node.title=node.dataset.learningBaseTitle;
    node.setAttribute("aria-label",node.dataset.learningBaseLabel);
    const verdict = enabled ? String(im?.verdict || "") : "";
    if (!(verdict in tones)) return;
    node.dataset.learningStatus=verdict;
    const val=Number(im?.recent),value=im?.recent!=null&&Number.isFinite(val)?" · relative loss "+val.toFixed(3):"";
    node.title=node.dataset.learningBaseTitle+" · "+tones[verdict]+value;
    node.setAttribute("aria-label",node.dataset.learningBaseLabel+", "+tones[verdict]);
  }
  function sort(items, map, mode) {
    const descending=mode==="loss-high";
    return [...items].sort((a,b)=>{
      const av=Number(forPath(map,a.path)?.recent),bv=Number(forPath(map,b.path)?.recent);
      const ah=forPath(map,a.path)?.recent!=null&&Number.isFinite(av);
      const bh=forPath(map,b.path)?.recent!=null&&Number.isFinite(bv);
      if (ah!==bh) return ah?-1:1;
      if (ah&&av!==bv) return descending?bv-av:av-bv;
      return a.name.localeCompare(b.name);
    });
  }
  return Object.freeze({pathKey,index,forPath,decorate,sort,tones});
})();
if(typeof module!=="undefined"&&module.exports)module.exports=CaptionStudioLearning;
