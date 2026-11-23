/* Native Qt image adapters. Injected only into the optional desktop Qt shell.
 * No remote API writes: all edits go through the existing Workbench actions.
 * Read-only sources: Caption Workbench, Samples gallery and Image Prep Look Filter.
 */
(() => {
  "use strict";
  if (window.FizgigNativeImages) return;
  const empty = {tab:"", key:"", items:[], selected:[], active:""};
  const tab = () => (document.querySelector(".tab-panel.active")?.id||"").replace(/^tab-/,"");
  const picture = name => /\.(png|jpe?g|webp|bmp|gif|tiff?|avif|jxl)$/i.test(String(name||""));
  const file = item => ({
    path:String(item.path||""),
    name:String(item.name||""),
    version:String(item.time??item.version??"")
  });
  const selectionSignature = (paths, active) => {
    // Stable lightweight fingerprint; don't transfer an entire 10k selection
    // across the Qt bridge on each 800ms status poll.
    let hash=2166136261;
    for (const path of paths) for(let i=0;i<path.length;i++){
      hash=Math.imul(hash^path.charCodeAt(i),16777619);
    }
    return paths.length+":"+hash+":"+String(active||"");
  };
  let sideSelection = {samples:new Set(),prep:new Set()};
  const sourceCache={samples:{ref:null,items:[]},prep:{ref:null,items:[]}};
  function pictureItems(kind,raw){
    const cache=sourceCache[kind];
    if(cache.ref!==raw){
      cache.ref=raw;
      cache.items=raw.filter(x=>picture(x.name)&&x.path);
      const allowed=new Set(cache.items.map(x=>x.path));
      sideSelection[kind]=new Set([...sideSelection[kind]].filter(p=>allowed.has(p)));
    }
    return cache.items;
  }
  function source(){
    if(typeof S==="undefined")return empty;
    const kind=tab();
    if(kind==="captions"&&Array.isArray(S.cap?.items)){
      // Constellation/Waterfall own the entire visual stage. Give them the
      // whole window and do not poll/decode the dormant native image rail.
      if((S.cap.studioExploreMode||"list")!=="list"||S.cap.imageEditing?.inline)
        return {tab:kind,key:"captions|explore|"+(S.cap.studioExploreMode||"list")+
          "|"+Boolean(S.cap.imageEditing?.inline),items:[],selected:[],active:""};
      const items=S.cap.nativeVisibleItems||S.cap.items;
      const selected=Array.from(S.cap.studioSelection||[]);
      return {
        tab:kind, items, selected, active:String(S.cap.studioPath||""),
        key:["captions",S.imageFolder||"",S.cap.nativeDatasetGeneration||0,
          S.cap.studioRevision||0,S.cap.nativeVisibleKey||""].join("|")
      };
    }
    if(kind==="samples"&&Array.isArray(S.gal?.items)){
      const items=pictureItems("samples",S.gal.items);
      return {
        tab:kind, items, selected:[...sideSelection.samples],
        active:[...sideSelection.samples][0]||"",
        key:["samples",S.gal.run||"",S.nativeGalleryGeneration||0].join("|")
      };
    }
    if(kind==="prep"&&S.prep?.look?.state==="done"&&
       S.prep.look.folder===S.imageFolder&&Array.isArray(S.prep.look.rows)){
      const items=pictureItems("prep",S.prep.look.rows);
      return {
        tab:kind, items, selected:[...sideSelection.prep],
        active:[...sideSelection.prep][0]||"",
        key:["prep",S.imageFolder||"",S.prep.look.rows.length,
          S.prep.nativeLookGeneration||0].join("|")
      };
    }
    return {tab:kind,key:kind,items:[],selected:[],active:""};
  }
  function status(){
    const src=source();
    return {tab:src.tab,key:src.key,count:src.items.length,
      selection:selectionSignature(src.selected,src.active)};
  }
  function snapshot(){
    const src=source();
    return {tab:src.tab,key:src.key,selected:src.selected,active:src.active,
      items:src.items.map(file)};
  }
  function selection(){
    const src=source();
    return {tab:src.tab,key:src.key,selected:src.selected,active:src.active};
  }
  function select(paths,active){
    const src=source();
    const visible=new Set(src.items.map(x=>x.path));
    const chosen=(Array.isArray(paths)?paths:[]).filter(p=>visible.has(p));
    const current=chosen.includes(active)?active:(chosen.at(-1)||"");
    if(src.tab==="captions"){
      // Mutate the original Set to preserve handlers captured during render.
      const selected=S.cap.studioSelection||(S.cap.studioSelection=new Set());
      selected.clear();
      for(const path of chosen)selected.add(path);
      if(current){
        S.cap.studioPath=current;
        S.cap.studioSelectionAnchor=current;
      }
      if(typeof renderCaptions==="function")renderCaptions();
      return true;
    }
    if(src.tab==="samples"||src.tab==="prep"){
      sideSelection[src.tab]=new Set(chosen);
      return true;
    }
    return false;
  }
  function action(name,path){
    if(tab()!=="captions")return false;
    if(typeof S==="undefined"||!S.cap?.nativeImageActions)return false;
    const fn=S.cap.nativeImageActions[name];
    if(typeof fn!=="function")return false;
    // An existing Fizgig handler validates review locks, drafts and removals.
    try {
      const result=fn(path);
      if(result?.catch)result.catch(error=>console.error("[native image action]",error));
      return true;
    }catch(error){console.error("[native image action]",error);return false;}
  }
  window.FizgigNativeImages={status,snapshot,selection,select,action};
  window.__FizgigQtNativeRail=true;
})();
