"use strict";
// This script is requested ONLY on entering Constellation. Closing or
// switching views terminates its worker before it can deliver stale graphs.
importScripts("/caption_studio_viz.js");
self.onmessage=event=>{
  try{
    const {items,categories,triggerWord}=event.data||{};
    const model=self.CaptionStudioViz.tagbenchModel(items||[],categories||{},triggerWord);
    self.postMessage({model});
  }catch(error){
    self.postMessage({error:String(error?.message||error)});
  }
};
