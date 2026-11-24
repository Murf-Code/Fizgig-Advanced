"use strict";
// This script is requested ONLY on entering Constellation. Closing or
// switching views terminates its worker before it can deliver stale graphs.
importScripts("/caption_studio_viz.js");
self.onmessage=event=>{
  try{
    const {items,categories,lossGroups}=event.data||{};
    const model=lossGroups?self.CaptionStudioViz.summarizeAtlasLoss(items||[],lossGroups):
      self.CaptionStudioViz.tagbenchModel(items||[],categories||{});
    self.postMessage({model});
  }catch(error){
    self.postMessage({error:String(error?.message||error)});
  }
};
