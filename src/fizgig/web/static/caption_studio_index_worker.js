"use strict";
importScripts("/caption_studio_tags.js");
self.onmessage=event=>{
  try{
    const {rows,delimiter}=event.data;
    const index=self.CaptionStudioTags.datasetIndex(delimiter);
    index.add(rows);
    self.postMessage(index.finish());
  }catch(error){self.postMessage({error:String(error?.message||error)});}
};
