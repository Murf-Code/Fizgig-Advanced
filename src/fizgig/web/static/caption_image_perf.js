"use strict";
// Shared performance helpers for the Image Prep editor and Caption Workbench.
// Keep the image/mask source and native LaMa endpoints unchanged.
const MaskEditorPerf=(()=>{
  const TILE=128;
  const MAX_PREVIEW_BYTES=32*1024*1024;
  const PREVIEW_PIXELS=1_500_000;

  function tiles(mask){
    return {kind:"tiles",width:mask.width,height:mask.height,items:new Map()};
  }
  function capture(entry,mask,x,y,radius){
    if(!entry||entry.kind!=="tiles")return;
    const left=Math.max(0,Math.floor((x-radius-3)/TILE));
    const top=Math.max(0,Math.floor((y-radius-3)/TILE));
    const right=Math.min(Math.ceil(mask.width/TILE)-1,Math.floor((x+radius+3)/TILE));
    const bottom=Math.min(Math.ceil(mask.height/TILE)-1,Math.floor((y+radius+3)/TILE));
    const ctx=mask.getContext("2d",{willReadFrequently:true});
    for(let ty=top;ty<=bottom;ty++)for(let tx=left;tx<=right;tx++){
      const key=ty*Math.ceil(mask.width/TILE)+tx;
      if(entry.items.has(key))continue;
      const px=tx*TILE,py=ty*TILE;
      const w=Math.min(TILE,mask.width-px),h=Math.min(TILE,mask.height-py);
      if(w>0&&h>0)entry.items.set(key,{x:px,y:py,data:ctx.getImageData(px,py,w,h)});
    }
  }
  function restore(entry,mask){
    if(entry.kind!=="tiles")throw new Error("Not a tile history step");
    if(entry.width!==mask.width||entry.height!==mask.height)
      throw new Error("Mask dimensions changed; cannot replay tile history");
    const ctx=mask.getContext("2d",{willReadFrequently:true});
    const other=tiles(mask);
    let rect=null;
    // All inverse pixels are read first, then the old pixels are written.
    for(const [key,tile] of entry.items){
      const width=tile.data.width,height=tile.data.height;
      other.items.set(key,{x:tile.x,y:tile.y,
        data:ctx.getImageData(tile.x,tile.y,width,height)});
    }
    for(const tile of entry.items.values()){
      ctx.putImageData(tile.data,tile.x,tile.y);
      const box={l:tile.x,t:tile.y,
        r:tile.x+tile.data.width,b:tile.y+tile.data.height};
      rect=rect?{l:Math.min(rect.l,box.l),t:Math.min(rect.t,box.t),
        r:Math.max(rect.r,box.r),b:Math.max(rect.b,box.b)}:box;
    }
    return {inverse:other,rect};
  }
  function historyBytes(entry){
    if(entry?.kind==="tiles"){
      let bytes=0;
      for(const tile of entry.items.values())bytes+=tile.data.data.byteLength;
      return bytes;
    }
    return entry?.step?0:Math.max(0,(entry?.width||0)*(entry?.height||0)*4);
  }
  function trimHistory(history,maxEntries=20,limit=128*1024*1024){
    let bytes=history.reduce((sum,item)=>sum+historyBytes(item),0);
    while(history.length>1&&(history.length>maxEntries||bytes>limit)){
      bytes-=historyBytes(history.shift());
    }
    return Math.max(0,bytes);
  }
  function pngDataURL(canvas){
    // toBlob performs compression asynchronously in Chromium; FileReader then
    // converts to the JSON-compatible payload expected by Fizgig's API.
    if(!canvas||typeof canvas.toBlob!=="function"||typeof FileReader==="undefined")
      return Promise.resolve(canvas.toDataURL("image/png"));
    return new Promise((resolve,reject)=>{
      try{
        canvas.toBlob(blob=>{
          if(!blob){reject(new Error("PNG encoding returned an empty image"));return;}
          const reader=new FileReader();
          reader.onerror=()=>reject(reader.error||new Error("Could not encode PNG"));
          reader.onload=()=>resolve(reader.result);
          reader.readAsDataURL(blob);
        },"image/png");
      }catch(error){reject(error);}
    });
  }
  class PreviewCache{
    constructor(maxBytes=MAX_PREVIEW_BYTES){
      this.maxBytes=maxBytes;this.map=new Map();this.bytes=0;
    }
    get(key){
      const entry=this.map.get(key);
      if(!entry)return null;
      this.map.delete(key);this.map.set(key,entry);
      return entry.bitmap;
    }
    put(key,bitmap){
      if(!bitmap)return;
      if(this.map.has(key))this.drop(key);
      const bytes=bitmap.width*bitmap.height*4;
      if(bytes>this.maxBytes){bitmap.close?.();return;}
      this.map.set(key,{bitmap,bytes});this.bytes+=bytes;
      while(this.bytes>this.maxBytes&&this.map.size)this.drop(this.map.keys().next().value);
    }
    drop(key){
      const entry=this.map.get(key);
      if(!entry)return;
      this.map.delete(key);this.bytes-=entry.bytes;
      entry.bitmap.close?.();
    }
    clear(){for(const key of [...this.map.keys()])this.drop(key);}
  }
  function previewSize(width,height){
    const pixels=width*height;
    if(pixels<=PREVIEW_PIXELS)return null;
    const scale=Math.sqrt(PREVIEW_PIXELS/pixels);
    return {resizeWidth:Math.max(1,Math.floor(width*scale)),
      resizeHeight:Math.max(1,Math.floor(height*scale)),
      resizeQuality:"high"};
  }
  return {tiles,capture,restore,historyBytes,trimHistory,
    pngDataURL,PreviewCache,previewSize,TILE};
})();
if(typeof module!=="undefined"&&module.exports)module.exports=MaskEditorPerf;
