"use strict";
// Keep the Caption Workbench image rail cheap while Chromium performs native
// scrolling. Pure reconciliation is also exercised by Node regression checks.
const CaptionThumbRail=(()=>{
  function reconcile(host,mounted,start,end,create){
    for(const [index,row] of mounted){
      if(index<start||index>=end){row.remove();mounted.delete(index);}
    }
    // Work backwards so survivors stay attached (and keep their decoded img).
    // Only rows entering the window are inserted into the DOM.
    let next=null;
    for(let index=end-1;index>=start;index--){
      let row=mounted.get(index);
      if(!row){
        row=create(index);
        host.insertBefore(row,next);
        mounted.set(index,row);
      }
      next=row;
    }
  }
  function prefetchRange(start,end,total,direction,count=8){
    if(direction<0)return [Math.max(0,start-count),start];
    return [end,Math.min(total,end+count)];
  }
  class ThumbPreloader{
    constructor(limit=40){this.limit=limit;this.scope="";this.images=new Map();}
    reset(scope){
      if(this.scope===scope)return;
      this.scope=scope;this.images.clear();
    }
    clear(){this.images.clear();}
    forget(key){this.images.delete(key);}
    prefetch(key,url){
      if(this.images.has(key)){
        const image=this.images.get(key);
        this.images.delete(key);this.images.set(key,image);
        return;
      }
      // Keep decoded thumbnails bounded. The actual on-screen <img> remains
      // owned by the rail, not by this cache.
      const image=new Image();
      image.decoding="async";
      this.images.set(key,image);
      image.src=url;
      if(typeof image.decode==="function")
        void image.decode().catch(()=>{
          if(this.images.get(key)===image)this.images.delete(key);
        });
      while(this.images.size>this.limit)
        this.images.delete(this.images.keys().next().value);
    }
  }
  return {reconcile,prefetchRange,ThumbPreloader};
})();
if(typeof module!=="undefined"&&module.exports)module.exports=CaptionThumbRail;
