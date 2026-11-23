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
  // Pixel budget is derived from the actual displayed row dimensions.
  // Portraits can reach the saved max-height setting even in a narrow rail;
  // landscapes usually only need 160 or 224px. Keep four stable cache tiers.
  function imageSizeForRow(railWidth,imageSize,pixelRatio=1,previewHeight=360){
    const width=Math.max(36,(Math.max(36,railWidth)-20)*.85-14);
    const ratio=imageSize?.[0]>0&&imageSize?.[1]>0?imageSize[1]/imageSize[0]:1;
    const cap=Math.max(160,Math.min(720,Number(previewHeight)||360))*.85;
    const height=Math.min(cap,width*ratio);
    const pixels=Math.max(width,height)*Math.max(1,Math.min(1.5,pixelRatio||1));
    return pixels<=160?160:pixels<=224?224:pixels<=320?320:384;
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
        if(image.src===url){
          this.images.delete(key);this.images.set(key,image);
          return;
        }
        // Same picture may be resized to a different rail resolution.
        this.images.delete(key);
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
  return {reconcile,prefetchRange,imageSizeForRow,ThumbPreloader};
})();
if(typeof module!=="undefined"&&module.exports)module.exports=CaptionThumbRail;
