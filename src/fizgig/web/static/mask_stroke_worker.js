"use strict";
// Offscreen Remove stroke painter. The main UI owns the authoritative canvas
// and applies only small patch bitmaps acknowledged by this worker.
let canvas=null,context=null,activeSession=0;
function dirtyBox(current,a,b,width,height,radius){
  const pad=Math.ceil(radius)+3;
  const rect={l:Math.max(0,Math.floor(Math.min(a.x,b.x)-pad)),
    t:Math.max(0,Math.floor(Math.min(a.y,b.y)-pad)),
    r:Math.min(width,Math.ceil(Math.max(a.x,b.x)+pad)),
    b:Math.min(height,Math.ceil(Math.max(a.y,b.y)+pad))};
  if(!current)return rect;
  return {l:Math.min(current.l,rect.l),t:Math.min(current.t,rect.t),
    r:Math.max(current.r,rect.r),b:Math.max(current.b,rect.b)};
}
self.onmessage=event=>{
  const data=event.data||{};
  try{
    if(data.type==="init"){
      if(data.bitmap)try{
        // Use the exact saved mask state, including earlier red strokes.
        canvas=new OffscreenCanvas(data.width,data.height);
        context=canvas.getContext("2d");
        context.drawImage(data.bitmap,0,0);
      }finally{data.bitmap.close?.();}
      else{
        canvas=new OffscreenCanvas(data.width,data.height);
        context=canvas.getContext("2d");
      }
      activeSession=data.session;
      self.postMessage({type:"ready",session:activeSession});
      return;
    }
    if(data.type!=="stroke"||data.session!==activeSession||!canvas||!context)return;
    let dirty=null;
    for(const s of data.segments||[]){
      const width=Math.max(1,s.width);
      context.globalCompositeOperation=s.erase?"destination-out":"source-over";
      context.strokeStyle="#fff";context.fillStyle="#fff";
      context.lineCap="round";context.lineJoin="round";context.lineWidth=width;
      context.beginPath();
      if(s.ax===s.bx&&s.ay===s.by){
        context.arc(s.ax,s.ay,width/2,0,Math.PI*2);context.fill();
      }else{
        context.moveTo(s.ax,s.ay);context.lineTo(s.bx,s.by);context.stroke();
      }
      dirty=dirtyBox(dirty,{x:s.ax,y:s.ay},{x:s.bx,y:s.by},
        canvas.width,canvas.height,width/2);
    }
    context.globalCompositeOperation="source-over";
    if(!dirty||dirty.r<=dirty.l||dirty.b<=dirty.t){
      self.postMessage({type:"patch",session:activeSession,seq:data.seq,rect:null});
      return;
    }
    const patch=new OffscreenCanvas(dirty.r-dirty.l,dirty.b-dirty.t);
    patch.getContext("2d").drawImage(canvas,dirty.l,dirty.t,
      patch.width,patch.height,0,0,patch.width,patch.height);
    const bitmap=patch.transferToImageBitmap();
    self.postMessage({type:"patch",session:activeSession,seq:data.seq,rect:dirty,bitmap},[bitmap]);
  }catch(error){
    self.postMessage({type:"error",session:data.session,seq:data.seq,
      message:String(error&&error.message||error)});
  }
};
