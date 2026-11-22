/* Caption Workbench's training-image context menu operations.
 * Pure helpers: share the existing validation picker and sample-prompt pairing
 * formats without adding a second storage format or touching source pictures.
 */
const CaptionStudioTrainingPicks = (() => {
  const norm = value => String(value || "").replace(/\\/g,"/").trim().toLocaleLowerCase();
  const names = value => {
    const raw=Array.isArray(value)?value:String(value||"").replace(/;/g,"\n").split(/\r?\n/);
    const used=new Set();
    return raw.map(v=>String(v).trim()).filter(v=>{
      if(!v||used.has(norm(v)))return false;
      used.add(norm(v));return true;
    });
  };
  const present = (picked,name) => picked.some(value=>norm(value)===norm(name));
  function toggleValidation(value, selected) {
    const picked=names(value), targets=names(selected);
    const remove=targets.length>0&&targets.every(name=>present(picked,name));
    return {
      remove,
      picked: remove ? picked.filter(n=>!present(targets,n))
        : [...picked,...targets.filter(n=>!present(picked,n))],
    };
  }
  const hash = text => {
    let h=0x811c9dc5;
    for(const b of new TextEncoder().encode(String(text||"").trim()))
      h=Math.imul(h^b,0x01000193)>>>0;
    return h.toString(16).padStart(8,"0");
  };
  const prompt = caption => String(caption||"").replace(/\s+/g," ").trim()
    .split(" --")[0].trim();
  const eligibleLines = raw => String(raw||"").split(/\r?\n/).map((text,rawIndex)=>{
    const clean=text.trim(),p=clean&&!clean.startsWith("#")?clean.split(" --")[0].trim():"";
    return {text,p,rawIndex};
  }).filter(row=>!!row.p);
  const hasPair = (pairs,name) => pairs.some(p=>norm(p?.image)===norm(name));
  function toggleSamples(rawPrompts,rawPairs,images) {
    const original=Array.isArray(rawPairs)?rawPairs.filter(x=>x&&x.image):
      (()=>{try{const v=JSON.parse(rawPairs||"[]");return Array.isArray(v)?v.filter(x=>x&&x.image):[];}
        catch(_){return [];}})();
    const selected=images.filter(it=>it&&it.name);
    const removing=selected.length>0&&selected.every(it=>hasPair(original,it.name));
    let pairs=[...original],lines=String(rawPrompts||"").split(/\r?\n/);
    const changed=[],skipped=[];
    if(removing) {
      const wanted=new Set(selected.map(it=>norm(it.name)));
      const candidates=pairs.filter(p=>wanted.has(norm(p.image)));
      pairs=pairs.filter(p=>!wanted.has(norm(p.image)));
      // Only remove a prompt if we created it and the prompt still matches.
      // Never delete an unrelated or user-edited preview instruction.
      const auto=candidates.filter(p=>p.fromWorkbench===true);
      const semantic=eligibleLines(lines.join("\n"));
      const removed=new Set();
      for(const p of auto) {
        const row=semantic[p.index];
        if(row&&hash(row.p)===p.hash)removed.add(row.rawIndex);
      }
      const removedIndices=[...removed].sort((a,b)=>b-a);
      for(const idx of removedIndices)lines.splice(idx,1);
      if(removedIndices.length){
        const removedSemantic=semantic.filter(r=>removed.has(r.rawIndex))
          .map(r=>semantic.indexOf(r));
        pairs=pairs.map(p=>({...p,index:Number.isInteger(p.index)?
          p.index-removedSemantic.filter(i=>i<p.index).length:p.index}));
      }
      changed.push(...candidates.map(p=>p.image));
    } else {
      let index=eligibleLines(lines.join("\n")).length;
      for(const it of selected) {
        if(hasPair(pairs,it.name))continue;
        const value=prompt(it.caption);
        if(!value){skipped.push(it.name);continue;}
        // Append only: existing manual sample prompts and pairings are retained.
        if(lines.length===1&&!lines[0].trim())lines=[];
        lines.push(value);
        pairs.push({index:index++,hash:hash(value),image:it.name,fromWorkbench:true});
        changed.push(it.name);
      }
    }
    return {removing,changed,skipped,prompts:lines.join("\n"),pairs};
  }
  return Object.freeze({names,present,toggleValidation,hash,eligibleLines,hasPair,toggleSamples});
})();
if(typeof module!=="undefined"&&module.exports)module.exports=CaptionStudioTrainingPicks;
