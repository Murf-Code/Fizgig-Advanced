/* TagBench-inspired frequency waterfall and interactive tag constellation.
 * Analysis uses local captions only; no API or GPU required.
 */
const CaptionStudioViz = (() => {
  const lower = tag => String(tag).trim().toLocaleLowerCase();
  // Stable, theme-derived tag hues. Category assignments can override these.
  const tagPalette = ["--accent","--data-teal","--data-blue","--special",
    "--data-green","--data-orange","--accent-hover","--special-soft"];
  let generatedTagColors=new Map();
  const setTagbenchPalette=palette=>{
    generatedTagColors=palette instanceof Map?palette:new Map();
  };
  function colorForTag(tag, categories) {
    const key=lower(tag);
    if(/^#[0-9a-f]{6}$/i.test(categories?.tagColors?.[key]||""))
      return categories.tagColors[key];
    const category=categories?.tags?.[key];
    if(category&&categories?.colors?.[category])return categories.colors[category];
    if(categories?.generatedColors?.has(key))return categories.generatedColors.get(key);
    if(generatedTagColors.has(key))return generatedTagColors.get(key);
    let hash=2166136261;
    for(const character of key)hash=Math.imul(hash^character.codePointAt(0),16777619)>>>0;
    return "var("+tagPalette[hash%tagPalette.length]+")";
  }


  function buildGraph(index, captions, maxTags = 48) {
    const top = index.slice(0, Math.min(maxTags, index.length));
    const selected = new Set(top.map(x => x[0]));
    const links = new Map();
    for (const tags of captions) {
      const present = [...new Set(tags.map(lower).filter(tag => selected.has(tag)))];
      for (let a = 0; a < present.length; a++)
        for (let b = a + 1; b < present.length; b++) {
          const pair = [present[a], present[b]].sort();
          const key = pair.join("\u0000");
          links.set(key, (links.get(key) || 0) + 1);
        }
    }
    // Rank links by how much more often two tags occur together than chance.
    // Raw co-occurrence counts would connect virtually every common tag.
    const frequencies=new Map(top.map(([tag,count])=>[lower(tag),count]));
    const total=Math.max(1,captions.length);
    const edges = [...links].map(([pair,count]) => {
      const [a,b] = pair.split("\u0000");
      const expected=(frequencies.get(a)||0)*(frequencies.get(b)||0)/total;
      return {a,b,count,strength:count/Math.max(1,expected)};
    }).filter(x=>x.count>=2&&x.strength>1.05)
      .sort((a,b)=>b.strength-a.strength||b.count-a.count).slice(0,90);
    const nodes = top.map(([tag,count],i) => {
      const angle = i * 2.399963229728653;
      const radius = 46 + 235 * Math.sqrt((i+1)/Math.max(top.length,1));
      return {tag,count,x:400+radius*Math.cos(angle),y:280+radius*Math.sin(angle)};
    });
    // Relax linked tags closer and unrelated tags apart, deterministically.
    const lookup = new Map(nodes.map((n,i) => [n.tag,i]));
    for(let tick=0; tick<48; tick++) {
      const fx=nodes.map(()=>0),fy=nodes.map(()=>0);
      for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[j].x-nodes[i].x,dy=nodes[j].y-nodes[i].y,d2=Math.max(100,dx*dx+dy*dy);
        const push=34000/d2;
        fx[i]-=dx*push/Math.sqrt(d2);fy[i]-=dy*push/Math.sqrt(d2);
        fx[j]+=dx*push/Math.sqrt(d2);fy[j]+=dy*push/Math.sqrt(d2);
      }
      for(const edge of edges){
        const a=lookup.get(edge.a),b=lookup.get(edge.b);
        const dx=nodes[b].x-nodes[a].x,dy=nodes[b].y-nodes[a].y;
        const pull=Math.min(0.015,0.003+Math.log1p(edge.count)*0.0015);
        fx[a]+=dx*pull;fy[a]+=dy*pull;fx[b]-=dx*pull;fy[b]-=dy*pull;
      }
      nodes.forEach((n,i)=>{n.x=Math.max(42,Math.min(758,n.x+fx[i]*0.7));
        n.y=Math.max(37,Math.min(523,n.y+fy[i]*0.7));});
    }
    return {nodes,edges};
  }

  function waterfall(index, onPick, categories) {
    const root = document.createElement("div");
    root.className = "studio-waterfall";
    root.setAttribute("aria-label","Tag frequency waterfall");
    if(!index.length){
      const empty=document.createElement("p");
      empty.className="field-hint";
      empty.textContent="No matching tags. Clear Search Tags to see frequencies.";
      root.append(empty);
      return root;
    }
    const max = Math.max(1,...index.slice(0,75).map(x=>x[1]));
    for(const [tag,count] of index.slice(0,75)) {
      const btn=document.createElement("button");
      btn.type="button";btn.className="studio-waterfall-row";
      btn.style.setProperty("--studio-tag-color",colorForTag(tag,categories));
      const label=document.createElement("span");
      label.textContent=tag;label.className="studio-waterfall-label";
      const track=document.createElement("span");
      track.className="studio-waterfall-track";
      const bar=document.createElement("span");
      bar.className="studio-waterfall-bar";
      bar.style.width=Math.max(1,(count/max)*100)+"%";
      track.append(bar);
      const value=document.createElement("span");
      value.textContent=String(count);value.className="studio-waterfall-count";
      btn.append(label,track,value);
      btn.title=tag+": "+count+" images · filter by exact tag";
      btn.addEventListener("click",()=>onPick(tag));
      root.append(btn);
    }
    return root;
  }

  function constellation(index, captions, onPick, categories) {
    const graph=buildGraph(index,captions);
    const holder=document.createElement("div");
    holder.className="studio-constellation";
    if(!graph.nodes.length){
      const empty=document.createElement("p");
      empty.className="field-hint";
      empty.textContent="No matching tags. Clear Search Tags to view the constellation.";
      holder.append(empty);
      return holder;
    }
    const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
    svg.setAttribute("role","group");svg.setAttribute("aria-label","Interactive tag co-occurrence constellation");
    const ns="http://www.w3.org/2000/svg";
    const make=(name,attrs={},parent=svg)=>{
      const el=document.createElementNS(ns,name);
      for(const [key,value] of Object.entries(attrs))el.setAttribute(key,String(value));
      parent.append(el);return el;
    };
    const lookup=new Map(graph.nodes.map(node=>[node.tag,node]));
    let box={x:0,y:0,w:800,h:560};
    const view=()=>svg.setAttribute("viewBox",[box.x,box.y,box.w,box.h].join(" "));
    view();
    for(const edge of graph.edges){
      const a=lookup.get(edge.a),b=lookup.get(edge.b);
      const line=make("line",{x1:a.x,y1:a.y,x2:b.x,y2:b.y,
        "stroke-width":Math.min(4,0.5+Math.sqrt(edge.count)/2),class:"studio-star-edge"});
      make("title",{},line).textContent=edge.a+" + "+edge.b+" · "+edge.count+" images";
    }
    for(const node of graph.nodes) {
      const g=make("g",{class:"studio-star-node",role:"button",tabindex:"0"});
      const r=Math.min(18,6+Math.sqrt(node.count)*0.85);
      const circle=make("circle",{cx:node.x,cy:node.y,r},g);
      circle.style.fill=colorForTag(node.tag,categories);
      make("text",{x:node.x+Math.min(r+4,23),y:node.y+4},g).textContent=node.tag;
      make("title",{},g).textContent=node.tag+" · "+node.count+" images";
      g.addEventListener("click",()=>onPick(node.tag));
      g.addEventListener("keydown",e=>{
        if(e.key==="Enter"||e.key===" "){e.preventDefault();onPick(node.tag);}
      });
    }
    let drag=null;
    svg.addEventListener("pointerdown",e=>{
      if(e.target.closest(".studio-star-node"))return;
      drag={x:e.clientX,y:e.clientY,ox:box.x,oy:box.y};
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener("pointermove",e=>{
      if(!drag)return;
      const rect=svg.getBoundingClientRect();
      box.x=drag.ox-(e.clientX-drag.x)*box.w/rect.width;
      box.y=drag.oy-(e.clientY-drag.y)*box.h/rect.height;view();
    });
    svg.addEventListener("pointerup",()=>drag=null);
    svg.addEventListener("pointercancel",()=>drag=null);
    svg.addEventListener("wheel",e=>{
      e.preventDefault();
      const next=Math.max(250,Math.min(1600,box.w*(e.deltaY>0?1.12:0.89)));
      const rect=svg.getBoundingClientRect(),fractionX=(e.clientX-rect.left)/rect.width,
        fractionY=(e.clientY-rect.top)/rect.height;
      box.x+=(box.w-next)*fractionX;
      box.y+=(box.h-next*0.7)*fractionY;
      box.w=next;box.h=next*0.7;view();
    },{passive:false});
    const hint=document.createElement("div");
    hint.className="field-hint";
    hint.textContent=graph.nodes.length+" tags · "+graph.edges.length+
      " above-chance connections · drag to pan, scroll to zoom, click tag to filter";
    holder.append(svg,hint);return holder;
  }

  // Adapted from the user-supplied TagBench 1.2 Explore.tsx. TagBench's
  // "Waterfall" is a picture masonry gallery, not a frequency bar chart.
  // Constellation uses its 120-node, link-weighted cluster/physics arrangement,
  // count-bearing orbs, label colors, focus relationships and pan/zoom controls.
  const TB_HUES=["#ff4d6d","#ff7a59","#f0b429","#d6e15a","#3ddc97",
    "#2ee6c7","#3ec6ff","#5b8cff","#9b6bff","#ff7ab6"];
  const TB_SPECIES=/wolf|fox|lion|feline|cat|dragon|horse|avian|bird|shark|rabbit|deer|tiger|leopard|anthro|feral|kobold|bear|otter|sergal/i;
  const TB_FAMILIES=[
    /\b(solo|duo|trio|group|multiple|couple|pair)\b/i,
    /\bfur\b|\beyes?\b|sclera|pupil|iris|\bskin\b|marking|\b(black|brown|white|grey|gray|tan|pale|pink|red|blue|green|yellow|orange|purple|dark|light)\s+body\b/i,
    /jewelry|piercing|bracelet|earring|anklet|necklace|\bring\b/i,
    /cloth|shirt|dress|\bwear\b|loincloth|glove|topless|nude|\bpant|skirt|jacket|collar|outfit|clothed/i,
    TB_SPECIES,
    /look|smile|\bmouth\b|standing|sitting|holding|blush|gaze/i,
    /\bmale\b|\bfemale\b|gender|\bboy\b|\bgirl\b/i,
    /outdoor|indoor|\bnight\b|background|\bsky\b|outside|\blight\b/i,
    /\b(book|bottle|cup|lamp|plant|container|fire|candle|box|bag|phone|camera|sword|spear|weapon|food|drink|chair|table|bed)\b/i,
    /\bfull body\b|\bupper body\b|\blower body\b|\bbust\b|\bportrait\b|\bclose[- ]?up\b|\bcowboy shot\b/i,
    /paw|claw|tail|whisker|fang|\bear\b|finger|toe|\bfeet\b|\bfoot\b|nipple|breast|pectoral|\babs\b|muscular|tuft|\bnose\b|muzzle|horn|wing|thigh|biped|\bbody\b|digitigrade|\bhair\b|navel|collarbone|eyebrow/i
  ];
  const tbFamily=tag=>{
    const name=lower(tag).replace(/_/g," ");
    const species=TB_SPECIES.test(name);
    if(species&&name.includes("character"))return 11;
    const i=TB_FAMILIES.findIndex(test=>test.test(name));
    return i<0?null:i;
  };
  const tbStem=tag=>{
    let w=lower(tag).replace(/[\s_]/g,"");
    if(w.endsWith("ing")&&w.length>6)w=w.slice(0,-3);
    else if(w.endsWith("ed")&&w.length>5)w=w.slice(0,-2);
    else if(w.endsWith("es")&&w.length>4)w=w.slice(0,-2);
    else if(w.endsWith("s")&&w.length>3)w=w.slice(0,-1);
    return w;
  };
  // Stable semantic cluster identity, blended toward the active theme's
  // foreground for readability on both light and dark surfaces.
  const tbHue=(id,colorMix=72)=>{
    const hue=id<TB_HUES.length?TB_HUES[id]:
      "hsl("+Math.round(id*137.508%360)+" 72% 62%)";
    return "color-mix(in srgb, "+hue+" "+colorMix+"%, var(--text-primary))";
  };
  function tagbenchGraph(items) {
    // Bound pair storage by the visible vocabulary. Search still indexes every tag.
    const counts=new Map(),covers=new Map(),coverSamples=new Map();
    for(const it of items){
      const unique=new Set((it.tags||[]).map(lower).filter(Boolean));
      for(const tag of unique){
        counts.set(tag,(counts.get(tag)||0)+1);
        if(Number.isInteger(it.coverIndex)){
          if(!covers.has(tag))covers.set(tag,it.coverIndex);
          else{
            // One fixed first cover and up to seven deterministic reservoir
            // samples. No unbounded per-tag lists, even for common tags.
            let samples=coverSamples.get(tag);
            if(!samples){samples=[covers.get(tag)];coverSamples.set(tag,samples);}
            if(samples.length<8)samples.push(it.coverIndex);
            else{
              const seen=counts.get(tag);
              const slot=(Math.imul(seen,0x9e3779b1)>>>0)%seen;
              if(slot>0&&slot<8)samples[slot]=it.coverIndex;
            }
          }
        }
      }
    }
    const all=[...counts].map(([tag,count])=>({tag,count}))
      .sort((a,b)=>b.count-a.count||a.tag.localeCompare(b.tag));
    const top=all.slice(0,120),keep=new Set(top.map(x=>x.tag)),pairs=new Map();
    for(const it of items){
      const tags=[...new Set((it.tags||[]).map(lower).filter(tag=>keep.has(tag)))];
      for(let i=0;i<tags.length;i++)for(let j=i+1;j<tags.length;j++){
        const a=tags[i],b=tags[j],key=a<b?a+"\0"+b:b+"\0"+a;
        pairs.set(key,(pairs.get(key)||0)+1);
      }
    }
    const edges=[...pairs].map(([key,w])=>{
      const [a,b]=key.split("\0");return {a,b,w};
    }).sort((a,b)=>b.w-a.w||a.a.localeCompare(b.a)||a.b.localeCompare(b.b));
    const linkCounts=new Map();
    for(const edge of edges){
      linkCounts.set(edge.a,(linkCounts.get(edge.a)||0)+1);
      linkCounts.set(edge.b,(linkCounts.get(edge.b)||0)+1);
    }
    return {all,top,edges,linkCounts,covers,coverSamples,imageCount:items.length};
  }
  function tagAffinity(shared,leftCount,rightCount){
    return shared/Math.sqrt(leftCount*rightCount)*shared/(shared+2);
  }
  // Multiscale atlas built in the existing bounded graph worker. Community
  // counts are tag occurrences, not unique images across several tags.
  function constellationAtlas(nodes,all,edges){
    if(!nodes.length)return {clusters:[],rivers:[],unmapped:0};
    const groups=new Map();
    for(const n of nodes){
      if(!groups.has(n.group))groups.set(n.group,[]);
      groups.get(n.group).push(n);
    }
    const maxCount=Math.max(1,...nodes.map(n=>n.count));
    const clusters=[...groups].map(([id,rows])=>{
      rows.sort((a,b)=>b.count-a.count||a.tag.localeCompare(b.tag));
      const x=rows.reduce((sum,n)=>sum+n.x,0)/rows.length;
      const y=rows.reduce((sum,n)=>sum+n.y,0)/rows.length;
      const count=rows.reduce((sum,n)=>sum+n.count,0);
      // Gaussian density on a bounded grid, sampled once in the worker.
      const sigma=27,step=9;
      const left=Math.min(...rows.map(n=>n.x))-sigma*3;
      const top=Math.min(...rows.map(n=>n.y))-sigma*3;
      const nx=Math.min(100,Math.max(2,Math.ceil((Math.max(...rows.map(n=>n.x))+sigma*3-left)/step)+1));
      const ny=Math.min(100,Math.max(2,Math.ceil((Math.max(...rows.map(n=>n.y))+sigma*3-top)/step)+1));
      const grid=new Float32Array(nx*ny);let peak=0;
      for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){
        const px=left+i*step,py=top+j*step;
        let value=0;
        for(const n of rows){
          const d2=(n.x-px)**2+(n.y-py)**2;
          if(d2<9*sigma*sigma)value+=Math.sqrt(n.count/maxCount)*Math.exp(-d2/(2*sigma*sigma));
        }
        grid[j*nx+i]=value;peak=Math.max(peak,value);
      }
      const contours=[.16,.34,.60].map(level=>{
        const segments=[],threshold=peak*level;
        for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){
          const v=[grid[j*nx+i],grid[j*nx+i+1],grid[(j+1)*nx+i+1],grid[(j+1)*nx+i]];
          const p=[[left+i*step,top+j*step],[left+(i+1)*step,top+j*step],
            [left+(i+1)*step,top+(j+1)*step],[left+i*step,top+(j+1)*step]];
          const crossings=[];
          for(let k=0;k<4;k++){
            const next=(k+1)%4;
            if((v[k]>=threshold)===(v[next]>=threshold))continue;
            const t=(threshold-v[k])/(v[next]-v[k]);
            crossings.push([p[k][0]+t*(p[next][0]-p[k][0]),p[k][1]+t*(p[next][1]-p[k][1])]);
          }
          for(let k=0;k+1<crossings.length;k+=2)segments.push([...crossings[k],...crossings[k+1]]);
        }
        return segments;
      });
      return {id,x,y,count,label:rows[0].tag,tags:rows.map(n=>n.tag),color:rows[0].color,
        spread:Math.max(45,...rows.map(n=>Math.hypot(n.x-x,n.y-y)+45)),contours};
    });
    const byTag=new Map(nodes.map(n=>[n.tag,n]));
    const unmapped=Math.max(0,all.length-nodes.length);
    if(unmapped)clusters.push({id:"other",label:"Other tags",count:all.reduce((sum,n)=>sum+(byTag.has(n.tag)?0:n.count),0),
      tags:[],color:"var(--text-muted)",unmapped});
    clusters.sort((a,b)=>b.count-a.count||a.label.localeCompare(b.label));
    // Deterministic nonoverlapping packing; compressed frequency scaling.
    const largest=clusters[0].count||1,placed=[];
    for(const c of clusters){
      const r=25+57*Math.sqrt(c.count/largest);
      let px=0,py=0;
      if(placed.length){
        for(let i=1;;i++){
          const angle=i*.43,dist=18*Math.sqrt(i);
          px=Math.cos(angle)*dist;py=Math.sin(angle)*dist;
          if(placed.every(p=>Math.hypot(p.px-px,p.py-py)>=p.r+r+9))break;
        }
      }
      Object.assign(c,{px,py,r});placed.push(c);
    }
    const left=Math.min(...placed.map(c=>c.px-c.r)),right=Math.max(...placed.map(c=>c.px+c.r));
    const top=Math.min(...placed.map(c=>c.py-c.r)),bottom=Math.max(...placed.map(c=>c.py+c.r));
    const scale=Math.min(760/(right-left),490/(bottom-top));
    for(const c of placed){c.px=500+(c.px-(left+right)/2)*scale;c.py=350+(c.py-(top+bottom)/2)*scale;c.r*=scale;}
    const bridges=new Map();
    for(const edge of edges){
      const a=byTag.get(edge.a),b=byTag.get(edge.b);
      if(!a||!b||a.group===b.group)continue;
      const lo=Math.min(a.group,b.group),hi=Math.max(a.group,b.group),key=lo+":"+hi;
      const old=bridges.get(key)||{a:lo,b:hi,weight:0,count:0};
      old.weight+=tagAffinity(edge.w,a.count,b.count);old.count+=edge.w;bridges.set(key,old);
    }
    const rivers=[...bridges.values()].sort((a,b)=>b.weight-a.weight||a.a-b.a||a.b-b.b).slice(0,14);
    return {clusters,rivers,unmapped};
  }
  function tagbenchModel(items,categories) {
    const graph=tagbenchGraph(items),{all,top,edges,linkCounts}=graph;
    const index=new Map(top.map((row,i)=>[row.tag,i]));
    const neighbors=top.map(()=>[]);
    // Cosine normalization discounts ubiquitous tags; shrink one-off coincidences.
    for(const edge of edges){
      const i=index.get(edge.a),j=index.get(edge.b);
      const weight=tagAffinity(edge.w,top[i].count,top[j].count);
      neighbors[i].push({peer:j,weight,edge});
      neighbors[j].push({peer:i,weight,edge});
    }
    const group=top.map((_,i)=>i);
    const degree=neighbors.map(list=>list.reduce((sum,n)=>sum+n.weight,0));
    const total=degree.reduce((a,b)=>a+b,0),volume=degree.slice();
    // Local modularity moves on observed caption pairs. Isolates stay independent.
    for(let pass=0;pass<24;pass++){
      let moved=false;
      for(let i=0;i<top.length;i++){
        if(!degree[i])continue;
        const old=group[i],weights=new Map();
        for(const {peer,weight} of neighbors[i]){
          const id=group[peer];weights.set(id,(weights.get(id)||0)+weight);
        }
        volume[old]-=degree[i];
        const gain=id=>(weights.get(id)||0)-degree[i]*volume[id]/total;
        let best=old,score=gain(old);
        for(const id of weights.keys()){
          const value=gain(id);
          if(value>score+1e-10){best=id;score=value;}
        }
        volume[best]+=degree[i];group[i]=best;if(best!==old)moved=true;
      }
      if(!moved)break;
    }
    const ids=[...new Set(group)],members=new Map();
    const nodes=top.map((row,i)=>{
      const id=ids.indexOf(group[i]),k=members.get(id)||0;
      members.set(id,k+1);
      const angle=id*2.39996,r=ids.length===1?0:180*Math.sqrt((id+.5)/ids.length);
      return {...row,group:id,links:linkCounts.get(row.tag)||0,
        x:Math.cos(angle)*r+Math.cos(k*2.39996)*10*Math.sqrt(k),
        y:Math.sin(angle)*r+Math.sin(k*2.39996)*10*Math.sqrt(k)};
    });
    const strands=new Map();
    neighbors.forEach((list,i)=>{
      list.sort((a,b)=>b.weight-a.weight||a.peer-b.peer);
      const bridge=list.find(n=>group[n.peer]!==group[i]);
      for(const n of [...list.slice(0,3),...(bridge?[bridge]:[])]){
        strands.set(n.edge,{i,j:n.peer,weight:n.weight});
      }
    });
    const chosen=[...strands.keys()];
    const fx=new Float64Array(nodes.length),fy=new Float64Array(nodes.length);
    const centers=ids.map(()=>({x:0,y:0}));
    for(let step=0;step<125;step++){
      for(const center of centers){center.x=0;center.y=0;}
      for(const n of nodes){centers[n.group].x+=n.x;centers[n.group].y+=n.y;}
      centers.forEach((center,id)=>{center.x/=members.get(id);center.y/=members.get(id);});
      fx.fill(0);fy.fill(0);
      for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y;
        const f=72/(dx*dx+dy*dy+2);
        fx[i]+=dx*f;fy[i]+=dy*f;fx[j]-=dx*f;fy[j]-=dy*f;
      }
      for(const {i,j,weight} of strands.values()){
        const dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y;
        const f=(.018+.15*weight)*(group[i]===group[j]?1:.55);
        fx[i]-=dx*f;fy[i]-=dy*f;fx[j]+=dx*f;fy[j]+=dy*f;
      }
      nodes.forEach((n,i)=>{
        const center=centers[n.group];
        const dx=fx[i]-n.x*.023+(center.x-n.x)*.12;
        const dy=fy[i]-n.y*.023+(center.y-n.y)*.12;
        const scale=Math.min(.45,(6-step*.035)/Math.max(1,Math.hypot(dx,dy)));
        n.x+=dx*scale;n.y+=dy*scale;
      });
    }
    // Make room for count discs after the community forces have settled.
    const radii=nodes.map(n=>countRadius(n.count,top[0].count));
    for(let pass=0;pass<48;pass++){
      let worst=0;
      for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
        const a=nodes[i],b=nodes[j],dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);
        const overlap=radii[i]+radii[j]+4-d;
        if(overlap<=0)continue;
        const x=d?dx/d:1,y=d?dy/d:0;
        a.x-=x*overlap/2;a.y-=y*overlap/2;b.x+=x*overlap/2;b.y+=y*overlap/2;
        worst=Math.max(worst,overlap);
      }
      if(worst<.5)break;
    }
    const colors=new Map();
    nodes.forEach(n=>{
      n.x+=500;n.y+=360;
      const category=categories?.tags?.[n.tag];
      const manual=categories?.tagColors?.[n.tag]||(category&&categories.colors?.[category]);
      n.color=manual||tbHue(n.group);colors.set(n.tag,n.color);
    });
    // Index node colors once; avoid O(all dataset tags × 120) scans.
    const stemColors=new Map(),familyColors=new Map();
    for(const n of nodes){
      const stem=tbStem(n.tag),family=tbFamily(n.tag);
      if(stem.length>=4&&!stemColors.has(stem))stemColors.set(stem,n.color);
      if(family!==null&&!familyColors.has(family))familyColors.set(family,n.color);
    }
    for(const tag of all)if(!colors.has(tag.tag)){
      const family=tbFamily(tag.tag),stem=tbStem(tag.tag);
      const friendColor=(stem.length>=4&&stemColors.get(stem))||
        (family!==null&&familyColors.get(family));
      const category=categories?.tags?.[tag.tag];
      const manual=categories?.tagColors?.[tag.tag]||(category&&categories.colors?.[category]);
      if(manual||friendColor)colors.set(tag.tag,manual||friendColor);
    }
    return {...graph,nodes,chosen,colors,atlas:constellationAtlas(nodes,all,edges)};
  }
  const waterfallScale=value=>{
    const n=Number(value);
    return value==null||value===""||!Number.isFinite(n)?100:Math.max(50,Math.min(150,Math.round(n/10)*10));
  };
  const waterfallColumns=(width,scale=100)=>Math.max(1,
    Math.floor((Math.max(100,width)-16+9)/(228*waterfallScale(scale)/100)));
  function tagbenchWaterfall(items,onSelect,onContextMenu,activePath,selection,token,onDecorate=null,onOpenTags=null,initialScale=100) {
    const stage=document.createElement("div");stage.className="studio-tb-waterfall";
    const flow=document.createElement("div");flow.className="studio-tb-flow";
    const pictures=items.filter(it=>it.size);
    let scale=waterfallScale(initialScale);
    const initialColumns=waterfallColumns(window.innerWidth-60,scale);
    const tiles=[]; // Dataset order, independent of the column DOM order.
    let columns=Array.from({length:initialColumns},()=>{
      const col=document.createElement("div");col.className="studio-tb-flow-column";
      flow.append(col);return col;
    });
    // Decode only the first page of thumbnails. The dataset may have tens of
    // thousands of images; creating every DOM node and image at once is costly.
    const batchSize=120;
    let count=0,disposed=false,resizeObserver=null,loadObserver=null,resizeFrame=null;
    const createTile=it=>{
      const tile=document.createElement("button");
      tile.type="button";tile.className="studio-tb-flow-tile";
      tile.dataset.path=it.path;
      tile.setAttribute("aria-label",it.name);
      tile.setAttribute("aria-pressed",String(!!selection?.has(it.path)));
      if(selection?.has(it.path))tile.classList.add("is-picked");
      if(it.path===activePath)tile.classList.add("active");
      const img=document.createElement("img");
      img.loading="lazy";img.decoding="async";
      img.src="/api/thumb?path="+encodeURIComponent(it.path)+"&t="+encodeURIComponent(token||"");
      img.alt="";
      if(it.size[0]>0&&it.size[1]>0){
        img.width=it.size[0];img.height=it.size[1];
      }
      tile.append(img);
      onDecorate?.(tile,it);
      tile.addEventListener("click",event=>onSelect(it,event));
      tile.addEventListener("dblclick",event=>{
        event.preventDefault();
        onOpenTags?.(it,event);
      });
      tile.addEventListener("contextmenu",event=>onContextMenu(event,it));
      return tile;
    };
    const sentinel=document.createElement("button");
    sentinel.type="button";sentinel.className="studio-tb-load-more btn btn-sm";
    sentinel.textContent="Load more images";sentinel.setAttribute("aria-label","Load more waterfall images");
    const loadMore=()=>{
      if(disposed||count>=pictures.length)return;
      const end=Math.min(pictures.length,count+batchSize);
      for(;count<end;count++){
        const tile=createTile(pictures[count]);
        tiles.push(tile);columns[count%columns.length].append(tile);
      }
      if(count>=pictures.length){
        sentinel.remove();
        loadObserver?.disconnect();
      }else sentinel.textContent="Load more images ("+count+" / "+pictures.length+")";
    };
    sentinel.addEventListener("click",loadMore);
    stage.append(flow);
    if(!pictures.length){
      const hint=document.createElement("p");hint.textContent="No pictures match the current dataset filter.";
      hint.className="field-hint";flow.append(hint);
    }else{
      stage.append(sentinel);
      loadMore();
      if(typeof IntersectionObserver!=="undefined"&&count<pictures.length){
        loadObserver=new IntersectionObserver(entries=>{
          if(entries.some(entry=>entry.isIntersecting))loadMore();
        },{root:stage,rootMargin:"450px 0px"});
        loadObserver.observe(sentinel);
      }
    }
    const resize=()=>{
      if(disposed||!stage.isConnected||!stage.clientWidth||!tiles.length)return;
      const wanted=waterfallColumns(stage.clientWidth,scale);
      if(wanted===columns.length)return;
      // Anchor the first visible image rather than a percentage of scroll
      // height. Binary search each column; no full-dataset layout scan.
      const top=stage.getBoundingClientRect().top;
      let anchor=null,offset=0,best=Infinity;
      for(const col of columns){
        const children=col.children;
        let lo=0,hi=children.length;
        while(lo<hi){
          const mid=(lo+hi)>>1;
          if(children[mid].getBoundingClientRect().bottom<=top)lo=mid+1;
          else hi=mid;
        }
        if(lo===children.length)continue;
        const tile=children[lo],rect=tile.getBoundingClientRect();
        const distance=Math.abs(rect.top-top);
        if(distance<best){best=distance;anchor=tile;offset=rect.top-top;}
      }
      const focused=document.activeElement;
      const nextColumns=Array.from({length:wanted},()=>{
        const col=document.createElement("div");col.className="studio-tb-flow-column";
        return col;
      });
      // Keep the original image nodes, selection classes and event handlers.
      // Use dataset order, not column-major querySelectorAll order.
      tiles.forEach((tile,i)=>nextColumns[i%wanted].append(tile));
      flow.replaceChildren(...nextColumns);columns=nextColumns;
      if(anchor)stage.scrollTop+=anchor.getBoundingClientRect().top-top-offset;
      if(focused&&tiles.includes(focused))focused.focus({preventScroll:true});
    };
    const queueResize=()=>{
      if(disposed||resizeFrame!==null)return;
      if(typeof requestAnimationFrame!=="function"){resize();return;}
      resizeFrame=requestAnimationFrame(()=>{resizeFrame=null;resize();});
    };
    stage.setScale=value=>{scale=waterfallScale(value);queueResize();};
    stage.restoreImage=({path,imageIndex,offset})=>{
      if(!pictures.length)return;
      const match=pictures.findIndex(it=>it.path===path);
      const index=match<0?Math.min(imageIndex,pictures.length-1):match;
      while(count<=index)loadMore();
      resize();
      const tile=tiles[index];
      stage.scrollTop+=tile.getBoundingClientRect().top-stage.getBoundingClientRect().top-offset;
      tile.focus({preventScroll:true});
    };
    if(typeof ResizeObserver!=="undefined"){
      resizeObserver=new ResizeObserver(queueResize);
      resizeObserver.observe(stage);
    }
    stage.dispose=()=>{
      disposed=true;
      if(resizeFrame!==null)cancelAnimationFrame(resizeFrame);
      resizeObserver?.disconnect();
      loadObserver?.disconnect();
    };
    return stage;
  }
  const constellationPrefs=p=>{
    const n=Number(p?.elasticity??20);
    return {elasticity:Number.isFinite(n)?Math.min(100,Math.max(0,n)):20,
      glassNodes:p?.glassNodes===true,celestial:p?.celestial!==false};
  };
  // Bound the tug in screen pixels. It is a transient visual offset, never a
  // change to the stored graph layout or to the co-occurrence data.
  function elasticOffset(dx,dy,weight,elasticity){
    const cap=25*constellationPrefs({elasticity}).elasticity/100;
    const length=Math.hypot(dx,dy);
    const strength=Math.max(0,Math.min(1,weight));
    const amount=Math.min(cap,length*.12)*strength;
    return length?{x:dx/length*amount,y:dy/length*amount}:{x:0,y:0};
  }
  function countRadius(count,maxCount){
    return 5+3.8*String(count).length+20*Math.sqrt(count/Math.max(1,maxCount));
  }
  function tagbenchConstellation(model,onPick,pickedTag=null,options={}){
    const stage=document.createElement("div");stage.className="studio-tb-constellation";
    if(!model.nodes.length&&!options.resume){stage.textContent="No tagged images in this dataset.";return stage;}
    const NS="http://www.w3.org/2000/svg",svg=document.createElementNS(NS,"svg");
    svg.setAttribute("role","group");svg.setAttribute("aria-label","Interactive tag constellation");
    const make=(type,attrs={},parent=svg)=>{
      const node=document.createElementNS(NS,type);
      for(const [k,v] of Object.entries(attrs))node.setAttribute(k,String(v));
      parent.append(node);return node;
    };
    const html=(tag,cls,text,parent)=>{
      const node=document.createElement(tag);if(cls)node.className=cls;
      if(text!==undefined)node.textContent=text;if(parent)parent.append(node);return node;
    };
    const button=(parent,label,title,fn)=>{
      const b=html("button","",label,parent);b.type="button";b.title=title;b.setAttribute("aria-label",title);
      b.addEventListener("click",fn);return b;
    };
    const defs=make("defs");
    const gradient=make("linearGradient",{id:"tb-sunlight",x1:"0%",y1:"0%",x2:"100%",y2:"100%"},defs);
    for(const [offset,color] of [["0%","#fffbe5"],["45%","#fff0b3"],["100%","#e4be72"]])
      make("stop",{offset,"stop-color":color},gradient);
    const canvas=html("canvas","studio-tb-strands");
    canvas.setAttribute("aria-hidden","true");
    const ctx=canvas.getContext("2d");
    const packLayer=make("g",{"aria-label":"Dataset community overview"});
    const edgeLayer=make("g"),nodeLayer=make("g"),labelLayer=make("g");
    const nodes=model.nodes.map(n=>({...n,ox:0,oy:0,
      color:n.color===tbHue(n.group)?tbHue(n.group,92):n.color}));
    const at=new Map(nodes.map(n=>[n.tag,n])),all=new Map(model.all.map(n=>[n.tag,n]));
    const adjacency=new Map(nodes.map(n=>[n.tag,new Map()]));
    for(const e of model.edges){adjacency.get(e.a)?.set(e.b,e.w);adjacency.get(e.b)?.set(e.a,e.w);}
    const maxCount=nodes[0]?.count||1;
    const visual=new Map();
    let labelOrder=nodes,overviewCamera=null,overviewZoom=1;
    let expanded=false,activePair=null,collection=null,collectionMotion=null;
    const wheelPause=180,wheelExitDistance=40;
    let closingWheel=null;
    const temporaryTags=new Set(),focusEdges=new Map();
    const focusedTags=new Set();
    let typeScale=1,packed=false,coverTimer=null;
    const activeCovers=new Set();
    const atlas=model.atlas||constellationAtlas(model.nodes,model.all,model.edges);
    const labelSize=n=>(n.tag===picked?22:13+8*Math.sqrt(n.count/maxCount))*typeScale/zoom;

    // Group strokes by color and width so each batch shares a Canvas path.
    const strandBatches=new Map();
    const overviewEdges=model.chosen.map(edge=>({
      ...edge,strength:tagAffinity(edge.w,at.get(edge.a).count,at.get(edge.b).count)
    })).sort((a,b)=>b.strength-a.strength).slice(0,150);
    const strongest=overviewEdges[0]?.strength||1;
    for(const edge of overviewEdges){
      const a=at.get(edge.a),b=at.get(edge.b);
      const tag=a.group===b.group&&a.color===b.color?a.tag:null;
      const color=tag?a.color:null;
      const band=Math.min(4,Math.floor(5*Math.sqrt(edge.strength/strongest)));
      const key=band+"|"+color;
      if(!strandBatches.has(key))strandBatches.set(key,{color,tag,band,edges:[]});
      strandBatches.get(key).edges.push(edge);
    }
    // Resolve theme-dependent SVG colors after mount, then reuse them during motion.
    const strandColors=new Map();
    let prefs=constellationPrefs(options.preferences),zoom=1,pan={x:0,y:0},picked=null,drag=null;
    let serial=0,shared=new Map(),frame=null,settling=null,camera=null,pulse=null,disposed=false,dirty=new Set(),lastClick=null;
    const media=typeof matchMedia==="function"?matchMedia("(prefers-reduced-motion: reduce)"):null;
    const reduced=()=>!!media?.matches;
    // Reserve room for the full total even when selection shows a shorter shared count.
    const radius=n=>countRadius(n.count,maxCount)*typeScale/zoom;
    const position=n=>({x:n.x+n.ox,y:n.y+n.oy});
    const view=()=>{
      packed=!picked&&atlas.clusters.length>1&&zoom<overviewZoom*.82;
      svg.setAttribute("viewBox",[500-500/zoom+pan.x,350-350/zoom+pan.y,1000/zoom,700/zoom].join(" "));
      packLayer.style.display=packed?"":"none";
      for(const layer of [edgeLayer,nodeLayer,labelLayer])layer.style.display=packed?"none":"";
    };
    const search=html("div","studio-tb-search"),input=html("input","",undefined,search),matches=html("div","studio-tb-suggest",undefined,search);
    input.placeholder="Search Tags";input.setAttribute("aria-label","Search all dataset tags");
    const zoomCtl=html("div","studio-tb-zoom"),spotlight=html("aside","studio-tb-cluster");
    spotlight.hidden=true;
    const menu=html("div","studio-tb-node-menu");menu.hidden=true;menu.setAttribute("role","group");
    const dismissMenu=()=>{menu.hidden=true;menu.replaceChildren();};
    const point=e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());};
    const scalePixels=()=>Math.max(.001,Math.abs(svg.getScreenCTM()?.a||1));
    const packedById=new Map(atlas.clusters.map(c=>[c.id,c]));
    for(const river of atlas.rivers){
      const a=packedById.get(river.a),b=packedById.get(river.b);
      if(!a||!b)continue;
      const dx=b.px-a.px,dy=b.py-a.py;
      const path=make("path",{d:"M "+a.px+" "+a.py+" Q "+((a.px+b.px)/2-dy*.18)+" "+((a.py+b.py)/2+dx*.18)+" "+b.px+" "+b.py,
        fill:"none",stroke:a.color,"stroke-opacity":.20,"stroke-width":2+14*Math.sqrt(river.weight/(atlas.rivers[0]?.weight||1)),
        "stroke-linecap":"round","pointer-events":"none"},packLayer);
      make("title",{},path).textContent=river.count+" cross-community tag-pair occurrences";
    }
    for(const c of atlas.clusters){
      const desc=c.unmapped?c.unmapped+" other tags; browse tags outside the graph":
        c.label+", "+c.tags.length+" tags, "+c.count+" tag occurrences";
      const g=make("g",{transform:"translate("+c.px+" "+c.py+")",role:"button",tabindex:0,
        class:"studio-tb-atlas-community","aria-label":desc,style:"cursor:pointer"},packLayer);
      make("circle",{r:c.r,fill:c.color,"fill-opacity":.16,stroke:c.color,"stroke-opacity":.72,"stroke-width":1.5},g);
      make("circle",{r:c.r*.83,fill:"none",stroke:c.color,"stroke-opacity":.22,"stroke-width":1},g);
      const font=Math.max(5,Math.min(17,c.r*.25));
      const title=make("text",{"text-anchor":"middle",y:-font*.1,fill:"var(--text-primary)","font-size":font,"font-weight":700,"pointer-events":"none"},g);
      const maxChars=Math.max(5,Math.floor(c.r*1.7/(font*.56)));
      title.textContent=c.label.length>maxChars?c.label.slice(0,maxChars-1)+"…":c.label;
      make("text",{"text-anchor":"middle",y:font*1.1,fill:"var(--text-primary)","font-size":font*.72,"pointer-events":"none"},g)
        .textContent=(c.unmapped||c.tags.length)+" tags";
      make("title",{},g).textContent=desc;
      const enter=()=>{
        if(c.unmapped){
          input.focus();matches.replaceChildren();
          let shown=0;
          for(const n of model.all){
            if(at.has(n.tag))continue;
            const b=button(matches,n.tag+" · "+n.count,"Focus "+n.tag,()=>{
              input.value=n.tag;choose(n.tag,false);center(n.tag);matches.replaceChildren();
            });
            b.style.color=model.colors.get(n.tag)||"var(--text-primary)";
            if(++shown>=8)break;
          }
          return;
        }
        cameraTo(Math.min(5,Math.max(overviewZoom*1.35,1.25)),{x:c.x-500,y:c.y-350});
      };
      g.addEventListener("pointerdown",e=>e.stopPropagation());
      g.addEventListener("click",enter);
      g.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();e.stopPropagation();enter();}});
    }
    const contourPaths=new Map();
    if(typeof Path2D==="function")for(const c of atlas.clusters){
      if(c.unmapped)continue;
      contourPaths.set(c.id,c.contours.map(segments=>{
        const path=new Path2D();
        for(const [x1,y1,x2,y2] of segments){path.moveTo(x1,y1);path.lineTo(x2,y2);}
        return path;
      }));
    }
    function drawStrands(){
      const width=svg.clientWidth,height=svg.clientHeight;
      if(!ctx||!width||!height)return;
      const ratio=Math.min(window.devicePixelRatio||1,2);
      const w=Math.round(width*ratio),h=Math.round(height*ratio);
      if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
      ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);
      const scale=Math.min(width/1000,height/700)*zoom*ratio;
      ctx.setTransform(scale,0,0,scale,w/2-(500+pan.x)*scale,h/2-(350+pan.y)*scale);
      if(packed)return;
      // Density regions share the existing invalidation lifecycle; no
      // additional physics or animation loop.
      if(!picked&&prefs.celestial){
        for(const c of atlas.clusters){
          if(c.unmapped)continue;
          const anchor=visual.get(c.tags[0]);
          if(!anchor)continue;
          const color=getComputedStyle(anchor.halo).fill;
          const gradient=ctx.createRadialGradient(c.x,c.y,0,c.x,c.y,c.spread);
          gradient.addColorStop(0,color);gradient.addColorStop(1,"transparent");
          ctx.globalAlpha=.15;ctx.fillStyle=gradient;ctx.beginPath();
          ctx.arc(c.x,c.y,c.spread,0,Math.PI*2);ctx.fill();
          ctx.strokeStyle=color;ctx.lineWidth=ratio/scale;
          c.contours.forEach((segments,i)=>{
            ctx.globalAlpha=.16+i*.075;
            const path=contourPaths.get(c.id)?.[i];
            if(path){ctx.stroke(path);return;}
            ctx.beginPath();
            for(const [x1,y1,x2,y2] of segments){ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);}
            ctx.stroke();
          });
        }
      }
      const curve=(a,b)=>{
        const p=position(at.get(a)),q=position(at.get(b));
        ctx.moveTo(p.x,p.y);
        ctx.quadraticCurveTo((p.x+q.x)/2-(q.y-p.y)*.09,(p.y+q.y)/2+(q.x-p.x)*.09,q.x,q.y);
      };
      const neutral=getComputedStyle(canvas).color;
      for(const {color,tag,band,edges} of strandBatches.values()){
        if(color&&!strandColors.has(color)){
          strandColors.set(color,getComputedStyle(visual.get(tag).halo).fill);
        }
        ctx.strokeStyle=strandColors.get(color)||neutral;
        ctx.lineWidth=(.65+band*.6)*ratio/scale;
        ctx.globalAlpha=picked?.05:(color?.18:.085)+band*.055;
        ctx.beginPath();
        for(const e of edges)curve(e.a,e.b);
        ctx.stroke();
      }
    }
    function updateFocusEdges(){
      for(const [tag,edge] of focusEdges){
        const from=position(at.get(picked)),to=position(at.get(tag));
        const dx=to.x-from.x,dy=to.y-from.y;
        const cx=(from.x+to.x)/2-dy*.09,cy=(from.y+to.y)/2+dx*.09;
        const path="M "+from.x+" "+from.y+" Q "+cx+" "+cy+" "+to.x+" "+to.y;
        const strength=associationStrength(tag,shared.get(tag));
        edge.line.setAttribute("d",path);edge.hit.setAttribute("d",path);
        edge.line.setAttribute("stroke-width",1.25+9.5*strength);
        edge.line.setAttribute("stroke-opacity",activePair===tag?1:.42+.48*strength);
        edge.caption.setAttribute("x",.04*from.x+.32*cx+.64*to.x);
        edge.caption.setAttribute("y",.04*from.y+.32*cy+.64*to.y-10*typeScale/zoom);
        edge.caption.setAttribute("font-size",13*typeScale/zoom);
        edge.caption.setAttribute("stroke-width",3*typeScale/zoom);
      }
    }
    const nodeGeometry=n=>{
      const v=visual.get(n.tag);if(!v)return;
      const p=position(n),r=radius(n),sz=labelSize(n);
      v.g.setAttribute("transform","translate("+p.x+" "+p.y+")");
      v.hit.setAttribute("r",Math.max(9/zoom,r));v.core.setAttribute("r",r);v.halo.setAttribute("r",r*1.34);
      v.ring.setAttribute("r",r+3/zoom);
      if(v.cover){
        v.coverClip.setAttribute("r",Math.max(0,r-1.8*typeScale/zoom));
        for(const [key,value] of Object.entries({x:-r,y:-r,width:r*2,height:r*2}))
          v.cover.setAttribute(key,value);
      }
      v.number.setAttribute("font-size",13*typeScale/zoom);
      v.number.setAttribute("stroke-width",1.2/zoom);
      v.label.setAttribute("font-size",sz);v.label.setAttribute("stroke-width",Math.max(sz*.28,2/zoom));
      v.ripple.setAttribute("r",r+4/zoom+(pulse?.tag===n.tag?pulse.amount*13/zoom:0));
    };
    const updateGeometry=(changed=null)=>{
      if(!changed){for(const n of at.values())nodeGeometry(n);}
      else for(const tag of changed){const n=at.get(tag);if(n)nodeGeometry(n);}
      drawStrands();
      updateFocusEdges();
      placeLabels();
    };
    function createNode(n){
      const id="tb-orb-"+serial++;
      const grad=make("radialGradient",{id,cx:"32%",cy:"25%",r:"85%"},defs);
      for(const [offset,color,opacity] of [["0%","#ffffff",".62"],["35%",n.color,"1"],["100%",n.color,".72"]])
        make("stop",{offset,"stop-color":color,"stop-opacity":opacity},grad);
      const g=make("g",{"data-tag":n.tag,role:"button",tabindex:"0",class:"studio-tb-orb"},nodeLayer);
      make("title",{},g).textContent=n.tag+" · "+n.count+" images";
      const halo=make("circle",{fill:n.color,opacity:0,class:"studio-tb-orb-halo"},g);
      const ripple=make("circle",{fill:"none",stroke:"url(#tb-sunlight)","vector-effect":"non-scaling-stroke","stroke-width":"1.2",opacity:0,"pointer-events":"none"},g);
      const ring=make("circle",{fill:"none",stroke:"#efd16e","vector-effect":"non-scaling-stroke","stroke-width":"1.5",opacity:0,"pointer-events":"none"},g);
      const core=make("circle",{fill:n.color,class:"studio-tb-orb-core"},g);
      const artwork=make("g",{"pointer-events":"none","aria-hidden":"true"},g);
      const number=make("text",{"text-anchor":"middle","dominant-baseline":"central",
        "font-family":"system-ui, sans-serif","font-weight":750,style:"font-variant-numeric:tabular-nums",fill:"#161823",stroke:"#fff",
        "stroke-opacity":.75,"paint-order":"stroke","pointer-events":"none","aria-hidden":"true"},g);
      const hit=make("circle",{fill:"transparent"},g);
      const label=make("text",{fill:"var(--text-primary)","font-family":"system-ui, sans-serif","font-weight":650,stroke:"var(--bg-deep)","paint-order":"stroke","pointer-events":"none"},labelLayer);
      label.textContent=n.tag;
      visual.set(n.tag,{g,core,halo,ripple,ring,hit,artwork,number,label,grad,id});
      g.addEventListener("pointerdown",e=>{
        if(e.button!==0)return;e.stopPropagation();dismissMenu();cancelMotion();
        const p=point(e);drag={kind:"node",tag:n.tag,dx:p.x-n.x,dy:p.y-n.y,x:e.clientX,y:e.clientY,moved:false};
        svg.setPointerCapture(e.pointerId);
      });
      g.addEventListener("keydown",e=>{
        if(e.key==="Enter"||e.key===" "){
          e.preventDefault();e.stopPropagation();
          if(n.tag===picked&&options.collectionItems)openCollection(n.tag);
          else choose(n.tag);
        }
        if(e.key==="ContextMenu"||(e.shiftKey&&e.key==="F10")){e.preventDefault();const r=g.getBoundingClientRect();openMenu(n.tag,r.left,r.bottom);}
      });
      g.addEventListener("contextmenu",e=>{e.preventDefault();e.stopPropagation();openMenu(n.tag,e.clientX,e.clientY);});
      nodeGeometry(n);
    }
    for(const n of nodes)createNode(n);
    function relationMap(tag){return adjacency.get(tag)||new Map();}
    function clearTemporaryTags(keep){
      for(const tag of temporaryTags){
        if(tag===keep)continue;
        const v=visual.get(tag);
        v.g.remove();v.label.remove();v.grad.remove();v.clip?.remove();
        visual.delete(tag);at.delete(tag);temporaryTags.delete(tag);
      }
    }
    function ensureTag(tag){
      if(at.has(tag))return at.get(tag);
      const entry=all.get(tag);if(!entry)return null;
      const n={...entry,x:500,y:350,ox:0,oy:0,color:model.colors.get(tag)||"var(--accent-text)"};
      at.set(tag,n);temporaryTags.add(tag);createNode(n);
      return n;
    }
    function placeLabels(){
      activeCovers.clear();
      if(packed){syncCoverTimer();return;}
      const boxes=[];
      let coverSlots=options.coverForTag?12:0;
      const pixels=coverSlots&&svg.isConnected?scalePixels():0;
      for(const n of at.values()){
        const v=visual.get(n.tag);
        v.artwork.style.display="none";
        const show=picked?focusedTags.has(n.tag):zoom>overviewZoom*1.15||n.count>=maxCount*.18;
        v.number.style.display=show?"":"none";
      }
      const left=500+pan.x-500/zoom,right=500+pan.x+500/zoom;
      const top=350+pan.y-350/zoom,bottom=350+pan.y+350/zoom;
      const discs=labelOrder.map(n=>({...position(n),r:radius(n),tag:n.tag}));
      for(const n of labelOrder){
        const v=visual.get(n.tag),p=position(n),sz=labelSize(n);
        const gap=radius(n)+3/zoom;
        if(p.x+gap<left||p.x-gap>right||p.y+gap<top||p.y-gap>bottom){
          v.label.setAttribute("visibility","hidden");continue;
        }
        const coverSrc=coverSlots>0&&(n.tag===picked||2*radius(n)*pixels>=32)?
          options.coverForTag?.(n.tag,v.coverFrame||0):null;
        const label=coverSrc?n.tag+" · "+n.count:n.tag;
        const width=(label.length*.60+.5)*sz;
        const candidates=[[p.x+gap,p.y],[p.x-gap-width,p.y],
          [p.x-width/2,p.y-gap-sz*.6],[p.x-width/2,p.y+gap+sz*.6]];
        let best=null,bestScore=Infinity;
        for(const [x,y] of candidates){
          const box={left:x,right:x+width,top:y-sz*.6,bottom:y+sz*.6};
          const overlaps=b=>b.left<box.right&&box.left<b.right&&b.top<box.bottom&&box.top<b.bottom;
          const clipped=box.left<left||box.right>right||box.top<top||box.bottom>bottom;
          const score=(clipped?20:0)+boxes.filter(overlaps).length*4+discs.filter(d=>d.tag!==n.tag&&
            d.x+d.r>box.left&&d.x-d.r<box.right&&d.y+d.r>box.top&&d.y-d.r<box.bottom).length;
          if(score<bestScore){best={box,x,y};bestScore=score;}
          if(!score)break;
        }
        const show=bestScore===0||n.tag===picked||(!picked&&n===labelOrder[0]);
        if(show){
          v.label.textContent=label;
          if(coverSrc){
            showCover(n,v,coverSrc);
            coverSlots--;
          }
          v.label.setAttribute("x",best.x);v.label.setAttribute("y",best.y+sz*.32);
          boxes.push(best.box);
        }
        v.label.setAttribute("visibility",show?"visible":"hidden");
      }
      syncCoverTimer();
    }
    function showCover(n,v,src){
      if(!v.cover){
        v.clip=make("clipPath",{id:v.id+"-cover"},defs);
        v.coverClip=make("circle",{},v.clip);
        v.cover=make("image",{href:src,"clip-path":"url(#"+v.id+"-cover)",
          preserveAspectRatio:"xMidYMid slice"},v.artwork);
        nodeGeometry(n);
      }
      v.artwork.style.display="";
      v.number.style.display="none";
      activeCovers.add(n.tag);
    }
    function stopCoverTimer(){
      if(coverTimer!==null)clearInterval(coverTimer);
      coverTimer=null;
    }
    function syncCoverTimer(){
      const visible=!disposed&&!packed&&!collection&&stage.isConnected&&!document.hidden&&
        (typeof stage.getClientRects!=="function"||stage.getClientRects().length>0);
      if(!visible||!options.coverCountForTag||reduced()||
        ![...activeCovers].some(tag=>options.coverCountForTag(tag)>1)){
        stopCoverTimer();return;
      }
      if(coverTimer!==null)return;
      // One timer for visible covers only, no graph redraw or relayout.
      coverTimer=setInterval(()=>{
        if(disposed||packed||collection||!stage.isConnected||document.hidden||
          (typeof stage.getClientRects==="function"&&!stage.getClientRects().length)||reduced()){
          stopCoverTimer();return;
        }
        for(const tag of activeCovers){
          const v=visual.get(tag),count=options.coverCountForTag(tag);
          if(!v?.cover||count<2)continue;
          v.coverFrame=((v.coverFrame||0)+1)%count;
          const src=options.coverForTag(tag,v.coverFrame);
          if(src)v.cover.setAttribute("href",src);
        }
      },2000);
    }
    const visibilityChanged=()=>syncCoverTimer();
    document.addEventListener?.("visibilitychange",visibilityChanged);
    media?.addEventListener?.("change",visibilityChanged);
    function updateAppearance(){
      const focus=picked,neighbors=shared;
      labelOrder=picked?[...focusedTags].map(tag=>at.get(tag)):nodes;
      for(const n of at.values()){
        const v=visual.get(n.tag),selected=n.tag===picked,related=!focus||focusedTags.has(n.tag);
        v.g.setAttribute("opacity",related?1:.23);
        v.label.style.display=related?"":"none";
        v.g.setAttribute("aria-pressed",String(selected));
        const sharedCount=neighbors.get(n.tag);
        v.number.textContent=String(sharedCount??n.count);
        v.g.setAttribute("aria-label",n.tag+", "+n.count+" images"+
          (selected?", selected":sharedCount!==undefined?", "+sharedCount+" shared with "+picked:""));
        v.core.setAttribute("fill",selected?"#efd16e":prefs.glassNodes?"url(#"+v.id+")":n.color);
        v.ring.setAttribute("opacity",selected?"1":"0");
        v.halo.setAttribute("opacity",n.tag===focus?".24":"0");
        v.label.setAttribute("fill",selected?"#efd16e":"color-mix(in srgb, "+n.color+" 82%, var(--text-primary))");
        v.label.setAttribute("font-weight",selected?750:650);
        v.label.setAttribute("opacity",related?"1":".38");
      }
      updateGeometry();
    }
    function renderPanel(){
      spotlight.replaceChildren();spotlight.hidden=!picked||!at.has(picked);if(spotlight.hidden)return;
      const n=at.get(picked),head=html("p","",undefined,spotlight);
      head.style.color=n.color;
      html("strong","",n.tag,head);
      html("span","",String(n.count),head);
      if(options.collectionItems)button(spotlight,"View "+n.count+" "+(n.count===1?"image":"images"),
        "View images tagged "+picked,()=>openCollection(picked)).className="studio-tb-pair-action";
      if(options.openWaterfall)button(spotlight,"Open in Waterfall","Open "+picked+" in Waterfall",
        ()=>options.openWaterfall(picked,null,collectionState(picked,null))).className="studio-tb-pair-action";
      if(!shared.size)html("div","studio-tb-pair-empty","No co-occurring tags",spotlight);
      if(activePair)renderPairPreview();
      if(shared.size>8)button(spotlight,expanded?"Show fewer connections":"Show more connections",
        expanded?"Show fewer connections":"Show more connections",()=>{
          expanded=!expanded;layoutNeighborhood();renderPanel();updateAppearance();
          spotlight.querySelector("button")?.focus({preventScroll:true});
        }).className="studio-tb-pair-action";
      const list=html("ul","",undefined,spotlight);
      for(const [tag,w] of shared){
        const item=html("li","",undefined,list);
        const b=button(item,tag,"Focus "+tag,()=>choose(tag));b.style.color=at.get(tag)?.color||"";
        const pair=button(item,String(w),"Preview "+w+" shared images with "+tag,()=>inspectPair(tag));
        pair.className="studio-tb-pair-count-button";
      }
    }
    function renderPairPreview(){
      const matches=options.previewPair?.(picked,activePair)||[];
      const preview=html("section","studio-tb-pair-preview",undefined,spotlight);
      html("strong","",picked+" + "+activePair,preview);
      const count=shared.get(activePair);
      html("div","studio-tb-pair-count",count+(count===1?" shared image":" shared images"),preview);
      const tiles=html("div","studio-tb-pair-thumbs",undefined,preview);
      for(const {src,name} of matches){
        const img=html("img","",undefined,tiles);
        img.src=src;img.alt=name;img.title=name;img.decoding="async";
      }
      if(options.collectionItems)button(preview,"View shared images","View shared images in Constellation",
        ()=>openCollection(picked,activePair)).className="studio-tb-pair-action";
      if(options.openWaterfall||options.openPair)button(preview,"Open in Waterfall","Open shared images in Waterfall",
        ()=>options.openWaterfall?options.openWaterfall(picked,activePair,collectionState(picked,activePair)):
          options.openPair(picked,activePair)).className="studio-tb-pair-action";
    }
    function inspectPair(tag){
      const fromPanel=spotlight.contains(document.activeElement);
      activePair=tag;
      renderPanel();spotlight.scrollTop=0;updateFocusEdges();
      if(fromPanel)spotlight.querySelector(".studio-tb-pair-preview button")?.focus({preventScroll:true});
    }
    function collectionKeyframes(tag,opening){
      const stageRect=stage.getBoundingClientRect();
      const node=visual.get(tag)?.core.getBoundingClientRect();
      const x=node?Math.max(0,Math.min(stageRect.width,node.left+node.width/2-stageRect.left)):stageRect.width/2;
      const y=node?Math.max(0,Math.min(stageRect.height,node.top+node.height/2-stageRect.top)):stageRect.height/2;
      const small="circle("+(node?.width/2||14)+"px at "+x+"px "+y+"px)";
      const large="circle("+Math.hypot(stageRect.width,stageRect.height)+"px at "+x+"px "+y+"px)";
      const frames=[{clipPath:small},{clipPath:large}];
      return opening?frames:frames.reverse();
    }
    function collectionState(tag,peer,image=null,imageIndex=0,offset=0){
      return {tag,peer,path:image?.path,imageIndex,offset,zoom,pan:{...pan},overviewCamera,
        query:input.value,picked,positions:nodes.map(n=>[n.tag,n.x,n.y])};
    }
    function openCollection(tag,peer=null,resume=null,animate=true){
      if(!options.collectionItems||!tag||collection)return;
      cancelMotion();stopCoverTimer();
      const pictures=options.collectionItems(tag,peer);
      const previousFocus=document.activeElement;
      const panel=html("section","studio-tb-collection");
      const header=html("header","studio-tb-collection-header",undefined,panel);
      const back=button(header,"← Back to map","Back to Constellation map",closeCollection);
      back.className="studio-tb-pair-action";
      back.title="Back to map · Scroll down over this header or scroll up at the top of the images";
      const heading=html("h3","",tag+(peer?" + "+peer:""),header);
      heading.style.color=at.get(tag)?.color||"";
      panel.setAttribute("aria-label",heading.textContent+" image collection");
      html("span","",pictures.length+" "+(pictures.length===1?"image":"images"),header);
      if(options.openWaterfall)button(header,"Open in Waterfall","Open this collection in Waterfall",()=>{
        const state=collectionState(tag,peer);
        options.openWaterfall(tag,peer,state);
      }).className="studio-tb-pair-action";
      const related=tag===picked?[...shared.keys()].slice(0,8):[];
      if(peer&&!related.includes(peer))related.push(peer);
      if(related.length){
        const filter=html("select","studio-tb-collection-filter",undefined,header);
        filter.setAttribute("aria-label","Filter by related tag");
        html("option","","All "+tag+" images",filter).value="";
        for(const other of related)html("option","",other,filter).value=other;
        filter.value=peer||"";
        filter.addEventListener("change",()=>{
          const next=filter.value||null;
          removeCollection(false);openCollection(tag,next,null,false);
          collection.panel.querySelector(".studio-tb-collection-filter")?.focus({preventScroll:true});
        });
      }
      const flow=tagbenchWaterfall(pictures,(image,event)=>{
        const tile=event.currentTarget;
        const offset=tile.getBoundingClientRect().top-flow.getBoundingClientRect().top;
        options.editCollectionImage?.(image,collectionState(tag,peer,image,pictures.indexOf(image),offset));
      },event=>event.preventDefault(),resume?.path,null,options.thumbToken,null,null,options.waterfallScale?.()??100);
      if(pictures.length===1)flow.classList.add("is-single");
      panel.append(flow);
      collection={panel,header,flow,tag,previousFocus,wheel:null};
      for(const part of [svg,search,zoomCtl,spotlight,menu])part.inert=true;
      stage.append(panel);
      options.onCollectionChange?.(true);
      if(animate&&!reduced()&&panel.animate&&!resume)
        collectionMotion=panel.animate(collectionKeyframes(tag,true),{duration:240,easing:"cubic-bezier(.2,.75,.25,1)"});
      back.focus({preventScroll:true});
      if(resume)flow.restoreImage(resume);
    }
    function removeCollection(restoreFocus){
      if(!collection)return;
      collectionMotion?.cancel();
      collectionMotion=null;
      const {panel,flow,tag,previousFocus}=collection;
      flow.dispose();panel.remove();collection=null;
      options.onCollectionChange?.(false);
      syncCoverTimer();
      for(const part of [svg,search,zoomCtl,spotlight,menu])part.inert=false;
      if(restoreFocus){
        const focusTarget=previousFocus?.isConnected&&stage.contains(previousFocus)?
          previousFocus:visual.get(tag)?.g||input;
        focusTarget.focus({preventScroll:true});
      }
    }
    function closeCollection(){
      if(!collection||collection.closing)return;
      collection.panel.inert=true;
      if(reduced()||!collection.panel.animate){removeCollection(true);return;}
      collection.closing=true;
      collectionMotion?.cancel();
      collectionMotion=collection.panel.animate(collectionKeyframes(collection.tag,false),
        {duration:190,easing:"ease-in",fill:"forwards"});
      collectionMotion.finished.then(()=>removeCollection(true),()=>{});
    }
    function associationStrength(tag,count){
      const baseline=all.get(tag).count/Math.max(1,model.imageCount||maxCount);
      const lift=count/(at.get(picked).count*baseline);
      // Shrink isolated coincidences before ranking or sizing a line.
      const logLift=Math.log1p(lift);
      return count/(count+2)*logLift/(1+logLift);
    }
    function layoutNeighborhood(){
      focusedTags.clear();focusEdges.clear();edgeLayer.replaceChildren();
      clearTemporaryTags(picked);
      if(!picked)return;
      const limit=expanded?12:8;
      const peers=[...shared.keys()].slice(0,limit);
      focusedTags.add(picked);
      const anchor=at.get(picked);
      peers.forEach((tag,i)=>{
        const exists=at.has(tag);
        const n=ensureTag(tag);
        if(!exists){
          // Tags outside the overview occupy free space close to the anchor.
          const distance=70/zoom,angle=i*2.39996;
          for(let ring=1;ring<=4;ring++){
            n.x=anchor.x+Math.cos(angle)*distance*ring;
            n.y=anchor.y+Math.sin(angle)*distance*ring;
            if(![...at.values()].some(other=>other!==n&&
              Math.hypot(other.x-n.x,other.y-n.y)<radius(n)+radius(other)+4/zoom))break;
          }
        }
        focusedTags.add(tag);
        const count=shared.get(tag);
        const control=make("g",{role:"button",tabindex:0,class:"studio-tb-pair-edge",
          "aria-label":picked+" and "+tag+", "+count+(count===1?" shared image":" shared images")},edgeLayer);
        const line=make("path",{fill:"none",stroke:n.color,"vector-effect":"non-scaling-stroke","pointer-events":"none"},control);
        const hit=make("path",{fill:"none",stroke:"transparent","stroke-width":16,
          "vector-effect":"non-scaling-stroke","pointer-events":"stroke"},control);
        const caption=make("text",{"text-anchor":"middle",fill:"var(--text-primary)",stroke:"var(--bg-deep)",
          "paint-order":"stroke","font-weight":750,"pointer-events":"none","aria-hidden":"true"},control);
        caption.textContent=count;
        const openRelationship=()=>options.collectionItems?openCollection(picked,tag):inspectPair(tag);
        control.addEventListener("click",openRelationship);
        control.addEventListener("keydown",event=>{
          if(event.key!=="Enter"&&event.key!==" ")return;
          event.preventDefault();openRelationship();
        });
        focusEdges.set(tag,{line,hit,caption});
      });
    }
    function selectNeighborhood(tag){
      if(!picked&&tag)overviewCamera={zoom,pan:{...pan}};
      camera=null;picked=tag;activePair=null;expanded=false;
      if(tag){
        const related=options.relationsForTag?.(tag)||relationMap(tag);
        shared=new Map([...related].filter(([peer])=>peer!==tag&&all.has(peer))
          .sort((a,b)=>associationStrength(b[0],b[1])-associationStrength(a[0],a[1])
            ||b[1]-a[1]||a[0].localeCompare(b[0])));
      }else{
        shared=new Map();
        if(overviewCamera){zoom=overviewCamera.zoom;pan=overviewCamera.pan;overviewCamera=null;}
      }
      layoutNeighborhood();view();renderPanel();updateAppearance();
    }
    function choose(tag,toggle=true){
      if(tag&&toggle&&tag===picked)tag=null;
      if(tag&&!ensureTag(tag))return;
      if(pulse){visual.get(pulse.tag)?.ripple.setAttribute("opacity","0");pulse=null;}
      selectNeighborhood(tag);
      onPick?.(picked);
      if(picked&&!reduced()&&prefs.elasticity>0){pulse={tag:picked,start:performance.now(),amount:0};wake();}
    }
    function openMenu(tag,x,y){
      dismissMenu();const rect=stage.getBoundingClientRect();
      menu.hidden=false;menu.style.left=Math.max(4,Math.min(x-rect.left,rect.width-185))+"px";
      menu.style.top=Math.max(4,Math.min(y-rect.top,rect.height-150))+"px";
      button(menu,"Focus tag","Focus "+tag,()=>{choose(tag,false);dismissMenu();});
      button(menu,"Center tag","Center "+tag,()=>{choose(tag,false);center(tag);dismissMenu();});
      if(options.collectionItems)button(menu,"View images","View images tagged "+tag,()=>{
        choose(tag,false);dismissMenu();openCollection(tag);
      });
      if(options.onNavigate)for(const [label,mode] of [["View in Tags","list"],["View in Waterfall","waterfall"]])
        button(menu,label,label,()=>{choose(tag,false);dismissMenu();options.onNavigate(mode,tag);});
      menu.querySelector("button")?.focus({preventScroll:true});
    }
    function cameraTo(nextZoom,nextPan,animate=true){
      nextZoom=Math.max(.35,Math.min(5,nextZoom));
      if(reduced()||!animate){camera=null;zoom=nextZoom;pan=nextPan;view();updateAppearance();return;}
      camera={start:performance.now(),fromZoom:zoom,fromPan:{...pan},toZoom:nextZoom,toPan:nextPan};wake();
    }
    function center(tag){const n=at.get(tag);if(n)cameraTo(Math.max(zoom,1.45),{x:n.x-500,y:n.y-350});}
    function fit(animate=true){
      const values=[...at.values()],minX=Math.min(...values.map(n=>n.x))-80,maxX=Math.max(...values.map(n=>n.x))+130;
      const minY=Math.min(...values.map(n=>n.y))-80,maxY=Math.max(...values.map(n=>n.y))+80;
      overviewZoom=Math.min(1.6,1000/(maxX-minX),700/(maxY-minY));
      cameraTo(overviewZoom,{x:(minX+maxX)/2-500,y:(minY+maxY)/2-350},animate);
    }
    function zoomBy(factor,e=null){
      const next=Math.min(5,Math.max(.35,zoom*factor));
      if(next===zoom)return;
      let target={...pan};
      if(e){const p=point(e),ratio=zoom/next;
        target={x:p.x-(p.x-500-pan.x)*ratio-500,y:p.y-(p.y-350-pan.y)*ratio-350};}
      if(!picked&&!packed&&next<overviewZoom*.82&&atlas.clusters.length>1)target={x:0,y:0};
      cameraTo(next,target,!e);
    }
    button(zoomCtl,"+","Zoom in",()=>zoomBy(1.2));button(zoomCtl,"−","Zoom out",()=>zoomBy(1/1.2));
    button(zoomCtl,"⛶","Fit full graph",()=>{if(picked)choose(null);fit();});
    const searchMatches=()=>{
      matches.replaceChildren();const q=input.value.toLowerCase().trim();if(!q)return;
      const found=model.all.filter(n=>n.tag.includes(q)).sort((a,b)=>(b.tag===q)-(a.tag===q)||b.count-a.count).slice(0,8);
      for(const n of found){const b=button(matches,n.tag+" · "+n.count,"Focus "+n.tag,()=>{
        input.value=n.tag;choose(n.tag,false);center(n.tag);matches.replaceChildren();
      });b.style.color=model.colors.get(n.tag)||"var(--text-primary)";}
      if(!found.length)html("span","studio-tb-panel-caption","No matching tags",matches);
    };
    input.addEventListener("input",searchMatches);
    input.addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();e.stopPropagation();matches.querySelector("button")?.click();}if(e.key==="Escape"){e.stopPropagation();matches.replaceChildren();input.value="";}});
    function clearOffsets(){
      const changed=new Set();
      for(const n of at.values())if(n.ox||n.oy){n.ox=0;n.oy=0;changed.add(n.tag);}
      settling=null;if(changed.size)updateGeometry(changed);
    }
    function cancelMotion(){camera=null;clearOffsets();}
    function wake(){if(frame===null&&!disposed)frame=requestAnimationFrame(tick);}
    function tick(now){
      frame=null;if(disposed)return;
      if(camera){
        const c=camera,t=Math.min(1,(now-c.start)/180),k=1-Math.pow(1-t,3);
        zoom=c.fromZoom+(c.toZoom-c.fromZoom)*k;pan={x:c.fromPan.x+(c.toPan.x-c.fromPan.x)*k,y:c.fromPan.y+(c.toPan.y-c.fromPan.y)*k};
        view();updateGeometry();if(t>=1)camera=null;
      }
      if(settling){
        const t=Math.min(1,(now-settling.start)/320),k=Math.pow(1-t,3)*Math.cos(t*Math.PI*1.1);
        for(const [tag,p] of settling.offsets){const n=at.get(tag);if(n){n.ox=p.x*k;n.oy=p.y*k;dirty.add(tag);}}
        if(t>=1){settling=null;clearOffsets();}
      }
      if(pulse){
        const t=Math.min(1,(now-pulse.start)/260),v=visual.get(pulse.tag);
        pulse.amount=t;dirty.add(pulse.tag);v?.ripple.setAttribute("opacity",String((1-t)*.65));
        if(t>=1)pulse=null;
      }
      if(dirty.size){updateGeometry(dirty);dirty.clear();}
      if(camera||settling||pulse)wake();
    }
    svg.addEventListener("pointerdown",e=>{
      if(e.button!==0||e.target.closest(".studio-tb-orb,.studio-tb-pair-edge,.studio-tb-atlas-community"))return;
      dismissMenu();cancelMotion();const p=point(e);
      drag={kind:"pan",x:e.clientX,y:e.clientY,world:p,px:pan.x,py:pan.y,moved:false};
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener("pointermove",e=>{
      if(!drag)return;
      const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
      if(!drag.moved&&Math.hypot(dx,dy)<4)return;drag.moved=true;
      if(drag.kind==="node"){
        const n=at.get(drag.tag),p=point(e);if(!n)return;
        n.x=p.x-drag.dx;n.y=p.y-drag.dy;dirty.add(n.tag);
        const scale=scalePixels();
        if(!reduced()&&prefs.elasticity>0)for(const [tag,w] of relationMap(n.tag)){
          const peer=at.get(tag);if(!peer)continue;
          const offset=elasticOffset(dx,dy,w/Math.max(1,n.count),prefs.elasticity);
          peer.ox=offset.x/scale;peer.oy=offset.y/scale;dirty.add(tag);
        }
        wake();
      }else{
        const scale=scalePixels();pan={x:drag.px-dx/scale,y:drag.py-dy/scale};view();drawStrands();placeLabels();
      }
    });
    function release(e,cancel=false){
      if(!drag)return;const prev=drag;drag=null;
      if(prev.kind==="node"&&!prev.moved&&!cancel){
        const now=performance.now();
        if(lastClick?.tag===prev.tag&&now-lastClick.time<330){choose(prev.tag,false);center(prev.tag);lastClick=null;}
        else{choose(prev.tag);lastClick={tag:prev.tag,time:now};}
      }else if(prev.kind==="pan"&&!prev.moved&&!cancel){choose(null);lastClick=null;}
      const offsets=new Map([...at.values()].filter(n=>n.ox||n.oy).map(n=>[n.tag,{x:n.ox,y:n.oy}]));
      if(offsets.size&&!reduced()&&prefs.elasticity>0){settling={start:performance.now(),offsets};wake();}
      else clearOffsets();
    }
    svg.addEventListener("pointerup",e=>release(e));svg.addEventListener("pointercancel",e=>release(e,true));
    svg.addEventListener("lostpointercapture",e=>release(e,true));
    svg.addEventListener("wheel",e=>{
      e.preventDefault();
      if(!e.deltaY)return;
      const overPicked=picked&&e.target.closest(".studio-tb-orb")===visual.get(picked)?.g;
      if(overPicked&&e.deltaY<0&&zoom*1.12>=Math.min(3.5,overviewZoom*1.7)&&options.collectionItems){
        openCollection(picked);
        // Ignore the rest of the wheel gesture that zoomed into the circle.
        collection.wheel={direction:-1,canExit:false,distance:0,at:e.timeStamp};
        return;
      }
      cancelMotion();zoomBy(e.deltaY<0?1.12:1/1.12,e);
    },{passive:false});
    stage.addEventListener("wheel",event=>{
      const {deltaX,deltaY,deltaMode,timeStamp,target}=event;
      if(event.ctrlKey||event.shiftKey||!deltaY||Math.abs(deltaX)>Math.abs(deltaY))return;
      const direction=Math.sign(deltaY);
      if(closingWheel){
        if(direction===closingWheel.direction&&timeStamp-closingWheel.at<wheelPause){
          closingWheel.at=timeStamp;
          event.preventDefault();event.stopPropagation();
          return;
        }
        closingWheel=null;
      }
      if(!collection||collection.closing||target.closest?.("select"))return;

      const {header,flow}=collection;
      const canExit=direction<0?flow.scrollTop<=0:header.contains(target);
      const previous=collection.wheel;
      if(!previous||direction!==previous.direction||timeStamp-previous.at>=wheelPause){
        // Only a gesture that starts at the boundary can leave the collection.
        collection.wheel={direction,canExit,distance:0,at:timeStamp};
      }
      const gesture=collection.wheel;
      gesture.at=timeStamp;
      if(!canExit){
        gesture.canExit=false;
        return;
      }
      event.preventDefault();
      if(!gesture.canExit)return;
      const unit=deltaMode===1?16:deltaMode===2?Math.max(wheelExitDistance,flow.clientHeight):1;
      gesture.distance+=Math.abs(deltaY)*unit;
      if(gesture.distance>=wheelExitDistance){
        closingWheel={direction,at:timeStamp};
        closeCollection();
      }
    },{capture:true,passive:false});
    stage.addEventListener("keydown",e=>{if(e.key==="Escape"){e.preventDefault();e.stopPropagation();if(collection)closeCollection();else if(!menu.hidden)dismissMenu();else choose(null);}});
    stage.addEventListener("pointerdown",e=>{if(!menu.contains(e.target))dismissMenu();if(!search.contains(e.target))matches.replaceChildren();});
    stage.setPreferences=value=>{
      prefs=constellationPrefs(value);stage.classList.toggle("no-celestial",!prefs.celestial);
      if(!prefs.elasticity||reduced()){
        clearOffsets();if(pulse){visual.get(pulse.tag)?.ripple.setAttribute("opacity","0");pulse=null;}
      }
      updateAppearance();
    };
    const resize=new ResizeObserver(()=>{
      const scale=Math.min(svg.clientWidth/1000,svg.clientHeight/700);
      if(scale>0){typeScale=Math.max(.8,Math.min(1.7,1/scale));updateGeometry();}
    });
    resize.observe(stage);
    const theme=new MutationObserver(()=>{strandColors.clear();drawStrands();});
    theme.observe(document.documentElement,{attributes:true,attributeFilter:["class","data-theme","style"]});
    stage.dispose=()=>{disposed=true;stopCoverTimer();document.removeEventListener?.("visibilitychange",visibilityChanged);
      media?.removeEventListener?.("change",visibilityChanged);removeCollection(false);resize.disconnect();theme.disconnect();
      if(frame!==null)cancelAnimationFrame(frame);frame=null;camera=null;settling=null;pulse=null;};
    stage.append(canvas,svg,search,zoomCtl,spotlight,menu);
    stage.setPreferences(prefs);view();
    if(nodes.length)fit(false);
    else html("p","studio-tb-empty-graph","No tagged images in this dataset.",stage);
    if(pickedTag&&ensureTag(pickedTag))selectNeighborhood(pickedTag);
    stage.resumeCollection=state=>{
      for(const [tag,x,y] of state.positions||[]){
        const node=at.get(tag);
        if(node){node.x=x;node.y=y;}
      }
      if(picked)layoutNeighborhood();
      overviewCamera=state.overviewCamera;
      input.value=state.query;
      cameraTo(state.zoom,state.pan,false);
      openCollection(state.tag,state.peer,state);
    };
    return stage;
  }

  return Object.freeze({buildGraph,waterfall,constellation,colorForTag,setTagbenchPalette,tagbenchModel,tagbenchWaterfall,tagbenchConstellation,waterfallScale,waterfallColumns,constellationPrefs,elasticOffset,constellationAtlas});
})();
// A classic WebWorker can import this existing pure graph implementation
// without maintaining a second constellation algorithm.
if(typeof self!=="undefined"&&typeof document==="undefined")
  self.CaptionStudioViz=CaptionStudioViz;
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioViz;
