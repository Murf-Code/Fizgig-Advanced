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
  const tbHue=id=>{
    const hue=id<TB_HUES.length?TB_HUES[id]:
      "hsl("+Math.round(id*137.508%360)+" 72% 62%)";
    return "color-mix(in srgb, "+hue+" 72%, var(--text-primary))";
  };
  function tagbenchGraph(items) {
    // Bound pair storage by the visible vocabulary. Search still indexes every tag.
    const counts=new Map();
    for(const it of items){
      const unique=new Set((it.tags||[]).map(lower).filter(Boolean));
      for(const tag of unique)counts.set(tag,(counts.get(tag)||0)+1);
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
    return {all,top,edges,linkCounts};
  }
  function tagbenchModel(items,categories) {
    const graph=tagbenchGraph(items),{all,top,edges,linkCounts}=graph;
    const index=new Map(top.map((row,i)=>[row.tag,i]));
    const neighbors=top.map(()=>[]);
    // Cosine normalization discounts ubiquitous tags; shrink one-off coincidences.
    for(const edge of edges){
      const i=index.get(edge.a),j=index.get(edge.b);
      const weight=edge.w/Math.sqrt(top[i].count*top[j].count)*edge.w/(edge.w+2);
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
    return {...graph,nodes,chosen,colors};
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
    return 4+3.8*String(count).length+10*Math.sqrt(count/maxCount);
  }
  function tagbenchConstellation(model,onPick,pickedTag=null,options={}){
    const stage=document.createElement("div");stage.className="studio-tb-constellation";
    if(!model.nodes.length){stage.textContent="No tagged images in this dataset.";return stage;}
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
    const nodeLayer=make("g"),labelLayer=make("g");
    const nodes=model.nodes.map(n=>({...n,ox:0,oy:0}));
    const at=new Map(nodes.map(n=>[n.tag,n])),all=new Map(model.all.map(n=>[n.tag,n]));
    const adjacency=new Map(nodes.map(n=>[n.tag,new Map()]));
    for(const e of model.edges){adjacency.get(e.a)?.set(e.b,e.w);adjacency.get(e.b)?.set(e.a,e.w);}
    const maxCount=nodes[0].count;
    const visual=new Map();
    const leaders=new Map();
    // The model is frequency-ranked; give each community a named landmark.
    for(const n of nodes)if(!leaders.has(n.group))leaders.set(n.group,n);
    let labelOrder=nodes,overviewCamera=null;
    const focusPositions=new Map();
    let typeScale=1;
    const labelSize=n=>(n.tag===picked?22:leaders.get(n.group)===n?19:15)*typeScale/zoom;

    const bridges=[],communityStrands=new Map();
    for(const edge of model.chosen){
      const a=at.get(edge.a),b=at.get(edge.b);
      if(a.group!==b.group||a.color!==b.color){bridges.push(edge);continue;}
      if(!communityStrands.has(a.color))communityStrands.set(a.color,[]);
      communityStrands.get(a.color).push(edge);
    }
    // Resolve theme-dependent SVG colors after mount, then reuse them during motion.
    const strandColors=new Map();
    let prefs=constellationPrefs(options.preferences),zoom=1,pan={x:0,y:0},picked=null,drag=null,rare=null;
    let serial=0,shared=new Map(),frame=null,settling=null,camera=null,pulse=null,disposed=false,dirty=new Set(),lastClick=null;
    const media=typeof matchMedia==="function"?matchMedia("(prefers-reduced-motion: reduce)"):null;
    const reduced=()=>!!media?.matches;
    // Reserve room for the full total even when selection shows a shorter shared count.
    const radius=n=>countRadius(n.count,maxCount)*typeScale/zoom;
    const location=n=>focusPositions.get(n.tag)||n;
    const position=n=>({x:location(n).x+n.ox,y:location(n).y+n.oy});
    const view=()=>svg.setAttribute("viewBox",[500-500/zoom+pan.x,350-350/zoom+pan.y,1000/zoom,700/zoom].join(" "));
    const search=html("div","studio-tb-search"),input=html("input","",undefined,search),matches=html("div","studio-tb-suggest",undefined,search);
    input.placeholder="Search Tags";input.setAttribute("aria-label","Search all dataset tags");
    const zoomCtl=html("div","studio-tb-zoom"),spotlight=html("aside","studio-tb-cluster");
    spotlight.hidden=true;
    const menu=html("div","studio-tb-node-menu");menu.hidden=true;menu.setAttribute("role","group");
    const dismissMenu=()=>{menu.hidden=true;menu.replaceChildren();};
    const point=e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());};
    const scalePixels=()=>Math.max(.001,Math.abs(svg.getScreenCTM()?.a||1));
    function drawStrands(){
      const width=svg.clientWidth,height=svg.clientHeight;
      if(!ctx||!width||!height)return;
      const ratio=Math.min(window.devicePixelRatio||1,2);
      const w=Math.round(width*ratio),h=Math.round(height*ratio);
      if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
      ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);
      const scale=Math.min(width/1000,height/700)*zoom*ratio;
      ctx.setTransform(scale,0,0,scale,w/2-(500+pan.x)*scale,h/2-(350+pan.y)*scale);
      const curve=(a,b)=>{
        const p=position(at.get(a)),q=position(at.get(b));
        ctx.moveTo(p.x,p.y);
        ctx.quadraticCurveTo((p.x+q.x)/2-(q.y-p.y)*.09,(p.y+q.y)/2+(q.x-p.x)*.09,q.x,q.y);
      };
      ctx.lineWidth=.85*ratio/scale;
      const stroke=(edges,color,alpha)=>{
        if(!edges.length)return;
        ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.beginPath();
        for(const e of edges)curve(e.a,e.b);
        ctx.stroke();
      };
      if(!picked){
        stroke(bridges,getComputedStyle(canvas).color,picked?.045:.14);
        for(const [color,edges] of communityStrands){
          if(!strandColors.has(color)){
            strandColors.set(color,getComputedStyle(visual.get(edges[0].a).halo).fill);
          }
          stroke(edges,strandColors.get(color),picked?.11:.48);
        }
      }else{
        ctx.strokeStyle="#e2b84e";
        for(const tag of focusPositions.keys()){
          if(tag===picked)continue;
          const count=shared.get(tag);
          const strength=Math.sqrt(count/at.get(picked).count);
          ctx.lineWidth=(.6+6*strength)*ratio/scale;
          ctx.globalAlpha=.24+.66*strength;
          ctx.beginPath();curve(picked,tag);ctx.stroke();
        }
      }
    }
    const nodeGeometry=n=>{
      const v=visual.get(n.tag);if(!v)return;
      const p=position(n),r=radius(n),sz=labelSize(n);
      v.g.setAttribute("transform","translate("+p.x+" "+p.y+")");
      v.hit.setAttribute("r",Math.max(9/zoom,r));v.core.setAttribute("r",r);v.halo.setAttribute("r",r*1.34);
      v.ring.setAttribute("r",r+3/zoom);
      v.number.setAttribute("font-size",13*typeScale/zoom);
      v.number.setAttribute("stroke-width",1.2/zoom);
      v.label.setAttribute("x",p.x+r+6/zoom);v.label.setAttribute("y",p.y+sz*.32);
      v.label.setAttribute("font-size",sz);v.label.setAttribute("stroke-width",Math.max(sz*.28,2/zoom));
      v.ripple.setAttribute("r",r+4/zoom+(pulse?.tag===n.tag?pulse.amount*13/zoom:0));
    };
    const updateGeometry=(changed=null)=>{
      if(!changed){for(const n of at.values())nodeGeometry(n);}
      else for(const tag of changed){const n=at.get(tag);if(n)nodeGeometry(n);}
      drawStrands();
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
      const number=make("text",{"text-anchor":"middle","dominant-baseline":"central",
        "font-family":"system-ui, sans-serif","font-weight":750,style:"font-variant-numeric:tabular-nums",fill:"#161823",stroke:"#fff",
        "stroke-opacity":.75,"paint-order":"stroke","pointer-events":"none","aria-hidden":"true"},g);
      const hit=make("circle",{fill:"transparent"},g);
      const label=make("text",{fill:"var(--text-primary)","font-family":"system-ui, sans-serif","font-weight":leaders.get(n.group)===n?750:600,stroke:"var(--bg-deep)","paint-order":"stroke","pointer-events":"none"},labelLayer);
      label.textContent=n.tag;
      visual.set(n.tag,{g,core,halo,ripple,ring,hit,number,label,grad,id});
      g.addEventListener("pointerdown",e=>{
        if(e.button!==0)return;e.stopPropagation();dismissMenu();cancelMotion();
        const p=point(e);drag={kind:"node",tag:n.tag,dx:p.x-location(n).x,dy:p.y-location(n).y,x:e.clientX,y:e.clientY,moved:false};
        svg.setPointerCapture(e.pointerId);
      });
      g.addEventListener("keydown",e=>{
        if(e.key==="Enter"||e.key===" "){e.preventDefault();e.stopPropagation();choose(n.tag);}
        if(e.key==="ContextMenu"||(e.shiftKey&&e.key==="F10")){e.preventDefault();const r=g.getBoundingClientRect();openMenu(n.tag,r.left,r.bottom);}
      });
      g.addEventListener("contextmenu",e=>{e.preventDefault();e.stopPropagation();openMenu(n.tag,e.clientX,e.clientY);});
      nodeGeometry(n);
    }
    for(const n of nodes)createNode(n);
    function relationMap(tag){return adjacency.get(tag)||new Map();}
    function removeRare(){
      if(!rare)return;
      const v=visual.get(rare.tag);v?.g.remove();v?.label.remove();v?.grad.remove();
      visual.delete(rare.tag);at.delete(rare.tag);adjacency.delete(rare.tag);rare=null;
    }
    function ensureTag(tag){
      if(at.has(tag))return at.get(tag);
      const entry=all.get(tag);if(!entry)return null;
      removeRare();
      const related=options.relationsForTag?.(tag)||new Map();
      const links=new Map([...related].filter(([key])=>at.has(key)));
      let x=500+pan.x,y=350+pan.y;
      const nearest=[...links].sort((a,b)=>b[1]-a[1])[0];
      if(nearest){const n=at.get(nearest[0]);x=n.x+45;y=n.y+35;}
      rare={...entry,links:links.size,x,y,ox:0,oy:0,color:model.colors.get(tag)||"var(--accent-text)"};
      at.set(tag,rare);adjacency.set(tag,links);createNode(rare);return rare;
    }
    function placeLabels(){
      const boxes=[];
      const discs=labelOrder.map(n=>({...position(n),r:radius(n),tag:n.tag}));
      for(const n of labelOrder){
        const v=visual.get(n.tag),p=position(n),sz=labelSize(n);
        const width=(n.tag.length*.60+.5)*sz,gap=radius(n)+3/zoom;
        const candidates=[[p.x+gap,p.y],[p.x-gap-width,p.y],
          [p.x-width/2,p.y-gap-sz*.6],[p.x-width/2,p.y+gap+sz*.6]];
        let best=null,bestScore=Infinity;
        for(const [left,y] of candidates){
          const box={left,right:left+width,top:y-sz*.6,bottom:y+sz*.6};
          const overlaps=b=>b.left<box.right&&box.left<b.right&&b.top<box.bottom&&box.top<b.bottom;
          const score=boxes.filter(overlaps).length*4+discs.filter(d=>d.tag!==n.tag&&
            d.x+d.r>box.left&&d.x-d.r<box.right&&d.y+d.r>box.top&&d.y-d.r<box.bottom).length;
          if(score<bestScore){best={box,left,y};bestScore=score;}
          if(!score)break;
        }
        const show=!!picked||bestScore===0||n===labelOrder[0];
        if(show){
          v.label.setAttribute("x",best.left);v.label.setAttribute("y",best.y+sz*.32);
          boxes.push(best.box);
        }
        v.label.setAttribute("visibility",show?"visible":"hidden");
      }
    }
    function updateAppearance(){
      const focus=picked,neighbors=relationMap(focus);
      labelOrder=picked?[...focusPositions.keys()].map(tag=>at.get(tag)):nodes;
      for(const n of at.values()){
        const v=visual.get(n.tag),selected=n.tag===picked,related=!focus||focusPositions.has(n.tag);
        v.g.style.display=v.label.style.display=related?"":"none";
        v.g.setAttribute("opacity",related?1:.22);
        v.g.setAttribute("aria-pressed",String(selected));
        const sharedCount=neighbors.get(n.tag);
        v.number.textContent=String(sharedCount??n.count);
        v.g.setAttribute("aria-label",n.tag+", "+n.count+" images"+
          (selected?", selected":sharedCount!==undefined?", "+sharedCount+" shared with "+picked:""));
        v.core.setAttribute("fill",selected?"#efd16e":prefs.glassNodes?"url(#"+v.id+")":n.color);
        v.ring.setAttribute("opacity",selected?"1":"0");
        v.halo.setAttribute("opacity",n.tag===focus?".24":"0");
        v.label.setAttribute("fill",selected?"#efd16e":leaders.get(n.group)===n?
          "color-mix(in srgb, "+n.color+" 50%, var(--text-primary))":"var(--text-primary)");
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
      const list=html("ul","",undefined,spotlight);
      for(const [tag,w] of shared){
        const item=html("li","",undefined,list);
        const b=button(item,tag,"Focus "+tag,()=>choose(tag));b.style.color=at.get(tag)?.color||"";
        html("span","",String(w),item);
      }
    }
    function selectNeighborhood(tag){
      if(!picked&&tag)overviewCamera={zoom,pan:{...pan}};
      camera=null;focusPositions.clear();picked=tag;shared=new Map([...relationMap(tag)].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])));
      if(tag){
        const peers=[...shared].slice(0,8);
        focusPositions.set(tag,{x:360,y:300});
        peers.forEach(([peer],i)=>focusPositions.set(peer,{x:590,y:350+(i-(peers.length-1)/2)*70}));
        zoom=1;pan={x:0,y:0};
      }else if(overviewCamera){
        zoom=overviewCamera.zoom;pan=overviewCamera.pan;overviewCamera=null;
      }
      view();renderPanel();updateAppearance();
    }
    function choose(tag,toggle=true){
      if(tag&&toggle&&tag===picked)tag=null;
      if(tag&&!ensureTag(tag))return;
      if(rare&&rare.tag!==tag)removeRare();
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
      if(options.onNavigate)for(const [label,mode] of [["View in Tags","list"],["View in Waterfall","waterfall"]])
        button(menu,label,label,()=>{choose(tag,false);dismissMenu();options.onNavigate(mode,tag);});
      menu.querySelector("button")?.focus({preventScroll:true});
    }
    function cameraTo(nextZoom,nextPan,animate=true){
      nextZoom=Math.max(.35,Math.min(5,nextZoom));
      if(reduced()||!animate){camera=null;zoom=nextZoom;pan=nextPan;view();updateAppearance();return;}
      camera={start:performance.now(),fromZoom:zoom,fromPan:{...pan},toZoom:nextZoom,toPan:nextPan};wake();
    }
    function center(tag){if(picked){cameraTo(1,{x:0,y:0});return;}const n=at.get(tag);if(n)cameraTo(Math.max(zoom,1.45),{x:n.x-500,y:n.y-350});}
    function fit(animate=true){
      const values=[...at.values()],minX=Math.min(...values.map(n=>n.x))-80,maxX=Math.max(...values.map(n=>n.x))+130;
      const minY=Math.min(...values.map(n=>n.y))-80,maxY=Math.max(...values.map(n=>n.y))+80;
      cameraTo(Math.min(1.6,1000/(maxX-minX),700/(maxY-minY)),{x:(minX+maxX)/2-500,y:(minY+maxY)/2-350},animate);
    }
    function zoomBy(factor,e=null){
      const next=Math.min(5,Math.max(.35,zoom*factor));
      if(next===zoom)return;
      let target={...pan};
      if(e){const p=point(e),ratio=zoom/next;
        target={x:p.x-(p.x-500-pan.x)*ratio-500,y:p.y-(p.y-350-pan.y)*ratio-350};}
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
      if(e.button!==0||e.target.closest(".studio-tb-orb"))return;
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
        location(n).x=p.x-drag.dx;location(n).y=p.y-drag.dy;dirty.add(n.tag);
        const scale=scalePixels();
        if(!reduced()&&prefs.elasticity>0)for(const [tag,w] of relationMap(n.tag)){
          const peer=at.get(tag);if(!peer)continue;
          const offset=elasticOffset(dx,dy,w/Math.max(1,n.count),prefs.elasticity);
          peer.ox=offset.x/scale;peer.oy=offset.y/scale;dirty.add(tag);
        }
        wake();
      }else{
        const scale=scalePixels();pan={x:drag.px-dx/scale,y:drag.py-dy/scale};view();drawStrands();
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
    svg.addEventListener("wheel",e=>{e.preventDefault();cancelMotion();zoomBy(e.deltaY<0?1.12:1/1.12,e);},{passive:false});
    stage.addEventListener("keydown",e=>{if(e.key==="Escape"){e.preventDefault();e.stopPropagation();if(!menu.hidden)dismissMenu();else choose(null);}});
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
    stage.dispose=()=>{resize.disconnect();theme.disconnect();disposed=true;if(frame!==null)cancelAnimationFrame(frame);frame=null;camera=null;settling=null;pulse=null;};
    stage.append(canvas,svg,search,zoomCtl,spotlight,menu);
    stage.setPreferences(prefs);view();
    fit(false);
    if(pickedTag&&ensureTag(pickedTag))selectNeighborhood(pickedTag);
    return stage;
  }

  return Object.freeze({buildGraph,waterfall,constellation,colorForTag,setTagbenchPalette,tagbenchModel,tagbenchWaterfall,tagbenchConstellation,waterfallScale,waterfallColumns,constellationPrefs,elasticOffset});
})();
// A classic WebWorker can import this existing pure graph implementation
// without maintaining a second constellation algorithm.
if(typeof self!=="undefined"&&typeof document==="undefined")
  self.CaptionStudioViz=CaptionStudioViz;
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioViz;
