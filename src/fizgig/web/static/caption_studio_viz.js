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
  // Constellation uses a compact, co-occurrence network map with focus relationships,
  // sparse curved strands, frequency-sized points and pan/zoom controls.
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
    // First count tags, then collect co-occurrences only for the top 120
    // that Constellation displays. The old one-pass implementation built
    // pairwise combinations of *every* dataset tag before discarding most.
    // This retains the exact same visible nodes and edge weights, but caps
    // the pair map at 7,140 entries even on very large datasets.
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
    }).sort((a,b)=>b.w-a.w);
    const linkCounts=new Map();
    for(const edge of edges){
      linkCounts.set(edge.a,(linkCounts.get(edge.a)||0)+1);
      linkCounts.set(edge.b,(linkCounts.get(edge.b)||0)+1);
    }
    return {all,top,edges,linkCounts};
  }
  // Cluster by normalized co-occurrence rather than predefined tag families.
  // The local modularity pass is deterministic and runs in the existing worker.
  function networkCommunities(top,edges,total){
    const n=top.length,at=new Map(top.map((t,i)=>[t.tag,i]));
    const adj=Array.from({length:n},()=>[]),degree=new Float64Array(n);
    const weighted=edges.map(e=>{
      const i=at.get(e.a),j=at.get(e.b),a=top[i].count,b=top[j].count;
      const expected=a*b/Math.max(1,total);
      const weight=e.w/Math.sqrt(a*b)*Math.max(.06,Math.min(1.5,Math.log1p(e.w/Math.max(expected,.01))-.5));
      return {...e,i,j,weight};
    }).sort((a,b)=>b.weight-a.weight||b.w-a.w||a.i-b.i||a.j-b.j);
    for(const e of weighted){
      adj[e.i].push([e.j,e.weight]);adj[e.j].push([e.i,e.weight]);
      degree[e.i]+=e.weight;degree[e.j]+=e.weight;
    }
    const group=top.map((_,i)=>i),mass=Float64Array.from(degree);
    const sum=degree.reduce((a,b)=>a+b,0)||1;
    for(let pass=0;pass<24;pass++){
      let moved=false;
      for(let i=0;i<n;i++){
        if(!degree[i])continue;
        const old=group[i],weights=new Map();
        for(const [j,w] of adj[i])weights.set(group[j],(weights.get(group[j])||0)+w);
        mass[old]-=degree[i];
        let best=old,score=(weights.get(old)||0)-degree[i]*mass[old]/sum;
        for(const [g,w] of weights){
          const gain=w-degree[i]*mass[g]/sum;
          if(gain>score+1e-10){best=g;score=gain;}
        }
        group[i]=best;mass[best]+=degree[i];moved||=best!==old;
      }
      if(!moved)break;
    }
    const ids=new Map();
    return {weighted,adj,group:group.map(g=>{if(!ids.has(g))ids.set(g,ids.size);return ids.get(g);})};
  }
  function networkRadius(count,maxCount){
    return 2.5+7.5*Math.sqrt(Math.max(0,count)/Math.max(1,maxCount));
  }
  function tagbenchModel(items,categories) {
    const graph=tagbenchGraph(items),{all,top,edges,linkCounts}=graph;
    const {weighted,adj,group}=networkCommunities(top,edges,items.length);
    const nodes=top.map((t,i)=>({tag:t.tag,count:t.count,links:linkCounts.get(t.tag)||0,
      group:group[i],x:0,y:0,color:null}));
    const maxCount=top[0]?.count||1,centers=new Map();
    const groups=new Map();
    nodes.forEach((node,i)=>{if(!groups.has(node.group))groups.set(node.group,[]);groups.get(node.group).push(i);});
    const clusters=[...groups].sort((a,b)=>b[1].length-a[1].length||a[0]-b[0]);
    // Seed the largest community at the center, then pack the others around
    // it on a compact spiral instead of leaving a hollow middle.
    clusters.forEach(([id,members],rank)=>{
      const a=rank*2.39996323,r=rank===0?0:31*Math.sqrt(rank);
      const center={x:Math.cos(a)*r,y:Math.sin(a)*r};centers.set(id,center);
      members.forEach((i,k)=>{
        const a=k*2.39996323,r=k?16*Math.sqrt(k):0;
        nodes[i].x=center.x+Math.cos(a)*r;nodes[i].y=center.y+Math.sin(a)*r;
      });
    });
    const chosen=[],seen=new Set();
    // Sparse network: each tag contributes its strongest local and bridge
    // edges. Full co-occurrence counts remain available on selection.
    for(let i=0;i<nodes.length;i++){
      const nearby=adj[i].slice().sort((a,b)=>b[1]-a[1]||a[0]-b[0]);
      const local=nearby.filter(([j])=>group[j]===group[i]).slice(0,5);
      const bridges=nearby.filter(([j])=>group[j]!==group[i]).slice(0,2);
      for(const [j] of [...local,...bridges])seen.add(Math.min(i,j)+":"+Math.max(i,j));
    }
    const springs=weighted.filter(e=>seen.has(Math.min(e.i,e.j)+":"+Math.max(e.i,e.j)));
    for(const e of springs)chosen.push({a:e.a,b:e.b,w:e.w,weight:e.weight,bridge:group[e.i]!==group[e.j]});
    const alike=[]; // no artificial semantic links: every visible strand is measured
    for(let step=0;step<180;step++){
      const fx=new Float64Array(nodes.length),fy=new Float64Array(nodes.length);
      for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
        const a=nodes[i],b=nodes[j],dx=a.x-b.x,dy=a.y-b.y;
        const d=Math.max(.5,Math.hypot(dx,dy)),min=networkRadius(a.count,maxCount)+networkRadius(b.count,maxCount)+7;
        const repulse=55/(d*d)+Math.max(0,min-d)*.13/d;
        fx[i]+=dx*repulse;fy[i]+=dy*repulse;fx[j]-=dx*repulse;fy[j]-=dy*repulse;
      }
      for(const e of springs){
        const a=nodes[e.i],b=nodes[e.j],dx=b.x-a.x,dy=b.y-a.y,d=Math.max(.5,Math.hypot(dx,dy));
        const same=a.group===b.group,target=same?22:52;
        const force=(d-target)*(.012+Math.min(1,e.weight)*.055)*(same?1:.45)/d;
        fx[e.i]+=dx*force;fy[e.i]+=dy*force;fx[e.j]-=dx*force;fy[e.j]-=dy*force;
      }
      const cooling=.8*(1-step/180)+.12;
      nodes.forEach((n,i)=>{
        const c=centers.get(n.group);
        // Gentle gravity closes the middle without changing community identity.
        fx[i]+=(c.x-n.x)*.018-n.x*.012;fy[i]+=(c.y-n.y)*.018-n.y*.012;
        const norm=Math.max(1,Math.hypot(fx[i],fy[i])/7);
        n.x+=fx[i]*cooling/norm;n.y+=fy[i]*cooling/norm;
      });
    }
    // Separate small hit targets without making giant empty spaces.
    for(let step=0;step<20;step++)for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.max(.5,Math.hypot(dx,dy));
      const min=networkRadius(a.count,maxCount)+networkRadius(b.count,maxCount)+8;
      if(d<min){const k=(min-d)/2/d;a.x+=dx*k;a.y+=dy*k;b.x-=dx*k;b.y-=dy*k;}
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
    return {...graph,nodes,chosen,alike,colors};
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
      glass:p?.glass!==false,celestial:p?.celestial!==false};
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
    const baseLayer=make("g"),focusLayer=make("g"),nodeLayer=make("g"),labelLayer=make("g");
    const nodes=model.nodes.map(n=>({...n,ox:0,oy:0}));
    const at=new Map(nodes.map(n=>[n.tag,n])),all=new Map(model.all.map(n=>[n.tag,n]));
    const adjacency=new Map(nodes.map(n=>[n.tag,new Map()]));
    for(const e of model.edges){adjacency.get(e.a)?.set(e.b,e.w);adjacency.get(e.b)?.set(e.a,e.w);}
    const maxCount=Math.max(1,...nodes.map(n=>n.count));
    const visual=new Map(),baseEdges=[],focusEdges=[];
    let prefs=constellationPrefs(options.preferences),zoom=1,pan={x:0,y:0},picked=null,drag=null,rare=null;
    let serial=0,shared=new Map(),frame=null,settling=null,camera=null,pulse=null,disposed=false,dirty=new Set(),lastClick=null;
    const media=typeof matchMedia==="function"?matchMedia("(prefers-reduced-motion: reduce)"):null;
    const reduced=()=>!!media?.matches;
    const radius=n=>networkRadius(n.count,maxCount)/Math.pow(zoom,.38);
    const position=n=>({x:n.x+n.ox,y:n.y+n.oy});
    const view=()=>svg.setAttribute("viewBox",[500-500/zoom+pan.x,350-350/zoom+pan.y,1000/zoom,700/zoom].join(" "));
    const search=html("div","studio-tb-search"),input=html("input","",undefined,search),matches=html("div","studio-tb-suggest",undefined,search);
    input.placeholder="Search Tags";input.setAttribute("aria-label","Search all dataset tags");
    const zoomCtl=html("div","studio-tb-zoom"),spotlight=html("aside","studio-tb-cluster");
    spotlight.hidden=true;
    const menu=html("div","studio-tb-node-menu");menu.hidden=true;menu.setAttribute("role","group");
    const dismissMenu=()=>{menu.hidden=true;menu.replaceChildren();};
    const point=e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());};
    const scalePixels=()=>Math.max(.001,Math.abs(svg.getScreenCTM()?.a||1));
    const edgeGeometry=edge=>{
      const a=at.get(edge.a),b=at.get(edge.b);if(!a||!b)return;
      const p=position(a),q=position(b),dx=q.x-p.x,dy=q.y-p.y,len=Math.hypot(dx,dy)||1;
      const start=Math.min(len/2,radius(a)),end=Math.min(len/2,radius(b));
      const x1=p.x+dx/len*start,y1=p.y+dy/len*start,x2=q.x-dx/len*end,y2=q.y-dy/len*end;
      const bend=Math.min(28,len*.16)*(edge.a<edge.b?1:-1);
      const cx=(x1+x2)/2-dy/len*bend,cy=(y1+y2)/2+dx/len*bend;
      edge.el.setAttribute("d",`M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`);
    };
    const addEdge=(e,layer,kind)=>{
      const line=make("path",{fill:"none","vector-effect":"non-scaling-stroke","pointer-events":"none",class:"studio-tb-edge "+kind},layer);
      const rec={...e,el:line};edgeGeometry(rec);return rec;
    };
    for(const e of model.chosen)if(at.has(e.a)&&at.has(e.b))baseEdges.push(addEdge(e,baseLayer,"is-base"));
    for(const e of model.alike)if(at.has(e.a)&&at.has(e.b))baseEdges.push(addEdge(e,baseLayer,"is-alike"));
    const nodeGeometry=n=>{
      const v=visual.get(n.tag);if(!v)return;
      const p=position(n),r=radius(n),sz=(n.count===maxCount?13:11)/Math.pow(zoom,.65);
      v.g.setAttribute("transform","translate("+p.x+" "+p.y+")");
      v.core.setAttribute("r",r);v.halo.setAttribute("r",r*1.55);v.hit.setAttribute("r",Math.max(r,9/scalePixels()));
      v.ring.setAttribute("r",r+4/zoom);
      v.star.setAttribute("transform","translate("+(r*.78+4/zoom)+" "+(-r*.78-4/zoom)+") scale("+1/zoom+")");
      v.label.setAttribute("x",p.x+r+5/zoom);v.label.setAttribute("y",p.y+sz*.32);
      v.label.setAttribute("font-size",sz);v.label.setAttribute("stroke-width",Math.max(sz*.24,1.5/zoom));
      v.ripple.setAttribute("r",r+4/zoom+(pulse?.tag===n.tag?pulse.amount*13/zoom:0));
    };
    const updateGeometry=(changed=null)=>{
      if(!changed){for(const n of at.values())nodeGeometry(n);}
      else for(const tag of changed){const n=at.get(tag);if(n)nodeGeometry(n);}
      for(const e of [...baseEdges,...focusEdges])if(!changed||changed.has(e.a)||changed.has(e.b))edgeGeometry(e);
    };
    function createNode(n){
      const id="tb-orb-"+serial++;
      const grad=make("radialGradient",{id,cx:"32%",cy:"25%",r:"85%"},defs);
      for(const [offset,color,opacity] of [["0%","#ffffff",".62"],["35%",n.color,"1"],["100%",n.color,".72"]])
        make("stop",{offset,"stop-color":color,"stop-opacity":opacity},grad);
      const g=make("g",{"data-tag":n.tag,role:"button",tabindex:"0",class:"studio-tb-orb"},nodeLayer);
      const hit=make("circle",{fill:"transparent",class:"studio-tb-hit","pointer-events":"all"},g);
      const halo=make("circle",{fill:n.color,class:"studio-tb-orb-halo","pointer-events":"none"},g);
      const ripple=make("circle",{fill:"none",stroke:"url(#tb-sunlight)","vector-effect":"non-scaling-stroke","stroke-width":"1.2",opacity:0,"pointer-events":"none"},g);
      const ring=make("circle",{fill:"none",stroke:"url(#tb-sunlight)","vector-effect":"non-scaling-stroke","stroke-width":"2",opacity:0,"pointer-events":"none"},g);
      const core=make("circle",{fill:n.color,class:"studio-tb-orb-core"},g);
      const star=make("path",{d:"M0 -7Q1 -1 7 0Q1 1 0 7Q-1 1 -7 0Q-1 -1 0 -7Z",fill:"url(#tb-sunlight)",stroke:"#141416","stroke-width":"1",opacity:0,"pointer-events":"none"},g);
      const label=make("text",{fill:n.color,"font-weight":650,stroke:"var(--bg-deep)","paint-order":"stroke","pointer-events":"none"},labelLayer);label.textContent=n.tag;
      visual.set(n.tag,{g,core,halo,hit,ripple,ring,star,label,grad,id});
      g.addEventListener("pointerdown",e=>{
        if(e.button!==0)return;e.stopPropagation();dismissMenu();cancelMotion();
        const p=point(e);drag={kind:"node",tag:n.tag,dx:p.x-n.x,dy:p.y-n.y,x:e.clientX,y:e.clientY,moved:false};
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
    function focusConnections(){
      focusLayer.replaceChildren();focusEdges.length=0;
      shared=picked?relationMap(picked):new Map();
      if(picked&&at.has(picked))for(const [tag,w] of shared)if(at.has(tag)){
        const e=addEdge({a:picked,b:tag,w},focusLayer,"is-focused");
        const strength=Math.min(1,w/Math.max(1,at.get(picked).count));
        e.el.setAttribute("stroke",at.get(tag).color);e.el.setAttribute("stroke-opacity",.22+strength*.73);
        e.el.setAttribute("stroke-width",.7+strength*2.5);focusEdges.push(e);
      }
    }
    function updateAppearance(){
      const focused=picked,neighbors=focused?relationMap(focused):null;
      const rankLimit=zoom<.8?10:zoom<1.25?22:zoom<2?45:zoom<3?80:120;
      for(const e of baseEdges){
        const touches=focused&&(e.a===focused||e.b===focused);
        e.el.setAttribute("stroke",at.get(e.a)?.color||"var(--text-muted)");
        e.el.setAttribute("stroke-opacity",focused?(touches?.3:.035):e.bridge?.15:.27);
        e.el.setAttribute("stroke-width",touches?.85:e.bridge?.48:.65);
      }
      // Labels are prioritized by selection, then cluster representatives,
      // then frequency. Collision rejection keeps the full map readable.
      const ordered=[...at.values()].sort((a,b)=>(b.tag===picked)-(a.tag===picked)||
        ((neighbors?.has(b.tag)?1:0)-(neighbors?.has(a.tag)?1:0))||b.count-a.count);
      const placed=[],represented=new Set();
      let labels=0;
      for(const n of ordered){
        const v=visual.get(n.tag),selected=n.tag===picked,related=!focused||selected||neighbors?.has(n.tag);
        v.g.setAttribute("opacity",related?1:.19);
        v.g.setAttribute("aria-pressed",String(selected));
        v.g.setAttribute("aria-label",n.tag+", "+n.count+" images"+(selected?", selected":""));
        v.core.setAttribute("fill",prefs.glass?"url(#"+v.id+")":n.color);
        v.ring.setAttribute("opacity",selected?"1":"0");v.star.setAttribute("opacity",selected?"1":"0");
        const p=position(n),sz=(n.count===maxCount?13:11)/Math.pow(zoom,.65),r=radius(n);
        const box={x:p.x+r+5/zoom,y:p.y-sz*.8,w:n.tag.length*sz*.56,h:sz*1.25};
        const collision=placed.some(b=>box.x<b.x+b.w&&box.x+box.w>b.x&&box.y<b.y+b.h&&box.y+box.h>b.y);
        const leader=!represented.has(n.group);
        const show=selected||(!collision&&related&&(labels<rankLimit||leader));
        v.label.setAttribute("visibility",show?"visible":"hidden");
        if(show){placed.push(box);represented.add(n.group);labels++;}
      }
      updateGeometry();
    }
    function renderPanel(){
      spotlight.replaceChildren();spotlight.hidden=!picked||!at.has(picked);if(spotlight.hidden)return;
      const n=at.get(picked),head=html("p","",undefined,spotlight);
      const name=html("strong","",n.tag,head);name.style.color=n.color;
      button(head,"×","Clear selected tag",()=>choose(null));
      html("div","studio-tb-panel-caption",n.count+" images · Related displayed tags",spotlight);
      const list=html("ul","",undefined,spotlight);
      for(const [tag,w] of [...shared].sort((a,b)=>b[1]-a[1])){
        const item=html("li","",undefined,list),row=html("div","studio-tb-related-row",undefined,item);
        const b=button(row,tag,"Focus "+tag,()=>choose(tag));b.style.color=at.get(tag)?.color||"";
        const percent=Math.round(w/Math.max(1,n.count)*100);
        html("span","",w+" · "+percent+"%",row);
        const track=html("div","studio-tb-related-track",undefined,item);
        const bar=html("div","",undefined,track);bar.style.width=Math.min(100,percent)+"%";bar.style.background=at.get(tag)?.color||"var(--accent)";
      }
      if(!shared.size)html("div","studio-tb-panel-caption","No shared tags in this displayed graph.",spotlight);
      if(options.onNavigate){
        const actions=html("div","studio-tb-panel-actions",undefined,spotlight);
        button(actions,"View in Tags","View matching images in Tags",()=>options.onNavigate("list",picked));
        button(actions,"Waterfall","View matching images in Waterfall",()=>options.onNavigate("waterfall",picked));
      }
    }
    function choose(tag,toggle=true){
      if(tag&&toggle&&tag===picked)tag=null;
      if(tag&&!ensureTag(tag))return;
      if(rare&&rare.tag!==tag)removeRare();
      if(pulse){visual.get(pulse.tag)?.ripple.setAttribute("opacity","0");pulse=null;}
      picked=tag;focusConnections();renderPanel();updateAppearance();
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
    function center(tag){const n=at.get(tag);if(n)cameraTo(Math.max(zoom,1.45),{x:n.x-500,y:n.y-350});}
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
    button(zoomCtl,"⛶","Fit full graph",()=>fit());
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
        view();updateAppearance();if(t>=1)camera=null;
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
        n.x=p.x-drag.dx;n.y=p.y-drag.dy;dirty.add(n.tag);
        const scale=scalePixels();
        if(!reduced()&&prefs.elasticity>0)for(const [tag,w] of relationMap(n.tag)){
          const peer=at.get(tag);if(!peer)continue;
          const offset=elasticOffset(dx,dy,w/Math.max(1,n.count),prefs.elasticity);
          peer.ox=offset.x/scale;peer.oy=offset.y/scale;dirty.add(tag);
        }
        wake();
      }else{
        const scale=scalePixels();pan={x:drag.px-dx/scale,y:drag.py-dy/scale};view();
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
    stage.dispose=()=>{disposed=true;if(frame!==null)cancelAnimationFrame(frame);frame=null;camera=null;settling=null;pulse=null;};
    stage.append(svg,search,zoomCtl,spotlight,menu);
    stage.setPreferences(prefs);view();
    if(pickedTag&&ensureTag(pickedTag)){picked=pickedTag;focusConnections();renderPanel();updateAppearance();}
    fit(false);
    return stage;
  }

  return Object.freeze({buildGraph,waterfall,constellation,colorForTag,setTagbenchPalette,tagbenchModel,tagbenchWaterfall,tagbenchConstellation,waterfallScale,waterfallColumns,constellationPrefs,elasticOffset});
})();
// A classic WebWorker can import this existing pure graph implementation
// without maintaining a second constellation algorithm.
if(typeof self!=="undefined"&&typeof document==="undefined")
  self.CaptionStudioViz=CaptionStudioViz;
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioViz;
