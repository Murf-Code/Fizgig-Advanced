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
  // TagBench's co-occurrence-profile similarity groups unknown tags, while
  // recognizable tag families remain aligned even on small datasets.
  function tbGroups(tags,edges) {
    const n=tags.length;
    const at=new Map(tags.map((t,i)=>[t.tag,i]));
    const shared=Array.from({length:n},()=>new Float64Array(n));
    for(const e of edges){
      const a=at.get(e.a),b=at.get(e.b);
      if(a!==undefined&&b!==undefined){shared[a][b]=e.w;shared[b][a]=e.w;}
    }
    const distance=Array.from({length:n},()=>new Float64Array(n));
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
      let dot=0,left=0,right=0;
      for(let k=0;k<n;k++){
        if(k===i||k===j||tags[k].count>tags[0].count*.5)continue;
        const rare=Math.log((n+1)/(tags[k].count+1));
        const a=shared[i][k]/Math.max(tags[i].count,1)*rare;
        const b=shared[j][k]/Math.max(tags[j].count,1)*rare;
        dot+=a*b;left+=a*a;right+=b*b;
      }
      distance[i][j]=distance[j][i]=1-(left&&right?dot/Math.sqrt(left*right):0);
    }
    const clusterAt=limit=>{
      const group=tags.map((_,i)=>i);
      const alive=tags.map(()=>true);
      const dist=distance.map(row=>Float64Array.from(row));
      for(let remaining=n;remaining>1;remaining--){
        let keep=-1,drop=-1,best=Infinity;
        for(let i=0;i<n;i++)if(alive[i])for(let j=i+1;j<n;j++){
          if(alive[j]&&dist[i][j]<best){best=dist[i][j];keep=i;drop=j;}
        }
        if(keep<0||best>limit)break;
        for(let k=0;k<n;k++)if(alive[k]&&k!==keep&&k!==drop)
          dist[keep][k]=dist[k][keep]=Math.max(dist[keep][k],dist[drop][k]);
        alive[drop]=false;
        for(let i=0;i<n;i++)if(group[i]===drop)group[i]=keep;
      }
      return group;
    };
    // Original TagBench silhouette-driven cut selection (sampled at five levels).
    const score=ids=>{
      const groups=new Map();
      ids.forEach((id,i)=>{if(!groups.has(id))groups.set(id,[]);groups.get(id).push(i);});
      if(groups.size<2)return -1;
      let total=0,used=0;
      for(let i=0;i<n;i++){
        const mine=groups.get(ids[i]);
        if(mine.length<2)continue;
        const near=mine.reduce((v,j)=>v+(j===i?0:distance[i][j]),0)/(mine.length-1);
        let far=Infinity;
        for(const [id,members] of groups)if(id!==ids[i])
          far=Math.min(far,members.reduce((v,j)=>v+distance[i][j],0)/members.length);
        total+=(far-near)/Math.max(near,far,1e-6);used++;
      }
      return used?total/used:-1;
    };
    let best=clusterAt(.45),bestScore=score(best);
    for(const level of [.25,.35,.55,.7]){
      const next=clusterAt(level),value=score(next);
      if(value>bestScore){best=next;bestScore=value;}
    }
    const familyMap=new Map(),learnedMap=new Map();
    let seq=20;
    tags.forEach((row,i)=>{
      const family=tbFamily(row.tag);
      const id=family===null?"learned:"+best[i]:"family:"+family;
      if(!learnedMap.has(id))learnedMap.set(id,seq++);
      familyMap.set(row.tag,learnedMap.get(id));
    });
    const stems=new Map();
    for(const row of tags){
      const stem=tbStem(row.tag);
      if(stem.length<4)continue;
      if(!stems.has(stem))stems.set(stem,[]);
      stems.get(stem).push(row);
    }
    const alike=[];
    for(const members of stems.values()){
      if(members.length<2)continue;
      members.sort((a,b)=>b.count-a.count);
      const first=members[0].tag;
      for(const m of members){
        familyMap.set(m.tag,familyMap.get(first));
        if(m.tag!==first)alike.push({a:first,b:m.tag,w:0});
      }
    }
    const ids=[...new Set(familyMap.values())],compact=new Map(ids.map((id,i)=>[id,i]));
    for(const [tag,id] of familyMap)familyMap.set(tag,compact.get(id));
    return {groups:familyMap,alike};
  }
  function tagbenchModel(items,categories) {
    const graph=tagbenchGraph(items),{all,top,edges,linkCounts}=graph;
    const {groups,alike}=tbGroups(top,edges);
    const lookup=new Map(top.map((t,i)=>[t.tag,i]));
    const neigh=new Map(top.map(t=>[t.tag,[]]));
    for(const e of edges){neigh.get(e.a).push({tag:e.b,w:e.w});neigh.get(e.b).push({tag:e.a,w:e.w});}
    const nodeGroup=new Map();
    const pods=new Map();let podIndex=0;
    for(const id of new Set(groups.values())){
      const members=top.filter(t=>groups.get(t.tag)===id).map(t=>t.tag);
      const left=new Set(members);
      while(left.size){
        const seed=[...left].sort()[0],chosen=[seed];left.delete(seed);
        const ranked=members.filter(tag=>left.has(tag))
          .map(tag=>({tag,w:neigh.get(seed).find(x=>x.tag===tag)?.w||0}))
          .sort((a,b)=>b.w-a.w);
        for(const next of ranked){if(chosen.length>=8)break;if(next.w<=0)continue;chosen.push(next.tag);left.delete(next.tag);}
        for(const tag of chosen)pods.set(tag,podIndex);podIndex++;
      }
    }
    const nodes=top.map(t=>({
      tag:t.tag,count:t.count,links:linkCounts.get(t.tag)||0,
      group:groups.get(t.tag),pod:pods.get(t.tag)||0,
      x:0,y:0,color:null
    }));
    const groupIds=[...new Set(nodes.map(n=>n.group))];
    const weights=new Map(groupIds.map(id=>[id,nodes.filter(n=>n.group===id).reduce((s,n)=>s+n.links,0)]));
    groupIds.sort((a,b)=>weights.get(b)-weights.get(a)).forEach((id,i)=>{
      const turn=i/Math.max(1,groupIds.length)*Math.PI*2-Math.PI/2;
      nodes.filter(n=>n.group===id).sort((a,b)=>b.links-a.links||b.count-a.count)
        .forEach((n,k)=>{
          const angle=k*2.399,rad=k?18+Math.sqrt(k)*12:0;
          n.x=Math.cos(turn)*210+Math.cos(angle)*rad;
          n.y=Math.sin(turn)*170+Math.sin(angle)*rad;
        });
    });
    const chosen=[],seen=new Set(),attract=[];
    for(const node of nodes){
      const related=edges.filter(e=>e.a===node.tag||e.b===node.tag);
      const picked=related.filter(e=>groups.get(e.a)===groups.get(e.b)).slice(0,3)
        .map(e=>({edge:e,inside:true}))
        .concat(related.filter(e=>groups.get(e.a)!==groups.get(e.b)).slice(0,1)
          .map(e=>({edge:e,inside:false})));
      for(const item of picked){
        const edge=item.edge,key=edge.a+"\0"+edge.b;
        if(seen.has(key))continue;seen.add(key);
        attract.push({i:lookup.get(edge.a),j:lookup.get(edge.b),
          w:(.45+.55*(edge.w/Math.max(1,edges[0]?.w||1)))*(item.inside?1:.08)});
        if(item.inside)chosen.push(edge);
      }
    }
    const mass=nodes.map(n=>1+Math.sqrt(n.links)),linkValues=nodes.map(n=>n.links);
    const minLinks=Math.min(...linkValues,0),maxLinks=Math.max(...linkValues,1);
    const radius=links=>4+(maxLinks===minLinks?1:(links-minLinks)/(maxLinks-minLinks))*22;
    for(let step=0;step<220;step++){
      const fx=nodes.map(()=>0),fy=nodes.map(()=>0);
      for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
        const dx=nodes[i].x-nodes[j].x,dy=nodes[i].y-nodes[j].y;
        const d=Math.hypot(dx,dy)||.01,fac=2*mass[i]*mass[j]/d/d;
        fx[i]+=dx*fac;fy[i]+=dy*fac;fx[j]-=dx*fac;fy[j]-=dy*fac;
      }
      for(const e of attract){
        const dx=nodes[e.i].x-nodes[e.j].x,dy=nodes[e.i].y-nodes[e.j].y;
        const d=Math.hypot(dx,dy)||.01,fac=-3*e.w*Math.log1p(d)/d;
        fx[e.i]+=dx*fac;fy[e.i]+=dy*fac;fx[e.j]-=dx*fac;fy[e.j]-=dy*fac;
      }
      let maxForce=1;
      nodes.forEach((n,i)=>{
        fx[i]-=n.x*.012*mass[i];fy[i]-=n.y*.012*mass[i];
        maxForce=Math.max(maxForce,Math.hypot(fx[i],fy[i]));
      });
      const scale=Math.min(.35,(8*Math.pow(.99,step)+.6)/maxForce);
      nodes.forEach((n,i)=>{n.x+=fx[i]*scale;n.y+=fy[i]*scale;});
    }
    for(let step=0;step<28;step++)for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.max(1,Math.hypot(dx,dy));
      const minimum=radius(a.links)+radius(b.links)+14;
      if(d>=minimum)continue;
      const k=(minimum-d)/2/d;a.x+=dx*k;a.y+=dy*k;b.x-=dx*k;b.y-=dy*k;
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
    const baseLayer=make("g"),focusLayer=make("g"),hoverLayer=make("g"),nodeLayer=make("g"),labelLayer=make("g");
    const nodes=model.nodes.map(n=>({...n,ox:0,oy:0}));
    const at=new Map(nodes.map(n=>[n.tag,n])),all=new Map(model.all.map(n=>[n.tag,n]));
    const adjacency=new Map(nodes.map(n=>[n.tag,new Map()]));
    for(const e of model.edges){adjacency.get(e.a)?.set(e.b,e.w);adjacency.get(e.b)?.set(e.a,e.w);}
    const vals=nodes.map(n=>n.links),lo=Math.min(...vals),hi=Math.max(...vals);
    const visual=new Map(),baseEdges=[],focusEdges=[],hoverEdges=[];
    let prefs=constellationPrefs(options.preferences),zoom=1,pan={x:0,y:0},picked=null,hover=null,drag=null,rare=null;
    let serial=0,shared=new Map(),frame=null,settling=null,camera=null,pulse=null,disposed=false,dirty=new Set(),lastClick=null;
    const media=typeof matchMedia==="function"?matchMedia("(prefers-reduced-motion: reduce)"):null;
    const reduced=()=>!!media?.matches;
    const radius=n=>(4+(hi===lo?1:Math.max(0,Math.min(1,(n.links-lo)/(hi-lo))))*22)*(zoom>1?1/Math.pow(zoom,.22):1)/zoom;
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
      edge.el.setAttribute("x1",p.x+dx/len*start);edge.el.setAttribute("y1",p.y+dy/len*start);
      edge.el.setAttribute("x2",q.x-dx/len*end);edge.el.setAttribute("y2",q.y-dy/len*end);
    };
    const addEdge=(e,layer,kind)=>{
      const line=make("line",{"vector-effect":"non-scaling-stroke","pointer-events":"none",class:"studio-tb-edge "+kind},layer);
      const rec={...e,el:line};edgeGeometry(rec);return rec;
    };
    for(const e of model.chosen)if(at.has(e.a)&&at.has(e.b))baseEdges.push(addEdge(e,baseLayer,"is-base"));
    for(const e of model.alike)if(at.has(e.a)&&at.has(e.b))baseEdges.push(addEdge(e,baseLayer,"is-alike"));
    const nodeGeometry=n=>{
      const v=visual.get(n.tag);if(!v)return;
      const p=position(n),r=radius(n),sz=12*(zoom>1?1/Math.pow(zoom,.22):1)/zoom;
      v.g.setAttribute("transform","translate("+p.x+" "+p.y+")");
      v.core.setAttribute("r",r);v.halo.setAttribute("r",r*1.34);
      v.ring.setAttribute("r",r+4/zoom);
      v.star.setAttribute("transform","translate("+(r*.78+4/zoom)+" "+(-r*.78-4/zoom)+") scale("+1/zoom+")");
      v.number.setAttribute("font-size",Math.max(4/zoom,r*(v.number.textContent.length>=3?.50:.70)));
      v.label.setAttribute("x",p.x+r+5/zoom);v.label.setAttribute("y",p.y+sz*.32);
      v.label.setAttribute("font-size",sz);v.label.setAttribute("stroke-width",Math.max(sz*.24,1.5/zoom));
      v.ripple.setAttribute("r",r+4/zoom+(pulse?.tag===n.tag?pulse.amount*13/zoom:0));
    };
    const updateGeometry=(changed=null)=>{
      if(!changed){for(const n of at.values())nodeGeometry(n);}
      else for(const tag of changed){const n=at.get(tag);if(n)nodeGeometry(n);}
      for(const e of [...baseEdges,...focusEdges,...hoverEdges])if(!changed||changed.has(e.a)||changed.has(e.b))edgeGeometry(e);
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
      const ring=make("circle",{fill:"none",stroke:"url(#tb-sunlight)","vector-effect":"non-scaling-stroke","stroke-width":"2",opacity:0,"pointer-events":"none"},g);
      const core=make("circle",{fill:n.color,class:"studio-tb-orb-core"},g);
      const number=make("text",{"text-anchor":"middle","dominant-baseline":"central",fill:"var(--text-on-light)","font-weight":700,"pointer-events":"none"},g);
      const star=make("path",{d:"M0 -7Q1 -1 7 0Q1 1 0 7Q-1 1 -7 0Q-1 -1 0 -7Z",fill:"url(#tb-sunlight)",stroke:"#141416","stroke-width":"1",opacity:0,"pointer-events":"none"},g);
      const label=make("text",{fill:n.color,"font-weight":650,stroke:"var(--bg-deep)","paint-order":"stroke","pointer-events":"none"},labelLayer);label.textContent=n.tag;
      visual.set(n.tag,{g,core,halo,ripple,ring,number,star,label,grad,id});
      g.addEventListener("pointerenter",()=>{if(drag)return;hover=n.tag;hoverConnections();updateAppearance();});
      g.addEventListener("pointerleave",()=>{if(hover===n.tag){hover=null;hoverConnections();updateAppearance();}});
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
    function hoverConnections(){
      hoverLayer.replaceChildren();hoverEdges.length=0;
      if(!hover||hover===picked||!at.has(hover))return;
      for(const [tag,w] of relationMap(hover))if(at.has(tag)){
        const e=addEdge({a:hover,b:tag,w},hoverLayer,"is-hovered");
        const strength=Math.min(1,w/Math.max(1,at.get(hover).count));
        e.el.setAttribute("stroke",at.get(tag).color);e.el.setAttribute("stroke-opacity",.16+strength*.54);
        e.el.setAttribute("stroke-width",.7+strength*1.4);hoverEdges.push(e);
      }
    }
    function updateAppearance(){
      const focused=hover||picked,neighbors=focused?relationMap(focused):null;
      const rankLimit=zoom<.7?15:zoom<1.1?45:zoom<1.6?85:120;
      for(const e of baseEdges){
        const touches=focused&&(e.a===focused||e.b===focused);
        e.el.setAttribute("stroke",e.el.classList.contains("is-alike")?"var(--text-secondary)":at.get(e.a)?.color||"var(--text-muted)");
        e.el.setAttribute("stroke-opacity",focused?(touches?.42:.045):e.el.classList.contains("is-alike")?.65:.23);
        e.el.setAttribute("stroke-width",touches?1.3:.7);
        if(e.el.classList.contains("is-alike"))e.el.setAttribute("stroke-dasharray","4 4");
      }
      focusLayer.setAttribute("opacity",hover&&hover!==picked?".22":"1");
      let rank=0;
      for(const n of at.values()){
        const v=visual.get(n.tag),selected=n.tag===picked,related=!focused||n.tag===focused||neighbors?.has(n.tag);
        v.g.setAttribute("opacity",related||selected?1:.15);
        v.g.setAttribute("aria-pressed",String(selected));
        v.g.setAttribute("aria-label",n.tag+", "+n.count+" images"+(selected?", selected":""));
        v.core.setAttribute("fill",prefs.glass?"url(#"+v.id+")":n.color);
        v.halo.setAttribute("opacity",hover===n.tag?".18":"0");
        v.ring.setAttribute("opacity",selected?"1":"0");v.star.setAttribute("opacity",selected?"1":"0");
        v.number.textContent=String(picked&&n.tag!==picked&&shared.has(n.tag)?shared.get(n.tag):n.count);
        v.label.setAttribute("opacity",related||selected?"1":".14");
        v.label.setAttribute("visibility",selected||n.tag===hover||(focused&&related)||rank<rankLimit?"visible":"hidden");
        rank++;
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
      picked=tag;hover=null;hoverConnections();focusConnections();renderPanel();updateAppearance();
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
