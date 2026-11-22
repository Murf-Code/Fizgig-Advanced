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
  const tbHue=id=>id<TB_HUES.length?TB_HUES[id]:
    "hsl("+Math.round(id*137.508%360)+" 72% 62%)";
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
      const category=categories?.tags?.[n.tag],manual=category&&categories.colors?.[category];
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
      const category=categories?.tags?.[tag.tag],manual=category&&categories.colors?.[category];
      if(manual||friendColor)colors.set(tag.tag,manual||friendColor);
    }
    return {...graph,nodes,chosen,alike,colors};
  }
  function tagbenchWaterfall(items,onSelect,onContextMenu,activePath,selection,token,onDecorate=null,onOpenTags=null) {
    const stage=document.createElement("div");stage.className="studio-tb-waterfall";
    const flow=document.createElement("div");flow.className="studio-tb-flow";
    const pictures=items.filter(it=>it.size);
    const initialColumns=Math.max(2,Math.floor(Math.max(300,window.innerWidth-60)/228));
    let columns=Array.from({length:initialColumns},()=>{
      const col=document.createElement("div");col.className="studio-tb-flow-column";
      flow.append(col);return col;
    });
    // Decode only the first page of thumbnails. The dataset may have tens of
    // thousands of images; creating every DOM node and image at once is costly.
    const batchSize=120;
    let count=0,disposed=false,resizeObserver=null,loadObserver=null;
    const createTile=it=>{
      const tile=document.createElement("button");
      tile.type="button";tile.className="studio-tb-flow-tile";
      tile.dataset.path=it.path;
      tile.setAttribute("aria-label",it.name);
      tile.setAttribute("aria-pressed",String(!!selection?.has(it.path)));
      if(it.path===activePath||selection?.has(it.path))tile.classList.add("is-picked");
      const img=document.createElement("img");
      img.loading="lazy";img.decoding="async";
      img.src="/api/thumb?path="+encodeURIComponent(it.path)+"&t="+encodeURIComponent(token||"");
      img.alt="";
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
      for(;count<end;count++)columns[count%columns.length].append(createTile(pictures[count]));
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
      if(typeof IntersectionObserver!=="undefined"&&sentinel.isConnected){
        loadObserver=new IntersectionObserver(entries=>{
          if(entries.some(entry=>entry.isIntersecting))loadMore();
        },{root:stage,rootMargin:"450px 0px"});
        loadObserver.observe(sentinel);
      }
    }
    const resize=()=>{
      if(disposed||!stage.isConnected)return;
      const wanted=Math.max(2,Math.floor(Math.max(400,stage.clientWidth)/228));
      if(wanted===columns.length)return;
      const tiles=[...flow.querySelectorAll(".studio-tb-flow-tile")];
      flow.replaceChildren();
      columns=Array.from({length:wanted},()=>{
        const col=document.createElement("div");col.className="studio-tb-flow-column";
        flow.append(col);return col;
      });
      tiles.forEach((tile,i)=>columns[i%wanted].append(tile));
    };
    if(typeof ResizeObserver!=="undefined"){
      resizeObserver=new ResizeObserver(resize);
      resizeObserver.observe(stage);
    }
    stage.dispose=()=>{
      disposed=true;
      resizeObserver?.disconnect();
      loadObserver?.disconnect();
    };
    return stage;
  }
  function tagbenchConstellation(model,onPick,pickedTag=null) {
    const stage=document.createElement("div");stage.className="studio-tb-constellation";
    if(!model.nodes.length){stage.textContent="No tagged images in this dataset.";return stage;}
    const NS="http://www.w3.org/2000/svg",svg=document.createElementNS(NS,"svg");
    svg.setAttribute("role","group");svg.setAttribute("aria-label","TagBench-style interactive tag constellation");
    const make=(type,attrs,parent=svg)=>{
      const node=document.createElementNS(NS,type);
      for(const [k,v] of Object.entries(attrs||{}))node.setAttribute(k,String(v));
      parent.append(node);return node;
    };
    const zoomCtl=document.createElement("div");zoomCtl.className="studio-tb-zoom";
    const search=document.createElement("div");search.className="studio-tb-search";
    const input=document.createElement("input");input.placeholder="Search Tags";
    input.setAttribute("aria-label","Search tags");
    const matches=document.createElement("div");matches.className="studio-tb-suggest";
    search.append(input,matches);
    let zoom=1,pan={x:0,y:0},picked=pickedTag,drag=null,popping=null;
    const nodes=model.nodes.map(n=>({...n}));
    const at=new Map(nodes.map(n=>[n.tag,n]));
    const radius=n=>{
      const vals=nodes.map(x=>x.links),lo=Math.min(...vals),hi=Math.max(...vals);
      const base=4+(hi===lo?1:(n.links-lo)/(hi-lo))*22;
      return base*(zoom>1?1/Math.pow(zoom,.22):1)/zoom;
    };
    const view=()=>{
      const w=1000/zoom,h=700/zoom;
      svg.setAttribute("viewBox",[500-w/2+pan.x,350-h/2+pan.y,w,h].join(" "));
    };
    const spotlight=document.createElement("aside");spotlight.className="studio-tb-cluster";spotlight.hidden=true;
    const draw=()=>{
      view();svg.replaceChildren();
      make("rect",{x:-3000,y:-3000,width:7000,height:7000,fill:"transparent"});
      for(const edge of model.chosen){
        const a=at.get(edge.a),b=at.get(edge.b);if(!a||!b)continue;
        if(picked&&(picked===a.tag||picked===b.tag))continue;
        make("line",{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:a.color,
          "stroke-opacity":picked?.08:.28,"stroke-width":.7,class:"studio-tb-edge"});
      }
      const shared=new Map();
      if(picked)for(const edge of model.edges){
        if(edge.a===picked)shared.set(edge.b,edge.w);
        else if(edge.b===picked)shared.set(edge.a,edge.w);
      }
      for(const edge of model.alike){
        const a=at.get(edge.a),b=at.get(edge.b);if(!a||!b)continue;
        make("line",{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:"#ffffff",
          "stroke-opacity":.9,"stroke-width":1.3/zoom,
          "stroke-dasharray":5/zoom+" "+3/zoom});
      }
      if(picked&&at.has(picked)){
        const self=at.get(picked);
        for(const [key,w] of shared){
          const n=at.get(key);if(!n)continue;
          const closeness=w/Math.max(1,self.count),opacity=.14+(1-Math.pow(1-closeness,2))*.86;
          const dx=n.x-self.x,dy=n.y-self.y,len=Math.hypot(dx,dy)||1,ux=dx/len,uy=dy/len;
          const begin=radius(self)+1.2/zoom,end=radius(n)+2/zoom;
          if(len<=begin+end+1)continue;
          const x1=self.x+ux*begin,y1=self.y+uy*begin,x2=n.x-ux*end,y2=n.y-uy*end;
          const px=-uy,py=ux,wa=.25/zoom,wb=(.45+closeness*5.6)/zoom;
          make("path",{d:"M "+(x1+px*wa)+" "+(y1+py*wa)+" L "+(x2+px*wb)+" "+(y2+py*wb)+
            " L "+(x2-px*wb)+" "+(y2-py*wb)+" L "+(x1-px*wa)+" "+(y1-py*wa)+" Z",
            fill:n.color,"fill-opacity":opacity});
        }
      }
      const drawLabel=(n,faded)=>{
        const r=radius(n),sz=12*(zoom>1?1/Math.pow(zoom,.22):1)/zoom;
        make("text",{x:n.x+r+6/zoom,y:n.y+sz*.3,fill:n.color,
          "font-size":sz,"font-weight":700,opacity:faded?.4:1,
          stroke:"#000","stroke-width":Math.max(sz*.28,2/zoom),
          "paint-order":"stroke","pointer-events":"none"}).textContent=n.tag;
      };
      for(const n of nodes)if(picked&&picked!==n.tag&&!shared.has(n.tag))drawLabel(n,true);
      for(const n of nodes){
        const r=radius(n),active=n.tag===picked,near=!picked||active||shared.has(n.tag);
        const g=make("g",{"data-tag":n.tag,role:"button",tabindex:"0",
          class:"studio-tb-orb"+(popping===n.tag?" is-popping":"")});
        make("title",{},g).textContent=n.tag+" · "+n.links+" links";
        const halo=make("circle",{cx:n.x,cy:n.y,r:r*1.55,fill:n.color,
          opacity:0,"pointer-events":"none",class:"studio-tb-orb-halo"},g);
        if(active)make("circle",{cx:n.x,cy:n.y,r:r+6/zoom,fill:"none",stroke:"white","stroke-width":1.5/zoom},g);
        const main=make("circle",{cx:n.x,cy:n.y,r,fill:n.color,opacity:near?1:.12,
          class:"studio-tb-orb-core"},g);
        g.addEventListener("pointerenter",()=>{
          if(!near)return;
          halo.setAttribute("opacity",".24");
          main.setAttribute("r",String(r*1.16));
        });
        g.addEventListener("pointerleave",()=>{
          halo.setAttribute("opacity","0");
          main.setAttribute("r",String(r));
        });
        const displayed=picked&&n.tag!==picked?(shared.get(n.tag)??n.count):n.count;
        make("text",{x:n.x,y:n.y,"text-anchor":"middle","dominant-baseline":"central",
          fill:"#141416","font-size":r*(displayed>=100?.48:displayed>=10?.58:.7),
          "font-weight":700,"pointer-events":"none",opacity:near?1:.35},g).textContent=String(displayed);
        g.addEventListener("pointerdown",event=>{
          event.stopPropagation();const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;
          const here=p.matrixTransform(svg.getScreenCTM().inverse());
          drag={kind:"node",tag:n.tag,dx:here.x-n.x,dy:here.y-n.y,startX:event.clientX,startY:event.clientY,moved:false};
          svg.setPointerCapture(event.pointerId);
        });
        g.addEventListener("keydown",event=>{
          if(event.key==="Enter"||event.key===" "){event.preventDefault();choose(n.tag);}
        });
      }
      for(const n of nodes)if(!picked||picked===n.tag||shared.has(n.tag))drawLabel(n,false);
      spotlight.replaceChildren();
      if(picked&&at.has(picked)){
        const n=at.get(picked),headline=document.createElement("p");
        headline.style.color=n.color;
        const label=document.createElement("strong");label.textContent=n.tag;
        const count=document.createElement("span");count.textContent=String(n.count);
        headline.append(label,count);spotlight.append(headline);
        const list=document.createElement("ul");
        for(const [tag,w] of [...shared].sort((a,b)=>b[1]-a[1])){
          const item=document.createElement("li"),b=document.createElement("button");
          b.type="button";b.textContent=tag;b.style.color=at.get(tag)?.color||"";
          b.addEventListener("click",()=>choose(tag));
          const number=document.createElement("span");number.textContent=String(w);
          item.append(b,number);list.append(item);
        }
        spotlight.append(list);spotlight.hidden=false;
      }else spotlight.hidden=true;
    };
    const choose=tag=>{
      picked=picked===tag?null:tag;
      popping=tag;
      onPick?.(picked);
      draw();
      setTimeout(()=>{
        popping=null;
        for(const node of svg.querySelectorAll(".studio-tb-orb.is-popping"))
          node.classList.remove("is-popping");
      },320);
    };
    const zoomBy=(factor,clientX=null,clientY=null)=>{
      const before=zoom,next=Math.min(5,Math.max(.35,zoom*factor));
      if(next===before)return;
      if(clientX!==null){
        const rect=svg.getBoundingClientRect();
        const mx=(clientX-rect.left)/rect.width,my=(clientY-rect.top)/rect.height;
        const w=1000/before,h=700/before,nextW=1000/next,nextH=700/next;
        const worldX=500-w/2+pan.x+mx*w,worldY=350-h/2+pan.y+my*h;
        pan={x:worldX-mx*nextW-500+nextW/2,y:worldY-my*nextH-350+nextH/2};
      }
      zoom=next;draw();
    };
    for(const [name,factor] of [["+",1.2],["−",1/1.2]]){
      const b=document.createElement("button");b.type="button";b.textContent=name;
      b.setAttribute("aria-label",factor>1?"Zoom in":"Zoom out");
      b.addEventListener("click",()=>zoomBy(factor));zoomCtl.append(b);
    }
    const searchMatches=()=>{
      matches.replaceChildren();
      const q=input.value.toLowerCase().trim();if(!q)return;
      for(const n of nodes.filter(n=>n.tag.includes(q)).sort((a,b)=>b.links-a.links).slice(0,8)){
        const b=document.createElement("button");b.textContent=n.tag;b.style.color=n.color;
        b.addEventListener("click",()=>{
          input.value=n.tag;pan={x:n.x-500,y:n.y-350};choose(n.tag);matches.replaceChildren();
        });matches.append(b);
      }
    };
    input.addEventListener("input",searchMatches);
    input.addEventListener("keydown",e=>{
      if(e.key==="Enter"){e.preventDefault();matches.querySelector("button")?.click();}
    });
    svg.addEventListener("pointerdown",e=>{
      if(e.target.closest(".studio-tb-orb"))return;
      drag={kind:"pan",x:e.clientX,y:e.clientY,px:pan.x,py:pan.y};
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener("pointermove",e=>{
      if(!drag)return;
      if(drag.kind==="node"){
        const distance=Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY);
        if(distance<4&&!drag.moved)return;
        drag.moved=true;
        const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
        const here=p.matrixTransform(svg.getScreenCTM().inverse()),n=at.get(drag.tag);
        n.x=here.x-drag.dx;n.y=here.y-drag.dy;draw();
      }else{
        const rect=svg.getBoundingClientRect();
        pan={x:drag.px-(e.clientX-drag.x)*1000/zoom/rect.width,
          y:drag.py-(e.clientY-drag.y)*700/zoom/rect.height};
        view();
      }
    });
    svg.addEventListener("pointerup",()=>{
      if(drag?.kind==="node"&&!drag.moved)choose(drag.tag);
      drag=null;
    });
    svg.addEventListener("pointercancel",()=>drag=null);
    svg.addEventListener("dblclick",e=>{if(!e.target.closest(".studio-tb-orb"))choose(picked);});
    svg.addEventListener("wheel",e=>{e.preventDefault();zoomBy(e.deltaY<0?1.12:1/1.12,e.clientX,e.clientY);},{passive:false});
    stage.append(svg,search,zoomCtl,spotlight);draw();
    return stage;
  }

  return Object.freeze({buildGraph,waterfall,constellation,colorForTag,setTagbenchPalette,tagbenchModel,tagbenchWaterfall,tagbenchConstellation});
})();
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioViz;
