const assert = require("node:assert/strict");
const Viz = require("../src/fizgig/web/static/caption_studio_viz.js");

assert.equal(Viz.waterfallScale(null),100);
assert.equal(Viz.waterfallScale(""),100);
assert.equal(Viz.waterfallScale("invalid"),100);
assert.equal(Viz.waterfallScale("111"),110);
assert.equal(Viz.waterfallScale(2),50);
assert.equal(Viz.waterfallScale(500),150);
assert.ok(Viz.waterfallColumns(1200,50)>Viz.waterfallColumns(1200,100));
assert.ok(Viz.waterfallColumns(1200,100)>Viz.waterfallColumns(1200,150));
assert.equal(Viz.waterfallColumns(250,150),1);
assert.deepEqual(Viz.constellationPrefs(),{elasticity:20,glassNodes:false,celestial:true});
assert.deepEqual(Viz.constellationPrefs({elasticity:0,glassNodes:true,celestial:false}),{elasticity:0,glassNodes:true,celestial:false});
assert.deepEqual(Viz.elasticOffset(1000,0,1,20),{x:5,y:0});
assert.deepEqual(Viz.elasticOffset(1000,0,.2,20),{x:1,y:0});
assert.deepEqual(Viz.elasticOffset(1000,0,1,0),{x:0,y:0});
const index = [["lion", 4], ["blue", 3], ["red", 2]];
const captions = [["Lion", "blue"], ["lion", "blue", "red"], ["lion", "blue"],
  ["lion", "red"], ["red"]];
const graph = Viz.buildGraph(index, captions);
assert.equal(graph.nodes.length, 3);
assert.equal(graph.edges.find(e=>e.a==="blue"&&e.b==="lion").count, 3);
assert.equal(graph.edges.find(e=>e.a==="lion"&&e.b==="red").count, 2);
assert.ok(graph.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));
assert.deepEqual(Viz.buildGraph(index, captions), graph);
assert.equal(Viz.buildGraph([], []).nodes.length, 0);
assert.ok(Viz.buildGraph(index, captions, 2).edges.every(e=>e.a!=="red"&&e.b!=="red"));
// The TagBench constellation uses the same exact visible co-occurrence
// weights after the two-pass memory optimization, including duplicate tags.
const captions2=[
  {tags:["lion","blue","lion","smiling"]},
  {tags:["LION","blue","sunset"]},
  {tags:["lion","smiling"]},
  {tags:["blue","sunset"]}
];
const model=Viz.tagbenchModel(captions2);
assert.equal(model.all.find(t=>t.tag==="lion").count,3);
assert.equal(model.edges.find(e=>e.a==="blue"&&e.b==="lion").w,2);
assert.equal(model.edges.find(e=>e.a==="lion"&&e.b==="smiling").w,2);
assert.equal(model.edges.find(e=>e.a==="blue"&&e.b==="sunset").w,2);
assert.equal(model.nodes.length,4);
assert.ok(model.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));

// Large sparse tag vocabularies must not increase the constellation's 120-node
// graph limit or create edges outside that set.
const sparse=Array.from({length:260},(_,i)=>({tags:["common","label_"+i,"rare_"+i]}));
const sparseModel=Viz.tagbenchModel(sparse);
const known=new Set(sparseModel.top.map(t=>t.tag));
assert.equal(sparseModel.top.length,120);
assert.ok(sparseModel.edges.every(e=>known.has(e.a)&&known.has(e.b)));
assert.ok(sparseModel.edges.length<=120*119/2);

// Waterfall tiles preserve ordinary single-click selection. Double-click is
// a distinct action that opens the exact tile in the Tags editor.
const savedDoc=global.document, savedWin=global.window;
const savedResize=global.ResizeObserver,savedMutation=global.MutationObserver,savedMotion=global.matchMedia;
try {
  const fakeElement=()=>{
    const children=[],handlers={},classes=new Set(),attrs={};
    return {children,handlers,dataset:{},classes,attrs,classList:{add(name){classes.add(name);},toggle(){}},
      style:{},isConnected:false,getContext(){return null;},contains(node){return this===node||children.some(child=>child.contains(node));},
      getBoundingClientRect(){return {top:0,left:0,width:100,height:100};},
      focus(){global.document.activeElement=this;},
      querySelector(selector){return descendants(this).find(node=>node.className===selector.slice(1));},
      getScreenCTM(){return {a:1,inverse(){return this;}};},
      createSVGPoint(){return {matrixTransform(){return {x:this.x,y:this.y};}};},
      append(...elements){children.push(...elements);for(const element of elements)element.parent=this;},
      setAttribute(name,value){attrs[name]=value;},addEventListener(name,fn){handlers[name]=fn;},
      remove(){const siblings=this.parent?.children;if(siblings)siblings.splice(siblings.indexOf(this),1);},
      replaceChildren(...elements){children.length=0;this.append(...elements);}
    };
  };
  global.document={createElement:()=>fakeElement(),createElementNS:()=>fakeElement()};
  global.window={innerWidth:850};
  const picked=[],opened=[];
  const it={path:"/dataset/portrait.webp",name:"portrait.webp",size:[512,768]};
  const stage=Viz.tagbenchWaterfall([it],item=>picked.push(item.path),
    ()=>{},null,new Set(),"",null,item=>opened.push(item.path));
  const tile=stage.children[0].children[0].children[0];
  const focused=Viz.tagbenchWaterfall([it],()=>{},()=>{},it.path,new Set(),"");
  const activeTile=focused.children[0].children[0].children[0];
  assert.equal(activeTile.classes.has("active"),true);
  assert.equal(activeTile.classes.has("is-picked"),false);
  assert.equal(activeTile.attrs["aria-pressed"],"false");
  assert.ok(tile.handlers.click);
  assert.ok(tile.handlers.dblclick);
  tile.handlers.click({detail:1});
  assert.deepEqual(picked,[it.path]);
  assert.deepEqual(opened,[]);
  let prevented=false;
  tile.handlers.dblclick({preventDefault(){prevented=true;}});
  assert.equal(prevented,true);
  assert.deepEqual(opened,[it.path]);

  // A focused relationship can include a tag omitted from the overview.
  global.ResizeObserver=global.MutationObserver=class {observe(){} disconnect(){}};
  const graph=Viz.tagbenchModel([
    {tags:["paws","white fur","digital art"],coverIndex:0},
    {tags:["digital art"]},
    {tags:["digital art"]}
  ]);
  graph.nodes=graph.nodes.filter(node=>node.tag!=="white fur");
  graph.chosen=[];graph.edges=[];
  global.matchMedia=()=>({matches:true});
  const pairCalls=[];
  const network=Viz.tagbenchConstellation(graph,()=>{},null,{
    relationsForTag:()=>new Map([["digital art",1],["white fur",1]]),
    previewPair:(...pair)=>{pairCalls.push(pair);return [{src:"/thumb",name:"sample.png"}];},
    openPair:(...pair)=>pairCalls.push(pair)
  });
  const descendants=node=>[node,...node.children.flatMap(descendants)];
  const paws=descendants(network).find(node=>node.attrs["data-tag"]==="paws");
  const art=descendants(network).find(node=>node.attrs["data-tag"]==="digital art");
  const before=[paws.attrs.transform,art.attrs.transform];
  paws.handlers.keydown({key:"Enter",preventDefault(){},stopPropagation(){}});
  assert.deepEqual([paws.attrs.transform,art.attrs.transform],before);
  const edges=descendants(network).filter(node=>node.attrs.class==="studio-tb-pair-edge");
  assert.equal(edges[0].attrs["aria-label"],"paws and white fur, 1 shared image");
  assert.ok(Number(edges[0].children[0].attrs["stroke-width"])>Number(edges[1].children[0].attrs["stroke-width"]));
  edges[0].handlers.click();
  assert.ok(descendants(network).some(node=>node.src==="/thumb"));
  descendants(network).find(node=>node.textContent==="Open in Waterfall").handlers.click();
  assert.deepEqual(pairCalls,[["paws","white fur"],["paws","white fur"]]);

  assert.equal(graph.covers.get("paws"),0);
  const pictures=Array.from({length:50},(_,index)=>({...it,name:"paws "+index,path:"/dataset/"+index+".webp"}));
  let edit=null,editedImage=null;
  const options={
    coverForTag:tag=>tag==="paws"?"/cover":null,
    collectionItems:(tag,peer)=>tag==="paws"?pictures.slice(0,peer?6:50):[],
    editCollectionImage:(image,state)=>{editedImage=image;edit=state;}
  };
  const atlas=Viz.tagbenchConstellation(graph,()=>{},"paws",options);
  const allIn=root=>descendants(root);
  const byLabel=(root,label)=>allIn(root).find(node=>node.attrs["aria-label"]===label);
  const gallery=root=>allIn(root).find(node=>node.className==="studio-tb-collection");
  const tiles=root=>allIn(root).filter(node=>node.className==="studio-tb-flow-tile");
  const map=atlas.children[1],mapView=map.attrs.viewBox;
  assert.ok(allIn(atlas).some(node=>node.attrs.href==="/cover"));
  byLabel(atlas,"View images tagged paws").handlers.click();
  assert.equal(tiles(gallery(atlas)).length,50);
  assert.equal(map.inert,true);
  const target=tiles(gallery(atlas)).find(tile=>tile.dataset.path===pictures[12].path);
  target.handlers.click({currentTarget:target});
  assert.equal(editedImage,pictures[12]);
  assert.equal(edit.tag,"paws");
  assert.equal(edit.path,pictures[12].path);
  atlas.handlers.keydown({key:"Escape",preventDefault(){},stopPropagation(){}});
  assert.equal(gallery(atlas),undefined);
  assert.equal(map.attrs.viewBox,mapView);
  assert.equal(map.inert,false);

  const resumed=Viz.tagbenchConstellation(graph,()=>{},"paws",options);
  resumed.resumeCollection(edit);
  assert.equal(tiles(gallery(resumed)).length,50);
  assert.equal(global.document.activeElement.dataset.path,pictures[12].path);
  assert.equal(resumed.children[1].attrs.viewBox,mapView);

  const sharedAtlas=Viz.tagbenchConstellation(graph,()=>{},"paws",{
    ...options,relationsForTag:()=>new Map([["white fur",6]])
  });
  byLabel(sharedAtlas,"paws and white fur, 6 shared images").handlers.click();
  assert.equal(tiles(gallery(sharedAtlas)).length,6);
  const filter=byLabel(sharedAtlas,"Filter by related tag");
  filter.value="";filter.handlers.change();
  assert.equal(tiles(gallery(sharedAtlas)).length,50);
  const afterEdit=Viz.tagbenchConstellation(graph,()=>{},"paws",{
    ...options,collectionItems:()=>pictures.filter(picture=>picture!==editedImage)
  });
  afterEdit.resumeCollection(edit);
  assert.equal(tiles(gallery(afterEdit)).length,49);
  assert.equal(global.document.activeElement.dataset.path,pictures[13].path);

  const wheelAtlas=Viz.tagbenchConstellation(graph,()=>{},"paws",options);
  const wheelTarget=allIn(wheelAtlas).find(node=>node.attrs["data-tag"]==="paws");
  const wheel=target=>wheelAtlas.children[1].handlers.wheel({
    deltaY:-1,timeStamp:100,clientX:0,clientY:0,target:{closest:()=>target},preventDefault(){}
  });
  for(let i=0;i<8;i++)wheel(null);
  assert.equal(gallery(wheelAtlas),undefined);
  wheel(wheelTarget);
  assert.equal(tiles(gallery(wheelAtlas)).length,50);

  function scroll(root,target,deltaY,timeStamp,extra={}){
    let prevented=false,stopped=false;
    root.handlers.wheel({
      deltaX:0,deltaY,deltaMode:0,timeStamp,target,...extra,
      preventDefault(){prevented=true;},stopPropagation(){stopped=true;}
    });
    return {prevented,stopped};
  }

  const wheelFlow=gallery(wheelAtlas).children[1];
  wheelFlow.scrollTop=0;
  scroll(wheelAtlas,wheelFlow,-80,120);
  assert.ok(gallery(wheelAtlas),"Momentum from zooming in leaves the collection open");
  const beforeReturn=wheelAtlas.children[1].attrs.viewBox;
  scroll(wheelAtlas,wheelFlow,-20,350);
  assert.ok(gallery(wheelAtlas),"A small trackpad movement stays in the collection");
  scroll(wheelAtlas,wheelFlow,-25,380);
  assert.equal(gallery(wheelAtlas),undefined);
  assert.equal(wheelAtlas.children[1].attrs.viewBox,beforeReturn);
  assert.deepEqual(scroll(wheelAtlas,wheelAtlas.children[1],-16,420),{prevented:true,stopped:true});
  assert.deepEqual(scroll(wheelAtlas,wheelAtlas.children[1],-16,650),{prevented:false,stopped:false});

  const largeAtlas=Viz.tagbenchConstellation(graph,()=>{},"paws",{
    ...options,relationsForTag:()=>new Map([["white fur",6]]),
    collectionItems:()=>Array.from({length:400},(_,index)=>
      ({...it,name:"paws "+index,path:"/dataset/large-"+index+".webp"}))
  });
  byLabel(largeAtlas,"View images tagged paws").handlers.click();
  const largeFlow=gallery(largeAtlas).children[1];
  assert.equal(tiles(largeFlow).length,120);
  largeFlow.scrollTop=600;
  assert.equal(scroll(largeAtlas,largeFlow,100,100).prevented,false);
  assert.equal(scroll(largeAtlas,largeFlow,-100,320).prevented,false);
  largeFlow.scrollTop=0;
  scroll(largeAtlas,largeFlow,-100,350);
  assert.ok(gallery(largeAtlas),"Scrolling to the top keeps the gallery open until a new gesture");
  scroll(largeAtlas,largeFlow,-3,600,{deltaMode:1});
  assert.equal(gallery(largeAtlas),undefined);

  byLabel(largeAtlas,"View images tagged paws").handlers.click();
  const [newHeader,newFlow]=gallery(largeAtlas).children;
  newFlow.scrollTop=600;
  const filterControl=byLabel(largeAtlas,"Filter by related tag");
  filterControl.closest=selector=>selector==="select"?filterControl:null;
  assert.equal(scroll(largeAtlas,filterControl,100,800).prevented,false);
  scroll(largeAtlas,newHeader,100,1020);
  assert.equal(gallery(largeAtlas),undefined,"Reverse scroll over the header works even deep in the gallery");

  // Every primary neighbor gets an edge, and only a bounded page of
  // additional rare tags is materialized. All visible nodes have an edge.
  const many=Viz.tagbenchModel(Array.from({length:155},(_,i)=>({tags:["anchor","peer_"+i]})));
  const manyRelations=new Map(many.all.filter(n=>n.tag!=="anchor").map(n=>[n.tag,1]));
  const manyMap=Viz.tagbenchConstellation(many,()=>{},"anchor",{
    relationsForTag:()=>manyRelations
  });
  const pairNodes=()=>allIn(manyMap).filter(n=>n.attrs.class==="studio-tb-pair-edge");
  const visibleNodes=()=>allIn(manyMap).filter(n=>n.attrs["data-tag"]&&n.style.display!=="none");
  assert.equal(pairNodes().length,119+16);
  assert.equal(visibleNodes().length,pairNodes().length+1);
  assert.ok(allIn(manyMap).some(n=>n.textContent==="135 of 155 connections shown"));
  const beforePeers=new Set(visibleNodes().map(n=>n.attrs["data-tag"]));
  byLabel(manyMap,"Next additional connections").handlers.click();
  assert.equal(pairNodes().length,119+16);
  const afterPeers=new Set(visibleNodes().map(n=>n.attrs["data-tag"]));
  assert.ok([...afterPeers].some(tag=>!beforePeers.has(tag)));
  assert.equal(visibleNodes().length,pairNodes().length+1);
  assert.ok(many.all.every(n=>many.colors.has(n.tag)));
  manyMap.dispose();

  // One shared timer, actual href advancement, pause while Waterfall
  // is open, and cleanup when the map is disposed.
  const oldInterval=global.setInterval,oldClear=global.clearInterval,oldMatch=global.matchMedia;
  let slideTick=null,starts=0,stops=0,waterfallState=null;
  try{
    global.setInterval=(fn,ms)=>{assert.equal(ms,2000);slideTick=fn;starts++;return starts;};
    global.clearInterval=()=>{slideTick=null;stops++;};
    global.matchMedia=()=>({matches:false});
    const rotating=Viz.tagbenchConstellation(graph,()=>{},"paws",{
      ...options,preferences:{elasticity:0},
      coverForTag:(tag,frame=0)=>tag==="paws"?"/slide-"+frame:null,
      coverCountForTag:tag=>tag==="paws"?3:0,
      waterfallScale:()=>80,
      openWaterfall:(tag,peer,state)=>{waterfallState={tag,peer,state};}
    });
    rotating.isConnected=true;
    rotating.setPreferences({elasticity:0});
    assert.equal(starts,1);
    const cover=allIn(rotating).find(node=>node.attrs.href==="/slide-0");
    assert.ok(cover);
    slideTick();assert.equal(cover.attrs.href,"/slide-1");
    slideTick();assert.equal(cover.attrs.href,"/slide-2");
    slideTick();assert.equal(cover.attrs.href,"/slide-0");
    rotating.setPreferences({elasticity:0});
    assert.equal(starts,1,"Repeated appearance updates must not start another timer");
    byLabel(rotating,"Open paws in Waterfall").handlers.click();
    assert.equal(waterfallState.tag,"paws");
    assert.equal(waterfallState.state.picked,"paws");
    byLabel(rotating,"View images tagged paws").handlers.click();
    assert.equal(slideTick,null);
    assert.equal(stops,1);
    byLabel(rotating,"Open this collection in Waterfall").handlers.click();
    assert.equal(waterfallState.state.imageIndex,0);
    rotating.handlers.keydown({key:"Escape",preventDefault(){},stopPropagation(){}});
    assert.equal(starts,2);
    rotating.dispose();
    assert.equal(stops,2);
    assert.equal(slideTick,null);
  } finally {
    global.setInterval=oldInterval;global.clearInterval=oldClear;global.matchMedia=oldMatch;
  }

} finally {
  global.document=savedDoc;
  global.window=savedWin;
  global.ResizeObserver=savedResize;global.MutationObserver=savedMutation;global.matchMedia=savedMotion;
}
// One palette powers Constellation, Image Tags and All Tags. Direct
// right-click colors override automatic and legacy category assignments.
const assigned={tagColors:{lion:"#1357ab"},tags:{lion:"legacy"},
  colors:{legacy:"#ff6600"}};
const colored=Viz.tagbenchModel(captions2,assigned);
assert.equal(colored.nodes.find(n=>n.tag==="lion").color,"#1357ab");
assert.equal(colored.colors.get("lion"),"#1357ab");
assert.equal(Viz.colorForTag("LION",assigned),"#1357ab");
const auto=Viz.tagbenchModel(captions2);
Viz.setTagbenchPalette(auto.colors);
assert.equal(Viz.colorForTag("lion",{tags:{},colors:{}}),
  auto.colors.get("lion"),"Image Tags and All Tags use the same graph hue");
Viz.setTagbenchPalette(null);

console.log("Caption Studio visualization co-occurrence and double-click checks passed");

assert.equal(Viz.constellationPrefs({glass:true}).glassNodes,false,"old default glass preference must not override restored appearance");

const covers=Viz.tagbenchModel(Array.from({length:200},(_,i)=>({tags:["common", "only_"+i],coverIndex:i})));
assert.equal(covers.covers.get("common"),0);
assert.equal(covers.coverSamples.get("common").length,8);
assert.equal(new Set(covers.coverSamples.get("common")).size,8);
assert.equal(covers.coverSamples.get("common")[0],0);
assert.equal(covers.coverSamples.has("only_0"),false);
assert.deepEqual(covers.coverSamples,Viz.tagbenchModel(Array.from({length:200},(_,i)=>({tags:["common","only_"+i],coverIndex:i}))).coverSamples);
console.log("Bounded deterministic slideshow samples passed");

const loss=Viz.summarizeAtlasLoss([
  {tags:["lion","lion","mane"],recent:2,verdict:"stuck"},
  {tags:["lion"],recent:0},
  {tags:["fox"],recent:-1,verdict:"suspect"},
  {tags:["lion"],recent:null},
  {tags:["lion"],recent:NaN}
],[{id:0,memberTags:["lion","mane"]},{id:1,memberTags:["fox"]}]);
assert.equal(loss.tags.get("lion").measured,2);
assert.equal(loss.tags.get("lion").mean,1);
assert.equal(loss.tags.get("lion").stuck,1);
assert.equal(loss.communities.get(0).measured,2);
assert.equal(loss.communities.get(0).mean,1);
assert.equal(loss.baseline.measured,3);
assert.equal(loss.baseline.mean,1/3);
assert.equal(Viz.lossTone(loss.tags.get("lion"),loss.baseline),"var(--text-muted)");
assert.equal(Viz.lossTone({measured:3,mean:2},loss.baseline),"var(--danger)");
assert.equal(Viz.lossTone({measured:3,mean:-1},loss.baseline),"var(--data-teal)");
const warningModel=Viz.tagbenchModel([{tags:["blue eyes","blue eyes","blue_eyes","lion"]},{tags:["blue-eyes","lion"]}]);
assert.equal(warningModel.warnings.get("blue eyes").duplicates,1);
assert.equal(warningModel.warnings.get("blue eyes").variantCount,2);
assert.equal(warningModel.warnings.has("lion"),false);
assert.equal(warningModel.atlas.clusters.reduce((sum,c)=>sum+c.imageCount,0)>=2,true);
console.log("Loss aggregation, no double counting, low-sample caution and caption warnings passed");
