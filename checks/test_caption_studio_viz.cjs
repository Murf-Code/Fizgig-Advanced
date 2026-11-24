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

// Large sparse tag vocabularies must not increase the constellation's 320-node
// graph limit or create edges outside that set.
const sparse=Array.from({length:260},(_,i)=>({tags:["common","label_"+i,"rare_"+i]}));
const sparseModel=Viz.tagbenchModel(sparse);
const known=new Set(sparseModel.top.map(t=>t.tag));
assert.equal(sparseModel.top.length,320);
assert.ok(sparseModel.edges.every(e=>known.has(e.a)&&known.has(e.b)));
assert.ok(sparseModel.edges.length<=320*319/2);

// Waterfall tiles preserve ordinary single-click selection. Double-click is
// a distinct action that opens the exact tile in the Tags editor.
const savedDoc=global.document, savedWin=global.window;
try {
  const fakeElement=()=>{
    const children=[],handlers={},classes=new Set(),attrs={};
    return {children,handlers,dataset:{},classes,attrs,classList:{add(name){classes.add(name);}},
      style:{},isConnected:false,
      append(...elements){children.push(...elements);},
      setAttribute(name,value){attrs[name]=value;},addEventListener(name,fn){handlers[name]=fn;},
      remove(){},replaceChildren(...elements){children.splice(0,children.length,...elements);}
    };
  };
  global.document={createElement:()=>fakeElement()};
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
} finally {
  global.document=savedDoc;
  global.window=savedWin;
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
