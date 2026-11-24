/* Caption Studio's pure parsing, formatting and Boolean tag-search helpers.
 * Shared by the UI and Node regression tests; no DOM and no external services.
 */
const CaptionStudioTags = (() => {
  const firstLine = text => String(text || "").split(/\r?\n/, 1)[0].trim();

  function kind(text, override = "auto", delimiter = ",") {
    if (override === "tags" || override === "prose") return override;
    const line = firstLine(text);
    if (!line) return "tags";
    // Auto-detect prose only when the caption has multiple complete
    // sentences. One trailing period is common in comma-separated tag lists
    // and must never switch an image to Natural Language automatically.
    // The manual Natural Language toggle still allows single-sentence prose.
    const bits = line.split(delimiter).map(x => x.trim()).filter(Boolean);
    const shortTags = bits.filter(x => x.split(/\s+/).length <= 3).length;
    if (bits.length >= 3 && shortTags >= Math.ceil(bits.length * 0.8)) return "tags";
    const sentenceEnds = (String(text || "").match(/[.!?]+(?:["'”’)]*)(?=\s|$)/g) || []).length;
    return sentenceEnds >= 2 ? "prose" : "tags";
  }

  function tags(text, mode = "auto", delimiter = ",") {
    if (kind(text, mode, delimiter) === "prose") return [];
    return firstLine(text).split(delimiter).map(x => x.trim()).filter(Boolean);
  }

  function compose(original, next, mode = "auto", delimiter = ",") {
    if (kind(original, mode, delimiter) === "prose") return String(original || "");
    const rest = String(original || "").match(/\r?\n[\s\S]*$/);
    return next.join(delimiter + " ") + (rest ? rest[0] : "");
  }

  function unique(values) {
    const seen = new Set();
    return values.filter(value => {
      const key = value.toLocaleLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  // Parse the Boolean filter once per render rather than once per image.
  // The compiled predicate retains TagBench's exact-tag AND/OR/NOT semantics.
  function compileSearch(expression) {
    const raw = String(expression || "").trim();
    if (!raw) return () => true;
    const lex = raw.match(/\(|\)|"(?:[^"\\]|\\.)*"|'[^']*'|[^\s()]+/g) || [];
    const tokens = lex.map(t => /^(AND|OR|NOT)$/i.test(t) ? t.toUpperCase() : t);
    let at = 0;
    function atom() {
      const token = tokens[at++];
      if (token === "(") {
        const value = or();
        if (tokens[at++] !== ")") throw new Error("Missing closing parenthesis");
        return value;
      }
      if (!token || token === ")" || token === "AND" || token === "OR" || token === "NOT")
        throw new Error("Expected a tag");
      const stripped = token.replace(/^['"]|['"]$/g, "").replace(/\\"/g, '"');
      if (!stripped.trim()) throw new Error("Empty tag");
      const key = stripped.toLocaleLowerCase();
      return have => have.has(key);
    }
    function not() {
      if (tokens[at] === "NOT") { at++; const value=not(); return have=>!value(have); }
      if (tokens[at]?.startsWith("-") && tokens[at].length > 1) {
        tokens[at] = tokens[at].slice(1);
        const value=not(); return have=>!value(have);
      }
      return atom();
    }
    function and() {
      let value = not();
      while (at < tokens.length && tokens[at] !== ")" && tokens[at] !== "OR") {
        if (tokens[at] === "AND") at++;
        const left=value,right=not();
        value=have=>left(have)&&right(have);
      }
      return value;
    }
    function or() {
      let value = and();
      while (tokens[at] === "OR") {
        at++;
        const left=value,right=and();
        value=have=>left(have)||right(have);
      }
      return value;
    }
    const predicate=or();
    if (at !== tokens.length) throw new Error("Unexpected search token: " + tokens[at]);
    return values => {
      const have=new Set(values.map(v=>v.trim().toLocaleLowerCase()));
      return predicate(have);
    };
  }
  function search(values, expression) {
    return compileSearch(expression)(values);
  }

  // Longest known multiword tags win; explicit Boolean operators delimit terms.
  function searchParts(expression, vocabulary=new Set()) {
    const raw=String(expression||"");
    const tokens=raw.match(/"(?:[^"\\]|\\.)*"|'[^']*'|\s+|[()]|[^\s()"]+|"/g)||[];
    const op=x=>/^(AND|OR|NOT)$/i.test(x||"");
    const boundary=x=>!x||op(x)||/^[()"']/.test(x)||x.startsWith("-");
    const parts=[];
    for(let i=0;i<tokens.length;i++){
      const token=tokens[i];
      if(/^\s+$/.test(token)||token==="("||token===")"){
        parts.push({text:token,syntax:token});continue;
      }
      if(op(token)){
        parts.push({text:token.toUpperCase(),syntax:token.toUpperCase(),operator:true});continue;
      }
      if(/^["']/.test(token)){
        let label=token;
        if(token[0]==='"'){try{label=JSON.parse(token);}catch(_){}}
        else if(token.endsWith("'")&&token.length>1)label=token.slice(1,-1);
        const known=vocabulary.has(String(label).toLocaleLowerCase());
        parts.push({text:token,syntax:token,tag:known?label:null});continue;
      }
      const negative=token.startsWith("-")&&token.length>1;
      let phrase=negative?token.slice(1):token,best=phrase,end=i;
      let found=vocabulary.has(phrase.toLocaleLowerCase());
      for(let j=i+1;j+1<tokens.length&&/^\s+$/.test(tokens[j])&&!boundary(tokens[j+1]);j+=2){
        phrase+=tokens[j]+tokens[j+1];
        if(vocabulary.has(phrase.toLocaleLowerCase())){best=phrase;end=j+1;found=true;}
      }
      if(found){
        if(negative)parts.push({text:"-",syntax:"NOT ",operator:true});
        parts.push({text:best,syntax:JSON.stringify(best),tag:best});
        i=end;
      }else parts.push({text:token,syntax:token});
    }
    return parts;
  }
  function naturalSearch(expression,vocabulary){
    return searchParts(expression,vocabulary).map(p=>p.syntax).join("").trim();
  }
  function displaySearch(expression,vocabulary){
    return searchParts(expression,vocabulary).map(p=>{
      if(!p.tag||!/^["']/.test(p.text))return p.text;
      const tag=String(p.tag);
      if(/[()"']/.test(tag)||/(^|\s)(AND|OR|NOT)(\s|$)/i.test(tag)||tag.startsWith("-"))return p.text;
      return tag;
    }).join("");
  }

  // TagBench context-menu Boolean actions: preserve the user's existing
  // expression and wrap it before combining to avoid changing precedence.
  function addFilter(expression, tag, mode = "and") {
    const raw=String(expression||"").trim();
    const label=String(tag||"").trim();
    if(!label)return raw;
    if(!["and","or","not"].includes(mode))throw new Error("Unknown filter mode");
    const term=JSON.stringify(label);
    const clause=(mode==="not"?"NOT ":"")+term;
    const next=!raw?clause:"("+raw+") "+(mode==="or"?"OR":"AND")+" "+clause;
    compileSearch(next);
    return next;
  }

  // Deterministic tag ordering. Front/back pins preserve user-supplied priority,
  // while the rest can use frequency or the model tokenizer's real token counts.
  function order(values, options = {}) {
    const rule = options.rule || "original";
    const key = x => String(x).trim().toLocaleLowerCase();
    const items = unique(values.map(x => String(x).trim()).filter(Boolean));
    const byKey = new Map(items.map(x => [key(x), x]));
    const front = unique(options.front || []).map(key);
    const back = unique(options.back || []).map(key).filter(x => !front.includes(x));
    const pinned = new Set([...front, ...back]);
    const lookup = (source, tag) => source instanceof Map ? source.get(key(tag)) :
      source?.[key(tag)];
    const remaining = items.filter(x => !pinned.has(key(x)));
    const alpha = (a, b) => a.localeCompare(b, undefined, {sensitivity:"base"});
    const count = (source, tag) => Number(lookup(source, tag) ?? 0);
    if(rule==="reverse")remaining.reverse();
    else if(rule==="alpha")remaining.sort(alpha);
    else if(rule==="alpha-desc")remaining.sort((a,b)=>alpha(b,a));
    else if(rule==="frequency"||rule==="frequency-asc")
      remaining.sort((a,b)=>(rule==="frequency" ? count(options.counts,b)-count(options.counts,a) :
        count(options.counts,a)-count(options.counts,b)) || alpha(a,b));
    else if(rule==="token-asc"||rule==="token-desc")
      remaining.sort((a,b)=>(rule==="token-asc" ? count(options.tokens,a)-count(options.tokens,b) :
        count(options.tokens,b)-count(options.tokens,a)) || alpha(a,b));
    return [...front,...back].filter(x=>byKey.has(x)).length===0 ? remaining :
      [...front.filter(x=>byKey.has(x)).map(x=>byKey.get(x)),...remaining,
        ...back.filter(x=>byKey.has(x)).map(x=>byKey.get(x))];
  }

  // TagBench prompt wildcards contain one COMPLETE caption per image, rather
  // than one distinct tag per line. Preserve dataset order and duplicates.
  const captionWildcards = captions => {
    const lines = captions.map(text => String(text || "").replace(/\s*\n\s*/g," ")
      .replace(/\s+/g," ").trim()).filter(Boolean);
    return lines.join("\n") + (lines.length ? "\n" : "");
  };
  const wildcards = (values, delimiter = ",") => unique(values).sort((a, b) => a.localeCompare(b))
    .join("\n") + (values.length ? "\n" : "");

  // Searches are called on every keystroke, including when very large
  // datasets have thousands of distinct tags. The caller prepares the third
  // tuple field (lowercase tag) once per dataset revision, instead of doing
  // locale normalization for every tag on every keypress.
  // Stop as soon as the short autocomplete dropdown has enough suggestions.
  function findTagMatches(rows, query, excluded=null, limit=Infinity) {
    const needle=String(query||"").trim().toLocaleLowerCase();
    if(!needle&&!excluded)return rows;
    const found=[];
    for(const row of rows){
      const lower=row[2]??String(row[0]).toLocaleLowerCase();
      if(needle&&!lower.includes(needle))continue;
      if(excluded?.has(lower))continue;
      found.push(row);
      if(found.length>=limit)break;
    }
    return found;
  }

  // A click in a menu should retain the selection until its action runs.
  // Only a tag button (or its inline rename input) owns list selection.
  function tagSelectionScope(target) {
    if(!target?.closest)return null;
    if(target.closest(".studio-image-context-menu, .studio-tag-context-menu"))return "menu";
    if(target.closest(".studio-image-tag, .studio-inline-image-tag"))return "image";
    if(target.closest(".studio-index-tag, .studio-inline-tag-edit"))return "all";
    return null;
  }

  function scopeTagSelections(imagePicked, allPicked, scope) {
    const cleared={image:scope!=="image"&&scope!=="menu",
                   all:scope!=="all"&&scope!=="menu"};
    if(cleared.image)imagePicked.clear();
    if(cleared.all)allPicked.clear();
    return cleared;
  }

  return Object.freeze({kind, tags, compose, unique, search, compileSearch, addFilter, searchParts, naturalSearch, displaySearch,
    order, captionWildcards, wildcards, findTagMatches, tagSelectionScope, scopeTagSelections});
})();
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioTags;
