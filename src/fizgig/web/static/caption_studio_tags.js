/* Caption Studio's pure parsing, formatting and Boolean tag-search helpers.
 * Shared by the UI and Node regression tests; no DOM and no external services.
 */
const CaptionStudioTags = (() => {
  const firstLine = text => String(text || "").split(/\r?\n/, 1)[0].trim();

  function kind(text, override = "auto", delimiter = ",") {
    if (override === "tags" || override === "prose") return override;
    const line = firstLine(text);
    if (!line) return "tags";
    // Favour prose when uncertain: accidentally treating a sentence as tags
    // would turn a sort/deduplicate click into a destructive rewrite.
    const bits = line.split(delimiter).map(x => x.trim()).filter(Boolean);
    if (/[.!?](?:\s|$)/.test(line)) return "prose";
    if (bits.some(x => x.split(/\s+/).length >= 8)) return "prose";
    if (bits.length === 1 && bits[0].split(/\s+/).length >= 5) return "prose";
    return "tags";
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

  // Exact, case-insensitive tag matching. Supports quoted multiword tags,
  // parentheses, NOT, AND, OR and implied AND between adjacent tag clauses.
  function search(values, expression) {
    const raw = String(expression || "").trim();
    if (!raw) return true;
    const lex = raw.match(/\(|\)|"(?:[^"\\]|\\.)*"|'[^']*'|[^\s()]+/g) || [];
    const tokens = lex.map(t => /^(AND|OR|NOT)$/i.test(t) ? t.toUpperCase() : t);
    const have = new Set(values.map(v => v.trim().toLocaleLowerCase()));
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
      return have.has(stripped.toLocaleLowerCase());
    }
    function not() {
      if (tokens[at] === "NOT") { at++; return !not(); }
      if (tokens[at]?.startsWith("-") && tokens[at].length > 1) {
        tokens[at] = tokens[at].slice(1);
        return !not();
      }
      return atom();
    }
    function and() {
      let value = not();
      while (at < tokens.length && tokens[at] !== ")" && tokens[at] !== "OR") {
        if (tokens[at] === "AND") at++;
        const right = not();  // always parse RHS, even when LHS is false
        value = value && right;
      }
      return value;
    }
    function or() {
      let value = and();
      while (tokens[at] === "OR") {
        at++;
        const right = and();
        value = value || right;
      }
      return value;
    }
    const result = or();
    if (at !== tokens.length) throw new Error("Unexpected search token: " + tokens[at]);
    return result;
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

  return Object.freeze({kind, tags, compose, unique, search, order, captionWildcards, wildcards});
})();
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioTags;
