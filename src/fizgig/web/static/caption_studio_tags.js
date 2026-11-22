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

  const wildcards = (values, delimiter = ",") => unique(values).sort((a, b) => a.localeCompare(b))
    .join("\n") + (values.length ? "\n" : "");

  return Object.freeze({kind, tags, compose, unique, search, wildcards});
})();
if (typeof module !== "undefined" && module.exports) module.exports = CaptionStudioTags;
