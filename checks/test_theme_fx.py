"""The bar effects (shimmer, light sweep, star specks, button glow) on every theme, each in its own colours, and all of
it off under reduced motion."""
import os
import re

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static")
ALL = ":is(:root:not([data-theme]), :root[data-theme])"


def _read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as f:
        return f.read()


def _fx_block(css):
    a = css.index("/* Every theme: buttons glow and lift")
    return css[a:css.index("animation-delay: var(--fx-d, 0s) !important; }", a)]


def _themes():
    js = _read("app.js")
    block = js[js.index("const THEMES = ["):js.index("];", js.index("const THEMES = ["))]
    return re.findall(r'key: "([a-z-]*)"', block)


def test_every_theme_has_its_own_bar_colours():
    fx = _fx_block(_read("style.css"))
    themes = _themes()
    assert "" in themes and "colour-safe" in themes and len(themes) >= 10
    assert re.search(r":root \{\s*--fx-glow:[^}]*--fx-bar:[^}]*--fx-sweep:", fx)       # the default (Fizgig) theme
    bars = {}
    for key in themes:
        if key:
            m = re.search(r':root\[data-theme="%s"\] \{[^}]*--fx-bar: ([^;]+);' % re.escape(key), fx)
            assert m, key
            bars[key] = m.group(1)
    assert len(set(bars.values())) == len(bars)                       # no two themes share a bar
    # the colour-blind safe theme: blues only (Okabe-Ito blue / sky blue), never red against green
    assert set(re.findall(r"#[0-9A-F]{6}", bars["colour-safe"])) <= {"#0072B2", "#2F8FD8", "#56B4E9"}


def test_the_effects_are_for_every_theme_not_two():
    fx = _fx_block(_read("style.css"))
    assert 'data-theme="cosmic-pigeon"], :root[data-theme="deep-space"])' not in fx
    for part in (":is(.btn, .runpod-btn, .theme-pick button):not(:disabled):hover", "> span, .fetch-bar > div)::before",
                 "> span, .fetch-bar > div)::after"):
        assert re.search(re.escape(ALL) + r"[^{]*" + re.escape(part), fx), part
    assert "radial-gradient(circle at 12% 30%, #fff" in fx            # the tiny white star specks
    assert "var(--fx-sweep) 70px" in fx and "rgba(255,255,255,.35) 70px" not in fx


def test_reduced_motion_turns_it_all_off():
    fx = _fx_block(_read("style.css"))
    rm = fx[fx.index("@media (prefers-reduced-motion: reduce)"):]
    rm = rm[:rm.index("\n}\n")]
    assert ALL + " :is(.run-bar > span, .meter-bar > span, .fetch-bar > div) { animation: none; }" in rm
    assert "::before { display: none; }" in rm
    assert ALL + " :is(.btn, .runpod-btn, .theme-pick button) { transition: none; transform: none !important; }" in rm
