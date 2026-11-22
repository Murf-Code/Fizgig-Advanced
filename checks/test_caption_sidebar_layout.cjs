"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");

const app=fs.readFileSync("src/fizgig/web/static/app.js","utf8");
const css=fs.readFileSync("src/fizgig/web/static/style.css","utf8");

// The TagBench-inspired sidebar has TWO visible lists instead of tabbing
// between Image Tags and All Tags. The bottom half may show AI tools.
assert.match(app,/const editorPane=el\("section",\{class:"studio-tag-editor-pane"\}\)/);
assert.match(app,/const tagIndexPane=el\("section",\{class:"studio-tag-index-pane"\}\)/);
assert.match(app,/right\.querySelector\("\.studio-image-tags-heading"\)/);
assert.match(app,/right\.querySelector\("\.studio-add-tag"\),tags,word/);
assert.match(app,/while\(right\.firstChild\)tagIndexPane\.append\(right\.firstChild\)/);
assert.match(app,/rightDock\.append\(showAll,showAi\)/);
assert.match(app,/right\.append\(editorPane,tagHeader,tagIndexPane,aiPane\)/);
assert.match(app,/const showAll=btn\("All Tags",\(\)=>\{/);
assert.match(app,/const showAi=btn\("Auto-Captioner",\(\)=>\{/);
assert.doesNotMatch(app,/editorPane\.hidden\s*=/);
assert.doesNotMatch(app,/showImage=btn\("Image Tags"/);
assert.match(app,/tagIndexPane\.hidden=S\.cap\.studioDock!=="all"/);
assert.match(app,/aiPane\.hidden=S\.cap\.studioDock!=="ai"/);

// Search, tag frequency sort with an independent ascending/descending
// control, auto-completion, and keyboard-accessible natural-language toggle.
assert.match(app,/const sortDirection=el\("select"/);
assert.match(app,/sortDirection\.addEventListener\("change",updateTagSort\)/);
assert.match(app,/indexSearch\.addEventListener\("input"/);
assert.match(app,/studio-add-tag-suggestions/);
assert.match(app,/class:"studio-natural-toggle"/);
assert.match(css,/#tab-captions \.studio-host \.studio-image-tag-list\s*\{/);
assert.match(css,/#tab-captions \.studio-host \.studio-tag-index-pane \.studio-explore>/);
assert.match(css,/min-height:0;\s*height:100%;\s*overscroll-behavior:contain/);
assert.match(css,/#tab-captions \.studio-tag-dock\[data-active="all"\]/);

// The user explicitly asked to leave the existing global selector alone.
// Waterfall and Constellation still replace the full workbench stage.
assert.match(app,/const viewButtons=el\("div",\{class:"studio-view-rail"/);
assert.match(app,/\[\["list","Tags"\],\["waterfall","Waterfall"\],\["constellation","Constellation"\]\]/);
assert.match(app,/const openView=\(mode\)=>/);
assert.match(app,/fullStage\.append\(flow\)/);
assert.match(app,/fullStage\.append\(network\)/);
assert.match(app,/const flow=CaptionStudioViz\.tagbenchWaterfall\(/);
assert.match(css,/#tab-captions \.studio-tb-explore\s*\{/);

// The left splitter starts at the smallest allowed 15% (rather than the old
// 19%) without discarding custom widths. Thumbnail cards render 15% smaller.
assert.match(app,/sizes=\{left:15,right:36\}/);
assert.match(app,/sizes\.left===19&&sizes\.right===36/);
assert.match(app,/Math\.max\(15,Math\.min\(30,sizes\.left\)\)/);
assert.match(css,/#tab-captions \.studio-thumb-rail \.studio-browser-row\s*\{[^}]*width:85%;/);
assert.match(css,/max-height:calc\(var\(--caption-settings-preview-height,360px\) \* \.85\)/);

// The center image fits available vertical space, preserving aspect ratio.
// The old 68vh/750px image ceiling prevented this on tall workbench layouts.
assert.match(css,/#tab-captions \.studio-host \.studio-editor\{[^}]*height:min\(82dvh,950px\)/);
assert.match(css,/#tab-captions \.studio-image-stage \.studio-preview\{[^}]*flex:1 1 0;[^}]*height:100%;/);
assert.match(css,/#tab-captions \.studio-image-stage \.studio-preview img,[\s\S]*?max-height:100%;[\s\S]*?object-fit:contain/);
assert.doesNotMatch(css,/\.studio-image-stage \.studio-preview img[^\n]*max-height:min\(68vh,750px\)/);

// Thumbnail wheel acceleration is limited to detented mouse wheels so
// precision touchpads continue using native scrolling. Programmatic rail
// repositioning must not animate when selecting or restoring an image.
assert.match(app,/thumbs\.addEventListener\("wheel",e=>\{/);
assert.match(app,/amount%100===0\|\|amount%120===0/);
assert.match(app,/\},\{passive:false\}\);/);
assert.match(css,/#tab-captions \.studio-host \.studio-thumb-rail,[\s\S]*?scroll-behavior:auto;/);
assert.doesNotMatch(css,/#tab-captions \.studio-host \.studio-thumb-rail,[\s\S]*?scroll-behavior:smooth;/);

// The Tags rail is continuous and progressively mounted. Cards/Table alone
// retain CAP_PAGE; arrow navigation and waterfall jumps reveal the active tile.
assert.match(app,/const thumbBatch=64/);
assert.match(app,/CaptionStudioSelection\.initialMountCount\(items\.length/);
assert.match(app,/CaptionStudioSelection\.nextMountCount\(thumbRendered,items\.length,thumbBatch\)/);
assert.match(app,/thumbs\.insertBefore\(fragment,moreThumbs\)/);
assert.match(app,/new IntersectionObserver\(/);
assert.doesNotMatch(app,/const pageItems=items\.slice\(S\.cap\.page\*CAP_PAGE/);
assert.match(app,/S\.cap\.studioThumbRevealPath=next\.path/);

// One AI caption action, automatically scoped to selected images, including
// Ctrl+A across the full dataset. No redundant selector or Selected Images card.
assert.match(app,/S\.cap\.studioCaptionSelected=captionSelected/);
assert.match(app,/CaptionStudioSelection\.captionTargets\(/);
assert.match(app,/if\(\$\("capView"\)\.value==="studio"\)/);
assert.match(app,/S\.cap\.studioCaptionSelected\?\.\(\)/);
assert.match(app,/S\.cap\.items\.forEach\(it=>selection\.add\(it\.path\)\)/);
assert.doesNotMatch(app,/const aiChoice=el\("select"/);
assert.doesNotMatch(app,/class:"studio-ai-selection"/);
assert.match(app,/action\.textContent="Caption selected \(AI\)"/);

// Image Tags and All Tags share density, colors, selection and glass markers.
assert.match(css,/Image Tags uses the same compact tag-list treatment as All Tags/);
assert.match(css,/#tab-captions \.studio-host \.studio-image-tag,[\s\S]*?#tab-captions \.studio-host \.studio-tag-list-item \.studio-index-tag\{/);
assert.match(css,/#tab-captions\.caption-theme-glass \.studio-host \.studio-image-tag::before/);

// Tag entry uses one field with Enter or a real clickable suggestion.
assert.match(app,/const addTag=value=>/);
assert.match(app,/newTag\.addEventListener\("keydown"/);
assert.match(app,/if\(e\.key==="Enter"&&!e\.isComposing\)/);
assert.match(app,/tagSuggestionList\.append\(option\)/);
assert.match(app,/option\.addEventListener\("click"/);
assert.doesNotMatch(app,/newTag,btn\("Add"/);
assert.doesNotMatch(app,/newTag\.parentElement\?\.querySelector\("button"\)/);

// No old category settings panel. Colors are edited from tag context menus,
// shared by Image Tags, All Tags and the cached Constellation palette.
assert.doesNotMatch(app,/const catPanel=el\("details"/);
assert.doesNotMatch(app,/const advancedTags=el\("details"/);
assert.match(app,/studioTagColorMenu\(menu,tag,categoryState,renderCaptions\)/);
assert.match(app,/tagColors:raw\.tagColors/);
assert.match(app,/categoryState\.generatedColors=themeGraph\.model\.colors/);
assert.match(app,/CaptionStudioViz\.setTagbenchPalette\(themeGraph\.model\.colors\)/);
assert.match(app,/const model=CaptionStudioViz\.tagbenchModel\(graphTags,categoryState\)/);
assert.match(css,/studio-tag-color-swatches/);

// Natural language replaces the lower All Tags list using the one autosaved
// textarea; clicking All Tags restores the list.
assert.match(app,/tagIndexPane\.replaceChildren\(/);
assert.match(app,/S\.cap\.studioNaturalMode=false;S\.cap\.studioDock="all";renderCaptions\(\)/);
assert.match(app,/const naturalEditor=el\("section",\{class:"studio-natural-pane"\},captionText,captionHelp\)/);
assert.match(css,/#tab-captions \.studio-host \.studio-tag-index-pane>\.studio-natural-pane/);
assert.match(css,/#tab-captions\.caption-theme-glass \.studio-host \.studio-natural-toggle:has\(input:checked\)/);

console.log("TagBench stacked sidebar and unchanged Tags/Waterfall/Constellation selector checks passed");

// Fast tag selection, single-pass filtering, aligned caption header, TagBench help.
assert.match(app,/const imageTagButtons=new Map\(\)/);
assert.match(app,/const syncImageTagSelection=before=>/);
assert.doesNotMatch(app,/for\(const el of tags\.querySelectorAll\("button"\)\)/);
assert.match(app,/const tagBatchSize=80/);
assert.match(app,/indexSearch\.addEventListener\("input",tagRows\)/);
assert.match(app,/if\(index\.parentElement!==explore\)explore\.replaceChildren\(index\)/);
assert.match(app,/const tagHeader=el\("div",\{class:"studio-tag-header"\},headerTitle,rightDock\)/);
assert.match(app,/Freeform caption\. Saved under the tags, after the first line break\./);
assert.match(app,/const captionHelp=el\("p",\{class:"studio-natural-help"/);
assert.match(css,/#tab-captions \.studio-host \.studio-tag-header\s*\{/);

assert.match(app,/CaptionStudioLearning\.filterByVerdict\(items,S\.cap\.learningMap,/);
assert.match(app,/const createLearningFilterSelect=\(\)=>/);
assert.match(app,/actionBar\.append\(learningToggle,learningFilter,viewButtons\)/);
assert.match(app,/const refilter=\(S\.cap\.studioLearningFilter\|\|"all"\)!=="all"/);
assert.match(css,/#tab-captions \.studio-learning-filter-label\{/);
