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

// The left splitter defaults to 12%, but can be dragged down to 6% with a
// true 96px minimum; custom widths still persist. Images keep their 85% size.
assert.match(app,/sizes=\{left:12,right:36\}/);
assert.match(app,/sizes\.left===19&&sizes\.right===36/);
assert.match(app,/Math\.max\(6,Math\.min\(30,sizes\.left\)\)/);
assert.match(app,/minmax\(96px,/);
assert.match(css,/#tab-captions \.studio-thumb-rail \.studio-browser-row\s*\{[^}]*width:85%;/);
assert.match(css,/max-height:calc\(var\(--caption-settings-preview-height,360px\) \* \.85\)/);

// The stage expands to fill the available notebook viewport, rather than
// being capped at 82dvh/950px. Short windows still scroll normally.
assert.match(css,/\.notebook-pane:has\(> #tab-captions\.active\)/);
assert.match(css,/#tab-captions\.active #captionsGrid\.studio-host > \.studio-layout\s*\{\s*flex:1 1 auto;/);
assert.match(css,/#tab-captions\.active #captionsGrid\.studio-host > \.studio-layout > :is\(\.studio-browser,\.studio-editor,\.studio-index\)/);
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

// Tags has a full-range virtualized scrollbar; only images in the visible
// window are mounted. Cards/Table alone retain CAP_PAGE.
assert.match(app,/function studioThumbGeometry\(items,railWidth,gap=14\)/);
assert.match(app,/function studioThumbIndex\(offsets,position\)/);
assert.match(app,/const paintWindow=\(\)=>/);
assert.match(app,/windowRows\.replaceChildren\(fragment\)/);
assert.match(app,/thumbs\.revealPath=path=>revealIndex\(/);
assert.match(app,/new ResizeObserver\(/);
assert.match(app,/thumbs\.addEventListener\("scroll",queueWindow,\{passive:true\}\)/);
assert.match(css,/#tab-captions \.studio-host \.studio-thumb-rail\.is-windowed/);
assert.doesNotMatch(app,/const moreThumbs=el\(/);
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

// Zero missing tags must never discard the full workbench. Empty searches
// get an explicit one-click recovery, including learning-status filters.
assert.match(app,/if\(filter==="missing"&&missingTotal===0&&all\.length\)/);
assert.match(app,/filter="all";/);
assert.match(app,/missingTotal===0\?"Missing Tags \(0\)"/);
assert.match(app,/class:"studio-missing-notice"/);
assert.match(app,/text:"Show all images"/);
assert.match(app,/S\.cap\.studioLearningFilter="all";/);
assert.match(css,/\.studio-empty-actions\{/);

// Scrolling Image Tags / All Tags must not flash hover rows under the cursor.
// Each mounted list is stabilized; All Tags lazy insertion is frame-batched.
assert.match(app,/const stabilizeTagScroll=list=>/);
assert.match(app,/stabilizeTagScroll\(tags\)/);
assert.match(app,/stabilizeTagScroll\(index\)/);
assert.match(app,/list\.classList\.add\("is-scrolling"\)/);
assert.match(app,/idle=setTimeout\(\(\)=>list\.classList\.remove\("is-scrolling"\),160\)/);
assert.match(app,/let tagScrollLoadPending=false/);
assert.match(app,/if\(!index\.isConnected\)return/);
assert.match(css,/\.studio-image-tag-list\.is-scrolling \.studio-image-tag:hover:not\(\.is-picked\):not\(:focus-visible\)/);
assert.match(css,/\.studio-tag-list\.is-scrolling \.studio-index-tag:hover:not\(\.is-picked\):not\(:focus-visible\)/);
assert.match(css,/scrollbar-gutter:stable/);
assert.match(css,/contain:paint/);
assert.match(css,/\.studio-file:is\(\.is-picked,\.active\) \{\s*animation:none;\s*filter:none;/);

// Stuck is a vivid, distinct red independent of the theme's muted danger token.
assert.match(css,/\[data-learning-status="stuck"\]\{--caption-learning-ring:#ff3344\}/);

// The loss status selector follows the Loss Status toggle rather than
// remaining visible (and potentially filtering invisibly) when switched off.
assert.match(app,/aria-label":"Loss Status"/);
assert.match(app,/el\("span",\{text:"Loss Status"\}\)/);
assert.doesNotMatch(app,/el\("span",\{text:"Learning colors"\}\)/);
assert.match(app,/learningFilter\.hidden=view==="constellation"\|\|!learningCheck\.checked/);
assert.match(app,/if\(!learningCheck\.checked\)S\.cap\.studioLearningFilter="all"/);
assert.match(app,/if\(!learningCheck\.checked&&wasFiltered\)\{/);
assert.match(app,/if\(\$\("capView"\)\.value==="studio"&&S\.cap\.studioLearningColors\)/);
assert.match(app,/filterLabel\.hidden=S\.cap\.studioExploreMode==="constellation"/);
assert.match(css,/#tab-captions \.studio-learning-filter-label\[hidden\]\{display:none!important\}/);

// Edit Tags must match TagBench's four dropdown items and shortcuts. Reorder
// opens a real dialog exposing previously hidden ordering rules.
assert.match(app,/menuAction\("Find and Replace…","Ctrl\+R",openFindReplace\)/);
assert.match(app,/menuAction\("Batch Reorder Tags…","Ctrl\+B",openBatchReorder\)/);
assert.match(app,/menuAction\("Remove Duplicate Tags","Ctrl\+D"/);
assert.match(app,/menuAction\("Remove Empty Tags","Ctrl\+E"/);
assert.match(app,/const openBatchReorder=\(\)=>/);
assert.match(app,/el\("label",\{class:"caption-order-field"\},el\("span",\{text:"Order"\}\),orderSelect\)/);
assert.match(app,/else if\(key==="b"\)\{handled\(\);openBatchReorder\(\);\}/);
assert.doesNotMatch(app,/History & advanced tag ordering/);
assert.match(css,/studio-edit-action kbd\{/);

// TagBench image menu keeps core copy/delete/review actions first, with the
// earlier Validation/Sample functionality grouped after them.
assert.match(app,/action\("Select All Images","Ctrl\+A"/);
assert.match(app,/action\("Invert Selection","Ctrl\+I"/);
assert.match(app,/action\("Copy Tags","Ctrl\+C"/);
assert.match(app,/action\("Paste Tags","Ctrl\+V"/);
assert.match(app,/action\("Delete "\+\(count===1\?"Image":"Images"\),"Ctrl\+Del"/);
assert.match(app,/action\(complete\?"Mark Incomplete":"Mark Complete","Ctrl\+K"/);
assert.match(app,/action\(inValidation\?"Remove "/);
assert.match(app,/action\(inSamples\?"Remove "/);
// Tag context menus are two distinct scopes with the same Boolean filter
// choices; global deletion explicitly confirms before writing.
assert.match(app,/const filterByTag=\(tag,mode\)=>/);
assert.match(app,/CaptionStudioTags\.addFilter\(S\.cap\.studioQuery,tag,mode\)/);
assert.match(app,/menuAction\("Add to Filter"/);
assert.match(app,/addAction\("Add to Filter"/);
assert.match(app,/menuAction\("Exclude from Filter"/);
assert.match(app,/addAction\("Exclude from Filter"/);
assert.match(app,/menuAction\("Delete Tag",removePickedTags,!editable\)/);
assert.match(app,/addAction\("Delete Tag",\(\)=>void deletePickedTags\(tag\)\)/);
assert.match(app,/Delete tag "'\+old\.trim\(\)\+'" from all eligible dataset captions/);
assert.match(app,/const category=el\("details",\{class:"studio-tag-color-category"\}\)/);
assert.match(css,/\.studio-tag-context-menu \.studio-tag-color-category>summary\{/);

// Selected-tag deletion is scoped to one active/selected image group. Image
// Tags can remove any number of selected tags, including the entire tag list.
assert.match(app,/menuAction\("Delete Tag",removePickedTags,!editable\)/);
assert.match(app,/const deletePickedTags=async\(contextTag=null\)=>/);
assert.match(app,/const chosen=imageTargets\(\);/);
assert.match(app,/Delete .*selected tag/);
assert.doesNotMatch(app,/Delete .*selected tag\(s\) from all unlocked dataset captions/);
assert.match(app,/if\(!imageTagPicked\.has\(tag\.toLowerCase\(\)\)\)imageTagPicked\.clear\(\)/);
assert.match(app,/if\(!selectionState\.has\(tag\)\)\{/);

// Double-click edits tag spelling inline. Empty edits and Escape restore the
// original row (including All Tags scroll position) without saving changes.
// The outer Image Tags Delete/Backspace shortcut must not eat keystrokes in
// the inline text editor.
assert.match(app,/button\.addEventListener\("dblclick",\(\)=>\{/);
assert.match(app,/if\(canceled\|\|!next\|\|next===t\)\{/);
assert.match(app,/row\.replaceChildren\(button\)/);
assert.match(app,/if\(e\.target\.closest\("input,textarea,\[contenteditable=true\]"\)\)return/);
assert.match(app,/const originalChildren=\[\.\.\.row\.childNodes\]/);
assert.match(app,/if\(canceled\|\|!next\|\|next===tag\)\{restore\(\);return;\}/);
assert.match(app,/if\(!ok\)restore\(\)/);
assert.match(app,/row\.isConnected\)row\.replaceChildren\(\.\.\.originalChildren\)/);
assert.match(app,/e\.key==="Escape"\)\{/);

// Auto-Captioner uses one reference-inspired card, retaining the original
// model, options, write/overwrite state and generation/progress controls.
assert.match(app,/const quickSettings=el\("div",\{class:"studio-ai-quick-settings"/);
assert.match(app,/overwrite\.checked=writeMode\.value==="replace"/);
assert.match(app,/overwrite\.dispatchEvent\(new Event\("change",\{bubbles:true\}\)\)/);
assert.match(app,/destination\.textContent=capTagger\(\)\|\|capDesc\(\)\?\.kind==="tags"\?"Tags":"Natural language"/);
assert.match(app,/target\.append\(el\("section",\{class:"studio-ai-caption-card"\},header,group\)\)/);
assert.match(app,/panel\?\.querySelector\("\.studio-ai-quick-settings"\)\?\.remove\(\)/);
assert.match(app,/text:count\+" selected picture"/);
assert.match(css,/#tab-captions \.studio-host \.studio-ai-caption-card\s*\{/);
assert.match(css,/#tab-captions \.studio-host \.studio-ai-quick-settings\s*\{/);

// The center image has a full-stage dark glass backdrop at wide aspect
// ratios, but no image distortion or expensive fullscreen backdrop blur.
assert.match(css,/#tab-captions \.studio-host \.studio-image-stage \.studio-preview\{/);
assert.match(css,/#tab-captions\.caption-theme-glass \.studio-host \.studio-image-stage \.studio-preview\{/);
assert.match(css,/\.studio-image-stage \.studio-preview::before\{/);
assert.match(css,/\.studio-image-stage \.studio-preview > :is\(img,video,\.song-media\)/);
assert.match(css,/object-fit:contain/);
// Remove the redundant visible Log card while retaining the hidden log sink
// still needed by the existing polling code.
const html=fs.readFileSync("src/fizgig/web/static/index.html","utf8");
assert.match(html,/<pre id="capLog" hidden aria-hidden="true" style="display:none"><\/pre>/);
assert.doesNotMatch(html,/<div class="card-title">Log<\/div>/);
assert.match(app,/const log = \$\("capLog"\)/);
