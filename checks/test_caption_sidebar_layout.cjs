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
assert.match(app,/right\.append\(editorPane,tagHeader,tagHeightGrip,tagIndexPane,aiPane\)/);
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
assert.match(app,/fullStage\.replaceChildren\(network\)/);
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

// The image total is shown once (in current/total progress), and the narrow
// thumbnail footer stacks its controls rather than letting them collide.
assert.match(app,/el\("strong",\{text:"Images"\}\)\)\)/);
assert.doesNotMatch(app,/el\("span",\{class:"field-hint",text:String\(items\.length\)\}\)/);
assert.match(app,/missingBtn\.dataset\.compact=/);
assert.match(app,/selectionCount\.hidden=selection\.size<=1/);
assert.match(css,/#tab-captions \.studio-host \.studio-thumb-footer\s*\{\s*display:grid;/);
assert.match(css,/@container \(max-width:170px\)/);
assert.match(css,/\.studio-missing-filter::after\s*\{\s*content:attr\(data-compact\)/);

// Saved is represented by a green light; pending writes, errors and required
// image actions still keep explanatory text. Status details remain accessible.
assert.match(app,/liveLabel\.textContent=label/);
assert.match(app,/liveSummary\.setAttribute\("aria-label","Dataset status: "\+label/);
assert.match(css,/\.studio-workspace-status\[data-state="saved"\] \.studio-status-label\s*\{/);
assert.match(css,/\.studio-workspace-status\[data-state="saved"\] summary::after\s*\{\s*display:none;/);
assert.match(css,/\.studio-workspace-status\[data-state="saved"\] \.studio-status-dot\s*\{/);
assert.match(css,/\.studio-workspace-status\[data-state="attention"\] \.studio-status-dot\s*\{/);

// The stage expands to fill the available notebook viewport, rather than
// being capped at 82dvh/950px. Short windows still scroll normally.
assert.match(css,/\.notebook-pane:has\(> #tab-captions\.active\)/);
assert.match(css,/#tab-captions\.active #captionsGrid\.studio-host > \.studio-layout\s*\{\s*flex:1 1 auto;/);
assert.match(css,/#tab-captions\.active #captionsGrid\.studio-host > \.studio-layout > :is\(\.studio-browser,\.studio-editor,\.studio-index\)/);
assert.match(css,/#tab-captions \.studio-image-stage \.studio-preview\{[^}]*flex:1 1 0;[^}]*height:100%;/);
assert.match(css,/#tab-captions \.studio-image-stage \.studio-preview img,[\s\S]*?max-height:100%;[\s\S]*?object-fit:contain/);
assert.doesNotMatch(css,/\.studio-image-stage \.studio-preview img[^\n]*max-height:min\(68vh,750px\)/);

// Keep a single production image renderer. The native Qt experiment was
// removed after GPU-enabled WebView2 passed every image-heavy Windows test.
assert.doesNotMatch(app,/__FizgigQtNativeRail|nativeVisibleItems|nativeImageActions|nativeDatasetGeneration/);
assert.match(app,/left\.append\(thumbs,nav\)/);
for (const retired of [
  "src/fizgig/web/qt_caption_trial.py",
  "src/fizgig/web/qt_images.py",
  "src/fizgig/web/static/qt_native_image_bridge.js",
  "Launch Fizgig Native Images.bat"
]) assert.equal(fs.existsSync(retired),false,`retired Qt renderer reintroduced: ${retired}`);

// Native Chromium wheel scrolling must not be blocked by JavaScript.
assert.doesNotMatch(app,/thumbs\.addEventListener\("wheel"/);
assert.match(app,/CaptionThumbRail\.reconcile\(windowRows,mountedThumbs/);
assert.match(css,/#tab-captions \.studio-host \.studio-thumb-rail,[\s\S]*?scroll-behavior:auto;/);
assert.doesNotMatch(css,/#tab-captions \.studio-host \.studio-thumb-rail,[\s\S]*?scroll-behavior:smooth;/);

// Tags has a full-range virtualized scrollbar; only images in the visible
// window are mounted. Cards/Table alone retain CAP_PAGE.
assert.match(app,/function studioThumbGeometry\(items,railWidth,gap=14,previewHeight=360\)/);
assert.match(app,/function studioThumbIndex\(offsets,position\)/);
assert.match(app,/const paintWindow=\(\)=>/);
assert.doesNotMatch(app,/windowRows\.replaceChildren\(fragment\)/);
assert.match(app,/const mountedThumbs=new Map\(\)/);
assert.match(app,/thumbPreloader\.prefetch\(it\.path,thumbUrl\(it\)\)/);
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

// Tag entry uses one field with Enter, click, or keyboard-navigable suggestions.
assert.match(app,/const addTag=value=>/);
assert.match(app,/newTag\.addEventListener\("keydown"/);
assert.match(app,/if\(e\.key==="Enter"&&!e\.isComposing\)/);
assert.match(app,/fragment\.append\(option\)/);
assert.match(app,/option\.addEventListener\("click"/);
assert.doesNotMatch(app,/newTag,btn\("Add"/);
assert.doesNotMatch(app,/newTag\.parentElement\?\.querySelector\("button"\)/);

// No old category settings panel. Colors are edited from tag context menus,
// shared by Image Tags, All Tags and the cached Constellation palette.
assert.doesNotMatch(app,/const catPanel=el\("details"/);
assert.doesNotMatch(app,/const advancedTags=el\("details"/);
assert.match(app,/studioTagColorMenu\(menu,tag,categoryState,renderCaptions\)/);
assert.match(app,/tagColors:raw\.tagColors/);
assert.match(app,/categoryState\.generatedColors=palette\|\|new Map\(\)/);
assert.match(app,/CaptionStudioViz\.setTagbenchPalette\(palette\)/);
assert.doesNotMatch(app,/const model=CaptionStudioViz\.tagbenchModel\(graphTags,categoryState\)/);
assert.match(app,/work=captionGraphModelAsync\(allTags,categoryState\)/);
assert.match(css,/studio-tag-color-swatches/);

// Natural language replaces the lower All Tags list using the one autosaved
// textarea; clicking All Tags restores the list.
assert.match(app,/tagIndexPane\.replaceChildren\(/);
assert.match(app,/naturalModes\.set\(current\.path,false\);S\.cap\.studioDock="all";renderCaptions\(\)/);
assert.match(app,/const naturalEditor=el\("section",\{class:"studio-natural-pane"\},captionText,captionHelp\)/);
assert.match(css,/#tab-captions \.studio-host \.studio-tag-index-pane>\.studio-natural-pane/);
assert.match(css,/#tab-captions\.caption-theme-glass \.studio-host \.studio-natural-toggle:has\(input:checked\)/);

console.log("TagBench stacked sidebar and unchanged Tags/Waterfall/Constellation selector checks passed");

// Fast tag selection, single-pass filtering, aligned caption header, TagBench help.
assert.match(app,/const imageTagButtons=new Map\(\)/);
assert.match(app,/const syncImageTagSelection=before=>/);
assert.doesNotMatch(app,/for\(const el of tags\.querySelectorAll\("button"\)\)/);
assert.match(app,/const tagBatchSize=80/);
assert.match(app,/indexSearch\.addEventListener\("input",scheduleTagRows\)/);
assert.match(app,/if\(index\.parentElement!==explore\)explore\.replaceChildren\(index\)/);
assert.match(app,/const tagHeader=el\("div",\{class:"studio-tag-header"\},headerTitle,rightDock\)/);
assert.match(app,/Freeform caption\. Saved under the tags, after the first line break\./);
assert.match(app,/const captionHelp=el\("p",\{class:"studio-natural-help"/);
assert.match(css,/#tab-captions \.studio-host \.studio-tag-header\s*\{/);

assert.match(app,/CaptionStudioLearning\.filterByVerdict\(items,S\.cap\.learningMap,/);
assert.match(app,/const createLearningFilterSelect=\(\)=>/);
assert.match(app,/actionBar\.append\(learningToggle,learningFilter,viewButtons,liveStatus\)/);
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
// The old window.confirm prompt was replaced by the scoped, keyboard-accessible
// removal dialog. Check the actual confirmation branch, not deleted wording.
assert.match(app,/if\(mode==="remove"&&!\(await captionConfirmRemoval\(/);
assert.match(app,/scope==="all"\?" across the dataset\?"/);
assert.match(app,/items:\[old\.trim\(\)\],confirmLabel:"Remove tag"/);
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

// CPU regression: neither searchable tags nor Constellation should be
// repeatedly rebuilt on every keystroke or ordinary image selection.
assert.match(app,/tagIndex\.searchRows\.map|tagIndex\.sorted\.map\(\(\[tag,count\]\)=>\[tag,count,tag\.toLocaleLowerCase\(\)\]\)/);
assert.match(app,/CaptionStudioTags\.findTagMatches\(/);
assert.match(app,/newTag\.addEventListener\("input",scheduleSuggestions\)/);
assert.match(app,/requestAnimationFrame\(\(\)=>\{\s*tagQueryFrame=0;/);
assert.match(app,/S\.cap\.studioGraphPending\?\.cancel\?\.\(\);/);
assert.match(app,/const palette=S\.cap\.studioTagbenchPalette\?\.key===colorScope/);


// Thumbnail and center-image context menus reuse one keyboard-accessible
// menu. The center preview targets exactly one image, never a bulk selection.
assert.match(app,/oncontextmenu:e=>openImageMenu\(e,it\)/);
assert.match(app,/const openImageMenu=\(e,it,\{preview=false\}=\{\}\)=>/);
assert.match(app,/const targets=preview\?\[it\]:imageTargets\(it.path\)/);
assert.match(app,/openImageMenu\(e,current,\{preview:true\}\)/);
assert.match(app,/if\(S\.cap\.imageEditing\?\.inline\|\|!e\.target\.closest\("img"\)\)return/);
assert.match(app,/action\("Open File Location","",\(\)=>revealImageFile\(it\)\)/);
assert.match(app,/action\("Copy Caption","",\(\)=>copyImageText\(ta\.value,"caption"\)/);
assert.match(app,/action\("Remove Tool","",\(\)=>selectInlineTool\("remove"\),!canEditPicture\)/);
assert.match(app,/action\("Crop Tool","",\(\)=>selectInlineTool\("crop"\),!canEditPicture\)/);
const server=fs.readFileSync("src/fizgig/web/server.py","utf8");
assert.match(server,/if path == "\/api\/captions\/reveal_file":/);
assert.match(server,/if on_pod\(\) or not self\._local_viewer\(\):/);
assert.match(server,/from fizgig\.web\.file_location import reveal_file/);
