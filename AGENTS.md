# AGENTS.md — Fizgig Advanced

These are standing project instructions for coding agents and contributors. Preserve experiment-specific protocols and findings in their research documents.

## 1. Development philosophy

- Prioritize training quality, numerical stability, performance and usability.
- Prefer intelligent automation over additional user settings.
- Remove unnecessary controls rather than introducing more complexity.
- Preserve proven functionality when introducing experimental features.
- Never assume lower training loss alone means better learning.
- Challenge assumptions using measured evidence.

## 2. UI and UX standards

Fizgig Advanced should feel like a polished, cohesive desktop application rather than a collection of technical utilities.

- Favor clean, compact, Apple-inspired interface design.
- Maintain consistent button heights, spacing, typography and visual hierarchy.
- All colors, borders, highlights and interactive states must respect the user's selected theme.
- Avoid unnecessary labels, duplicated controls and excessive explanatory text.
- Automatically populate settings when their values are already known.
- Prefer intuitive interactions over requiring users to read instructions.
- Maintain responsive performance with datasets exceeding 15,000 images.

### Dialogs and popups

All dialogs must follow the established custom Fizgig dialog design. Reference: the Selective optimizer comparison dialog in `src/fizgig/web/static/selective_optimizer_smoke.js` at commit `6dbe6532195b8819ac410f5250642fb0298db778`.

- Never introduce native browser `alert()`, `prompt()` or `confirm()` dialogs.
- Use theme-aware modal windows with consistent styling, rounded corners and button heights.
- Use checkboxes, selectors and appropriate controls instead of comma-separated text input.
- Display relevant defaults automatically.
- Show validation errors inside the dialog.
- Enter confirms appropriate actions; Escape cancels. Preserve multiline editing and avoid accidental destructive submissions.
- Destructive operations require two-stage confirmation.
- Reuse shared dialog components wherever possible.

### Keyboard and interaction behavior

- Support expected keyboard shortcuts.
- Right-click menus should provide contextually relevant actions.
- Avoid hover-triggered selection.
- Undo and redo must restore both the visible interface and underlying saved data.
- Keep selection, thumbnails and image previews synchronized.

## 3. Dataset and caption workbench

- Use TagBench-inspired workflows where they improve usability.
- Automatically save caption changes.
- Preserve tag colors across relevant views.
- Support AND, OR and NOT tag filtering.
- Display token counts rather than word counts, highlighting counts above 75.
- Preserve filter and selection context when switching views.
- Recache only images or captions that have changed, except when cache validity requires regeneration.
- Never silently delete or overwrite dataset content.

## 4. Training architecture

- Consider 12 GB VRAM hardware an important development target.
- Prefer mathematically sound memory reductions over arbitrary feature restrictions.
- Preserve optimizer-specific mathematics and internal state.
- Respect native optimizer learning-rate management.
- Adaptive LR should start from the user's configured learning rate.
- Avoid unnecessary manual Adaptive LR minimum and maximum controls.
- Maintain mask compatibility across training methods wherever technically supported.
- Resume operations must preserve training progress, optimizer state and valid caches.

## 5. Research and experimental features

- Isolate experimental features in dedicated research or test branches.
- Keep production defaults unchanged until experiments are validated.
- Compare optimizers using effective learning strength rather than identical numerical learning rates.
- Use matched initialization, datasets, seeds, validation images and sample prompts.
- Distinguish measured findings from theoretical estimates.
- Evaluate visual learning, validation loss, training speed, memory consumption and numerical stability.
- Preserve complete experiment reports, including failed and terminated runs.
- Automate test configuration to minimize user error.
- Provide safe ways to terminate individual queued experiments.
- Do not claim GPU validation based solely on CPU tests.

### Selective Global Replay

- Preserve the global-loss gradient propagation through suspended blocks.
- Suspended adapter weights and local optimizer state must remain unchanged. Distinguish local history from deliberately shared native optimizer controllers.
- Verify correct block reactivation.
- Respect each optimizer's native learning-rate and update mechanisms.
- Validate compensation independently from basic Selective compatibility.
- Reuse established optimizer-family implementations rather than duplicating logic.

## 6. Independent peer review — Astra and other AI models

Treat Astra as an independent technical research reviewer, particularly for complex mathematical, architectural and experimental decisions.

### When to consult Astra

- Developing new training methods or optimizer architectures.
- Modifying matrix mathematics, numerical precision or gradient propagation.
- Investigating unexpected experimental results.
- Designing GPU memory optimizations.
- Reviewing substantial architectural changes before implementation.
- Investigating persistent bugs when conventional solutions have failed.
- Reviewing experimental features before integration into WIP.

Routine bug fixes and small UI changes do not require Astra review. If Astra is not directly accessible, prepare a self-contained review request for the user to submit; do not claim a review occurred before receiving actual feedback.

### Review methodology

When requesting an independent review, provide:

- Repository URL, branch and exact commit hash.
- Direct links to relevant source files.
- Research objectives and hypotheses.
- Previous experiments, measured results and known limitations.
- Hardware requirements and constraints.
- Specific questions and unresolved concerns.

Ask Astra to independently challenge the implementation rather than simply validate the proposed approach.

### Handling peer-review findings

- Critically evaluate Astra's recommendations rather than automatically accepting them.
- Verify proposed mathematical corrections against the actual implementation.
- Distinguish verified defects, theoretical concerns and speculative improvements.
- Use Grok or other independent models for additional perspectives when appropriate.
- Investigate disagreements between reviewers using mathematics, source inspection and controlled experiments.
- Preserve significant peer-review findings in the repository's research documentation.
- Never substitute AI consensus for empirical validation.

### Division of responsibilities

- **Primary coding agent:** Implementation, repository management, debugging, testing and research continuity.
- **Astra:** Independent mathematical reasoning, architectural evaluation, experimental design and technical peer review.
- **Additional reviewers:** Alternative perspectives, independent verification and challenging assumptions.
- **User:** Real-world GPU testing, visual learning assessment, usability feedback and final project direction.

For particularly complex research, follow an iterative cycle of implementation, independent review, correction, controlled experimentation and evaluation before production integration.

## 7. Git and branch management

- `fizgig-advanced` is the production branch.
- Use `WIP` for integration and development.
- Use dedicated branches for substantial experimental changes.
- Verify the current branch before editing or pushing.
- Do not overwrite unrelated changes.
- Check whether upstream improvements already exist locally before merging them.
- Retain the better implementation when upstream and Advanced offer competing solutions.
- Give appropriate credit to upstream developers and contributors.
- Do not merge unfinished research into production without approval.

## 8. Testing and reporting

- Test changes before declaring them complete whenever the required environment is available.
- Add regression tests for discovered bugs.
- Report exactly what was tested and what remains unverified.
- Disclose known limitations, assumptions and unresolved issues.
- Never claim a test passed if it was not executed.
- Provide the commit hash and branch after pushing changes.

## 9. Project continuity

- Consult existing research reports before repeating experiments.
- Preserve previously validated findings and user-approved design decisions.
- Do not reopen settled decisions without new evidence.
- Update project documentation when significant architectural or design decisions are established.
- Keep this document synchronized across active branches. Record changing numerical baselines, benchmark results and experiment-specific settings in research documentation rather than here.

The goal is to make Fizgig Advanced increasingly intelligent, reliable, efficient and intuitive without burdening the user with unnecessary complexity.
