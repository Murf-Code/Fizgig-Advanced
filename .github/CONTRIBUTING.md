# Contributing to Fizgig

Pull requests are welcome: bug fixes, new models, features, docs, pod and RunPod improvements.

## Start with a conversation

- **New model:** open a post in [New models & fine-tuning](https://github.com/Murf-Code/Fizgig-Advanced/discussions/new?category=new-models-fine-tuning) before writing code: which model, where its weights and reference code live, and what you plan to support. Then follow the [driver guide](../docs/drivers/README.md).
- **New feature, or anything more than a small fix:** open an [issue](https://github.com/Murf-Code/Fizgig-Advanced/issues) first, so we can agree the scope and where it fits before you put the hours in. The last thing I want is to turn down work you've already spent days on.
- **Small fixes and docs:** go straight to a pull request.

## What Fizgig is for

Fizgig is a trainer and LoRA workbench: dataset essentials (captions, resizing, face crops), training, previews, and the tools for testing and fixing a LoRA afterwards. Deeper dataset curation and general image tools are better served by dedicated apps, so features in that direction are likely to be turned down, however well they're made.

## What helps a pull request land

- **New behaviour is opt-in.** What people get today stays the default: models, memory and precision plans, presets, the face and likeness tools. A new option starts switched off, or off by default, unless we've agreed otherwise in the issue.
- **One install.** No second Python environment, worker process or change to Fizgig's pinned dependencies. If a model needs code from a newer library, port the parts it uses into its own package (the driver guide explains how).
- **Model pull requests add the model.** Training features any model could use (batch size, caption dropout, timestep options and so on) are their own issue and pull request, so they're designed once for every model. These are welcome as their own proposal, not as part of a model.
- **Plain-English UI.** Say what a control is for first, in words a photographer would use, and keep technical detail for the docs. Names say what the thing does.
- **One concern per pull request** where possible.
- **Say what you tested and on what hardware.** GPU and VRAM matter a lot here: real runs, peaks, and before/after numbers where something changed.
- **Tests are welcome in [`checks/`](../checks/README.md).** Checks there run on every pull request, so they need to be CPU only, self-contained (no model downloads or personal paths) and quick. Anything that needs a GPU or real model files to prove stays on your side: describe what you measured in the pull request.
- **Match the surrounding code's style** and use the shared helpers rather than one-off copies.

## What you can expect

- **Your authorship is preserved.** Pull requests are merged with a real GitHub merge: your name on the commits, the Merged badge and contribution-graph credit. Accepted work is credited by @handle in the release notes, and substantial contributions go in [CONTRIBUTORS.md](../docs/CONTRIBUTORS.md).
- **A real review.** Pull requests here get read properly, including testing on real cards where it's warranted. Feedback comes as review comments, or as fixup commits on your branch that keep your work as yours, never as a silent rewrite.
- **Fast turnaround.** Fizgig ships often, and good pull requests usually land in the next release.

## Bugs

Open an issue with the console output and your setup (GPU, OS, the app or a RunPod pod). Detailed reports here have often gone from filed to fixed and released in a day.
