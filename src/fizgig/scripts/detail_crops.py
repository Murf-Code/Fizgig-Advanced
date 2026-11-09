"""The Detail crops stage (families/launch.py, before caching): each size the run trains at gets its crops
(dataset/detail_crops.py), and each crop with no caption gets one at its own size - a smaller crop shows only part of
what a bigger one does, so every size is captioned from its own crop. With Image Prep's "Copy the picture's caption to
its crops" on, a crop gets its picture's caption; the rest are captioned with the Captions tab's model (--recaption_with:
its worker config - Qwen3-VL, Florence-2 or a tagger such as JTP-3; none: Qwen3-VL with --captioner), Qwen3-VL with
the Captions tab's Style instruction, trigger word first.

    python -m fizgig.scripts.detail_crops --folder <training folder> --mp 1.0 [--mp 0.25] [--captioner <file>]
        [--recaption_with <json>] --instruction_file <file> [--trigger zwxem] [--ext .txt]"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fizgig.dataset import detail_crops  # noqa: E402


def captioner_config(recaption_with, captioner):
    """The caption worker config to load: the Captions tab's model, else Qwen3-VL from the captioner file; None when
    there is neither."""
    cfg = dict(recaption_with or {})
    if cfg.get("backend") in ("tagger", "florence"):
        return cfg
    path = cfg.get("text_encoder") or captioner or ""
    return {"backend": "qwen", "text_encoder": path} if path and os.path.isfile(path) else None


def model_name(cfg):
    if cfg["backend"] == "tagger":
        return cfg.get("model") or cfg.get("tagger") or "the tagger"
    if cfg["backend"] == "florence":
        return cfg.get("model") or cfg.get("florence_model") or "Florence-2"
    return "Qwen3-VL"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--mp", type=float, action="append", required=True)
    ap.add_argument("--captioner", default="")
    ap.add_argument("--recaption_with", default="")
    ap.add_argument("--instruction_file", default="")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    a = ap.parse_args()
    say = lambda t: print(t, flush=True)
    for mp in a.mp:
        detail_crops.make(a.folder, mp, log=say, ext=a.ext)
        detail_crops.copy_captions(a.folder, mp, log=say, ext=a.ext)    # Image Prep's Copy the caption, when on
    todo = detail_crops.to_caption(a.folder, a.ext, a.mp)
    if not todo:
        say("[detail crops] every crop has its caption")
        return 0
    cfg = captioner_config(json.loads(a.recaption_with) if a.recaption_with else None, a.captioner)
    if cfg is None:
        say(f"[detail crops] {len(todo)} crop(s) have no caption and no captioner is set: pick a model on the "
            "Captions tab, or set the Krea 2 text encoder (or a secondary captioning model) in Preferences")
        return 1
    import torch
    from fizgig.scripts import batch_caption as bc
    trig = a.trigger if a.trigger and a.trigger.lower() != "trigger_word" else ""
    say(f"[detail crops] captioning {len(todo)} crop(s) with {model_name(cfg)} (the Captions tab's model), each at "
        "its own size")
    state = bc._load_state_from_config(cfg, "cuda" if torch.cuda.is_available() else "cpu")
    try:
        caption_one = bc._caption_fn(state, {"tags": cfg.get("tags"), "florence_task": cfg.get("florence_task"),
                                             "instruction_file": a.instruction_file}, trig, a.max_new_tokens)
        for i, p in enumerate(todo, 1):
            try:
                cap = caption_one(p)
                if cap:
                    bc._write_caption(p, cap, trig, a.ext)
                say(f"[cache] captions {i}/{len(todo)}")       # the run card's count
            except Exception as e:
                say(f"[detail crops] {os.path.basename(p)}: no caption ({e})")
    finally:
        if state.backend == "tagger":
            state.model[1].close()
        del state.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
