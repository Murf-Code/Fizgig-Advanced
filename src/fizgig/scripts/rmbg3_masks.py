"""Masks for masked training from RMBG-3.0, for a whole folder, from the command line (dataset/rmbgmask.py). The
masks card's RMBG-3.0 choice runs the same code; this is for a folder outside the app, other model paths, or a run
left going on its own.

    python src/fizgig/scripts/rmbg3_masks.py "D:/datasets/my_fox"                      # soft masks, FP8, Auto
    python src/fizgig/scripts/rmbg3_masks.py "D:/datasets/my_fox" --threshold 0.35 --grow 3 --feather 2
    python src/fizgig/scripts/rmbg3_masks.py "D:/datasets/my_fox" --refinish --threshold 0.5   # no model: new edges
    python src/fizgig/scripts/rmbg3_masks.py --download fp8                               # fetch the files, then quit

Weights: CC BY-NC 4.0 (Bria's non-commercial terms)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parser():
    from fizgig.dataset import rmbgmask as R
    p = argparse.ArgumentParser(description="RMBG-3.0 masks for masked training (photo.png -> photo-masklabel.png)")
    p.add_argument("folder", nargs="?", help="the training folder (its top level: one mask per picture)")
    p.add_argument("--variant", default=R.VARIANT_DEFAULT, choices=sorted(list(R.VARIANTS) + list(R.UNSUPPORTED)),
                   help="the RMBG-3.0 transformer: fp8 (7.6 GB, default), bf16 (15.1 GB) or diffusers (Bria's gated "
                        "repo); nvfp4 can't be read")
    p.add_argument("--dit", help="the RMBG-3.0 transformer: a single file or the diffusers repo / transformer dir "
                                 "(default: the models folder's)")
    p.add_argument("--te", help="Fibo 2.0's text encoder, qwen3vl_4b_bf16.safetensors (default: the models folder's)")
    p.add_argument("--vae", help="the FLUX.2 autoencoder, flux2-vae.safetensors (default: the models folder's)")
    p.add_argument("--models-dir", help="Fizgig's models folder (default: user_data/models)")
    p.add_argument("--kind", default="subject", choices=list(R.KINDS), help="what is white: the subject (default) "
                                                                           "or the background")
    p.add_argument("--threshold", type=float, default=None,
                   help="black and white at this matte value (0-1); left out, the greys stay (soft masks)")
    p.add_argument("--grow", type=int, default=0, help="widen the white by this many pixels (keeps fur edges in)")
    p.add_argument("--feather", type=float, default=0, help="soft edge added outside the white, in pixels")
    p.add_argument("--megapixels", type=float, default=R.MEGAPIXELS, help="render size (1.0 = Fibo 2.0's own; 0.5 "
                                                                         "is quicker and lighter)")
    p.add_argument("--steps", type=int, default=R.STEPS, help="denoising steps (the model is distilled to 4)")
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "int8", "nf4"],
                   help="the transformer on the card (Auto: bf16 when it fits, else INT8 with block swap)")
    p.add_argument("--blocks-to-swap", type=int, default=-1, help="blocks streamed from RAM (-1: Auto)")
    p.add_argument("--chunk", type=int, default=R.CHUNK, help="pictures per text-encoder pass (RAM: ~60 MB each)")
    p.add_argument("--overwrite", action="store_true", help="replace masks already there (off: they're kept)")
    p.add_argument("--no-mattes", action="store_true", help="don't keep the raw mattes in rmbg3_mattes/")
    p.add_argument("--refinish", action="store_true", help="remake the masks from the saved mattes with these edge "
                                                           "settings: no model, CPU only (replaces masks)")
    p.add_argument("--fallback", action="store_true", help="when RMBG-3.0 can't run here, use the character finder "
                                                           "(anime-seg) instead of stopping")
    p.add_argument("--download", metavar="VARIANT", help="fetch what VARIANT still needs, then quit")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    from fizgig.dataset import rmbgmask as R
    if a.models_dir:
        models_dir = a.models_dir
    else:
        from fizgig.user_data import MODELS_DIR as models_dir
    log = lambda s: print(s, end="", flush=True)    # noqa: E731
    if a.download:
        import time
        R.download(models_dir, a.download)
        while True:
            st = R.fetch_state(models_dir)
            if st["state"] != "running":
                break
            log(f"\rDownloading: {st['gb']:.2f} GB ...")
            time.sleep(2)
        if st["state"] == "failed":
            log(f"\nThe download failed: {st['error']}\n")
            return 1
        log("\nDone.\n")
        return 0
    if not a.folder or not os.path.isdir(a.folder):
        log("Give the training folder (it isn't a folder here).\n")
        return 2
    try:
        if a.refinish:
            c = R.refinish_folder(a.folder, kind=a.kind, threshold=a.threshold, grow=a.grow, feather=a.feather,
                                  log=log)
        else:
            c = R.make_folder(a.folder, models_dir, variant=a.variant, kind=a.kind, threshold=a.threshold,
                              grow=a.grow, feather=a.feather, overwrite=a.overwrite, keep_mattes=not a.no_mattes,
                              fallback=a.fallback, dit=a.dit, te=a.te, vae=a.vae, precision=a.precision,
                              blocks_to_swap=a.blocks_to_swap, megapixels=a.megapixels, steps=a.steps, chunk=a.chunk,
                              log=log)
    except ValueError as e:
        log(f"{e}\n")
        return 1
    log(f"\nDone: {c['made']} mask(s) written" + (f", {c['kept']} already there kept" if c["kept"] else "")
        + (f", {c['no_person']} with nothing to mask" if c["no_person"] else "")
        + (f", {c['skipped']} skipped" if c["skipped"] else "") + ".\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
