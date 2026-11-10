"""LaMa inpainting for the picture editor's Remove (web/image_edit.py).

LaMa ("Resolution-robust Large Mask Inpainting with Fourier Convolutions", Suvorov et al. 2021, advimman/lama,
Apache-2.0) rebuilds what was under a mask from the picture around it. Fizgig uses the big-lama checkpoint as a
TorchScript file (JosephCatrambone/big-lama-torchscript, Apache-2.0), so it loads with torch.jit.load and needs no
other package. The file (about 206 MB) is fetched on first use into user_data/models/lama/, pinned to one revision
and checked by size. Nothing else is downloaded and Fizgig never ships the file.

It runs on the GPU when the editor says it may (not while a run is training) and the card has room, else on the
CPU. The model stays loaded while Remove is being used (so Remove all watermarks loads it once) and is dropped,
with its GPU memory, IDLE_S seconds after the last use. Our own wrapper; no LaMa or IOPaint code is copied."""
import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

REPO = "JosephCatrambone/big-lama-torchscript"
REVISION = "7dfbdc6dc27602d370909e8688f8f84404a1f995"
FILENAME = "lama.pt"
SIZE = 206134115                # bytes, from the pinned revision
SHA256 = "79ed9fa18680ca19a269eb88fc4f708fb6580197b2220a51d3e000eb0ffdf33b"
LICENCE = "Apache-2.0"
IDLE_S = 90                     # drop the model this long after the last Remove
GPU_FREE_GB = 2.5               # use the GPU only with at least this much free on it

_lock = threading.RLock()
_model = None
_device = None
_timer = None


class LamaUnavailable(RuntimeError):
    """The LaMa file couldn't be downloaded or loaded: Remove can't run, and nothing was changed."""


def _folder():
    from fizgig.user_data import MODELS_DIR
    return os.path.join(MODELS_DIR, "lama")


def model_path(download=True):
    """The checkpoint's path, downloaded first when it isn't there yet (and `download`)."""
    path = os.environ.get("FIZGIG_LAMA_PATH") or os.path.join(_folder(), FILENAME)
    if os.path.isfile(path) and (os.environ.get("FIZGIG_LAMA_PATH") or os.path.getsize(path) == SIZE):
        return path
    if not download:
        raise LamaUnavailable("The LaMa model isn't downloaded yet.")
    logger.info("Remove: downloading the LaMa inpainting model (about 206 MB, once) into %s", _folder())
    from fizgig.utils import hf_public
    try:
        got = hf_public.download(repo_id=REPO, filename=FILENAME, revision=REVISION, local_dir=_folder())
    except Exception as e:
        raise LamaUnavailable("Couldn't download the LaMa model for Remove (" + hf_public.friendly_error(e) +
                              "). Check the internet connection and try again; the picture wasn't changed.") from e
    if os.path.getsize(got) != SIZE:
        raise LamaUnavailable("The LaMa model download is incomplete. Try Remove again; the picture wasn't changed.")
    return got


def _pick_device(gpu):
    import torch
    if gpu and torch.cuda.is_available():
        try:
            if torch.cuda.mem_get_info()[0] / 2 ** 30 >= GPU_FREE_GB:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def _drop():
    global _model, _device, _timer
    with _lock:
        was_cuda = _device == "cuda"
        _model, _device, _timer = None, None, None
    if was_cuda:
        import torch
        torch.cuda.empty_cache()


def _touch():
    global _timer
    if _timer is not None:
        _timer.cancel()
    _timer = threading.Timer(IDLE_S, _drop)
    _timer.daemon = True
    _timer.start()


def load(gpu=True):
    """The loaded model and its device; raises LamaUnavailable."""
    global _model, _device
    with _lock:
        if _model is not None:
            if _device == "cuda" and not gpu:     # a run started since: move off the card
                import torch
                _model, _device = _model.to("cpu"), "cpu"
                torch.cuda.empty_cache()
            return _model, _device
        want = _pick_device(gpu)
        path = model_path()
        import torch
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)      # torch.jit.load's "deprecated" note
                m = torch.jit.load(path, map_location=want).eval()
        except Exception as e:
            raise LamaUnavailable(f"Couldn't load the LaMa model for Remove ({type(e).__name__}: {e}). Delete "
                                  f"{path} to fetch it again; the picture wasn't changed.") from e
        _model, _device = m, want
        logger.info("Remove: LaMa loaded on the %s", "GPU" if want == "cuda" else "CPU")
        return _model, _device


def inpaint(rgb, hole, gpu=True):
    """rgb uint8 HxWx3 (H, W multiples of 8) and hole bool HxW -> rgb uint8 with the hole rebuilt by LaMa."""
    import torch
    with _lock:
        model, device = load(gpu)
        try:
            img = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1)[None].float() / 255.0
            mask = torch.from_numpy(np.asarray(hole, np.float32))[None, None].to(device)
            with torch.inference_mode():
                out = model(img, mask)
            res = (out[0].permute(1, 2, 0).clamp(0, 1) * 255.0).round().byte().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            img = mask = None
            _drop()
            logger.info("Remove: the GPU ran out of memory for LaMa; running it on the CPU")
            return inpaint(rgb, hole, gpu=False)
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()
            _touch()
        return res
