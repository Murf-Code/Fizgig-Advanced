"""Experimental cross-checkpoint block mixer.

Offline, CPU-only, bounded-RAM checkpoint mixing for compatible *same-family*
MOSS-Image, Anima, Krea 2, and SDXL/Illustrious base-model safetensors files.
It never loads the model or changes the training/cache path. H3 uses its
separate live ConvRot/AdaLN-compatible Frankenstein implementation.

The base file's tensor layout, metadata, and every unselected tensor are
preserved; floating-point block weights are interpolated in FP32 in chunks.
Different architectures, missing block tensors, incompatible shapes, and
quantized model weights are refused instead of silently producing a broken
checkpoint. A new file is written atomically without editing either input.

Inspired by shootthesound/Fizgig 9.8.3 Frankenstein (DiTBlend), adapted for
the four distinct checkpoint architectures in Fizgig Advanced.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import tempfile
from collections import defaultdict
from contextlib import contextmanager

from fizgig.utils.st_meta import _header

FAMILIES = {
    "moss_image": "MOSS-Image",
    "anima": "Anima",
    "krea2": "Krea 2",
    "sdxl": "SDXL / Illustrious",
}
FLOAT_DTYPES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
SIZES = {**FLOAT_DTYPES, "I8": 1, "U8": 1, "BOOL": 1, "I16": 2,
         "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E4M3FN": 1, "F8_E5M2": 1}
CHUNK_BYTES = 8 << 20
COPY_BYTES = 8 << 20


class MixCancelled(RuntimeError):
    """The user cancelled; incomplete temporary output was discarded."""


def block_of(family: str, key: str) -> str | None:
    """Name a complete model block, not just an attention linear.

    Preserve every weight in that block together. SDXL uses different block
    modules in the LDM and Diffusers checkpoint layouts; never mix the two
    layouts unless the exact keys and shapes match.
    """
    # Only top-level diffusion-transformer blocks. In Anima, for example,
    # llm_adapter.blocks.N is a separate architecture and must not be grafted.
    # Official Anima checkpoints also use net.blocks.N or model.diffusion_model.blocks.N.
    root = r"^(?:(?:model\.diffusion_model|diffusion_model|model|net|transformer)\.)?"
    if family == "moss_image":
        match = re.match(root + r"layers\.(\d+)\.", key)
        return "layer_" + match.group(1) if match else None
    if family in {"anima", "krea2"}:
        match = re.match(root + r"blocks\.(\d+)\.", key)
        return "block_" + match.group(1) if match else None
    if family == "sdxl":
        # LDM format used by Illustrious and most SDXL single-file checkpoints.
        match = re.search(r"(?:^|\.)(input_blocks|output_blocks)\.(\d+)\.", key)
        if match:
            return match.group(1) + "_" + match.group(2)
        if re.search(r"(?:^|\.)middle_block\.", key):
            return "middle_block"
        # Diffusers split UNet files are also possible.
        match = re.search(r"(?:^|\.)(down_blocks|up_blocks)\.(\d+)\.", key)
        if match:
            return match.group(1) + "_" + match.group(2)
        if re.search(r"(?:^|\.)mid_block\.", key):
            return "mid_block"
    return None


@contextmanager
def _file_header(path):
    with open(path, "rb") as f:
        n, h = _header(f)
        if not isinstance(h, dict):
            raise ValueError("Invalid safetensors header (expected an object).")
        size = os.fstat(f.fileno()).st_size
        data_size = size - 8 - n
        tensors = {}
        offset_pairs = []
        for key, info in h.items():
            if key == "__metadata__":
                continue
            if not isinstance(info, dict):
                raise ValueError("Invalid tensor header: " + key)
            dtype, shape = info.get("dtype"), info.get("shape")
            offs = info.get("data_offsets")
            if dtype not in SIZES or not isinstance(shape, list) or not isinstance(offs, list) or len(offs) != 2:
                raise ValueError("Unsupported or damaged tensor header: " + key)
            if any(not isinstance(d, int) or d < 0 for d in shape):
                raise ValueError("Invalid tensor dimensions: " + key)
            start, end = offs
            expect = SIZES[dtype] * math.prod(shape)
            if not all(isinstance(v, int) for v in offs) or start < 0 or end < start or end > data_size or end - start != expect:
                raise ValueError("Invalid tensor bounds: " + key)
            tensors[key] = info
            offset_pairs.append((start, end, key))
        offset_pairs.sort()
        if not tensors:
            raise ValueError("No tensors in checkpoint.")
        prev = 0
        for start, end, key in offset_pairs:
            if start < prev:
                raise ValueError("Overlapping tensor offsets: " + key)
            prev = end
        yield f, n, h, tensors, offset_pairs, data_size


def _groups(family, tensors):
    groups = defaultdict(set)
    for key in tensors:
        name = block_of(family, key)
        if name is not None:
            groups[name].add(key)
    return groups


def _inspect(base, donor, family):
    if family not in FAMILIES:
        raise ValueError("Choose a supported model family: " + ", ".join(FAMILIES))
    if not os.path.isfile(base) or not os.path.isfile(donor):
        raise ValueError("Both base model checkpoint files must exist.")
    if os.path.samefile(base, donor):
        raise ValueError("Choose two different checkpoints of the same model family.")
    if not str(base).lower().endswith(".safetensors") or not str(donor).lower().endswith(".safetensors"):
        raise ValueError("Only .safetensors checkpoints can be mixed.")
    with _file_header(base) as (_a, _na, ah, at, _order, _bytes), _file_header(donor) as (_b, _nb, bh, bt, _bo, _bytes_b):
        ag, bg = _groups(family, at), _groups(family, bt)
        if len(ag) < 2 or len(bg) < 2:
            raise ValueError("The checkpoints do not expose the expected " + FAMILIES[family] + " blocks.")
        if set(ag) != set(bg):
            missing = sorted(set(ag) ^ set(bg))
            raise ValueError("Different block layouts; cannot merge these checkpoints (" + ", ".join(missing[:5]) + ").")
        blocks = []
        for group in sorted(ag, key=lambda x: (re.sub(r"[_0-9]+$", "", x), int(re.search(r"\d+$", x).group()) if re.search(r"\d+$", x) else -1)):
            if ag[group] != bg[group]:
                raise ValueError("Block " + group + " has different tensor names; models aren't interchangeable.")
            size = 0
            for key in ag[group]:
                a, b = at[key], bt[key]
                if a["shape"] != b["shape"] or a["dtype"] != b["dtype"]:
                    raise ValueError("Incompatible tensor shape or dtype in " + group + ": " + key)
                if a["dtype"] not in FLOAT_DTYPES and a["dtype"] in {"I8", "F8_E4M3", "F8_E4M3FN", "F8_E5M2"}:
                    raise ValueError("Quantized block weights need a model-specific blend adapter (" + key + "). "
                                     "Use matching BF16/FP16/FP32 checkpoints instead.")
                size += a["data_offsets"][1] - a["data_offsets"][0]
            blocks.append({"id": group, "tensors": len(ag[group]), "bytes": size})
        return {
            "family": family, "name": FAMILIES[family], "blocks": blocks,
            "base": os.path.abspath(base), "donor": os.path.abspath(donor),
            "base_bytes": os.path.getsize(base), "donor_bytes": os.path.getsize(donor),
            "mixed_tensor_count": sum(b["tensors"] for b in blocks),
            "outside_blocks": len(set(at) - set().union(*ag.values())),
            "compatible": True,
            "note": "Only full, shape-compatible model blocks are blended. Other weights remain from Base.",
        }


def inspect_pair(base: str, donor: str, family: str) -> dict:
    """Quick preflight using only bounded safetensors headers; reads no tensor payloads."""
    return _inspect(os.fspath(base), os.fspath(donor), family)


def _copy_bytes(src, dst, n, cancel=None, progress=None):
    while n:
        if cancel is not None and cancel.is_set():
            raise MixCancelled("Model merge cancelled.")
        buf = src.read(min(n, COPY_BYTES))
        if not buf:
            raise ValueError("Checkpoint was truncated while mixing.")
        dst.write(buf)
        n -= len(buf)
        if progress:
            progress(len(buf))


def _blend_bytes(a, b, out, n, dtype, weight, cancel=None, progress=None):
    """Bound FP32 mixing to ~8 MB per input block; never stage an entire layer."""
    import torch
    typ = {"F16": torch.float16, "BF16": torch.bfloat16,
           "F32": torch.float32, "F64": torch.float64}[dtype]
    elem = FLOAT_DTYPES[dtype]
    if not 0.0 < weight < 1.0:
        src = b if weight == 1.0 else a
        return _copy_bytes(src, out, n, cancel, progress)
    chunk = max(elem, (CHUNK_BYTES // elem) * elem)
    while n:
        if cancel is not None and cancel.is_set():
            raise MixCancelled("Model merge cancelled.")
        grab = min(n, chunk)
        x, y = bytearray(a.read(grab)), bytearray(b.read(grab))
        if len(x) != grab or len(y) != grab:
            raise ValueError("Checkpoint changed or was truncated during mixing.")
        ta = torch.frombuffer(x, dtype=typ).float()
        tb = torch.frombuffer(y, dtype=typ).float()
        # one bounded FP32 result, not an entire model in RAM, with exact endpoints
        mixed = torch.lerp(ta, tb, weight).to(typ).contiguous()
        out.write(mixed.view(torch.uint8).numpy().tobytes())
        del x, y, ta, tb, mixed
        n -= grab
        if progress:
            progress(grab)


def merge_pair(base: str, donor: str, output: str, family: str, weights: dict,
               *, overwrite=False, cancel=None, progress=None) -> dict:
    """Export an ordinary standalone checkpoint; the UI can run this on a CPU worker.

    A's untouched tensor bytes are copied without torch decoding. For selected
    blocks A/B are blended at FP32 in bounded chunks; no GPU allocation or model
    inference. The output is atomic and neither original is ever edited.
    """
    plan = inspect_pair(base, donor, family)
    if not isinstance(weights, dict):
        raise ValueError("Provide a per-block weight map.")
    allowed = {item["id"] for item in plan["blocks"]}
    unknown = set(weights) - allowed
    if unknown:
        raise ValueError("Unknown block(s): " + ", ".join(sorted(unknown)[:5]))
    w = {}
    for group in allowed:
        value = float(weights.get(group, 0.0))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Blend strengths must be between 0 and 1.")
        w[group] = value
    if not any(v > 0.0 for v in w.values()):
        raise ValueError("Nothing to mix: choose at least one block with a strength above zero.")
    output = os.path.abspath(os.fspath(output))
    if not output.lower().endswith(".safetensors"):
        raise ValueError("Output must end in .safetensors.")
    if os.path.abspath(base) == output or os.path.abspath(donor) == output:
        raise ValueError("Output cannot replace either input checkpoint.")
    if os.path.exists(output) and not overwrite:
        raise FileExistsError("Output exists. Choose another filename or confirm overwrite.")
    folder = os.path.dirname(output)
    if not os.path.isdir(folder):
        raise ValueError("The output folder does not exist.")
    temp = None
    done = 0
    try:
        with _file_header(base) as (afile, na, ah, at, order, data_size), \
             _file_header(donor) as (bfile, nb, bh, bt, _order_b, _data_size_b):
            # Revalidate after preflight, especially if a user replaced an input file.
            for item in plan["blocks"]:
                group = item["id"]
                akeys = {k for k in at if block_of(family, k) == group}
                bkeys = {k for k in bt if block_of(family, k) == group}
                if akeys != bkeys or any(at[k]["shape"] != bt[k]["shape"] or at[k]["dtype"] != bt[k]["dtype"] for k in akeys):
                    raise ValueError("One checkpoint changed during preflight: " + group)
            metadata = dict(ah.get("__metadata__") or {})
            metadata["fizgig_checkpoint_mix"] = json.dumps({
                "family": family, "base": os.path.basename(base),
                "donor": os.path.basename(donor),
                "blocks": {k: round(v, 6) for k, v in w.items() if v > 0},
                "engine": "Fizgig Advanced CPU streamed block mixer v1"
            }, separators=(",", ":"))
            header = {"__metadata__": metadata, **{k: v for k, v in ah.items() if k != "__metadata__"}}
            raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
            raw += b" " * (-len(raw) % 8)
            if len(raw) > 100 << 20:
                raise ValueError("Output safetensors header is too large.")
            fd, temp = tempfile.mkstemp(prefix=".fizgig-mix-", suffix=".tmp", dir=folder)
            with os.fdopen(fd, "wb") as out:
                out.write(struct.pack("<Q", len(raw)))
                out.write(raw)
                pos = 0
                def advanced(amount):
                    nonlocal done
                    done += amount
                    if progress is not None:
                        progress(done, data_size)
                for start, end, key in order:
                    if start > pos:
                        afile.seek(8 + na + pos)
                        _copy_bytes(afile, out, start - pos, cancel, advanced)
                    n = end - start
                    afile.seek(8 + na + start)
                    group = block_of(family, key)
                    strength = w.get(group, 0.0)
                    if strength and at[key]["dtype"] in FLOAT_DTYPES:
                        bfile.seek(8 + nb + bt[key]["data_offsets"][0])
                        _blend_bytes(afile, bfile, out, n, at[key]["dtype"], strength, cancel, advanced)
                    else:
                        _copy_bytes(afile, out, n, cancel, advanced)
                    pos = end
                if pos < data_size:
                    afile.seek(8 + na + pos)
                    _copy_bytes(afile, out, data_size - pos, cancel, advanced)
                out.flush()
                os.fsync(out.fileno())
            if cancel is not None and cancel.is_set():
                raise MixCancelled("Model merge cancelled.")
            # Both the final write and any overwrite happen only after all tensors succeeded.
            if os.path.exists(output) and not overwrite:
                raise FileExistsError("Output appeared during export; nothing was overwritten.")
            os.replace(temp, output)
            temp = None
        return {"path": output, "family": family, "bytes": os.path.getsize(output),
                "groups_changed": sum(v > 0 for v in w.values()), "ok": True}
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
