"""Persistent full Q6 archive cache and per-comparison stock reference cache."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from native_codec import ELIGIBLE, FORMAT, validate, sha256


def identity(raw_hash, codec, rank, group):
    here = Path(__file__).resolve().parent
    return {"raw_sha256":raw_hash,"codec":codec,"rank":rank,"group":group,
            "format":FORMAT,"torch":str(torch.__version__),
            "sources":{n:sha256(here/n) for n in ("native_codec.py","geometry_codec.py")}}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def cache_path(root, ident):
    directory = Path(os.environ.get("FIZGIG_LITTLEBIT_CACHE_DIR") or root.parent/"_compressed_cache")
    return directory/(digest(ident)+".safetensors")


def inspect_archive(reader, ident, shapes):
    meta = reader.metadata() or {}
    if meta.get("source_sha256") != ident["raw_sha256"] or meta.get("littlebit_cache_identity") != json.dumps(ident,sort_keys=True):
        raise ValueError("compressed cache provenance mismatch")
    specs = json.loads(meta[FORMAT])
    if set(specs) != ELIGIBLE or set(reader.keys()) != ELIGIBLE:
        raise ValueError("compressed cache requires all eligible projections")
    for key,spec in specs.items():
        if spec.get("codec")!=ident["codec"] or spec.get("group")!=ident["group"] or spec.get("shape")!=shapes[key]:
            raise ValueError("compressed cache spec mismatch: "+key)
        validate(reader.get_tensor(key),spec)
    return specs


def link_or_copy(source,dest):
    """Publish a complete file without overwriting; use a hardlink when possible."""
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    try:
        os.link(source,dest)
        return "hardlink"
    except FileExistsError:
        raise
    except OSError:
        tmp = dest.with_name(dest.name+"."+uuid.uuid4().hex+".tmp")
        try:
            if shutil.disk_usage(dest.parent).free < source.stat().st_size+2*1024**3:
                raise ValueError("insufficient disk space to copy compressed cache")
            shutil.copyfile(source,tmp)
            if dest.exists():
                raise FileExistsError(dest)
            # os.rename is no-overwrite on Windows; use exclusive link on POSIX.
            if os.name=="nt":
                os.rename(tmp,dest)
            else:
                os.link(tmp,dest)
            return "copy"
        finally:
            tmp.unlink(missing_ok=True)


def reference_path(root, ident, key):
    return root/"stock-projection-reference"/digest(ident)/(key+".safetensors")


def references(path, probes, compute):
    expected = {f"{i}.{kind}" for i in range(len(probes)) for kind in ("y","dx")}
    if path.is_file():
        with safe_open(str(path),framework="pt",device="cpu") as f:
            if set(f.keys())!=expected:
                raise ValueError("invalid cached stock reference keys")
            result = [(f.get_tensor(f"{i}.y"),f.get_tensor(f"{i}.dx")) for i in range(len(probes))]
        for (y,dx),(x,dy) in zip(result,probes):
            if y.shape!=dy.shape or dx.shape!=x.shape or y.dtype!=x.dtype or dx.dtype!=x.dtype or not torch.isfinite(y).all() or not torch.isfinite(dx).all():
                raise ValueError("invalid cached stock reference tensors")
        return result,True
    result = [compute(x,dy) for x,dy in probes]
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        save_file({f"{i}.{kind}":t.contiguous() for i,pair in enumerate(result) for kind,t in zip(("y","dx"),pair)},str(tmp))
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)
    return result,False
