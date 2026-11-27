"""Opt-in text cache numeric audit. No changes to production tensor values."""
import json
import logging
from pathlib import Path

_path = None
_totals = {}
logger = logging.getLogger(__name__)


def active():
    return _path is not None


def enable(path):
    global _path
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    with _path.open("x", encoding="utf-8"):
        pass
    _totals.clear()


def _write(row):
    with _path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record(phase, kind, label, sd):
    if not active():
        return
    import torch
    for name, tensor in sd.items():
        if not name.endswith("hidden_states") or not tensor.is_floating_point():
            continue
        x = tensor.detach().cpu()
        mask = sd.get(name[:-len("hidden_states")] + "attention_mask")
        if mask is None or mask.ndim != 1 or x.ndim < 2 or x.shape[0] != mask.shape[0]:
            raise ValueError(f"Cache audit cannot identify real tokens: {label} / {name}")
        m = mask.detach().cpu().bool()
        valid = x[m]
        rows = valid.flatten(1)
        n = int(valid.numel())
        row = {"phase": phase, "kind": kind, "source": str(label), "tensor": name,
               "valid_tokens": int(m.sum()), "valid_values": n,
               "valid_nan": int(torch.isnan(valid).sum()), "valid_inf": int(torch.isinf(valid).sum()),
               "valid_zero": int((valid == 0).sum()), "all_zero_rows": int((rows == 0).all(dim=1).sum()),
               "total_nan": int(torch.isnan(x).sum()), "total_inf": int(torch.isinf(x).sum()),
               "empty_mask": int(n == 0), "all_zero_embedding": int(n > 0 and not bool(torch.count_nonzero(valid)))}
        if valid.ndim == 3 and n and bool(torch.isfinite(valid).all()):
            f = valid.float()
            row["layer_rms"] = f.square().mean(dim=(0, 2)).sqrt().tolist()
            row["layer_std"] = f.std(dim=(0, 2), unbiased=False).tolist()
            row["layer_zero_fraction"] = (f == 0).float().mean(dim=(0, 2)).tolist()
        _write(row)
        totals = _totals.setdefault(phase + "/" + kind, {"embeddings": 0})
        totals["embeddings"] += 1
        for key in ("valid_tokens", "valid_values", "valid_nan", "valid_inf", "valid_zero",
                    "all_zero_rows", "total_nan", "total_inf", "empty_mask", "all_zero_embedding"):
            totals[key] = totals.get(key, 0) + row[key]
        if row["valid_nan"] or row["valid_inf"] or row["all_zero_rows"] or row["empty_mask"]:
            logger.warning("[cache-audit] suspicious %s %s: %s", phase, kind, row)


def finish():
    if not active():
        return
    _write({"summary": _totals})
    logger.info("[cache-audit] %s", json.dumps(_totals))
    if _path.name != "latent-audit.jsonl" and not _totals.get("readback/variant", {}).get("embeddings"):
        logger.warning("[cache-audit] No shuffled variant embeddings were saved; shuffle path was not exercised.")
    logger.info("[cache-audit] report: %s", _path)
    combined_report()


def check(phase, label, **fields):
    if not active():
        return
    row = {"phase": phase, "source": str(label), **fields}
    _write(row)
    totals = _totals.setdefault(phase, {"checks": 0, "mismatches": 0})
    totals["checks"] += 1
    if fields.get("ok") is False:
        totals["mismatches"] += 1
        logger.warning("[cache-audit] %s: %s", phase, row)


def compare(expected, actual, label, phase):
    """Compare training tokens, requiring matching full masks and shapes."""
    import torch
    x, y = expected.get("hidden_states"), actual.get("hidden_states")
    m, n = expected.get("attention_mask"), actual.get("attention_mask")
    if any(t is None for t in (x, y, m, n)):
        check(phase, label, ok=False, reason="missing conditioning tensor")
        return
    shape_ok = x.shape == y.shape and m.shape == n.shape and m.ndim == 1 and x.shape[0] == m.shape[0]
    mask_ok = bool(shape_ok and torch.equal(m, n))
    row = {"shape_equal": bool(shape_ok), "mask_equal": mask_ok, "dtype_equal": x.dtype == y.dtype}
    if mask_ok:
        a, b = x.detach().cpu()[m.cpu().bool()], y.detach().cpu()[n.cpu().bool()]
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        row.update(finite=finite, exact=bool(torch.equal(a, b)), valid_values=a.numel())
        if finite and a.numel():
            af, bf = a.float().flatten(), b.float().flatten()
            d = (af - bf).abs()
            row.update(max_abs=float(d.max()), mean_abs=float(d.mean()),
                       rms_error=float(d.square().mean().sqrt()),
                       cosine=float(torch.nn.functional.cosine_similarity(af, bf, dim=0)))
    check(phase, label, ok=bool(row.get("exact") and row.get("finite") and row.get("valid_values") and row["dtype_equal"]), **row)


def roundtrip(path, original, restored):
    if not active():
        return
    check("roundtrip_keys", path, ok=set(original) == set(restored))
    for name in original:
        if name.endswith("hidden_states"):
            pre = name[:-len("hidden_states")]
            mask = pre + "attention_mask"
            compare({"hidden_states": original.get(name), "attention_mask": original.get(mask)},
                    {"hidden_states": restored.get(name), "attention_mask": restored.get(mask)},
                    str(path) + "/" + pre, "compact_roundtrip")


def inspect_text(datasets, encode, encoder, workers, variants=True):
    """Re-encode independently against dataset captions, not cached metadata."""
    if not active():
        return
    from safetensors import safe_open
    from fizgig.dataset import compact_text, tag_variants as tv
    for ds in datasets:
        items = sorted((it for batch in ds.retrieve_text_encoder_output_cache_batches(workers) for it in batch),
                       key=lambda it: it.item_key)
        settings = tv.settings_of(ds) if variants else None
        picker = tv.VariantPicker(settings) if settings else None
        for i, it in enumerate(items):
            p, caption = it.text_encoder_output_cache_path, it.caption
            with safe_open(p, framework="pt") as f:
                md = f.metadata() or {}
            check("base_provenance", p, ok=md.get("caption1") == caption and md.get("text_encoder") == encoder,
                  caption_match=md.get("caption1") == caption, encoder_match=md.get("text_encoder") == encoder,
                  encoder=md.get("text_encoder"))
            if i < 4:
                base = compact_text.load(p)
                compare(encode([caption])[0], {k[6:]: v for k, v in base.items() if k.startswith("cond__")},
                        p, "fresh_base")
            wanted = tv.variant_captions(caption, settings) if settings else []
            if not wanted:
                continue
            vpath = tv.variants_path(p)
            if not Path(vpath).is_file():
                check("variant_provenance", vpath, ok=False, reason="missing variant cache")
                continue
            with safe_open(vpath, framework="pt") as f:
                vm = f.metadata() or {}
            n, _ = picker.info(p)
            aliases = [j for j, cap in enumerate(wanted) if cap == caption]
            check("variant_provenance", vpath, ok=vm.get("source") == caption and
                  vm.get("recipe") == settings.recipe() and json.loads(vm.get("captions", "[]")) == wanted and
                  json.loads(vm.get("orig", "[]")) == aliases and n == len(wanted),
                  encoder_present=bool(vm.get("text_encoder")), encoder=vm.get("text_encoder"),
                  expected_variants=len(wanted), picker_variants=n, original_aliases=aliases)
            for j, cap in enumerate(wanted):
                cached = compact_text.load(p) if j in aliases else picker.load(p, j)
                compare(encode([cap])[0], {k[6:]: v for k, v in cached.items() if k.startswith("cond__")},
                        vpath + f"#{j}", "fresh_variant")


def latent(phase, path, x, expected_shape):
    if not active():
        return
    import torch
    x = x.detach().cpu()
    f = x.float()
    finite = bool(torch.isfinite(f).all())
    check(phase, path, ok=bool(finite and x.numel() and torch.count_nonzero(x) and
                              tuple(x.shape) == tuple(expected_shape)),
          shape=list(x.shape), expected_shape=list(expected_shape), dtype=str(x.dtype),
          nan=int(torch.isnan(f).sum()), inf=int(torch.isinf(f).sum()), zeros=int((f == 0).sum()),
          rms=float(f.square().mean().sqrt()) if finite and x.numel() else None,
          min=float(f.min()) if finite and x.numel() else None,
          max=float(f.max()) if finite and x.numel() else None)


def provenance_probe(cache_matches):
    """Synthetic metadata probe, not an actual encoder change."""
    if not active():
        return
    import tempfile
    import torch
    from safetensors.torch import save_file
    from fizgig.dataset import tag_variants as tv
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "probe_te.safetensors")
        cap = "trigger, red, blue, green, yellow"
        settings = tv.TagSettings(shuffle=True)
        wanted = tv.variant_captions(cap, settings)
        h = torch.ones(2, 1, 2)
        m = torch.ones(2, dtype=torch.bool)
        save_file({"cond__hidden_states": h, "cond__attention_mask": m}, p,
                  metadata={"caption1": cap, "text_encoder": "new"})
        sd = {}
        for i, v in enumerate(wanted):
            if v != cap:
                sd[f"v{i}__hidden_states"] = h.clone()
                sd[f"v{i}__attention_mask"] = m.clone()
        save_file(sd, tv.variants_path(p), metadata={"fizgig_tags": "1", "source": cap,
                  "recipe": settings.recipe(), "captions": json.dumps(wanted),
                  "orig": json.dumps([i for i, v in enumerate(wanted) if v == cap]),
                  "empty": "0", "text_encoder": "old"})
        needed = tv._needed(p, cap, settings)
        count, _ = tv.VariantPicker(settings).info(p)
        check("synthetic_encoder_change", p, ok=bool(needed) and count == 0,
              base_rejects_old=not cache_matches(p, cap, "", "old"),
              variants_requested=len(needed), stale_picker_variants=count,
              note="synthetic metadata, no second real model loaded")
        save_file({"cond__hidden_states": h, "cond__attention_mask": m}, p,
                  metadata={"caption1": cap})
        check("synthetic_missing_encoder", p, ok=not cache_matches(p, cap, "", "new"),
              accepts_unstamped_base=cache_matches(p, cap, "", "new"))


def combined_report():
    if not active() or _path.name != "cache-audit.jsonl":
        return
    latent_path = _path.with_name("latent-audit.jsonl")
    latent_summary = None
    if latent_path.exists():
        with latent_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "summary" in row:
                    latent_summary = row["summary"]
    report = {"text": _totals, "latents": latent_summary,
              "scope": "fresh isolated cache; exact roundtrip; independent re-encoding; synthetic provenance probes"}
    _path.with_name("cache-integrity-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
