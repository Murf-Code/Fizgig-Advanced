"""The per-image loss watch for described families: detect problem images, per-image LR, look-outlier warm-up and
auto-recaption, on the shared PerImageLossWatch (training/loss_logger.py).

This mirrors Krea 2's trainer integration (krea2/trainer.py) step for step - fresh-run cleanup, exclusion preflight,
look-outlier warm-up, resume replay from the run's own JSONL and the applied-captions ledger, and the between-epoch
caption repair. Stuck images are recaptioned by the model picked on the Captions tab (Qwen3-VL - Krea 2's text
encoder, the default -, Florence-2 or a tagger) with that tab's instructions or settings, for every family - and
captions are re-encoded with the family's own driver.encode_text into the family cache (cache.save_cond). The GUI's Problem Images window reads the same loss_log/ files as for Krea 2.
"""
import gc
import json
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

def _picture_exts():
    from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS      # every format the training set reads
    return tuple(dict.fromkeys(e.lower() for e in IMAGE_EXTENSIONS))


def _picture(folder, name, listings):
    """The training picture called `name` (its key's file name, no extension) in `folder`, matched as the dataset does:
    the extension in any case (IMG_0001.JPG: on Linux pods names are case-sensitive, and a lowercase guess missed
    it) and in every format training reads (.avif). listings: {folder: its file names}, filled on first use."""
    if folder not in listings:
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            names = []
        have = {}
        for n in names:
            stem, ext = os.path.splitext(n)
            have.setdefault((stem, ext.lower()), n)
        listings[folder] = have
    have = listings[folder]
    for ext in _picture_exts():
        if (name, ext) in have:
            return os.path.join(folder, have[(name, ext)])
    return ""


def _find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _folders(group, fallback=None):
    """{key prefix: folder} of the run's watched datasets (every block but the regularisation set's): the training
    folder's own pictures have no prefix, a sub-folder's or another subject's have theirs (ImageDataset.item_prefix)."""
    out = {}
    for ds in getattr(group, "datasets", None) or []:
        d = getattr(ds, "image_directory", None)
        if d and os.path.isdir(d) and not getattr(ds, "is_reg", False):
            out.setdefault(str(getattr(ds, "item_prefix", "") or ""), d)
    if not out and fallback:
        out[""] = fallback
    return out


def _tag_settings(group):
    """item_key -> its dataset's active TagSettings (tag shuffle / caption dropout), for the items that have them."""
    from fizgig.dataset.tag_variants import settings_of
    out = {}
    for ds in getattr(group, "datasets", None) or []:
        ts, bm = settings_of(ds), getattr(ds, "batch_manager", None)
        if ts is not None and bm is not None:
            for bucket in list(bm.buckets.values()) + list(getattr(bm, "live_pools", {}).values()):
                for it in bucket:
                    out[str(it.item_key)] = ts
    return out


def _items(group):
    """item_key -> ItemInfo over every watched training bucket (not the regularisation set's: its pictures are an
    anchor at their own fixed LR, never judged, throttled, excluded or recaptioned)."""
    out = {}
    for ds in group.datasets:
        if getattr(ds, "is_reg", False):
            continue
        bm = getattr(ds, "batch_manager", None)
        if bm is not None:
            for bucket in list(bm.buckets.values()) + list(getattr(bm, "live_pools", {}).values()):
                for it in bucket:
                    out[str(it.item_key)] = it
    return out


def second_try_min(min_score):
    """A tagger's second-try min score: 25% lower, but never more than 0.1 below the first try's."""
    return round(max(min_score * 0.75, min_score - 0.1), 3)
def _reg_keys(group):
    """The item keys of the run's regularisation set (dataset blocks marked is_reg)."""
    out = set()
    for ds in getattr(group, "datasets", None) or []:
        if getattr(ds, "is_reg", False) and getattr(ds, "batch_manager", None) is not None:
            for bucket in ds.batch_manager.buckets.values():
                out.update(str(it.item_key) for it in bucket)
    return out


class Watch:
    """The run's loss watch and the state around it. `watch` is None when every toggle is off."""
    reg = frozenset()                       # the regularisation set's keys (set per run in __init__)

    def __init__(self, output_dir, group, user_config, driver, *, log=False, per_image_lr=False, auto_recaption=False,
                 warmup_look=False, resume=False, start_epoch=0, te_path=None, trigger_word=None,
                 trigger_position="start", recaption_triggers=None, recaption_instruction=None, recaption_instruction_detailed=None,
                 captioner_path=None, recaption_task=None, recaption_task_detailed=None, recaption_with=None, vae_path=None,
                 folder_weights=None):
        from fizgig.training.loss_logger import PerImageLossWatch, is_enabled as env_on
        self.output_dir, self.group, self.driver, self.te_path = output_dir, group, driver, te_path
        self.vae_path = vae_path
        self.folder_weights = folder_weights
        self.removed_keys = set()
        self.invalid_mask_keys = set()  # mask errors are distinct from deleted/stale image latents
        self.trigger_word, self.trigger_position = trigger_word, trigger_position
        self.triggers = dict(recaption_triggers or {})     # another folder's own trigger word, by key prefix
        self.instr, self.instr_detailed = recaption_instruction, recaption_instruction_detailed
        self.recaptioned = {}
        self.captioner_path = captioner_path
        # auto-recaption always uses Qwen3-VL (--captioner); the Captions tab's other models never load in a run
        self.recap = dict(recaption_with or {})
        self.watch = None
        self.reg = _reg_keys(group)         # the regularisation set's pictures: never recorded or judged
        if not resume:
            _cleanup(output_dir)
        if auto_recaption and not (captioner_path and os.path.isfile(captioner_path)):
            logger.warning("[auto-recaption] needs the captioner (--captioner: the Krea 2 Qwen3-VL-4B text encoder "
                           "file) - off")
            auto_recaption = False
        # Caption Studio can enqueue changes even with every loss-watch feature off.
        # Keep the epoch-boundary worker alive so edits are never silently ignored.
        if not (log or per_image_lr or auto_recaption or warmup_look or env_on()):
            logger.info("[caption-fix] epoch-boundary caption synchronization enabled")
        self.image_dir = _find_key(user_config, "image_directory")
        self.caption_ext = _find_key(user_config, "caption_extension") or ".txt"
        if not (self.image_dir and os.path.isdir(self.image_dir)):
            self.image_dir = None
        self.folders = _folders(group, self.image_dir)
        if self.folders and not self.image_dir:
            self.image_dir = next(iter(self.folders.values()))
        # Baseline the source files at trainer startup. The GUI queue covers immediate
        # edits; these fingerprints additionally catch AI captions, bulk tools and
        # external file edits that bypass the Caption Studio save route.
        self._source_snapshot = self._source_state()
        self._mask_snapshot = self._mask_state()
        self._unscheduled_last = None
        self._pending_reference_text = {}
        self.unsupported_media_keys = set()
        self.stale_caption_keys = set()
        if auto_recaption and not self.image_dir:
            logger.warning("[auto-recaption] image_directory not found in the dataset config - auto-recaption off")
            auto_recaption = False
        self.auto_recaption = auto_recaption
        if auto_recaption:
            logger.info(f"[auto-recaption] ON - stuck images re-captioned in their own folders "
                        f"({len(self.folders)}: {', '.join(self.folders.values())})"
                        + (f" (trigger: '{trigger_word}')" if trigger_word else ""))
            first = recaption_task or ("an instruction given on the command line" if recaption_instruction
                                       else '"Training caption" (as shipped)')
            second = recaption_task_detailed or ("an instruction given on the command line"
                                                 if recaption_instruction_detailed
                                                 else '"Exhaustive detail" (as shipped)')
            logger.info(f"[auto-recaption] first try: {first}; second try: {second} (Qwen3-VL, Captions tab instructions)")
        if log or per_image_lr or auto_recaption or warmup_look or env_on():
            self.watch = PerImageLossWatch(output_dir, apply_lr=per_image_lr, write_jsonl=True,
                                           dataset_dir=self.image_dir, caption_ext=self.caption_ext,
                                           family=self.driver.description.key, folders=self.folders)
            keys = set(_items(group))
            self.watch.preflight(keys)
            logger.info(f"[loss-watch] per-image loss watch ON (per_image_lr={per_image_lr})")
            if warmup_look:
                self._warmup(keys)
            if resume and start_epoch > 0:
                self._resume(start_epoch)

    # ---- setup helpers -------------------------------------------------------------------------------
    def _warmup(self, keys):
        """Each folder's own Look Filter scores (a sub-folder or subject scanned on its own); a folder not scanned
        simply has no warm-up."""
        found, use = 0, set()
        folders = getattr(self, "folders", None) or {"": self.image_dir or ""}
        for prefix, folder in folders.items():
            path = os.path.join(folder, "fizgig_look_scores.json")
            try:
                with open(path, encoding="utf-8") as f:
                    look = json.load(f)
            except FileNotFoundError:
                if len(folders) > 1:
                    logger.info(f"[look-warmup] {folder}: no Look Filter scan - no warm-up for its pictures")
                continue
            except Exception as e:
                logger.warning(f"[look-warmup] could not load {path} ({e}) - no warm-up for its pictures")
                continue
            found += 1
            use |= _warmup_keys(keys, look, prefix)
        if not found:
            logger.warning("[look-warmup] fizgig_look_scores.json not found in the dataset folder - run the Look "
                           "Consistency Filter (Image Prep tab) first; warm-up off this run")
        if use:
            self.watch.set_warmup_keys(use)

    def _resume(self, start_epoch):
        resets = {}
        try:
            with open(os.path.join(self.output_dir, "loss_log", "caption_updates_applied.json"),
                      encoding="utf-8") as f:
                for k, info in json.load(f).items():
                    for e in (info if isinstance(info, list) else [info]):
                        att, auto = int(e.get("attempt", 0) or 0), bool(e.get("auto"))
                        if auto:
                            self.recaptioned[k] = max(self.recaptioned.get(k, 0), att)
                        resets.setdefault(k, []).append((int(e.get("epoch", 0) or 0), att, auto))
        except Exception:
            pass
        self.watch.resume_from_jsonl(up_to_epoch=start_epoch, resets=resets)



    def _new_image_sources(self):
        """New eligible stills in existing dataset blocks (holdouts never join training)."""
        known = set(_items(self.group))
        results = []
        allowed = set(_picture_exts())
        for ds in getattr(self.group, "datasets", ()) or ():
            if getattr(ds, "is_reg", False) or getattr(ds, "holdout_only", False):
                continue
            folder = getattr(ds, "image_directory", None)
            bm = getattr(ds, "batch_manager", None)
            if not folder or not bm or not os.path.isdir(folder):
                continue
            prefix = str(getattr(ds, "item_prefix", "") or "")
            held = set(getattr(ds, "holdout_stems", ()) or ())
            seen = set()
            for filename in sorted(os.listdir(folder)):
                stem, ext = os.path.splitext(filename)
                if ext.lower() not in allowed or stem in held or stem in seen:
                    continue
                seen.add(stem)
                key = prefix + stem
                if key in known:
                    continue
                source = os.path.join(folder, filename)
                if not os.path.isfile(source):
                    continue
                cext = getattr(ds, "caption_extension", None)
                caption_path = os.path.join(folder, stem + cext) if cext else ""
                if cext and not os.path.isfile(caption_path) and not getattr(ds.datasource, "captions_optional", False):
                    logger.warning("[live-dataset] %s not yet eligible: missing caption %s", key, caption_path)
                    continue
                results.append((ds, key, source, caption_path))
        return results

    def _admit_new_images(self, epoch, dit, device, parkable):
        """Encode and admit stills into existing buckets without changing training step counts.

        A new shape with no existing bucket, unsupported source type or failed encoding
        stays out of the run and is reported by the status endpoint. No partial cache
        can enter the sampler.
        """
        candidates = self._new_image_sources()
        if not candidates:
            return
        if not self.vae_path or not self.te_path:
            logger.warning("[live-dataset] %s new images await VAE/text encoder paths", len(candidates))
            return
        from fizgig.dataset.image_dataset import ItemInfo, BucketSelector, resize_image_to_bucket
        from fizgig.dataset.masks import mask_for_stem, mask_problem
        from fizgig.dataset.caption_file import read_caption
        from fizgig.utils.upright import open_upright
        from fizgig.families import cache, quant
        import copy
        prepared = []
        for ds, key, source, cap_path in candidates[:64]:
            try:
                im = open_upright(source)
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                selector = BucketSelector(ds.resolution, ds.enable_bucket, ds.bucket_no_upscale, ds.reso_steps)
                bucket = tuple(selector.get_bucket_resolution(im.size))
                if bucket not in ds.batch_manager.buckets:
                    logger.warning("[live-dataset] %s needs a new bucket %s; restart to include this aspect ratio",
                                   key, bucket)
                    continue
                if getattr(ds, "has_control", False) and not getattr(self.driver, "supports_references", False):
                    logger.warning("[live-dataset] %s uses a reference format this driver cannot live-encode", key)
                    continue
                item = ItemInfo(key, read_caption(cap_path) if cap_path else "", im.size, bucket)
                # Dataset cache filenames are based on source file basename, not
                # subject prefixes such as "s2:" (invalid in Windows filenames).
                import copy
                cache_item = copy.copy(item)
                cache_item.item_key = source
                item.latent_cache_path = ds.get_latent_cache_path(cache_item)
                item.text_encoder_output_cache_path = ds.get_text_encoder_output_cache_path(cache_item)
                mask_root = ds.mask_directory or ds.image_directory
                if getattr(ds, "loss_masks", False):
                    mask = mask_for_stem(mask_root, os.path.splitext(os.path.basename(source))[0])
                    if mask:
                        why = mask_problem(mask, im.size)
                        if why:
                            logger.warning("[live-dataset] %s has invalid mask (%s); not admitted", key, why)
                            continue
                        item.mask_path = mask
                self._owner_dataset(key)  # initialize the owner map before adding this entry
                self._owner_cache[key] = ds
                references = (self._reference_images(key, source, bucket)
                              if getattr(ds, "has_control", False) else [])
                item.control_content = references or None
                source_stat = (os.stat(source).st_size, os.stat(source).st_mtime_ns)
                cap_stat = (os.stat(cap_path).st_size, os.stat(cap_path).st_mtime_ns) if cap_path else None
                prepared.append((ds, key, source, cap_path, item, im, references, source_stat, cap_stat))
            except Exception:
                logger.warning("[live-dataset] cannot prepare new image %s", key, exc_info=True)
        if not prepared:
            return
        if parkable:
            quant.move(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
        latent_prepared = []
        admitted = []
        try:
            vae = self.driver.load_vae(self.vae_path, device)
            try:
                with torch.no_grad():
                    for row in prepared:
                        ds, key, source, cap_path, item, image, refs, source_stat, cap_stat = row
                        scratch = copy.copy(item)
                        scratch.latent_cache_path += ".live-update"
                        try:
                            pixels = resize_image_to_bucket(image, tuple(item.bucket_size))
                            target = self.driver.encode_images(vae, [pixels])[0]
                            controls = [self.driver.encode_images(vae, [ref])[0] for ref in refs]
                            cache.save_latents(self.driver.description, scratch, target, controls)
                            latent_prepared.append(row)
                        except Exception:
                            _remove(scratch.latent_cache_path)
                            logger.warning("[live-dataset] failed encoding latent of %s", key, exc_info=True)
            finally:
                del vae
                gc.collect()
                torch.cuda.empty_cache()

            plain, reference = [], []
            from safetensors import safe_open
            for row in latent_prepared:
                ds, key, *_ = row
                if not getattr(ds, "has_control", False):
                    plain.append(row)
                    continue
                # Edit-pair text has a vision reference; a slider's paired text
                # does not. Read this format from an existing cache in the block.
                ref_text = False
                for bucket in ds.batch_manager.buckets.values():
                    if bucket:
                        with safe_open(bucket[0].text_encoder_output_cache_path,
                                       framework="pt", device="cpu") as f:
                            ref_text = bool((f.metadata() or {}).get("reference_sizes"))
                        break
                (reference if ref_text else plain).append(row)

            def activate(row, condition, refs_text=""):
                ds, key, source, cap_path, item, _image, _refs, source_stat, cap_stat = row
                original = os.stat(source)
                now = (original.st_size, original.st_mtime_ns)
                caption_now = (os.stat(cap_path).st_size, os.stat(cap_path).st_mtime_ns) if cap_path else None
                if now != source_stat or caption_now != cap_stat:
                    raise ValueError("Source or caption changed while encoding; will retry next boundary")
                text_tmp = copy.copy(item)
                text_tmp.text_encoder_output_cache_path += ".live-update"
                try:
                    cache.save_cond(self.driver.description, text_tmp, condition,
                                    refs=refs_text, encoder=cache.encoder_stamp(self.te_path))
                    os.replace(item.latent_cache_path + ".live-update", item.latent_cache_path)
                    os.replace(text_tmp.text_encoder_output_cache_path, item.text_encoder_output_cache_path)
                    if ds.batch_manager.add_live_item(tuple(item.bucket_size), item, ds.num_repeats):
                        admitted.append(key)
                        if self.folder_weights is not None and float(getattr(ds, "loss_weight", 1.0)) != 1.0:
                            self.folder_weights[key] = float(ds.loss_weight)
                        logger.info("[live-dataset] epoch %s: admitted new image %s (fixed steps/epoch)", epoch, key)
                finally:
                    _remove(text_tmp.text_encoder_output_cache_path)
                    _remove(item.latent_cache_path + ".live-update")

            if plain:
                te = self.driver.load_text_encoder(self.te_path, device)
                try:
                    for row in plain:
                        try:
                            activate(row, self.driver.encode_text(te, [row[4].caption])[0])
                        except Exception:
                            logger.warning("[live-dataset] deferred new image %s", row[1], exc_info=True)
                finally:
                    self.driver.unload_text_encoder(te)
                    del te
                    gc.collect()
                    torch.cuda.empty_cache()
            if reference:
                rte = self.driver.load_reference_text_encoder(self.te_path, device)
                try:
                    for row in reference:
                        try:
                            refs = row[6]
                            cond = self.driver.encode_text_with_references(rte, [row[4].caption], [refs])[0]
                            refs_text = ",".join(f"{r.shape[1]}x{r.shape[0]}" for r in refs)
                            activate(row, cond, refs_text)
                        except Exception:
                            logger.warning("[live-dataset] deferred new paired image %s", row[1], exc_info=True)
                finally:
                    self.driver.unload_text_encoder(rte)
                    del rte
                    gc.collect()
                    torch.cuda.empty_cache()
            # Baseline ONLY newly admitted keys. Rebaselining existing keys could
            # erase an edit made while the encoder was occupied.
            if admitted:
                fresh = self._source_state()
                self._source_snapshot.update({k: fresh[k] for k in admitted if k in fresh})
        finally:
            for row in latent_prepared:
                _remove(row[4].latent_cache_path + ".live-update")
            if parkable:
                quant.move(dit, device)

    def _unscheduled_inputs(self):
        """Keys of newly present source media outside the fixed active batch schedule.

        Changing step counts/buckets mid-run without caching and reconciling the
        optimizer schedule would be unsafe. Surface these files explicitly rather
        than pretending they joined the run.
        """
        from fizgig.dataset.masks import is_mask_file
        scheduled = set(_items(self.group))
        found = set()
        for ds in getattr(self.group, "datasets", ()) or ():
            if getattr(ds, "is_reg", False) or getattr(ds, "holdout_only", False):
                continue
            folder = getattr(ds, "image_directory", None)
            if not folder or not os.path.isdir(folder):
                continue
            prefix = str(getattr(ds, "item_prefix", "") or "")
            extensions = set(_picture_exts()) | {x.lower() for x in getattr(ds, "extra_extensions", ())}
            held = set(getattr(ds, "holdout_stems", ()) or ())
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            for filename in names:
                stem, ext = os.path.splitext(filename)
                if ext.lower() not in extensions or is_mask_file(filename) or stem in held:
                    continue
                if prefix + stem not in scheduled and os.path.isfile(os.path.join(folder, filename)):
                    found.add(prefix + stem)
        return sorted(found)

    def _report_unscheduled_inputs(self):
        pending = self._unscheduled_inputs()
        unsupported = sorted(getattr(self, "unsupported_media_keys", set()))
        invalid = sorted(getattr(self, "invalid_mask_keys", set()))
        signature = (tuple(pending), tuple(unsupported), tuple(invalid))
        if signature == getattr(self, "_unscheduled_last", None):
            return
        if pending:
            logger.warning("[live-dataset] %s file(s) not admitted to the active batch schedule "
                           "(examples: %s). Missing captions, cache errors or unseen bucket "
                           "shapes may require correction or a restart.",
                           len(pending), ", ".join(pending[:12]))
        status_path = os.path.join(self.output_dir, "loss_log", "live_dataset_status.json")
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        payload = {
            "requires_restart": bool(unsupported),
            "attention": bool(pending or unsupported or invalid),
            "unscheduled_count": len(pending),
            "unscheduled_examples": pending[:40],
            "unsupported_media_count": len(unsupported),
            "unsupported_media_examples": unsupported[:30],
            "invalid_mask_count": len(invalid),
            "invalid_mask_examples": invalid[:30],
            "reason": ("Edited video/audio sources cannot yet be live-re-encoded; their "
                       "old cached batches have been excluded until restart." if unsupported else
                       "Some new files have not been admitted yet. Check captions, matching "
                       "bucket shapes and the training log." if pending else
                       "Fix the invalid masks to let those images train." if invalid else ""),
        }
        with open(status_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(status_path + ".tmp", status_path)
        self._unscheduled_last = signature

    def _mask_state(self):
        """{item key: (path, size, mtime)} for masks of active still-image datasets.

        Masks are read by the batch loader on the latent grid, not VAE-encoded.
        Checking them separately also detects masks added after training starts.
        """
        from fizgig.dataset.masks import folder_names, mask_for_stem
        state = {}
        for ds in getattr(self.group, "datasets", ()) or ():
            if getattr(ds, "is_reg", False) or not getattr(ds, "loss_masks", False):
                continue
            folder = getattr(ds, "mask_directory", None) or getattr(ds, "image_directory", None)
            manager = getattr(ds, "batch_manager", None)
            if not folder or manager is None:
                continue
            names = folder_names(folder)
            prefix = str(getattr(ds, "item_prefix", "") or "")
            for bucket in list(manager.buckets.values()) + list(getattr(manager, "live_pools", {}).values()):
                for item in bucket:
                    key = str(item.item_key)
                    if key in state or getattr(item, "still_only", False):
                        continue
                    stem = key[len(prefix):] if key.startswith(prefix) else key
                    stem = stem.rsplit("/", 1)[-1]
                    mask = mask_for_stem(folder, stem, names)
                    try:
                        st = os.stat(mask) if mask else None
                        state[key] = (os.path.realpath(mask), st.st_size, st.st_mtime_ns) if st else ("", 0, 0)
                    except OSError:
                        state[key] = ("", 0, 0)
        return state

    def _refresh_masks(self, changed_images=()):
        """Validate added/edited masks and atomically switch each scheduled item's mask path.

        An invalid new mask must never silently make a masked picture train unmasked.
        Deleting a mask is an intentional switch to whole-picture loss at this boundary.
        """
        current = self._mask_state()
        previous = getattr(self, "_mask_snapshot", {})
        keys = {k for k, stamp in current.items() if stamp != previous.get(k)}
        keys.update(k for k in changed_images if k in current)
        if not keys:
            self._mask_snapshot = current
            return
        from fizgig.dataset.masks import mask_problem
        items = _items(self.group)
        for key in sorted(keys):
            item = items.get(key)
            if item is None:
                continue
            path = current[key][0]
            if path:
                try:
                    reason = mask_problem(path, item.original_size)
                except Exception as exc:
                    reason = str(exc) or "failed validation"
                if reason:
                    self.invalid_mask_keys.add(key)
                    item.mask_path = None
                    logger.warning("[live-mask] %s: mask %s (%s); image excluded until mask is fixed",
                                   key, path, reason)
                    continue
            item.mask_path = path or None
            self.invalid_mask_keys.discard(key)
            logger.info("[live-mask] %s: %s at epoch boundary", key,
                        "validated " + path if path else "mask removed; training unmasked")
        self._mask_snapshot = current


    def _owner_dataset(self, key):
        """Key -> dataset block map, built once so large source scans stay linear."""
        owners = getattr(self, "_owner_cache", None)
        if owners is None:
            owners = {}
            for ds in getattr(self.group, "datasets", ()) or ():
                if getattr(ds, "is_reg", False):
                    continue
                bm = getattr(ds, "batch_manager", None)
                for bucket in list((getattr(bm, "buckets", {}) or {}).values()) + list(getattr(bm, "live_pools", {}).values()):
                    for it in bucket:
                        owners[str(it.item_key)] = ds
            self._owner_cache = owners
        return owners.get(str(key))

    def _source_for_key(self, key, listings=None):
        """A training media path, with that dataset's permitted video/audio extensions."""
        from fizgig.dataset.masks import is_mask_file
        ds = self._owner_dataset(key)
        if ds:
            folder = getattr(ds, "image_directory", None) or ""
            prefix = str(getattr(ds, "item_prefix", "") or "")
        else:
            prefix, folder = max(((p, d) for p, d in self.folders.items() if str(key).startswith(p)),
                                 key=lambda pair: len(pair[0]), default=("", self.image_dir or ""))
        stem = str(key)[len(prefix):].rsplit("/", 1)[-1].removesuffix("#still")
        extensions = set(_picture_exts()) | {str(x).lower() for x in getattr(ds, "extra_extensions", ())}
        if not folder:
            return ""
        if listings is not None and folder in listings:
            names = listings[folder]
        else:
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                names = []
            if listings is not None:
                listings[folder] = names
        for n in names:
            base, ext = os.path.splitext(n)
            if base == stem and ext.lower() in extensions and not is_mask_file(n):
                p = os.path.join(folder, n)
                if os.path.isfile(p):
                    return p
        return ""

    def _reference_paths(self, key, source, listings=None):
        """Find controls currently on disk; datasource's startup list may now be stale."""
        from fizgig.dataset.masks import is_mask_file
        ds = self._owner_dataset(key)
        if not ds or not getattr(ds, "has_control", False):
            return []
        stem = os.path.splitext(os.path.basename(source))[0].casefold()
        extensions = {".mp4"} if source.lower().endswith(".mp4") else set(_picture_exts())
        matches = []
        for folder in (getattr(ds, "control_directory", None),
                       getattr(ds, "extra_reference_directory", None)):
            if not folder or not os.path.isdir(folder):
                continue
            if listings is not None and folder in listings:
                names = listings[folder]
            else:
                names = os.listdir(folder)
                if listings is not None:
                    listings[folder] = names
            for n in names:
                nstem, ext = os.path.splitext(n)
                if (ext.lower() in extensions and not is_mask_file(n) and
                    (nstem.casefold() == stem or nstem.casefold().startswith(stem + "_"))):
                    p = os.path.join(folder, n)
                    if os.path.isfile(p):
                        matches.append(p)
        def order(path):
            suffix = os.path.splitext(os.path.basename(path))[0].casefold()[len(stem):]
            return (0 if not suffix else 1, int(suffix[1:]) if suffix.startswith("_") and suffix[1:].isdigit() else 999999, path)
        return sorted(set(matches), key=order)

    def _reference_images(self, key, source, bucket):
        """Resize before-images identically to the original cache phase."""
        from fizgig.dataset.image_dataset import BucketSelector, _own_shape, resize_image_to_bucket
        from fizgig.utils.upright import open_upright
        ds = self._owner_dataset(key)
        paths = self._reference_paths(key, source)
        if not ds or not paths:
            raise ValueError("Paired references are missing or their dataset cannot be found")
        selector = BucketSelector(ds.resolution, ds.enable_bucket, ds.bucket_no_upscale, ds.reso_steps)
        arrays = []
        for i, path in enumerate(paths):
            im = open_upright(path)
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            target = bucket if i == 0 else _own_shape(im, bucket, selector.reso_steps)
            arrays.append(resize_image_to_bucket(im, target))
        return arrays

    def _source_state(self):
        """Fingerprint every active item's media, caption and paired controls."""
        seen, listings, ref_listings = {}, {}, {}
        for key in _items(self.group):
            ds = self._owner_dataset(key)
            source = self._source_for_key(key, listings)
            folder = getattr(ds, "image_directory", None) if ds else None
            prefix = str(getattr(ds, "item_prefix", "") or "") if ds else ""
            stem = str(key)[len(prefix):].rsplit("/", 1)[-1].removesuffix("#still")
            folder = folder or os.path.dirname(source) or self.image_dir or ""
            caption_ext = getattr(ds, "caption_extension", None) or self.caption_ext
            caption = os.path.splitext(source)[0] + caption_ext if source else os.path.join(folder, stem + caption_ext)
            def stamp(path):
                try:
                    st = os.stat(path)
                    return (os.path.realpath(path), st.st_size, st.st_mtime_ns)
                except OSError:
                    return ("", 0, 0)
            refs = tuple(stamp(p) for p in self._reference_paths(key, source, ref_listings)) if source else ()
            seen[str(key)] = (stamp(source), stamp(caption), caption, refs)
        return seen

    def _scan_sources(self):
        """Detect on-disk edits even when UI routes did not enqueue a change."""
        previous, latest = getattr(self, "_source_snapshot", {}), self._source_state()
        changed_images, changed_text = {}, {}
        for key, (picture, cap, caption_path, refs) in latest.items():
            before = previous.get(key)
            if before is None:
                continue
            if picture != before[0] or refs != (before[3] if len(before) >= 4 else ()):
                changed_images[key] = "updated" if picture[0] else "removed"
            if cap != before[1]:
                try:
                    from fizgig.dataset.caption_file import read_caption
                    changed_text[key] = read_caption(caption_path) if cap[0] else ""
                except OSError:
                    changed_text[key] = ""
        self._source_snapshot = latest
        return changed_images, changed_text

    # ---- per step --------------------------------------------------------------------------------------
    def excluded(self, batch):
        """The step is skipped: its picture is excluded (a batch's: every one of them)."""
        keys = batch.get("item_keys")
        # Never train a removed image, including a mixed batch whose family lacks
        # per-item losses. Skipping the whole batch is safer than training stale pixels.
        keys_to_check = keys if isinstance(keys, (list, tuple)) else [keys]
        if any(str(k) in self.removed_keys or str(k) in getattr(self, "invalid_mask_keys", ())
               or str(k) in getattr(self, "unsupported_media_keys", ())
               or str(k) in getattr(self, "stale_caption_keys", ()) for k in keys_to_check):
            return True
        if self.watch is None or self._reg_only(keys):
            return False
        if isinstance(keys, (list, tuple)) and len(keys) > 1:
            return all(self.watch.is_excluded([str(k)]) for k in keys)
        return self.watch.is_excluded(keys)

    def multiplier(self, batch):
        keys = batch.get("item_keys")
        blocked = self.removed_keys | getattr(self, "invalid_mask_keys", set()) | getattr(self, "unsupported_media_keys", set()) | getattr(self, "stale_caption_keys", set())
        if (isinstance(keys, (list, tuple)) and len(keys) == 1 and str(keys[0]) in blocked):
            return 0.0
        if not isinstance(keys, (list, tuple)) and str(keys) in blocked:
            return 0.0
        return self.watch.multiplier(keys) if self.watch is not None and not self._reg_only(keys) else 1.0

    def observe(self, epoch, step, batch, t, loss):
        if self.watch is not None and not self._reg_only(batch.get("item_keys")):
            self.watch.observe(epoch=epoch, step=step, item_keys=batch.get("item_keys"), timestep=t, loss=loss)

    def _reg_only(self, keys):
        """Every picture of the step is a regularisation one (the watch leaves those alone)."""
        keys = [str(k) for k in (keys if isinstance(keys, (list, tuple)) else [keys]) if k is not None]
        return bool(self.reg) and bool(keys) and all(k in self.reg for k in keys)

    # ---- Batch Size above 1: each picture its own weight and record, as its own step would have them (the driver
    # hands back each picture's loss, info["per"] - the idea from @Murf-Code's Fizgig Advanced) ------------------
    def weigh(self, info, keys, reg_keys=(), reg_mult=1.0):
        """(loss to backpropagate, the step's recorded loss, kept indices) for a batched step: each picture at its
        per-image LR / look warm-up (a regularisation picture at reg_mult), an excluded one at 0 and left out of the
        average and the records; None when the step has no per-picture losses."""
        per = info.get("per") if isinstance(info, dict) else None
        keys = [str(k) for k in keys or []]
        if per is None or len(keys) < 2 or int(per.shape[0]) != len(keys):
            return None
        raw = info.get("per_raw") or [float(x) for x in per.detach().float().cpu()]
        blocked = self.removed_keys | getattr(self, "invalid_mask_keys", set()) | getattr(self, "unsupported_media_keys", set()) | getattr(self, "stale_caption_keys", set())
        out = [k in blocked or (self.watch is not None and k not in self.reg and self.watch.is_excluded([k])) for k in keys]
        kept = [i for i, o in enumerate(out) if not o]
        mult = [0.0 if out[i] else float(reg_mult) if reg_keys and k in reg_keys else
                (self.watch.multiplier([k]) if self.watch is not None else 1.0) for i, k in enumerate(keys)]
        if all(m == 1.0 for m in mult):
            scaled = per.mean()
        else:
            scaled = (per * torch.tensor(mult, device=per.device, dtype=per.dtype)).sum() / max(1, len(kept))
        return scaled, torch.tensor(sum(raw[i] for i in kept) / max(1, len(kept))), kept

    def observe_each(self, epoch, step, keys, ts, raw, kept):
        """One record per kept picture of a batched step, with its own t and raw loss."""
        if self.watch is None:
            return
        for i in kept:
            if str(keys[i]) in self.reg:     # a regularisation picture: not recorded
                continue
            self.watch.observe(epoch=epoch, step=step, item_keys=[str(keys[i])], timestep=float(ts[i]),
                               loss=float(raw[i]))

    # ---- epoch boundary --------------------------------------------------------------------------------
    def boundary(self, epoch, dit, device, parkable=True):
        """Reclassify, detect dataset edits, and apply changes between epochs."""
        # A crash/stop can leave a claimed queue behind. Fold it back into the
        # pending queue before starting any new encoding work.
        for name in ("image_updates.json", "caption_updates.json"):
            _recover_processing(os.path.join(self.output_dir, "loss_log", name))
        if self.watch is not None:
            self.watch.epoch_boundary(epoch)
        try:
            changed_images, changed_text = self._scan_sources()
        except Exception:
            logger.warning("[live-dataset] source scan failed; queued UI edits still apply", exc_info=True)
            changed_images, changed_text = {}, {}
        try:
            self._images(epoch, dit, device, parkable, changed_images)
        except Exception:
            _recover_processing(os.path.join(self.output_dir, "loss_log", "image_updates.json"), changed_images)
            logger.warning("[image-fix] image refresh failed this boundary - training continues", exc_info=True)
        try:
            self._admit_new_images(epoch, dit, device, parkable)
        except Exception:
            logger.warning("[live-dataset] new-image admission failed; no unencoded images were scheduled",
                           exc_info=True)
        try:
            self._refresh_masks(changed_images)
        except Exception:
            logger.warning("[live-mask] mask refresh failed; old mask state retained", exc_info=True)
        try:
            self._captions(epoch, dit, device, parkable, changed_text)
        except Exception:
            _recover_processing(os.path.join(self.output_dir, "loss_log", "caption_updates.json"), changed_text)
            logger.warning("[caption-fix] caption repair failed this boundary - training continues", exc_info=True)
        try:
            self._report_unscheduled_inputs()
        except Exception:
            logger.warning("[live-dataset] could not report outstanding source changes", exc_info=True)

    def _images(self, epoch, dit, device, parkable, detected=None):
        """Rebuild only edited still-image latents at the boundary, retaining their training bucket."""
        path = os.path.join(self.output_dir, "loss_log", "image_updates.json")
        processing = path + ".processing"
        if not os.path.exists(path) and not detected:
            return
        has_file = os.path.exists(path)
        if has_file:
            os.replace(path, processing)
        try:
            updates = {}
            if has_file:
                with open(processing, encoding="utf-8") as f:
                    updates = json.load(f)
            if not isinstance(updates, dict):
                raise ValueError("Image queue must be a mapping")
            updates = {**(detected or {}), **updates}
        except Exception:
            logger.warning("[image-fix] could not read queued images; queue preserved", exc_info=True)
            raise
        if not updates:
            _remove(processing)
            return
        items = _items(self.group)
        listings = {}
        todo = []
        for k, action in updates.items():
            if k not in items:
                logger.warning("[image-fix] %s not in current training schedule; added images require a new run", k)
                continue
            # Select the dataset folder from the item's key prefix, not a filename guessed from the UI.
            folder = ""
            stem = str(k).rsplit("/", 1)[-1]
            best = -1
            for prefix, d in self.folders.items():
                if str(k).startswith(prefix) and len(prefix) > best:
                    folder, best = d, len(prefix)
                    stem = str(k)[len(prefix):].rsplit("/", 1)[-1]
            if not folder:
                folder = self.image_dir or ""
            source = self._source_for_key(str(k), listings)
            if action == "removed" or not source:
                self.removed_keys.add(str(k))
                logger.info("[image-fix] %s excluded from subsequent epochs", k)
            elif action == "updated":
                # Never train on stale cache while the edited source is waiting to refresh.
                self.removed_keys.add(str(k))
                if os.path.splitext(source)[1].lower() not in _picture_exts():
                    self.unsupported_media_keys.add(str(k))
                    logger.error("[live-dataset] %s was edited: audio/video recaching during an active "
                                 "run is not supported for this family; excluded until restart", k)
                else:
                    todo.append((str(k), items[k], source))
        if not todo:
            _remove(processing)
            return
        if not self.vae_path:
            logger.warning("[image-fix] no VAE available; retaining pending images")
            _requeue(path, processing, {k: "updated" for k, _, _ in todo})
            return
        from fizgig.families import cache, quant
        from fizgig.dataset.image_dataset import resize_image_to_bucket
        from fizgig.utils.upright import open_upright
        pending = {}
        if parkable:
            quant.move(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
        try:
            vae = self.driver.load_vae(self.vae_path, device)
            try:
                for k, item, source in todo:
                    try:
                        # A paired edit must refresh target, every before-image latent and
                        # reference-aware text together. Never discard control latents.
                        from safetensors import safe_open
                        with safe_open(item.latent_cache_path, framework="pt", device="cpu") as existing:
                            had_controls = any(key.startswith("latent_control_") for key in existing.keys())
                        ds = self._owner_dataset(k)
                        paired = bool(getattr(ds, "has_control", False) or had_controls)
                        if paired and not getattr(self.driver, "supports_references", False):
                            raise ValueError("The driver cannot live-refresh paired references")
                        with torch.no_grad():
                            upright = open_upright(source)
                            new_source_size = upright.size
                            image = resize_image_to_bucket(upright, tuple(item.bucket_size))
                            encoded = self.driver.encode_images(vae, [image])[0]
                            references = self._reference_images(k, source, tuple(item.bucket_size)) if paired else []
                            limit = max(1, getattr(self.driver.description, "edit_references", 1))
                            if len(references) > limit:
                                raise ValueError(f"Paired references exceed this model's limit of {limit}")
                            encoded_controls = [self.driver.encode_images(vae, [ref])[0] for ref in references]
                            import copy
                            scratch = copy.copy(item)
                            scratch.original_size = new_source_size
                            scratch.control_content = references if paired else None
                            owner = self._owner_dataset(k)
                            if owner is not None:
                                cache_name = copy.copy(scratch)
                                cache_name.item_key = source
                                new_cache = owner.get_latent_cache_path(cache_name)
                            else:
                                new_cache = item.latent_cache_path
                            scratch.latent_cache_path = new_cache + ".live-update"
                            old_cache = item.latent_cache_path
                            try:
                                cache.save_latents(self.driver.description, scratch, encoded,
                                                   encoded_controls)
                                os.replace(scratch.latent_cache_path, new_cache)
                                item.latent_cache_path = new_cache
                                item.original_size = new_source_size
                                if paired:
                                    item.control_content = references
                                    # A paired text cache is vision-conditioned on the
                                    # before-images. Keep this key excluded until text
                                    # also refreshes successfully.
                                    from fizgig.dataset.caption_file import read_caption
                                    caption_file = os.path.splitext(source)[0] + (
                                        getattr(owner, "caption_extension", None) or self.caption_ext)
                                    try:
                                        cap = read_caption(caption_file)
                                    except OSError:
                                        cap = item.caption
                                    self._pending_reference_text[k] = cap
                                if old_cache != new_cache:
                                    _remove(old_cache)
                            finally:
                                _remove(scratch.latent_cache_path)
                        if not paired:
                            self.removed_keys.discard(k)
                        logger.info("[image-fix] epoch %s: %s latent re-encoded at %s%s", epoch, k,
                                    item.bucket_size, " with paired controls (text pending)" if paired else "")
                    except Exception:
                        pending[k] = "updated"
                        logger.warning("[image-fix] could not refresh %s; retaining pending update", k, exc_info=True)
            finally:
                del vae
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pending.update({k: "updated" for k, _, _ in todo})
            logger.warning("[image-fix] VAE load/encode failed; retry at next boundary", exc_info=True)
        finally:
            if parkable:
                quant.move(dit, device)
        if pending:
            _requeue(path, processing, pending)
        else:
            _remove(processing)

    def _captions(self, epoch, dit, device, parkable, detected=None):
        path = os.path.join(self.output_dir, "loss_log", "caption_updates.json")
        processing = path + ".processing"
        updates = {}
        if os.path.exists(path):
            try:
                os.replace(path, processing)             # atomic claim: GUI edits made meanwhile land in a fresh file
                with open(processing, encoding="utf-8") as f:
                    updates = {str(k): str(v) for k, v in json.load(f).items()}
            except Exception:
                logger.warning("[caption-fix] could not read caption_updates.json - skipping", exc_info=True)
                raise
        updates = {**getattr(self, "_pending_reference_text", {}), **(detected or {}), **updates}
        if not hasattr(self, "stale_caption_keys"):
            self.stale_caption_keys = set()
        self.stale_caption_keys.update(updates)
        auto = []
        if self.auto_recaption and self.watch is not None and self.image_dir:
            listings = {}
            for k in sorted(k for k, v in self.watch.verdicts.items() if v == "stuck"):
                if k in updates or self.recaptioned.get(k, 0) >= 2:
                    continue
                _p, folder, name = self.watch.where(k)
                p = _picture(folder, name, listings) if folder else ""
                if p:
                    auto.append((k, p, self.recaptioned.get(k, 0) + 1))
        if not updates and not auto:
            _remove(processing)
            return
        if not self.te_path:
            logger.warning("[caption-fix] caption work is pending but no --text_encoder was given; left queued")
            _requeue(path, processing, updates)
            return
        items = _items(self.group)
        todo = [(k, items[k], c, 0) for k, c in updates.items() if k in items]
        for k in updates:
            if k not in items:
                logger.warning(f"[caption-fix] '{k}' not in the training set - skipped")
        auto = [a for a in auto if a[0] in items]
        if not todo and not auto:
            _remove(processing)
            return
        logger.info(f"[caption-fix] epoch boundary {epoch}: {len(todo)} manual edit(s), {len(auto)} stuck image(s) "
                    f"to auto-recaption...")
        from fizgig.families import cache, quant
        if parkable:
            quant.move(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
        ok = False
        try:
            if auto:
                todo += self._recaption(auto, items, device)
            if not todo:        # every stuck image failed to caption: nothing to re-encode, they retry next time
                _remove(processing)
                return
            from safetensors import safe_open
            import copy
            desc = self.driver.description
            tag_sets = _tag_settings(self.group)
            plain_rows, paired_rows = [], []
            for row in todo:
                k, item, cap, attempt = row
                ds = self._owner_dataset(k)
                use_reference_text = False
                if ds and getattr(ds, "has_control", False):
                    with safe_open(item.text_encoder_output_cache_path, framework="pt", device="cpu") as f:
                        use_reference_text = bool((f.metadata() or {}).get("reference_sizes"))
                (paired_rows if use_reference_text else plain_rows).append(row)

            def write(row, cond, refs=""):
                k, item, cap, _attempt = row
                scratch = copy.copy(item)
                scratch.caption = cap
                scratch.text_encoder_output_cache_path = item.text_encoder_output_cache_path + ".live-update"
                try:
                    cache.save_cond(desc, scratch, cond, refs=refs,
                                    encoder=cache.encoder_stamp(self.te_path))
                    os.replace(scratch.text_encoder_output_cache_path,
                               item.text_encoder_output_cache_path)
                finally:
                    _remove(scratch.text_encoder_output_cache_path)
                item.caption = cap
                self.stale_caption_keys.discard(k)
                self.removed_keys.discard(k)
                getattr(self, "_pending_reference_text", {}).pop(k, None)

            if plain_rows:
                te = self.driver.load_text_encoder(self.te_path, device)
                try:
                    for i in range(0, len(plain_rows), 4):
                        rows = plain_rows[i:i + 4]
                        conds = self.driver.encode_text(te, [r[2] for r in rows])
                        if len(conds) != len(rows):
                            raise ValueError("Text encoder returned the wrong number of captions")
                        for row, cond in zip(rows, conds):
                            write(row, cond)
                    # Tag variants are text-only and cannot represent vision references.
                    for k, item, cap, _ in plain_rows:
                        ts = tag_sets.get(k)
                        if ts is None:
                            continue
                        try:
                            from fizgig.dataset.tag_variants import write_variants
                            write_variants(lambda caps: self.driver.encode_text(te, caps),
                                           item.text_encoder_output_cache_path, cap, ts, arch=desc.arch_id,
                                           compact=getattr(desc, "text_cache_trim", False))
                        except Exception:
                            logger.warning("[caption-fix] could not rebuild tag variants of %s", k, exc_info=True)
                finally:
                    self.driver.unload_text_encoder(te)
                    del te
                    gc.collect()
                    torch.cuda.empty_cache()

            if paired_rows:
                if not getattr(self.driver, "supports_references", False):
                    raise ValueError("Reference-aware captions cannot refresh with this driver")
                ref_te = self.driver.load_reference_text_encoder(self.te_path, device)
                try:
                    for row in paired_rows:
                        k, item, cap, _ = row
                        source = self._source_for_key(k)
                        refs = (getattr(item, "control_content", None) or
                                self._reference_images(k, source, tuple(item.bucket_size)))
                        cond = self.driver.encode_text_with_references(ref_te, [cap], [refs])[0]
                        write(row, cond, cache._ref_sizes(type("References", (), {"control_content": refs})()))
                finally:
                    self.driver.unload_text_encoder(ref_te)
                    del ref_te
                    gc.collect()
                    torch.cuda.empty_cache()
            ok = True
        finally:
            if parkable:
                quant.move(dit, device)
        if not ok:     # already-written AI captions re-queue as if manual: no need to regenerate them
            _requeue(path, processing, {**updates, **{k: c for k, _, c, a in todo if a > 0}})
            return
        for k, _, _, attempt in todo:
            if attempt > 0:
                self.recaptioned[k] = max(self.recaptioned.get(k, 0), attempt)
        if self.watch is not None:
            for k, _, _, attempt in todo:
                self.watch.reset_key(k)
            for k, _, _, attempt in todo:
                if attempt >= 2:
                    self.watch.mark_incorrigible(k)
        _ack(self.output_dir, todo, epoch)
        _remove(processing)
        logger.info(f"[caption-fix] {len(todo)} caption(s) re-encoded - next epoch trains on the fixed text. "
                    f"Loss-watch history reset for: " + ", ".join(k for k, _, _, _ in todo))

    def _recaption(self, auto, items, device):
        """Caption the stuck images with the shared captioner (Krea 2's Qwen3-VL-4B, loaded for the call and freed
        before the family's encoder loads). Writes each .txt and returns (key, item, caption, attempt) rows to
        re-encode. A captioner that does not fit the card turns auto-recaption off for the run.

        Only this model, loaded directly: the Captions tab's other captioners (fizgig/captioners/) never load in a run,
        so the training memory plans only ever fit around Qwen3-VL (the rule in fizgig/captioners/__init__.py)."""
        try:
            from fizgig.krea2.embedder import generate_caption
            from fizgig.krea2.utils import load_krea2_text_encoder
            enc = load_krea2_text_encoder(self.captioner_path, dtype=torch.bfloat16, device=device)

            def caption(img, attempt):
                instr = (self.instr_detailed if attempt >= 2 else self.instr) or None
                # 160 tokens on the first try: room for the Detailed and Style tasks (a caption ends where it ends)
                return generate_caption(enc, img, detailed=attempt >= 2, instruction=instr, max_new_tokens=160)
        except torch.OutOfMemoryError:
            logger.warning("[auto-recaption] off for the rest of this run: the captioner does not fit this card's "
                           "VRAM. Training continues; manual caption edits still apply.")
            self.auto_recaption = False
            gc.collect()
            torch.cuda.empty_cache()
            return []
        except Exception:
            logger.warning("[auto-recaption] off for the rest of this run: the captioner could not load. "
                           "Training continues; manual caption edits still apply.", exc_info=True)
            self.auto_recaption = False
            return []
        rows = []
        try:
            for k, img, attempt in auto:
                try:
                    cap = caption(img, attempt)
                    prefix = self.watch.where(k)[0] or ""
                    trig = self.triggers.get(prefix, "") if prefix else self.trigger_word   # the picture's folder's
                    if trig:
                        cap = (f"{cap}, {trig}" if str(self.trigger_position) == "end" else f"{trig}, {cap}")
                    try:
                        _p, folder, name = self.watch.where(k)
                        with open(os.path.join(folder, name + self.caption_ext), "w", encoding="utf-8") as f:
                            f.write(cap)
                    except OSError:
                        logger.warning(f"[auto-recaption] could not write the .txt for {k} - this run is fixed, "
                                       "a future re-cache will use the old caption")
                    rows.append((k, items[k], cap, attempt))
                    logger.info(f"[auto-recaption] {k} (attempt {attempt}/2"
                                f"{', detailed' if attempt >= 2 else ''}): \"{cap}\"")
                except Exception:
                    logger.warning(f"[auto-recaption] captioning failed for {k} - retry next "
                                   f"boundary", exc_info=True)
        finally:
            if hasattr(enc, "to"):
                enc.to("cpu")
            caption = enc = None
            gc.collect()
            torch.cuda.empty_cache()
        return rows


def _warmup_keys(keys, look, prefix):
    """One folder's look outliers among the training keys (its Look Filter scores, named as in that folder)."""
    cut = look.get("cutoff")
    # Scores carry the embedder that made them (ArcFace / DINOv2 - different scales); the cutoff was computed on that
    # same scale when the scan ran, so it is used as is. Files from before the face backends have no "embedder" and
    # are ArcFace.
    emb = look.get("embedder_label") or look.get("embedder") or "ArcFace"
    if cut is None:
        logger.warning("[look-warmup] no cutoff in fizgig_look_scores.json (too few scored faces) - warm-up off")
        return set()
    outliers = {prefix + k for k, v in (look.get("scores") or {}).items() if isinstance(v, (int, float)) and v < float(cut)}
    gone, use = sorted(outliers - keys), outliers & keys
    if gone:
        logger.info(f"[look-warmup] {len(gone)} scored outlier(s) not in the dataset (moved via the Look "
                    f"Filter) - skipped")
    if use:
        logger.info(f"[look-warmup] {len(use)} look-outlier image(s) on LR warm-up x0.4 -> x1.0 "
                    f"({emb} scores, cutoff {float(cut):.2f})")
    return use


def _cleanup(output_dir):
    """Fresh run: clear the previous run's Problem Images artifacts; rotate (not delete) the research JSONL."""
    ll = os.path.join(output_dir, "loss_log")
    for f in ("problem_images.json", "problem_images.json.tmp", "caption_updates_applied.json",
              "caption_updates_applied.json.tmp", "caption_updates.json", "caption_updates.json.processing",
               "image_updates.json", "image_updates.json.processing"):
        _remove(os.path.join(ll, f))
    jl = os.path.join(ll, "per_image_loss.jsonl")
    if os.path.exists(jl):
        try:
            os.replace(jl, jl + "." + time.strftime("%Y%m%d%H%M%S") + ".bak")
        except OSError:
            pass


def _remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass                     # the GUI may hold it open; it is consumed next boundary


def _recover_processing(path, pending=None):
    """Recover a claimed queue after interrupted encoder work, preserving newer UI saves."""
    processing = path + ".processing"
    changes = {}
    if os.path.isfile(processing):
        try:
            with open(processing, encoding="utf-8") as f:
                changes = json.load(f)
            if not isinstance(changes, dict):
                raise ValueError("queue is not a mapping")
        except (OSError, ValueError):
            logger.warning("[live-dataset] unreadable claimed queue %s; keeping it for inspection", processing,
                           exc_info=True)
            return
    changes.update(pending or {})
    if changes:
        _requeue(path, processing, changes)
    elif os.path.isfile(processing):
        _remove(processing)


def _requeue(path, processing, updates):
    """Put a claimed batch back, merged with anything the GUI queued meanwhile (newer edits win)."""
    if not updates:
        _remove(processing)
        return
    try:
        newer = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                newer = json.load(f)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({**updates, **newer}, f, indent=2)
        os.replace(path + ".tmp", path)
        _remove(processing)
    except Exception:
        pass


def _ack(output_dir, todo, epoch):
    """Per-fix history per key (the GUI's row badge, and the resume replay's reset timeline)."""
    path = os.path.join(output_dir, "loss_log", "caption_updates_applied.json")
    applied = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                applied = {k: (v if isinstance(v, list) else [v]) for k, v in json.load(f).items()}
    except Exception:
        applied = {}
    for k, _, cap, attempt in todo:
        applied.setdefault(k, []).append({"epoch": epoch, "caption": cap, "attempt": attempt, "auto": attempt > 0})
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(applied, f, indent=2)
        os.replace(path + ".tmp", path)
    except Exception:
        logger.warning("[caption-fix] could not write caption_updates_applied.json", exc_info=True)
