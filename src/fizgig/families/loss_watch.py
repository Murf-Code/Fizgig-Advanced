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
            for bucket in bm.buckets.values():
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
            for bucket in bm.buckets.values():
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
                 captioner_path=None, recaption_task=None, recaption_task_detailed=None, recaption_with=None, vae_path=None):
        from fizgig.training.loss_logger import PerImageLossWatch, is_enabled as env_on
        self.output_dir, self.group, self.driver, self.te_path = output_dir, group, driver, te_path
        self.vae_path = vae_path
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
        if pending == getattr(self, "_unscheduled_last", None):
            return
        if pending:
            logger.warning("[live-dataset] %s input(s) exist outside this run's cached batch schedule "
                           "(examples: %s). They require a new cache/optimizer schedule and are NOT "
                           "being trained. Restart the run to include them.",
                           len(pending), ", ".join(pending[:12]))
        status_path = os.path.join(self.output_dir, "loss_log", "live_dataset_status.json")
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        payload = {"requires_restart": bool(pending), "unscheduled_count": len(pending),
                   "unscheduled_examples": pending[:40],
                   "reason": ("New or uncached media cannot be added safely to the active fixed training "
                              "schedule. Restart training after preparing the cache." if pending else "")}
        with open(status_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(status_path + ".tmp", status_path)
        self._unscheduled_last = pending

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
            for bucket in manager.buckets.values():
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

    def _source_state(self):
        """Current source-file fingerprints, using the same item keys as cached training."""
        seen, listings = {}, {}
        for key in _items(self.group):
            folder, stem, best = "", str(key).rsplit("/", 1)[-1], -1
            for prefix, directory in self.folders.items():
                if str(key).startswith(prefix) and len(prefix) > best:
                    folder, best = directory, len(prefix)
                    stem = str(key)[len(prefix):].rsplit("/", 1)[-1]
            if not folder:
                folder = self.image_dir or ""
            pic = _picture(folder, stem, listings) if folder else ""
            caption = os.path.splitext(pic)[0] + self.caption_ext if pic else os.path.join(folder, stem + self.caption_ext)
            def stamp(path):
                try:
                    st = os.stat(path)
                    return (os.path.realpath(path), st.st_size, st.st_mtime_ns)
                except OSError:
                    return ("", 0, 0)
            seen[str(key)] = (stamp(pic), stamp(caption), caption)
        return seen

    def _scan_sources(self):
        """Detect file edits independently of how they were made; no GPU work here."""
        previous = getattr(self, "_source_snapshot", {})
        latest = self._source_state()
        changed_images, changed_text = {}, {}
        for key, (picture, cap, caption_path) in latest.items():
            before = previous.get(key)
            if before is None:
                continue
            if picture != before[0]:
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
               for k in keys_to_check):
            return True
        if self.watch is None or self._reg_only(keys):
            return False
        if isinstance(keys, (list, tuple)) and len(keys) > 1:
            return all(self.watch.is_excluded([str(k)]) for k in keys)
        return self.watch.is_excluded(keys)

    def multiplier(self, batch):
        keys = batch.get("item_keys")
        blocked = self.removed_keys | getattr(self, "invalid_mask_keys", set())
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
        blocked = self.removed_keys | getattr(self, "invalid_mask_keys", set())
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
            self._report_unscheduled_inputs()
        except Exception:
            logger.warning("[live-dataset] could not report new source media", exc_info=True)
        try:
            self._images(epoch, dit, device, parkable, changed_images)
        except Exception:
            _recover_processing(os.path.join(self.output_dir, "loss_log", "image_updates.json"), changed_images)
            logger.warning("[image-fix] image refresh failed this boundary - training continues", exc_info=True)
        try:
            self._refresh_masks(changed_images)
        except Exception:
            logger.warning("[live-mask] mask refresh failed; old mask state retained", exc_info=True)
        try:
            self._captions(epoch, dit, device, parkable, changed_text)
        except Exception:
            _recover_processing(os.path.join(self.output_dir, "loss_log", "caption_updates.json"), changed_text)
            logger.warning("[caption-fix] caption repair failed this boundary - training continues", exc_info=True)

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
            source = _picture(folder, stem, listings)
            if action == "removed" or not source:
                self.removed_keys.add(str(k))
                logger.info("[image-fix] %s excluded from subsequent epochs", k)
            elif action == "updated":
                # Never train on stale cached pixels if the VAE cannot refresh this item.
                self.removed_keys.add(str(k))
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
                        # The still-image update path must not silently discard edit reference latents.
                        from safetensors import safe_open
                        with safe_open(item.latent_cache_path, framework="pt", device="cpu") as existing:
                            if any(key.startswith("latent_control_") for key in existing.keys()):
                                raise ValueError("Paired-reference cache requires a full recache")
                        # The old bucket is fixed for the active run, even if an edit changes aspect.
                        with torch.no_grad():
                            upright = open_upright(source)
                            new_source_size = upright.size
                            image = resize_image_to_bucket(upright, tuple(item.bucket_size))
                            encoded = self.driver.encode_images(vae, [image])[0]
                            import copy
                            scratch = copy.copy(item)
                            scratch.original_size = new_source_size
                            scratch.latent_cache_path = item.latent_cache_path + ".live-update"
                            try:
                                cache.save_latents(self.driver.description, scratch, encoded)
                                os.replace(scratch.latent_cache_path, item.latent_cache_path)
                                item.original_size = new_source_size
                            finally:
                                _remove(scratch.latent_cache_path)
                        self.removed_keys.discard(k)
                        logger.info("[image-fix] epoch %s: %s latent re-encoded at %s", epoch, k, item.bucket_size)
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
        updates = {**(detected or {}), **updates}
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
            te = self.driver.load_text_encoder(self.te_path, device)
            try:
                desc = self.driver.description
                for i in range(0, len(todo), 4):
                    chunk = todo[i:i + 4]
                    for (_, item, cap, _), cond in zip(chunk, self.driver.encode_text(te, [c for _, _, c, _ in chunk])):
                        import copy
                        scratch = copy.copy(item)
                        scratch.caption = cap
                        scratch.text_encoder_output_cache_path = item.text_encoder_output_cache_path + ".live-update"
                        try:
                            cache.save_cond(desc, scratch, cond, encoder=cache.encoder_stamp(self.te_path))
                            os.replace(scratch.text_encoder_output_cache_path, item.text_encoder_output_cache_path)
                        finally:
                            _remove(scratch.text_encoder_output_cache_path)
                        item.caption = cap
                ok = True
                # tag shuffle / caption dropout: the old caption's variants no longer apply (the loader already
                # ignores them - their source caption differs); the new caption's are made now, with the encoder here
                tag_sets = _tag_settings(self.group)
                for k, item, cap, _ in todo:
                    ts = tag_sets.get(k)
                    if ts is None:
                        continue
                    try:
                        from fizgig.dataset.tag_variants import write_variants
                        write_variants(lambda caps: self.driver.encode_text(te, caps),
                                       item.text_encoder_output_cache_path, cap, ts, arch=desc.arch_id,
                                       compact=getattr(desc, "text_cache_trim", False))
                    except Exception:
                        logger.warning(f"[caption-fix] could not remake the tag variants of {k} - it trains on its "
                                       "new caption as written", exc_info=True)
            finally:
                self.driver.unload_text_encoder(te)
                del te
                gc.collect()
                torch.cuda.empty_cache()
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
