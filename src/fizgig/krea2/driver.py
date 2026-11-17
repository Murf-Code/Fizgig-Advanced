"""Krea 2 driver for Fizgig's standard layer (families/driver.py).

The original Krea 2 trainer's rules behind the FamilyDriver interface, so the generic cache / train / preview code
trains Krea 2 the way src/fizgig/krea2/trainer.py does:
* latents: Qwen-Image VAE (16 channels, 8x), normalised with its latents_mean / std, (16, h, w)
* conditioning: Qwen3-VL-4B multi-layer hidden stack + validity mask, padded to max_length
  -> {"hidden_states": (seq, layers, dim), "attention_mask": (seq,)}; valid tokens are gathered per forward
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0; t logit-normal with the resolution-
  dependent shift exp(mu), mu linear in the image-token count (0.5 at 256 to 1.15 at 6400); unweighted MSE
* sampling: Euler on the same schedule (sampling.timesteps), mu pinned at 1.15 for the turbo previews
* LoRA: every Linear in the DiT (264: the 28 blocks, the text-fusion stack, the input / timestep / output layers);
  quantisation (INT8 / NF4) covers the blocks' Linears only, as the original's fp8 / int8 / NF4 paths
"""
import math
import re

import numpy as np

import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver, picture_losses, prior_base, weighted_loss, with_prior
from fizgig.training import timestep_spread as TS

DTYPE = torch.bfloat16
_BLOCK_MODULES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
_IO = ("first", "tmlp.0", "tmlp.2", "txtmlp.1", "txtmlp.3", "tproj.1", "last.linear")


def _mu(num_img_tokens):
    """musubi's get_lin_function through (256, 0.5) and (6400, 1.15): the flow shift's log for this token count."""
    m = (1.15 - 0.5) / (6400 - 256)
    return m * num_img_tokens + (0.5 - m * 256)


class Krea2Driver(FamilyDriver):

    # INT8 scales computed in bf16, exactly as the original trainer's apply_int8_training quantises Krea 2 - so a
    # driver run starts from the same INT8 weights as a Krea 2 run.
    int8_fp32_scales = False

    def ft_spec(self, dit):
        # the original's component windows (balanced: attention ~30% of a block, the MLP split in its three
        # matrices), text fusion trained throughout, and its measured NF4-trunk planner calibration
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="blocks", components=("attn", "mlp.gate", "mlp.up", "mlp.down"),
                      always_on=("txtfusion",), overhead_gb=9.5, trunk_gb_per_block=0.217,
                      stream_base_gb=3.2,      # measured 30 Sep: 2.7 GB worst window (12 GB card) + 0.5 slack
                      calib_mp=0.25, act_gb_per_mp=3.2)   # 0.25 -> 0.98 MP measured +2.2 GB (3.0 GB/MP)

    def compile_targets(self, dit):
        return dit.blocks

    def compile_blocks(self, dit, boundary="inside", blocks_to_swap=0):
        """The shared compile, then the attention stops trimming to each sequence's own length (it would recompile
        every block for every new length; krea2/attention.py COMPILED_BLOCKS)."""
        super().compile_blocks(dit, boundary=boundary, blocks_to_swap=blocks_to_swap)
        import torch
        from fizgig.krea2 import attention as _attn
        if any(isinstance(b, torch._dynamo.eval_frame.OptimizedModule)
               or isinstance(getattr(b, "block", None), torch._dynamo.eval_frame.OptimizedModule)
               for b in self.compile_targets(dit)):
            _attn.COMPILED_BLOCKS = True

    def compile_plan(self, mode, total_steps, precision, blocks_to_swap, mp=0.25):
        """Krea 2's own measured rule (utils/capabilities: should_compile / compile_boundary) - its compiled-memory
        figures place the checkpoint inside the graph where it fits and outside where it does not."""
        from fizgig.utils.capabilities import compile_boundary, should_compile
        q4, q8 = precision == "nf4", ("int8" if precision == "int8" else "")
        if mode == "auto":
            return should_compile(total_steps, q4, q8, blocks_to_swap, mp=mp)
        if mode == "outside":
            return "outside", ""
        b = compile_boundary(q4, q8, mp=mp)
        return b, ("on: inside-the-graph won't fit at this token load - compiling with the checkpoint OUTSIDE the "
                   "region instead." if b == "outside" else "")

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.krea2.utils import load_krea2_dit
        dit = load_krea2_dit(path, device=device, dtype=DTYPE, fp8_scaled=False, loading_device=device)
        return dit.eval().requires_grad_(False)

    def stream_dit(self, path):
        """For a short-RAM load (families/quant.py _stream_load): the model on meta, the open file and the dtype
        load_dit reads it at. None for a pre-quantised fp8 file, which load_dit keeps fp8. An int8 / older fp8
        ComfyUI build is decoded a tensor at a time as it is read."""
        from fizgig.krea2.model import SingleStreamDiT
        from fizgig.krea2.utils import ComfyDecodeReader, dit_layout, single_mmdit_large_wide
        from fizgig.utils.safetensors import warm_file_cache
        if dit_layout(path) == "fp8":
            return None
        warm_file_cache(path)
        with torch.device("meta"):
            dit = SingleStreamDiT(single_mmdit_large_wide, attn_mode="torch", split_attn=False)
        return dit, ComfyDecodeReader(path, dit.state_dict().keys()), DTYPE

    def dit_gb(self, path):
        """An int8 / older fp8 ComfyUI build is decoded to bf16 at load: about twice its file."""
        from fizgig.krea2.utils import dit_layout
        size = super().dit_gb(path)
        return size * 2 if dit_layout(path) == "decode" else size

    def max_blocks_to_swap(self, dit=None):
        return (len(dit.blocks) if dit is not None else self.description.n_blocks) - 2

    send_only_block_swap = True

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True, send_only=False):
        from fizgig.krea2.offloading import BlockSwapConfig
        dit.enable_block_swap(num_blocks, BlockSwapConfig(torch.device(device), supports_backward=supports_backward,
                                                          h2d_only=bool(send_only)))
        dit.move_to_device_except_swap_blocks(torch.device(device))
        dit.switch_block_swap_for_training()

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        from fizgig.krea2.vae_loader import load_vae
        vae = load_vae(path, input_channels=3, device=device, disable_mmap=True)
        return vae.to(device)

    def load_text_encoder(self, path, device):
        from fizgig.krea2.utils import load_krea2_text_encoder
        return load_krea2_text_encoder(path, dtype=DTYPE, device=device)

    def unload_text_encoder(self, te):
        try:
            te.to("cpu")
        except Exception:
            pass
        del te
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def enable_gradient_checkpointing(self, dit, on=True):
        if on:
            dit.enable_gradient_checkpointing()
        else:
            dit.disable_gradient_checkpointing()

    # ---- encoding (the original's caching.py, per item) -----------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).unsqueeze(2) / 127.5 - 1.0                      # (B, C, 1, H, W)
        z = vae.encode_pixels_to_latents(x.to(vae.device, dtype=vae.dtype))
        return [(zi.squeeze(1) if zi.dim() == 4 else zi).cpu() for zi in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        """One caption per forward, as the original cache script: a batched forward rounds a bf16 step or two
        differently, so a caption's conditioning would depend on which captions shared its batch."""
        out = []
        for cap in captions:
            hiddens, mask = te([cap])                                            # (1, seq, L, D), (1, seq)
            out.append({"hidden_states": hiddens[0].cpu(), "attention_mask": mask[0].cpu().to(torch.bool)})
        return out

    def encode_text_with_image(self, te, captions, image, megapixels=1.0):
        """The original's preview reference: the image through Qwen3-VL's vision path (1 MP for training previews;
        the workbench passes its MP setting), one caption per forward."""
        out = []
        for cap in captions:
            hiddens, mask = te([cap], images=[[image]], vision_megapixels=float(megapixels))
            out.append({"hidden_states": hiddens[0].cpu(), "attention_mask": mask[0].cpu().to(torch.bool)})
        return out

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(num_img_tokens, generator, min_t=0.0, max_t=1.0, spread="default"):
        """sample_krea2_timesteps: logit-normal base, shift exp(mu), rescaled INTO the window (never clamped). spread:
        the Timestep focus preset (fizgig.training.timestep_spread; "default" = exactly the above)."""
        shift = float(np.exp(_mu(num_img_tokens)))
        t = TS.logit_normal(torch.randn(1, generator=generator), spread, min_t, max_t)
        t = (t * shift) / (1.0 + (shift - 1.0) * t)
        if min_t > 0.0 or max_t < 1.0:
            lo, hi = max(0.0, float(min_t)), min(1.0, float(max_t))
            t = lo + t * max(hi - lo, 1e-6)
        return t

    @staticmethod
    def _text(cond, device):
        from fizgig.krea2.sampling import gather_valid_text
        h, m = cond["hidden_states"], cond["attention_mask"]
        if h.dim() == 3:                      # one caption as encoded (previews); training batches carry a batch dim
            h, m = h[None], m[None]
        return gather_valid_text(h.to(device=device, dtype=DTYPE), m.to(device).bool())

    def _forward(self, dit, noised, t, cond):
        """The DiT's velocity for a noised latent (B, 16, h, w) at t (B,), image tokens only."""
        from fizgig.krea2.sampling import prepare
        device = noised.device
        txt, txtmask = self._text(cond, device)
        img, pos, mask = prepare(noised, txt.shape[1], dit.config.patch, txtmask)
        with torch.autocast(device_type=device.type, dtype=DTYPE):
            return dit(img=img, context=txt, t=t.to(DTYPE), pos=pos, mask=mask)

    @property
    def selective_global_replay(self):
        method = str(getattr(self, "options", {}).get("training_method", "") or "").lower()
        return method in ("selective_global_replay", "selective_global_replay_h2d")

    @property
    def residual_path_hybrid(self):
        method = str(getattr(self, "options", {}).get("training_method", "") or "").lower()
        return method in ("residual_path_hybrid", "residual_path_hybrid_h2d",
                          "residual_path_hybrid_h2d_selective_a")

    @property
    def residual_path_selective_a(self):
        """Whether this hybrid variant periodically includes approximate LoRA A-factor gradients."""
        method = str(getattr(self, "options", {}).get("training_method", "") or "").lower()
        return method == "residual_path_hybrid_h2d_selective_a"

    @property
    def h2d_block_swap(self):
        """The H2D-labelled choices use the existing host-to-device-only Krea 2 block offloader."""
        method = str(getattr(self, "options", {}).get("training_method", "") or "").lower()
        return method in ("block_local_h2d", "selective_global_replay_h2d", "residual_path_hybrid_h2d",
                          "residual_path_hybrid_h2d_selective_a")

    @property
    def block_local_flow(self):
        """Experimental Krea 2 modes that train only the main blocks with a manual backward path."""
        method = str(getattr(self, "options", {}).get("training_method", "") or "").lower()
        return method in ("block_local", "selective_global_replay",
                          "block_local_h2d", "selective_global_replay_h2d",
                          "residual_path_hybrid", "residual_path_hybrid_h2d",
                          "residual_path_hybrid_h2d_selective_a")

    @property
    def manual_loss_backward(self):
        """The generic trainer must call backward_training_loss after applying its per-image scale."""
        return self.block_local_flow

    def configure_residual_path_hybrid(self, dit, net, *, exact_every=8, approx_lr_multiplier=2.0):
        """Configure identity-path credit for Krea 2's residual output projections.

        Approximate updates train the LoRA B factors on attn.wo and mlp.down. In the selective-A variant,
        original inputs are additionally staged to pinned host RAM every fourth approximate update and used to
        calculate A-factor gradients. A readout-only gradient is passed along residual identity paths, gated at
        each residual branch. Every ``exact_every``-th optimizer update uses exact global-loss replay.
        """
        from fizgig.families.lora import LoRALinear, TRAINABLE

        if not self.residual_path_hybrid:
            raise RuntimeError("Residual-path hybrid can only be configured for Residual-path hybrid training.")
        exact_every = int(exact_every)
        if exact_every < 2:
            raise ValueError("Residual-path hybrid exact calibration interval must be at least 2 optimizer updates.")
        approx_lr_multiplier = float(approx_lr_multiplier)
        if not math.isfinite(approx_lr_multiplier) or approx_lr_multiplier <= 0.0:
            raise ValueError("Residual-path approximate-update LR multiplier must be a finite number greater than zero.")

        by_block = {}
        for index, block in enumerate(dit.blocks):
            values = {}
            for suffix in ("attn.wo", "mlp.down"):
                module = block
                for part in suffix.split("."):
                    module = getattr(module, part)
                if not isinstance(module, LoRALinear) or TRAINABLE not in module.adapters:
                    raise RuntimeError("Residual-path hybrid needs a trainable standard LoRA on every block's "
                                       f"{suffix} projection; missing blocks.{index}.{suffix}.")
                adapter = module.adapters[TRAINABLE]
                if not isinstance(adapter, torch.nn.Sequential) or len(adapter) != 2:
                    raise RuntimeError("Residual-path hybrid currently supports standard two-factor LoRA only; "
                                       f"unsupported adapter at blocks.{index}.{suffix}.")
                values[suffix] = module
            by_block[index] = values

        self._residual_path_modules = by_block
        self._residual_path_exact_every = exact_every
        self._residual_path_approx_lr_multiplier = approx_lr_multiplier
        self._residual_path_optimizer_steps = 0
        self._residual_path_approx_steps = 0
        self._residual_path_a_every = 4 if self.residual_path_selective_a else 0
        self._residual_path_last_step_exact = False
        self._residual_path_d2h_stream = None

    @property
    def residual_path_exact_due(self):
        """Whether the next optimizer update is the scheduled exact-gradient calibration update."""
        if not self.residual_path_hybrid or not hasattr(self, "_residual_path_optimizer_steps"):
            return False
        return (self._residual_path_optimizer_steps + 1) % self._residual_path_exact_every == 0

    @property
    def residual_path_update_lr_multiplier(self):
        """LR multiplier for the next update: approximate-only; exact calibration always uses 1.0."""
        if not self.residual_path_hybrid or not hasattr(self, "_residual_path_approx_lr_multiplier"):
            return 1.0
        return 1.0 if self.residual_path_exact_due else self._residual_path_approx_lr_multiplier

    @property
    def residual_path_a_update_due(self):
        """True on every fourth approximate update in the selective-A variant (exact updates do not count)."""
        return bool(self.residual_path_selective_a and self._residual_path_a_every > 0
                    and not self.residual_path_exact_due
                    and (self._residual_path_approx_steps + 1) % self._residual_path_a_every == 0)

    def finish_residual_path_optimizer_step(self):
        """Advance the optimizer and approximate-update counters after a real optimizer update."""
        if not self.residual_path_hybrid or not hasattr(self, "_residual_path_optimizer_steps"):
            return None
        if not self._residual_path_last_step_exact:
            self._residual_path_approx_steps += 1
        self._residual_path_optimizer_steps += 1
        upcoming = self._residual_path_optimizer_steps + 1
        upcoming_exact = upcoming % self._residual_path_exact_every == 0
        upcoming_a = (self.residual_path_selective_a and not upcoming_exact
                      and (self._residual_path_approx_steps + 1) % max(1, self._residual_path_a_every) == 0)
        return {"optimizer_steps": self._residual_path_optimizer_steps,
                "approximate_steps": self._residual_path_approx_steps,
                "next_exact": upcoming_exact, "next_a_update": upcoming_a,
                "exact_every": self._residual_path_exact_every}

    def residual_path_state_dict(self):
        if not self.residual_path_hybrid or not hasattr(self, "_residual_path_optimizer_steps"):
            return None
        return {"version": 3, "exact_every": self._residual_path_exact_every,
                "approx_lr_multiplier": self._residual_path_approx_lr_multiplier,
                "optimizer_steps": self._residual_path_optimizer_steps,
                "approximate_steps": self._residual_path_approx_steps,
                "selective_a_every": self._residual_path_a_every}

    def load_residual_path_state_dict(self, state):
        # v1 predates the LR multiplier; v2 predates selective A updates. Both remain resumable.
        if not self.residual_path_hybrid or not isinstance(state, dict) or state.get("version") not in (1, 2, 3):
            raise RuntimeError("The saved Residual-path hybrid state is missing or unsupported.")
        if int(state.get("exact_every", -1)) != self._residual_path_exact_every:
            raise RuntimeError("Residual-path hybrid exact-calibration settings differ from the saved run state.")
        saved_multiplier = state.get("approx_lr_multiplier")
        if saved_multiplier is not None and not math.isclose(
                float(saved_multiplier), self._residual_path_approx_lr_multiplier, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError("Residual-path hybrid approximate-update LR multiplier differs from the saved run state.")
        saved_a_every = state.get("selective_a_every")
        if saved_a_every is not None and int(saved_a_every) != self._residual_path_a_every:
            raise RuntimeError("Residual-path hybrid selective-A cadence differs from the saved run state.")
        self._residual_path_optimizer_steps = max(0, int(state.get("optimizer_steps", 0)))
        if "approximate_steps" in state:
            self._residual_path_approx_steps = max(0, int(state.get("approximate_steps", 0)))
        else:
            # Every Nth optimizer update is exact; older states did not track approximate updates separately.
            steps = self._residual_path_optimizer_steps
            self._residual_path_approx_steps = max(0, steps - (steps // self._residual_path_exact_every))

    def _residual_path_training_loss(self, dit, latents, noise, t, cond):
        """One no-graph transformer forward with low-rank captures for identity-path credit assignment.

        The only autograd operation is the small final readout derivative with respect to its input. No
        transformer block is recorded or replayed on these updates.
        """
        from fizgig.krea2.attention import AttentionParams
        from fizgig.krea2.model import temb
        from fizgig.krea2.sampling import patchify_block, prepare
        from fizgig.families.lora import TRAINABLE

        if getattr(self, "_pending_block_local", None) is not None:
            raise RuntimeError("A previous residual-path training loss has not been backpropagated.")
        if not hasattr(self, "_residual_path_modules"):
            raise RuntimeError("Residual-path hybrid was not configured before training.")

        device = latents.device
        patch = dit.config.patch
        latent = latents.to(dtype=DTYPE)
        noise = noise.to(device=device, dtype=DTYPE)
        t = t.to(device)
        t_ = t.view(-1, 1, 1, 1).to(DTYPE)
        noised = (1.0 - t_) * latent + t_ * noise
        target, _, _ = patchify_block(noise - latent, patch)
        txt, txtmask = self._text(cond, device)
        img, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)

        captures = {}
        update_a = self.residual_path_a_update_due
        self._residual_path_last_step_exact = False
        hook_handles = []
        offloader = getattr(dit, "offloader", None) if dit.blocks_to_swap else None
        h2d_stream = False
        if offloader is not None and hasattr(offloader, "set_forward_only"):
            from fizgig.krea2.offloading import LoRAStreamOffloader
            h2d_stream = isinstance(offloader, LoRAStreamOffloader)
            # No backward hooks run on these approximate updates. Switch the swapper to its forward-only
            # rotation so the ring is wrapped back to the first blocks for the next training step.
            offloader.set_forward_only(True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=DTYPE):
            img = dit.first(img)
            t_emb = dit.tmlp(temb(t.to(DTYPE), dit.config.tdim, device=img.device, dtype=img.dtype))
            tvec = dit.tproj(t_emb)
            imglen = img.shape[1]
            txtmask = mask[:, imglen:]
            txt_attn_params_nomask = AttentionParams.create_attention_params_from_mask(
                dit.attn_mode, dit.split_attn, 0, None)
            txt_attn_params = AttentionParams.create_attention_params_from_mask(
                dit.attn_mode, dit.split_attn, 0, txtmask)
            context = dit.txtfusion(txt, txt_attn_params_nomask, txt_attn_params)
            context = dit.txtmlp(context)
            combined = torch.cat((img, context), dim=1)
            fulllen = combined.shape[1]
            padlen = (-fulllen) % 256
            if padlen:
                combined = F.pad(combined, (0, 0, 0, padlen))
                pos = F.pad(pos, (0, 0, 0, padlen))
                txtmask = F.pad(txtmask, (0, padlen), value=False)
            attn_params = AttentionParams.create_attention_params_from_mask(
                dit.attn_mode, dit.split_attn, imglen, txtmask)
            freqs = dit.posemb(pos)

            try:
                for index, block in enumerate(dit.blocks):
                    if dit.blocks_to_swap:
                        dit.offloader.wait_for_block(index)

                    _, _, pre_gate, _, _, post_gate = block.mod(tvec)
                    captures[index] = {
                        "attn_gate": pre_gate.detach(),
                        "mlp_gate": post_gate.detach(),
                        "projections": {},
                    }

                    for suffix, module in self._residual_path_modules[index].items():
                        def _capture_input(layer, inputs, *, _index=index, _suffix=suffix):
                            x = inputs[0].detach()
                            adapter = layer.adapters[TRAINABLE]
                            a, b = adapter[0], adapter[1]
                            # Match LoRALinear's low-precision forward: both factor matmuls use x's dtype.
                            ax = F.linear(x, a.weight.to(dtype=x.dtype))
                            projection_capture = {
                                "ax": ax.detach(), "a": a, "b": b,
                                "scale": float(layer.scales.get(TRAINABLE, 0.0)),
                            }
                            if update_a:
                                # Full-width inputs are too expensive to retain on the GPU across all 28 blocks.
                                # Stage them asynchronously into pinned host RAM, then bring one back at a time
                                # for the A-gradient GEMM after the final readout credit is known.
                                if x.device.type == "cuda":
                                    stream = getattr(self, "_residual_path_d2h_stream", None)
                                    if stream is None:
                                        stream = torch.cuda.Stream(device=x.device)
                                        self._residual_path_d2h_stream = stream
                                    ready = torch.cuda.Event()
                                    ready.record(torch.cuda.current_stream(x.device))
                                    try:
                                        x_host = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
                                    except RuntimeError:
                                        # Some Windows/PyTorch setups impose a pinned-host-memory quota. Fall back
                                        # to a synchronous pageable copy rather than failing the training run.
                                        stream.synchronize()
                                        x_host = x.to("cpu")
                                        self._residual_path_pin_fallback = True
                                    else:
                                        with torch.cuda.stream(stream):
                                            stream.wait_event(ready)
                                            x_host.copy_(x, non_blocking=True)
                                        x.record_stream(stream)
                                    projection_capture["x_host"] = x_host
                                else:
                                    projection_capture["x_host"] = x.detach().to("cpu")
                            captures[_index]["projections"][_suffix] = projection_capture
                        hook_handles.append(module.register_forward_pre_hook(_capture_input))

                    combined = block(combined, tvec, freqs, attn_params).detach()
                    # The next block's hooks are installed only for its own forward.
                    for handle in hook_handles[-2:]:
                        handle.remove()
                    if dit.blocks_to_swap:
                        dit.offloader.submit_move_blocks_forward(dit.blocks, index)
            finally:
                for handle in hook_handles:
                    handle.remove()
                if offloader is not None and hasattr(offloader, "set_forward_only"):
                    offloader.set_forward_only(False)
                    # The H2D ring explicitly preloads its first window at the end of a forward-only pass.
                    # The classic offloader instead needs its ordinary device layout restored here.
                    if not h2d_stream:
                        dit.prepare_block_swap_before_forward()

        if len(captures) != len(dit.blocks) or any(len(v["projections"]) != 2 for v in captures.values()):
            raise RuntimeError("Residual-path hybrid did not capture both output projections for every Krea 2 block.")

        # The final readout's input derivative is exact, but the transformer-wide derivative is not computed.
        # It is then copied through the residual identity paths as the approximate block credit signal.
        final_hidden = combined.detach().requires_grad_(True)
        with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=DTYPE):
            pred = dit.last(final_hidden, t_emb)[:, :imglen, :]
            loss = F.mse_loss(pred.float(), target.float())
            readout_grad = torch.autograd.grad(loss, final_hidden, retain_graph=False, create_graph=False)[0]

        self._pending_block_local = {
            "dit": dit,
            "phase": "residual_path_approx",
            "captures": captures,
            "update_a": update_a,
            "readout_grad": readout_grad.detach(),
            "device": device,
        }
        return loss.detach()

    def configure_selective_global_replay(self, dit, net, *, warmup_updates=8, reactivate_every=8, score_decay=0.5):
        """Track the direction of each block's actual optimizer update and selectively compute adapter gradients.

        All blocks participate in the initial probe window. At each later probe update, all blocks are briefly
        reactivated and each block's displacement since its previous probe is scored against the fresh global-loss
        gradient at the probe, before the probe's own step: cos(-g_now, theta_now - theta_last_probe). A negative
        score means the block's recent updates carried it past where the loss now wants it (uphill); such blocks
        are skipped between probes. (Scoring a step against the gradient that produced it is always positive for
        SGD, Adam or Muon, so it can never find an uphill block.) The frozen block's input gradient is still
        propagated during replay, preserving the exact chain rule for earlier blocks.
        """
        from fizgig.families.lora import TRAINABLE

        if not self.selective_global_replay:
            raise RuntimeError("Selective replay can only be configured for Selective global replay training.")
        n = len(dit.blocks)
        by_block = {i: [] for i in range(n)}
        for full, wrapped in net.wrapped.items():
            if TRAINABLE not in wrapped.adapters:
                continue
            name = str(full)
            if not name.startswith("blocks."):
                continue
            parts = name.split(".", 2)
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            index = int(parts[1])
            if index in by_block:
                by_block[index].extend(list(wrapped.adapters[TRAINABLE].parameters()))
        missing = [i for i, values in by_block.items() if not values]
        if missing:
            raise RuntimeError("Selective global replay could not find trainable LoRA parameters in transformer "
                               f"blocks: {missing}.")

        self._selective_params_by_block = by_block
        self._selective_block_count = n
        self._selective_warmup_updates = max(1, int(warmup_updates))
        self._selective_reactivate_every = max(1, int(reactivate_every))
        self._selective_score_decay = min(0.99, max(0.0, float(score_decay)))
        self._selective_optimizer_steps = 0
        self._selective_scores = [None] * n
        self._selective_active = set(range(n))
        # each block's weights at its last probe (CPU), the start of the displacement the next probe scores
        self._selective_anchor = {i: self._selective_flat(by_block[i]) for i in range(n)}
        self._set_selective_active_parameters()

    @staticmethod
    def _selective_flat(params):
        return torch.cat([p.detach().float().reshape(-1) for p in params]).cpu()

    def _set_selective_active_parameters(self):
        active = getattr(self, "_selective_active", set())
        for index, parameters in getattr(self, "_selective_params_by_block", {}).items():
            enabled = index in active
            for parameter in parameters:
                parameter.requires_grad_(enabled)

    @property
    def selective_probe_due(self):
        """Whether the optimizer update about to happen is a full-block probe that should be scored."""
        if not self.selective_global_replay or not hasattr(self, "_selective_optimizer_steps"):
            return False
        upcoming = self._selective_optimizer_steps + 1
        warmup = self._selective_warmup_updates
        if upcoming < warmup:
            return False
        return upcoming == warmup or (upcoming > warmup
                                      and (upcoming - warmup) % self._selective_reactivate_every == 0)

    def selective_snapshot_before_step(self):
        """At a probe, score each block's displacement since its last probe against this probe's fresh gradient
        (before the step), on CPU so the check does not increase VRAM use. Then re-anchor at the current weights."""
        if not self.selective_probe_due:
            return None
        cosines = {}
        for index in range(self._selective_block_count):
            block_params = self._selective_params_by_block[index]
            if not all(p.requires_grad and p.grad is not None for p in block_params):
                continue
            now = self._selective_flat(block_params)
            grad = torch.cat([p.grad.detach().float().reshape(-1) for p in block_params]).cpu()
            anchor = self._selective_anchor.get(index)
            self._selective_anchor[index] = now
            if anchor is None or anchor.shape != now.shape:
                continue                     # just resumed: this probe only sets the anchor
            delta = now - anchor
            denom = grad.norm() * delta.norm()
            if not torch.isfinite(denom) or float(denom) <= 1e-20:
                continue
            cosine = float((-torch.dot(grad, delta) / denom).clamp(-1.0, 1.0))
            if np.isfinite(cosine):
                cosines[index] = cosine
        return {"cosines": cosines}

    def finish_selective_optimizer_step(self, snapshot):
        """Score a probe and choose active blocks for the next optimizer window. Returns a status for logging."""
        if not self.selective_global_replay:
            return None
        self._selective_optimizer_steps += 1
        step = self._selective_optimizer_steps
        if snapshot is not None:
            for index, cosine in snapshot.get("cosines", {}).items():
                old = self._selective_scores[index]
                decay = self._selective_score_decay
                self._selective_scores[index] = cosine if old is None else decay * old + (1.0 - decay) * cosine
            # Keep unknown blocks active; only a measured uphill direction is eligible to be skipped. If every
            # measured score is uphill, keep the least harmful block alive.
            active = {i for i, score in enumerate(self._selective_scores) if score is None or score > 0.0}
            if not active and self._selective_scores:
                known = [(score, i) for i, score in enumerate(self._selective_scores) if score is not None]
                active = {max(known)[1]} if known else set(range(self._selective_block_count))
            self._selective_active = active or set(range(self._selective_block_count))
            self._set_selective_active_parameters()
            return {"kind": "probe", "active": len(self._selective_active), "total": self._selective_block_count,
                    "scores": list(self._selective_scores)}

        # Wake every block for the next update immediately before a scheduled probe. That probe then refreshes
        # the scores and selects the following window.
        if (step > self._selective_warmup_updates
                and (step - self._selective_warmup_updates) % self._selective_reactivate_every
                == self._selective_reactivate_every - 1):
            self._selective_active = set(range(self._selective_block_count))
            self._set_selective_active_parameters()
            return {"kind": "reactivation", "active": self._selective_block_count,
                    "total": self._selective_block_count, "scores": list(self._selective_scores)}
        return None

    def selective_state_dict(self):
        if not self.selective_global_replay or not hasattr(self, "_selective_optimizer_steps"):
            return None
        return {"version": 1, "optimizer_steps": self._selective_optimizer_steps,
                "warmup_updates": self._selective_warmup_updates,
                "reactivate_every": self._selective_reactivate_every,
                "score_decay": self._selective_score_decay,
                "scores": list(self._selective_scores), "active_blocks": sorted(self._selective_active)}

    def load_selective_state_dict(self, state):
        if not self.selective_global_replay or not isinstance(state, dict) or state.get("version") != 1:
            raise RuntimeError("The saved Selective global replay state is missing or unsupported.")
        if (int(state.get("warmup_updates", -1)) != self._selective_warmup_updates
                or int(state.get("reactivate_every", -1)) != self._selective_reactivate_every
                or float(state.get("score_decay", -1)) != self._selective_score_decay):
            raise RuntimeError("Selective global replay settings differ from the saved run state.")
        scores = state.get("scores")
        if not isinstance(scores, list) or len(scores) != self._selective_block_count:
            raise RuntimeError("The saved Selective global replay scores do not match the Krea 2 block count.")
        active = {int(i) for i in state.get("active_blocks", []) if 0 <= int(i) < self._selective_block_count}
        if not active:
            raise RuntimeError("The saved Selective global replay state has no active blocks.")
        self._selective_scores = [None if v is None else float(v) for v in scores]
        self._selective_active = active
        self._selective_optimizer_steps = max(0, int(state.get("optimizer_steps", 0)))
        # the anchors were taken before the resumed weights loaded: the first probe after a resume re-anchors only
        self._selective_anchor = {i: None for i in range(self._selective_block_count)}
        self._set_selective_active_parameters()

    def selective_active_blocks(self):
        return sorted(getattr(self, "_selective_active", set()))

    def make_block_local_heads(self, dit, device):
        """Create temporary per-block velocity readouts used only during the optional head warm-up.

        They are not attached to the DiT or FamilyLoRA and are never exported. After the warm-up, they are frozen
        and the LoRA is trained against the real Krea 2 final readout with exact blockwise gradient replay.
        """
        import copy
        from fizgig.families.lora import LoRALinear

        heads = []
        for _ in dit.blocks:
            head = copy.deepcopy(dit.last)
            # Strip adapters from the copied readout. Frozen context/speed adapters must not be duplicated into
            # temporary heads or accidentally made trainable with the heads.
            for module in head.modules():
                if isinstance(module, LoRALinear):
                    module.adapters = torch.nn.ModuleDict()
                    module.scales = {}
            heads.append(head.float())
        heads = torch.nn.ModuleList(heads).to(device)
        heads.requires_grad_(True)
        self._block_local_heads = heads
        self._block_local_head_only = True
        self._pending_block_local = None
        return heads

    def set_block_local_head_only(self, enabled):
        """Select the auxiliary-head calibration phase or the real global-loss replay phase."""
        enabled = bool(enabled)
        self._block_local_head_only = enabled
        heads = getattr(self, "_block_local_heads", None)
        if heads is not None:
            heads.requires_grad_(enabled)

    def _block_local_training_loss(self, dit, latents, noise, t, cond):
        """Run a no-grad Krea 2 forward and save CPU activations for one-block-at-a-time backward replay.

        During the optional head-only warm-up, each temporary readout is calibrated from that block's output.
        After warm-up, the returned loss is the ordinary full-model Krea 2 flow-matching loss from ``dit.last``;
        backward_training_loss then computes its exact gradient by replaying the transformer blocks in reverse.
        """
        from fizgig.krea2.attention import AttentionParams
        from fizgig.krea2.model import temb
        from fizgig.krea2.sampling import patchify_block, prepare

        head_only = bool(getattr(self, "_block_local_head_only", False))
        heads = getattr(self, "_block_local_heads", None)
        if head_only and heads is None:
            raise RuntimeError("Block-local head warm-up was requested but temporary heads were not initialized.")
        if getattr(self, "_pending_block_local", None) is not None:
            raise RuntimeError("A previous block-local loss has not been backpropagated.")

        device = latents.device
        patch = dit.config.patch
        latent = latents.to(dtype=DTYPE)
        noise = noise.to(device=device, dtype=DTYPE)
        t = t.to(device)
        t_ = t.view(-1, 1, 1, 1).to(DTYPE)
        noised = (1.0 - t_) * latent + t_ * noise
        target, _, _ = patchify_block(noise - latent, patch)
        txt, txtmask = self._text(cond, device)
        img, pos, mask = prepare(noised, txt.shape[1], patch, txtmask)

        block_inputs, block_outputs, local_losses = [], [], []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=DTYPE):
            img = dit.first(img)
            # Match _forward / SingleStreamDiT.forward exactly: Krea 2 casts t to BF16 before temb.
            # Building temb from the FP32 sampler value changes t_emb/tvec and every block's activations.
            t_emb = dit.tmlp(temb(t.to(DTYPE), dit.config.tdim, device=img.device, dtype=img.dtype))
            tvec = dit.tproj(t_emb)
            imglen = img.shape[1]
            txtmask = mask[:, imglen:]
            txt_attn_params_nomask = AttentionParams.create_attention_params_from_mask(
                dit.attn_mode, dit.split_attn, 0, None)
            txt_attn_params = AttentionParams.create_attention_params_from_mask(
                dit.attn_mode, dit.split_attn, 0, txtmask)
            context = dit.txtfusion(txt, txt_attn_params_nomask, txt_attn_params)
            context = dit.txtmlp(context)
            combined = torch.cat((img, context), dim=1)
            fulllen = combined.shape[1]
            padlen = (-fulllen) % 256
            if padlen:
                combined = F.pad(combined, (0, 0, 0, padlen))
                pos = F.pad(pos, (0, 0, 0, padlen))
                txtmask = F.pad(txtmask, (0, padlen), value=False)
            attn_params = AttentionParams.create_attention_params_from_mask(
                dit.attn_mode, dit.split_attn, imglen, txtmask)
            freqs = dit.posemb(pos)

            for index, block in enumerate(dit.blocks):
                if dit.blocks_to_swap:
                    dit.offloader.wait_for_block(index)
                if head_only:
                    # Warm-up backward only needs block outputs. Saving these avoids a second transformer pass and
                    # avoids relying on autograd block-swap hooks while every LoRA adapter is frozen.
                    combined = block(combined, tvec, freqs, attn_params)
                    block_outputs.append(combined.detach().to("cpu").contiguous())
                    pred = heads[index](combined, t_emb)[:, :imglen, :]
                    local_losses.append(F.mse_loss(pred.float(), target.float()))
                else:
                    block_inputs.append(combined.detach().to("cpu").contiguous())
                    # Keep grad mode enabled for the block's forward so attention chooses the same kernels as the
                    # ordinary LoRA training forward. Immediately detach the output: the next block gets the exact
                    # activation but no cross-block graph is retained in VRAM.
                    with torch.enable_grad():
                        combined = block(combined.detach(), tvec, freqs, attn_params).detach()
                if dit.blocks_to_swap:
                    dit.offloader.submit_move_blocks_forward(dit.blocks, index)

            if head_only:
                loss = torch.stack(local_losses).mean().detach()
                final_hidden = None
            else:
                # This is the exact readout used by normal training, validation, and sampling.
                pred = dit.last(combined, t_emb)[:, :imglen, :]
                loss = F.mse_loss(pred.float(), target.float()).detach()
                final_hidden = combined.detach().to("cpu").contiguous()

        self._pending_block_local = {
            "dit": dit,
            "phase": "head_warmup" if head_only else "global_replay",
            "block_inputs": block_inputs,
            "block_outputs": block_outputs,
            "final_hidden": final_hidden,
            "tvec": tvec.detach(),
            "t_emb": t_emb.detach(),
            "freqs": freqs.detach(),
            "attn_params": attn_params,
            "target": target.detach(),
            "imglen": imglen,
            "device": device,
        }
        return loss

    def backward_training_loss(self, scale=1.0):
        """Apply local identity-path credit, warm-up gradients, or exact global-loss block replay."""
        state = getattr(self, "_pending_block_local", None)
        if state is None:
            raise RuntimeError("No pending block-local flow loss is available for backward.")
        dit = state["dit"]
        n = len(dit.blocks)
        try:
            if state["phase"] == "residual_path_approx":
                # B is updated on every approximate step. The selective-A variant also computes dL/dA
                # every fourth approximate update, from the cached full-width input and current B factor.
                credit = state["readout_grad"] * float(scale)
                if state.get("update_a") and getattr(self, "_residual_path_d2h_stream", None) is not None:
                    # D2H activation copies are submitted during the forward; wait once after readout work
                    # has had an opportunity to overlap them.
                    self._residual_path_d2h_stream.synchronize()
                for index, values in state["captures"].items():
                    for suffix, projection in values["projections"].items():
                        gate = values["attn_gate"] if suffix == "attn.wo" else values["mlp_gate"]
                        ax = projection["ax"]
                        a = projection["a"]
                        b = projection["b"]
                        local_grad = credit * gate.unsqueeze(1)
                        # Match _LoraLowp.backward's activation precision and scale placement.
                        g2 = (local_grad.to(dtype=ax.dtype) * projection["scale"]).reshape(-1, local_grad.shape[-1])
                        ax2 = ax.reshape(-1, ax.shape[-1])
                        grad_b = (g2.transpose(0, 1) @ ax2).to(dtype=b.weight.dtype)
                        if b.weight.grad is None:
                            b.weight.grad = grad_b
                        else:
                            b.weight.grad.add_(grad_b)

                        if state.get("update_a"):
                            x_host = projection["x_host"]
                            x = x_host.to(device=b.weight.device, dtype=ax.dtype,
                                          non_blocking=bool(getattr(x_host, "is_pinned", lambda: False)()))
                            x2 = x.reshape(-1, x.shape[-1])
                            grad_hidden = g2 @ b.weight.to(dtype=x.dtype)
                            grad_a = (grad_hidden.transpose(0, 1) @ x2).to(dtype=a.weight.dtype)
                            if a.weight.grad is None:
                                a.weight.grad = grad_a
                            else:
                                a.weight.grad.add_(grad_a)
                return

            if state["phase"] == "head_warmup":
                heads = getattr(self, "_block_local_heads", None)
                outputs = state["block_outputs"]
                if heads is None or len(outputs) != n or len(heads) != n:
                    raise RuntimeError("Block-local warm-up state does not match the Krea 2 block count.")
                # Every head is independent. The frozen transformer is not replayed during this phase.
                for index, output in enumerate(outputs):
                    hidden = output.to(state["device"], non_blocking=True)
                    with torch.autocast(device_type=state["device"].type, dtype=DTYPE):
                        pred = heads[index](hidden, state["t_emb"])[:, :state["imglen"], :]
                        local_loss = F.mse_loss(pred.float(), state["target"].float())
                    # Match the reported mean over blocks: each independent head contributes 1/N of the total
                    # warm-up objective. Without this factor, the manual replay would apply N times the gradient
                    # implied by the scalar loss (28x on Krea 2's full transformer).
                    (local_loss * (float(scale) / n)).backward()
                # Frozen block weights do not build an autograd graph, so the offloader's normal backward hooks
                # cannot rotate its residency window during head-only calibration. Restore the forward layout here
                # so the next training batch starts with the expected resident/offloaded blocks.
                if dit.blocks_to_swap:
                    dit.prepare_block_swap_before_forward()
                return

            inputs = state["block_inputs"]
            if n == 0 or len(inputs) != n or state["final_hidden"] is None:
                raise RuntimeError("Global block-replay state does not match the Krea 2 block count.")

            # Start at the actual frozen Krea 2 readout. Its input requires grad, so autograd computes the exact
            # gradient that must enter the final transformer block, without retaining the model's full graph.
            final_hidden = state["final_hidden"].to(state["device"], non_blocking=True).detach().requires_grad_(True)
            with torch.autocast(device_type=state["device"].type, dtype=DTYPE):
                pred = dit.last(final_hidden, state["t_emb"])[:, :state["imglen"], :]
                global_loss = F.mse_loss(pred.float(), state["target"].float())
            (global_loss * float(scale)).backward()
            grad = final_hidden.grad
            if grad is None:
                raise RuntimeError("Krea 2 final readout did not produce a gradient for its input.")

            # Recompute one block at a time in reverse order. Each replay receives the downstream gradient, so the
            # LoRA gradients are the same chain-rule gradients as an ordinary full-model backward pass.
            for index in range(n - 1, -1, -1):
                if dit.blocks_to_swap:
                    dit.offloader.wait_for_block(index)
                x = inputs[index].to(state["device"], non_blocking=True).detach().requires_grad_(True)
                with torch.autocast(device_type=state["device"].type, dtype=DTYPE):
                    hidden = dit.blocks[index](x, state["tvec"], state["freqs"], state["attn_params"])
                torch.autograd.backward(hidden, grad)
                grad = x.grad
                if grad is None:
                    raise RuntimeError(f"Krea 2 block {index} did not propagate a gradient to its input.")
        finally:
            self._pending_block_local = None

    def loss_at(self, dit, latents, noise, t, cond, *, diff_ref=None, diff_weight=0.0, loss_mask=None, prior=None,
                per=False):
        """The training loss for a given latent, noise and t - the original compute_loss's arithmetic, in the same
        dtype order. training_loss draws noise and t; tests call this directly to compare with the original. per:
        each picture's own loss, (B,) (training_loss_batch)."""
        from fizgig.krea2.sampling import patchify_block
        patch = dit.config.patch
        latent = latents.to(dtype=DTYPE)
        noise = noise.to(device=latent.device, dtype=DTYPE)
        t = t.to(latent.device)
        t_ = t.view(-1, 1, 1, 1).to(DTYPE)
        noised = (1.0 - t_) * latent + t_ * noise
        target, _, _ = patchify_block(noise - latent, patch)
        base = prior_base(prior, loss_mask, lambda: self._forward(dit, noised, t, cond))
        pred = self._forward(dit, noised, t, cond)
        w = None
        if diff_ref is not None and diff_weight > 0.0:
            d, _, _ = patchify_block((latent - diff_ref.to(device=latent.device, dtype=DTYPE)).abs(), patch)
            d = d.float().mean(dim=-1)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            w = (1.0 - float(diff_weight)) + float(diff_weight) * r
            w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
            w = torch.where(dm > 1e-6, w, torch.ones_like(w))      # identical pair: uniform, never all-zero
        mw = None
        if loss_mask is not None:                                  # a token's weight: its patch's average
            mw = patchify_block(loss_mask.to(latent.device).float(), patch)[0].mean(dim=-1)
        if per:
            return picture_losses((pred.float() - target.float()).pow(2).mean(dim=-1), w, mw)
        if w is None and mw is None:
            return F.mse_loss(pred.float(), target.float())
        loss = weighted_loss((pred.float() - target.float()).pow(2).mean(dim=-1), w, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred.float() - base).pow(2).mean(dim=-1), mw, w)
        return loss

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        if refs:
            raise RuntimeError("Krea 2 has no edit training (its DiT takes no reference latents)")
        patch = dit.config.patch
        n_tokens = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
        noise = torch.randn(latents.shape, generator=generator)
        t = self._sample_t(n_tokens, generator, min_t, max_t, self.timestep_spread)
        if self.block_local_flow and torch.is_grad_enabled():
            if diff_ref is not None or diff_weight:
                raise RuntimeError("Block-local flow matching does not support paired/difference loss yet.")
            if loss_mask is not None or prior is not None:
                raise RuntimeError("Block-local flow matching does not support masked/prior loss yet.")
            if self.residual_path_hybrid:
                exact = self.residual_path_exact_due
                self._residual_path_last_step_exact = exact
                if exact:
                    loss = self._block_local_training_loss(dit, latents, noise, t, cond)
                    phase = "exact_calibration"
                else:
                    loss = self._residual_path_training_loss(dit, latents, noise, t, cond)
                    phase = "identity_credit"
                return loss, {"t": float(t.mean()), "block_local": True, "block_local_phase": phase,
                              "residual_path_hybrid": True}
            loss = self._block_local_training_loss(dit, latents, noise, t, cond)
            phase = "head_warmup" if bool(getattr(self, "_block_local_head_only", False)) else "global_replay"
            return loss, {"t": float(t.mean()), "block_local": True, "block_local_phase": phase}
        loss = self.loss_at(dit, latents, noise, t, cond, diff_ref=diff_ref, diff_weight=diff_weight,
                            loss_mask=loss_mask, prior=prior)
        return loss, {"t": float(t.mean())}

    def training_loss_batch(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, loss_mask=None):
        """Batch Size above 1: each picture's noise and t drawn as training_loss draws them, picture by picture, then
        one loss_at over the batch. Captions stack as encoded (512 tokens + validity mask each); _text gathers each
        one's valid tokens and pads to the batch's longest with a mask, as the shared attention expects."""
        patch = dit.config.patch
        n_tokens = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
        noises, ts = [], []
        for i in range(latents.shape[0]):
            noises.append(torch.randn(latents[i:i + 1].shape, generator=generator))
            ts.append(self._sample_t(n_tokens, generator, min_t, max_t))
        t = torch.cat(ts)
        per = self.loss_at(dit, latents, torch.cat(noises), t, cond, loss_mask=loss_mask, per=True)
        return per.mean(), {"t": float(t.mean()), "per": per, "ts": [float(v) for v in t]}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        patch = 2
        n_tokens = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
        noise = torch.randn(latents.shape, generator=generator).to(latents.device, DTYPE)
        t = self._sample_t(n_tokens, generator, min_t, max_t, self.timestep_spread).to(latents.device)
        t_ = t.view(-1, 1, 1, 1).to(DTYPE)
        return {"xt": (1.0 - t_) * latents.to(DTYPE) + t_ * noise, "t": t}

    def map_target(self, latents, state):
        """The target in predict()'s layout (training maps): noise - x0, patchified as loss_at does."""
        from fizgig.krea2.sampling import patchify_block
        t = state["t"].float().view(-1, 1, 1, 1).to(latents.device)
        return patchify_block((state["xt"].float() - latents.float()) / t, 2)[0]

    def predict(self, dit, state, cond):
        return self._forward(dit, state["xt"], state["t"], cond)

    def trigger_text(self, tokenizer, caption):
        """The encoder's system + user prefix (dropped: its 34 tokens never reach the DiT), the caption, the suffix
        (krea2/embedder.py: prompt_template_encode_prefix / _suffix)."""
        from fizgig.krea2.caption_text import ENCODE_SYSTEM_DESCRIPTOR
        prefix = "<|im_start|>system\n" + ENCODE_SYSTEM_DESCRIPTOR + "<|im_end|>\n<|im_start|>user\n"
        return prefix + caption + "<|im_end|>\n<|im_start|>assistant\n", prefix

    # ---- sampling (sampling.sample, split at the decode) ----------------------------------------
    @staticmethod
    def _grid(width, height):
        from fizgig.krea2.sampling import roundup
        align = 16                                                      # VAE 8 x patch 2
        return roundup(width, align, "width"), roundup(height, align, "height")

    def pad_conditioning(self, conds):
        """Prompt travel: every Krea 2 prompt is already the encoder's fixed 512 tokens with a validity mask, so the
        shapes match as they are; blending the masks (as weights) fades in the tokens only one prompt has."""
        return [dict(c) for c in conds]

    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        width, height = self._grid(width, height)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=dev).manual_seed(int(seed))
        return torch.randn(1, 16, height // 8, width // 8, device=dev, dtype=DTYPE, generator=g).float().cpu()

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        from fizgig.krea2.sampling import prepare, timesteps
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        width, height = self._grid(width, height)
        opts = dict(options)
        patch = dit.config.patch
        txt, txtmask = self._text(cond, device)
        guided = cfg > 1.0 and neg_cond is not None
        if guided:
            untxt, untxtmask = self._text(neg_cond, device)
        x0 = (noise if noise is not None else self.initial_noise(seed, width, height)).to(device, DTYPE)
        img, pos, mask = prepare(x0, txt.shape[1], patch, txtmask)
        if guided:
            _, unpos, unmask = prepare(x0, untxt.shape[1], patch, untxtmask)
        x1, x2 = (256 // 16) ** 2, (1280 // 16) ** 2
        ts = list(sigmas) + [0.0] if sigmas is not None and len(sigmas) == steps else \
            timesteps(img.shape[1], steps, x1, x2, mu=opts.get("mu"))
        with torch.autocast(device_type=device.type, dtype=DTYPE):
            for i, (tc, tp) in enumerate(zip(ts[:-1], ts[1:])):
                if on_step is not None:
                    on_step(i, len(ts) - 1)
                t = torch.full((1,), tc, dtype=img.dtype, device=device)
                v = dit(img=img, context=txt, t=t, pos=pos, mask=mask)
                if guided:
                    u = dit(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                    v = u + cfg * (v - u)
                img = img + (tp - tc) * v
        from einops import rearrange
        return rearrange(img, "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)", ph=patch, pw=patch,
                         h=height // 16, w=width // 16)

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        dev = next(vae.parameters()).device
        pixels = vae.decode_to_pixels(latents.to(dev, torch.bfloat16))                # [0, 1], (B, C, H, W)
        arr = (pixels[0].float().permute(1, 2, 0).clamp(0, 1) * 255.0).byte().cpu().numpy()
        return Image.fromarray(arr)

    # ---- LoRA and the block map -------------------------------------------------------------------
    def block_map(self, dit=None):
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        blocks = [Block(f"block_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                  for i in range(self.description.n_blocks)]
        # the text-fusion stack as the original's four sliders (its layerwise and refiner blocks); its projector
        # rides with the input / output layers
        fusion = [Block(f"txt_{short}_{i}", f"Text fusion {name} {i}",
                        keep([f"txtfusion.{stack}_blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                  for short, name, stack in (("lw", "layerwise", "layerwise"), ("rf", "refiner", "refiner"))
                  for i in range(2)]
        return [BlockGroup("Blocks", blocks),
                BlockGroup("Text fusion", fusion),
                BlockGroup("Input and output", [Block("io", "Input, timestep and output layers",
                                                      keep(list(_IO) + ["txtfusion.projector"]))])]

    # diffusers naming (OneTrainer, AI-Toolkit, diffusers exports) -> Krea 2's own, on flattened names; the same
    # table the original Krea 2 loader used (networks.lora._KREA2_DIFFUSERS_RENAMES, issue #50)
    _ALIASES = ((r"^transformer_blocks_(\d+)_", r"blocks_\1_"), (r"^text_fusion_", "txtfusion_"),
                (r"_attn_to_out_0$", "_attn_wo"), (r"_attn_to_gate$", "_attn_gate"), (r"_attn_to_q$", "_attn_wq"),
                (r"_attn_to_k$", "_attn_wk"), (r"_attn_to_v$", "_attn_wv"), (r"_ff_up$", "_mlp_up"),
                (r"_ff_gate$", "_mlp_gate"), (r"_ff_down$", "_mlp_down"), (r"^img_in$", "first"),
                (r"^final_layer_linear$", "last_linear"), (r"^txt_in_linear_1$", "txtmlp_1"),
                (r"^txt_in_linear_2$", "txtmlp_3"), (r"^time_embed_linear_1$", "tmlp_0"),
                (r"^time_embed_linear_2$", "tmlp_2"), (r"^time_mod_proj$", "tproj_1"))

    def alias_flat(self, flat):
        out = flat
        for pat, new in self._ALIASES:
            out = re.sub(pat, new, out)
        return out if out != flat else None

    def quant_target_names(self, dit):
        """The blocks' Linears only: the text-fusion stack and the I/O layers stay bf16, as in the original."""
        return [m for b in self.block_map(dit)[0].blocks for m in b.modules]
