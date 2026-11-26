"""Fixed real-training diagnostic matrix for Krea 2 NaNs (not a tuning benchmark)."""
import os

CASES = (
    ("a-original", "outside", True, True, "polora", "INT8 (8-bit, fastest)", "0"),
    ("b-polora-eager", "outside", False, True, "polora", "INT8 (8-bit, fastest)", "0"),
    ("c-model-eager", "Off", True, True, "polora", "INT8 (8-bit, fastest)", "0"),
    ("d-both-eager", "Off", False, True, "polora", "INT8 (8-bit, fastest)", "0"),
    ("e-no-masks", "Off", False, False, "polora", "INT8 (8-bit, fastest)", "0"),
    ("f-adamw8bit", "Off", False, True, "adamw8bit", "INT8 (8-bit, fastest)", "0"),
    ("g-bf16", "Off", False, True, "polora", "bf16 (full precision)", "Auto (detect from GPU)"),
)

def settings(group, root):
    common = {
        "FAMILY_KIND": "standard", "NETWORK_TYPE": "LoRA (standard)",
        "NETWORK_DIM": "8", "NETWORK_ALPHA": "8", "GRADIENT_ACCUMULATION": "1",
        "BATCH_SIZE": "1", "MAX_GRAD_NORM": "1.0", "MAX_TRAIN_EPOCHS": "1",
        "SAVE_EVERY_N_EPOCHS": "1", "SAVE_STATE": False, "SAVE_STATE_ON_TRAIN_END": False,
        "SEED": "42", "DATASET_MEGAPIXELS": "1.0", "ENABLE_BUCKET": True,
        "BUCKET_NO_UPSCALE": True, "DATASET_CAPTION_EXT": ".txt", "ENABLE_CACHE": True,
        "TAG_SHUFFLE": True, "TAG_KEEP_FIRST": "1", "TAG_VARIANTS": "4",
        "MASK_FLOOR": "0.1 (recommended)", "MASK_PRIOR": "Off", "FAMILY_EMA": "0.98 (recommended)",
        "VALIDATION_LOSS": False, "VALIDATION_EARLY_STOP": False,
        "SAMPLE_ENABLED": False, "SAMPLE_AT_FIRST": False, "TRAINING_MAPS": False,
        "KREA2_AUTO_RECAPTION": False, "KREA2_LOSS_WATCH": True,
        "ADAPTIVE_LR": True, "ADAPTIVE_LR_MIN": "0.01", "ADAPTIVE_LR_MAX": "0.04",
        "KREA2_PER_IMAGE_LR": True, "KREA2_TRAINING_METHOD": "Standard",
        "KREA2_SELECTIVE_BLOCK_CHECK": False, "INT8_ROTATED_BACKWARD": False,
        "RESUME_TRAINING": "", "CONTEXT_LORA_PATH": "",
    }
    result = []
    for case, model_compile, opt_compile, masks, opt, prec, swap in CASES:
        v = dict(common, __polora_diagnostic=case, LORA_NAME=f"{group}-{case}",
                 LORA_OUTPUT_DIR=os.path.join(root, group, case),
                 COMPILE_BLOCKS=model_compile, POLORA_COMPILE=opt_compile,
                 MASKED_TRAINING=masks, OPTIMIZER_TYPE=opt, FAMILY_PRECISION=prec,
                 BLOCKS_SWAP=swap, LEARNING_RATE="0.02",
                 OPTIMIZER_ARGS=f"magnitude_scale=3 curvature_beta=0.99 compile={opt_compile}")
        if opt != "polora":
            v.update(LEARNING_RATE="0.0002", OPTIMIZER_ARGS="", ADAPTIVE_LR=False,
                     KREA2_PER_IMAGE_LR=False)
        result.append(v)
    return result
