"""Turn a Civitai detector (.pt, ultralytics YOLO) into the .onnx Image Prep's Body-part and watermark detectors run.

Fizgig itself never imports ultralytics or opens a .pt. This script makes a THROWAWAY Python venv in a temp folder,
pip-installs ultralytics + onnx there (never into Fizgig's venv), exports the model with
src/fizgig/scripts/convert_yolo_to_onnx.py, saves the .onnx where Fizgig looks for it, then deletes the venv.
Needs an internet connection for the pip install (about 1 GB with torch, removed afterwards).

    python tools/convert_detector_to_onnx.py aio          C:\\Downloads\\animeNSFWAllInOne_v50.pt
    python tools/convert_detector_to_onnx.py assdetailer  C:\\Downloads\\assDetailer_v2.pt
    python tools/convert_detector_to_onnx.py watermark    C:\\Downloads\\watermark_s.pt
    python tools/convert_detector_to_onnx.py nsfw_watermark C:\\Downloads\\nsfw_watermark_s.pt

-> user_data/models/detectors/<name>.onnx (or --out PATH). Then reopen Image Prep: the model shows as Installed.
A .pt file is a pickle: loading it can run code. Only convert files you downloaded yourself from the model's page.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = {"aio": "anime_nsfw_aio_v5.onnx", "assdetailer": "assdetailer_v2.onnx",
         "watermark": "watermark_s.onnx", "nsfw_watermark": "nsfw_watermark_s.onnx",
         "booru": "booru_yolov8s_aa11.onnx"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(FILES), help="which detector this .pt is")
    ap.add_argument("pt", help="the downloaded .pt file")
    ap.add_argument("--out", help="where to save the .onnx (default: user_data/models/detectors/<name>.onnx)")
    ap.add_argument("--keep-venv", action="store_true", help="keep the temp venv (to convert several in a row)")
    a = ap.parse_args(argv)
    pt = os.path.abspath(a.pt)
    if not os.path.isfile(pt):
        sys.exit(f"not found: {pt}")
    out = os.path.abspath(a.out) if a.out else os.path.join(ROOT, "user_data", "models", "detectors", FILES[a.model])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), "fizgig-convert-venv")
    py = os.path.join(tmp, "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")
    try:
        if not os.path.isfile(py):
            print(f"[convert] making a throwaway venv in {tmp} (not Fizgig's)")
            venv.create(tmp, with_pip=True)
            subprocess.check_call([py, "-m", "pip", "install", "--quiet", "ultralytics", "onnx"])
        subprocess.check_call([py, os.path.join(ROOT, "src", "fizgig", "scripts", "convert_yolo_to_onnx.py"), pt,
                               "--out", out, "--detector"])
        print(f"[convert] saved {out}")
    finally:
        if not a.keep_venv:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
