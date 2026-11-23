"""WebView2 renderer defaults for Fizgig's Windows desktop shell.

Keep WebView2's Chromium rendering configuration as close as possible to the
normal Edge/Chrome browser. GPU disabling must be an explicit user choice;
software video decoding remains available as a diagnostic/workaround for
machines affected by a separate video decoder freeze.

This module intentionally imports no pywebview/Windows dependencies so it can
be regression-tested without a desktop runtime.
"""
import re


def _has_flag(arguments, flag):
    """Recognize a complete Chromium switch, not an arbitrary substring."""
    return bool(re.search(r"(?<!\S)" + re.escape(flag) + r"(?=\s|=|$)", arguments or ""))


def browser_arguments(prefs=None, inherited="", software_video=False):
    """Build WebView2 flags without overriding explicitly inherited flags.

    Prior versions disabled accelerated video decoding for every user.
    That workaround is opt-in now so video, canvas and image-heavy views use
    Chromium's normal acceleration by default.
    """
    prefs = prefs or {}
    result = str(inherited or "").strip()
    requested = []
    if str(prefs.get("qt_disable_gpu") or "") == "1":
        requested.append("--disable-gpu")
    if software_video:
        requested.append("--disable-accelerated-video-decode")
    for flag in requested:
        if not _has_flag(result, flag):
            result = (result + " " + flag).strip()
    return result


def acceleration_warning(arguments):
    """Describe only known performance switches, never print arbitrary env data."""
    args = str(arguments or "")
    details = []
    for flag, description in (
        ("--disable-gpu", "GPU acceleration DISABLED"),
        ("--disable-gpu-compositing", "GPU compositing DISABLED"),
        ("--disable-software-rasterizer", "software raster fallback DISABLED"),
        ("--disable-accelerated-2d-canvas", "accelerated 2D canvas DISABLED"),
        ("--disable-accelerated-video-decode", "hardware video decode DISABLED"),
    ):
        if _has_flag(args, flag):
            details.append(description + " (" + flag + ")")
    if "--disable-gpu" in args and not _has_flag(args, "--disable-gpu"):
        # Chromium flags can come from config files with unusual whitespace.
        details.append("custom disable-gpu argument present")
    return "; ".join(details) if details else "No known GPU/video acceleration disabling flags"
