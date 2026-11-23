"""WebView2 renderer defaults for Fizgig's Windows desktop shell.

Keep WebView2's Chromium rendering defaults independently configurable
from Edge/Chrome. Windows and Linux use hardware acceleration by default.
The GPU-disable preference is an opt-in compatibility workaround; software
video decoding is independently opt-in.

This module intentionally imports no pywebview/Windows dependencies so it can
be regression-tested without a desktop runtime.
"""
import re
import sys


def _has_flag(arguments, flag):
    """Recognize a complete Chromium switch, not an arbitrary substring."""
    return bool(re.search(r"(?<!\S)" + re.escape(flag) + r"(?=\s|=|$)", arguments or ""))


def gpu_disabled(prefs=None, platform=None):
    """Disable embedded desktop acceleration only when explicitly requested.

    The default remains hardware acceleration on both Windows and Linux.
    This option has no effect on standalone Chrome or Edge.
    """
    prefs = prefs or {}
    setting = str(prefs.get("qt_disable_gpu", "")).strip()
    if setting == "1":
        return True
    if setting == "0":
        return False
    return False


def browser_arguments(prefs=None, inherited="", software_video=False, platform=None):
    """Build WebView2 flags without overriding explicitly inherited flags.

    Both Windows and Linux retain browser-like GPU acceleration by default.
    The user can explicitly disable GPU acceleration for embedded desktop
    compatibility. Hardware video decoding remains separately configurable.
    """
    prefs = prefs or {}
    result = str(inherited or "").strip()
    requested = []
    if gpu_disabled(prefs, platform):
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
