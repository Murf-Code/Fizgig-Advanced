"""Reveal an existing dataset file in the local desktop's file manager.

This is intentionally an OS action rather than a browser download. The caller
must check local-viewer access; never expose this to remote browser clients.
"""
import os
import subprocess
import sys


def file_manager_command(path, os_name=None, platform=None):
    """An argument vector (no shell), selecting the file where supported."""
    os_name = os.name if os_name is None else os_name
    platform = sys.platform if platform is None else platform
    if os_name == "nt":
        return ["explorer.exe", "/select,", path]
    if platform == "darwin":
        return ["open", "-R", path]
    # Linux file managers vary; opening the containing folder is portable.
    return ["xdg-open", os.path.dirname(path)]


def reveal_file(path, folder, name):
    """Open the containing folder and select the requested file when possible.

    A folder/name/path triple is required so a mismatched or traversal filename
    cannot accidentally open an unrelated path. The endpoint itself is
    accessible only from the computer running Fizgig.
    """
    if not all(isinstance(value, str) and value for value in (path, folder, name)):
        raise ValueError("Choose a dataset image first.")
    if name in (".", "..") or os.path.basename(name) != name:
        raise ValueError("Invalid file name.")
    root = os.path.realpath(folder)
    target = os.path.realpath(os.path.join(root, name))
    if not os.path.isabs(path) or os.path.normcase(os.path.realpath(path)) != os.path.normcase(target):
        raise ValueError("The selected file does not match its dataset folder.")
    if not os.path.isfile(target):
        raise ValueError("The selected image no longer exists on disk.")
    if os.name != "nt" and sys.platform != "darwin" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("A desktop file manager is not available on this machine.")
    command = file_manager_command(target)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        subprocess.Popen(command, creationflags=flags)
    except (OSError, ValueError) as error:
        raise ValueError("Unable to open the file manager: " + str(error)) from error
    return {"ok": True}
