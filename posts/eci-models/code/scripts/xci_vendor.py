"""The published XCI code (github.com/hugetim/xci) at a pinned commit, in vendor/xci (git ignores vendor/), so the
daily fit and post 1A's XCI view run the same code as the public repo (Tim, Nov 23, 2026). ensure() clones it if
missing and checks out XCI_COMMIT, refusing a copy with local changes; it returns the folder, for sys.path or for
running xci.py. To move to a new XCI release, change XCI_COMMIT.

    python scripts/xci_vendor.py      # set it up; prints the folder and commit
"""
import os
import subprocess
import sys

XCI_URL = "https://github.com/hugetim/xci.git"
XCI_COMMIT = "47066a179c6aaeaaef90e42dd104dc1a75554612"  # tag v0.1.1
XCI_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "xci"))


def _git(*args):
    return subprocess.run(["git", "-C", XCI_DIR, *args], check=True, capture_output=True, text=True).stdout.strip()


def ensure():
    if not os.path.isdir(os.path.join(XCI_DIR, ".git")):
        os.makedirs(os.path.dirname(XCI_DIR), exist_ok=True)
        subprocess.run(["git", "clone", "-q", XCI_URL, XCI_DIR], check=True)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"{XCI_DIR} has local changes; the fit must run the published code")
    if _git("rev-parse", "HEAD") != XCI_COMMIT:
        try:
            _git("checkout", "-q", XCI_COMMIT)
        except subprocess.CalledProcessError:
            _git("fetch", "-q", "origin")
            _git("checkout", "-q", XCI_COMMIT)
    return XCI_DIR


if __name__ == "__main__":
    print(ensure(), _git("rev-parse", "--short", "HEAD"))
    sys.exit(0)
