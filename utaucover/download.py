"""Model downloads that report progress to the window / CLI / MCP log.

The libraries' own progress bars only draw on a terminal, so on a fresh install the window sat on
"downloading the model" for minutes with no sign of life. These helpers report MB and percent instead.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Callable

Log = Callable[[str], None]


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class Progress:
    """A tqdm-shaped progress reporter (total=, update(n), close()) that logs a line every 5% / 10 s."""

    def __init__(self, log: Log, label: str, total: int = 0, **_):
        self.log, self.label, self.total = log, label, int(total or 0)
        self.done, self._last_pct, self._last_t = 0, -1, time.monotonic()
        self.log(f"[download] {label}: starting" + (f" ({self.total / 1e6:,.0f} MB)" if self.total else ""))

    def update(self, n: int):
        self.done += n
        now = time.monotonic()
        pct = int(100 * self.done / self.total) if self.total else -1
        if (pct >= 0 and pct // 5 > self._last_pct // 5) or now - self._last_t > 10:
            self._last_pct, self._last_t = pct, now
            of = f" / {self.total / 1e6:,.0f} MB ({pct}%)" if self.total else " MB"
            self.log(f"[download] {self.label}: {self.done / 1e6:,.0f}{of}")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fetch(url: str, target: Path, log: Log, label: str, sha256: str | None = None, timeout: float = 60):
    """Download url to target (via a .partial file, so an interrupted download never looks finished)."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "UtauCover"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r, open(partial, "wb") as f:
            bar = Progress(log, label, int(r.headers.get("Content-Length") or 0))
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                bar.update(len(chunk))
        if sha256 and h.hexdigest() != sha256:
            raise RuntimeError(f"{label} downloaded incompletely (checksum mismatch). Build again to retry.")
        os.replace(partial, target)
        log(f"[download] {label}: done")
    finally:
        if partial.exists():
            partial.unlink()


def fetch_whisper(name: str, root: Path, log: Log):
    """The same file and checksum Whisper's own downloader uses, with progress."""
    import whisper
    url = whisper._MODELS[name]
    fetch(url, Path(root) / os.path.basename(url), log, f"Whisper {name} model", sha256=url.split("/")[-2])


def report_separator_downloads(separator, log: Log):
    """Make audio-separator's model downloads report progress (it otherwise draws a terminal-only bar)."""
    import audio_separator.separator.separator as mod
    orig = separator.download_file_if_not_exists

    def download(url, output_path, *args, **kwargs):
        name = os.path.basename(str(output_path))
        saved = mod.tqdm
        mod.tqdm = lambda total=0, **kw: Progress(log, name, total)
        try:
            return orig(url, output_path, *args, **kwargs)
        finally:
            mod.tqdm = saved

    separator.download_file_if_not_exists = download
