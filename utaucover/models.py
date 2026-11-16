"""Which analysis models are available: note engines and Whisper sizes (for the window, CLI and MCP)."""

from __future__ import annotations

from .paths import whisper_dir


# (name passed to Whisper, file it downloads, size GB, one-line description)
WHISPER_MODELS = [
    ("turbo", "large-v3-turbo.pt", 1.6, "default: large-v3 quality at a fraction of the time"),
    ("large-v3", "large-v3.pt", 3.1, "the full large model: slowest"),
    ("medium", "medium.pt", 1.5, "older mid-size model"),
    ("small", "small.pt", 0.5, "fastest, least accurate"),
]


def whisper_models() -> list[dict]:
    return [{"name": n, "file": f, "size_gb": gb, "downloaded": (whisper_dir() / f).exists(), "about": about}
            for n, f, gb, about in WHISPER_MODELS]


def note_engines() -> list[dict]:
    from .game import GAME_DOWNLOAD_URL, INSTALL_HINT, find_game, game_version
    g = find_game()
    return [
        {"name": "builtin", "label": "Built-in", "installed": True,
         "about": "UtauCover's pitch tracking (FCPE) and note segmentation: less accurate, kept as a fallback"},
        {"name": "game", "label": f"GAME {game_version(g[1])}" if g else "GAME", "installed": g is not None,
         "about": "UtauV's GAME note transcriber (openvpi): the recommended engine, adds 2-3 min per build"
                  if g else "not installed. " + INSTALL_HINT, "download_url": GAME_DOWNLOAD_URL, "recommended": True},
    ]
