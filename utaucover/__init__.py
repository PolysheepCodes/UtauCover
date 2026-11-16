"""UtauCover: turn a song into an OpenUTAU / UtauV cover project.

Pipeline: stem separation -> vocal pitch + lyric analysis -> notes on tracks -> .ustx
"""

import os as _os

__version__ = "0.7.0"

if _os.environ.get("UTAUCOVER_APP_DIR"):  # running from the packaged app
    from . import _bundle  # noqa: F401
