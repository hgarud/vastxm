from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "vastxm") -> logging.Logger:
    """Return a vastxm logger, configuring the parent ``vastxm`` logger on first call.

    The handler is always attached to the parent ``vastxm`` logger so that all
    child loggers (e.g. ``vastxm.bundle``, ``vastxm.workflow``) propagate their
    records up to a single configured handler.

    Levels: DEBUG via VASTXM_DEBUG=1 env var, otherwise INFO.
    Output: stderr only. stdout is reserved for streamed remote output.
    """
    parent = logging.getLogger("vastxm")
    if not parent.handlers:
        level = logging.DEBUG if os.environ.get("VASTXM_DEBUG") == "1" else logging.INFO
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[vastxm %(levelname)s] %(message)s"))
        parent.addHandler(handler)
        parent.setLevel(level)
        parent.propagate = False
    return logging.getLogger(name)
