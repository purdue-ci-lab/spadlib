"""
spadlib: reusable utilities for SPAD / single-photon (quanta) imaging data.

Logging follows the standard library convention for libraries: a NullHandler is
attached to the top-level ``spadlib`` logger so the package is silent by default
and the application controls handlers/levels. Call :func:`enable_logging` for a
quick console handler during interactive use.
"""
import logging
from pathlib import Path

# Silent by default: the application is responsible for configuring handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())


def enable_logging(level=logging.INFO, stream=None):
    """
    Attach a StreamHandler to the top-level ``spadlib`` logger for quick output.

    Intended for interactive/notebook use; applications should normally configure
    logging themselves rather than calling this.

    Args:
        level: logging level for the spadlib logger and handler (default INFO).
        stream: stream to write to (default ``sys.stderr`` via StreamHandler).

    Returns:
        The configured ``logging.StreamHandler``.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s,%(msecs)d | %(name)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(handler)
    return handler


def add_file_handler(path, level=logging.DEBUG):
    """
    Attach a FileHandler to the top-level ``spadlib`` logger.

    Opt-in convenience for users who want spadlib logs written to a file; the
    parent directory is created if needed.

    Args:
        path: file path to log to.
        level: logging level for the handler (default DEBUG).

    Returns:
        The configured ``logging.FileHandler``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s,%(msecs)d | %(name)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(handler)
    return handler
