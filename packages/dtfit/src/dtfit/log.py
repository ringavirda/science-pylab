"""Library logging, under the ``"dtfit"`` logger.

The logger carries only a ``NullHandler``: importing dtfit never configures
the root logger or prints anything. ``enable_logging`` is the opt-in for
notebooks and scripts.
"""

import logging
from typing import Any

logger = logging.getLogger("dtfit")
logger.addHandler(logging.NullHandler())


def enable_logging(
    level: int = logging.INFO,
    fmt: str = "%(name)s %(levelname)s: %(message)s",
) -> logging.Logger:
    """Attach a stream handler to the dtfit logger and set its level.

    Repeat calls reconfigure the existing dtfit stream handler instead of
    stacking another one, so a re-run notebook cell does not double its
    records.

    Args:
        level: Logging level, e.g. ``logging.INFO``.
        fmt: Format string for the stream handler.

    Returns:
        The configured ``"dtfit"`` logger.
    """
    handler = next(
        (h for h in logger.handlers if isinstance(h, logging.StreamHandler)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    handler.setFormatter(logging.Formatter(fmt))
    logger.setLevel(level)
    return logger


def echo(
    message: str,
    value: list[Any] | Any | None = None,
    *,
    level: int = logging.DEBUG,
) -> None:
    """Emit a fitting-detail message through the dtfit logger.

    Silent until the application configures logging; call
    ``enable_logging(logging.DEBUG)`` to see these.

    Args:
        message: Message to log before any values.
        value: Value(s) to log after the message.
        level: Logging level.
    """
    if not logger.isEnabledFor(level):
        return

    logger.log(level, message)
    if value is not None:
        values = value if isinstance(value, list) else [value]
        for val in values:
            logger.log(level, "%s", val)
