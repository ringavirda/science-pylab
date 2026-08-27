"""Logging behaviour of the dtfit echo helper and logger."""

import logging

from dtfit.log import echo, enable_logging, logger


def test_echo_logs_message_and_values_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="dtfit"):
        echo("visible-message", [1, 2])
    assert "visible-message" in caplog.text
    assert "1" in caplog.text


def test_echo_is_silent_below_its_level(caplog):
    # echo defaults to DEBUG, below the INFO level captured here
    with caplog.at_level(logging.INFO, logger="dtfit"):
        echo("hidden-message")
    assert "hidden-message" not in caplog.text


def test_echo_respects_explicit_level(caplog):
    with caplog.at_level(logging.INFO, logger="dtfit"):
        echo("info-message", level=logging.INFO)
    assert "info-message" in caplog.text


def test_enable_logging_returns_dtfit_logger():
    assert enable_logging() is logger
