"""Unit tests for biometric.utils.logging."""

from biometric.utils.logging import get_logger


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger(self) -> None:
        """get_logger returns a bound logger."""
        logger = get_logger(__name__)
        assert logger is not None

    def test_logger_accepts_structured_args(self) -> None:
        """Logger accepts keyword args for structured logging."""
        logger = get_logger(__name__)
        # Should not raise
        logger.info("test_event", key="value", count=1)

    def test_log_level_from_env(self) -> None:
        """LOG_LEVEL env var is respected (smoke test)."""
        # Re-import to pick up env; conftest sets WARNING so INFO may not appear
        # Just verify no crash when logging
        logger = get_logger("biometric.utils.logging")
        logger.debug("debug_msg", x=1)
        logger.info("info_msg", y=2)
        logger.warning("warn_msg", z=3)
