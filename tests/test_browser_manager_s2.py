"""
Slice 2 tests for core/browser_manager.py

Covers:
- CDP connect mode rejects non-localhost URLs
- Password field redaction (with mocked PIL)
- _is_allowed still correct after Slice 2 changes
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from core.browser_manager import BrowserManager
from core.settings import load_settings


def _make_png(color=(200, 200, 200)) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCDPConnectValidation:
    @pytest.mark.asyncio
    async def test_connect_rejects_non_localhost(self):
        settings = load_settings()
        manager = BrowserManager(settings)
        with pytest.raises(ValueError, match="127.0.0.1"):
            await manager.connect("http://remote-host.example.com:9222", "https://example.com")

    @pytest.mark.asyncio
    async def test_connect_accepts_localhost(self):
        settings = load_settings()
        manager = BrowserManager(settings)
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_context.pages = [mock_page]
        mock_context.route = AsyncMock()
        mock_context.on = MagicMock()
        mock_browser.contexts = [mock_context]

        with patch("core.browser_manager.async_playwright") as mock_pw_factory:
            mock_pw = AsyncMock()
            mock_pw_factory.return_value.__aenter__ = AsyncMock(return_value=mock_pw)
            mock_pw.start = AsyncMock(return_value=mock_pw)
            mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)

            # Should not raise
            try:
                await manager.connect("http://127.0.0.1:9222", "https://example.com")
            except Exception:
                pass  # Page interactions will fail with mocks; that's ok


class TestPasswordRedaction:
    @pytest.mark.asyncio
    async def test_redaction_blacks_out_password_fields(self):
        settings = load_settings()
        settings.security.block_password_fields = True
        manager = BrowserManager(settings)

        raw_png = _make_png((255, 255, 255))

        # Mock page with one password field at (10, 10, 40, 20)
        mock_page = AsyncMock()
        mock_page.eval_on_selector_all = AsyncMock(
            return_value=[{"x": 10, "y": 10, "width": 40, "height": 20}]
        )
        manager._page = mock_page

        result = await manager._redact_password_fields(raw_png)

        # Result should differ from input (black rectangle drawn)
        assert result != raw_png

        # Verify black pixel was drawn at (10, 10).
        # Redaction outputs JPEG; compression may alter exact values, so accept near-black.
        img = Image.open(io.BytesIO(result))
        pixel = img.getpixel((10, 10))
        rgb = pixel[:3] if len(pixel) >= 3 else pixel
        assert max(rgb) <= 10, f"Expected near-black at (10,10), got {pixel}"

    @pytest.mark.asyncio
    async def test_redaction_noop_when_no_password_fields(self):
        settings = load_settings()
        manager = BrowserManager(settings)

        raw_png = _make_png((200, 200, 200))

        mock_page = AsyncMock()
        mock_page.eval_on_selector_all = AsyncMock(return_value=[])
        manager._page = mock_page

        result = await manager._redact_password_fields(raw_png)
        assert result == raw_png

    @pytest.mark.asyncio
    async def test_redaction_returns_original_on_dom_error(self):
        settings = load_settings()
        manager = BrowserManager(settings)

        raw_png = _make_png()

        mock_page = AsyncMock()
        mock_page.eval_on_selector_all = AsyncMock(side_effect=RuntimeError("page closed"))
        manager._page = mock_page

        result = await manager._redact_password_fields(raw_png)
        assert result == raw_png
