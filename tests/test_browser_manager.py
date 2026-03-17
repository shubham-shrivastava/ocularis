"""
Tests for core/browser_manager.py

Covers:
- _is_allowed domain check with various patterns
- Domain allowlist blocks disallowed requests (unit test without real browser)
"""
from __future__ import annotations

import pytest

from core.browser_manager import BrowserManager
from core.settings import load_settings


class TestDomainAllowlist:
    def test_allowed_when_list_empty(self):
        # Empty list means allow all
        assert BrowserManager._is_allowed("https://anything.com/path", [])

    def test_exact_domain_match(self):
        assert BrowserManager._is_allowed("https://example.com/page", ["example.com"])

    def test_subdomain_not_matched_without_wildcard(self):
        # Exact host match: "example.com" does NOT match "sub.example.com"
        assert not BrowserManager._is_allowed("https://sub.example.com/page", ["example.com"])

    def test_subdomain_matched_with_wildcard(self):
        assert BrowserManager._is_allowed("https://sub.example.com/page", ["*.example.com"])

    def test_evil_domain_not_matched(self):
        # "example.com" must not match "evil-example.com" (no substring matching)
        assert not BrowserManager._is_allowed("https://evil-example.com/page", ["example.com"])

    def test_wildcard_domain(self):
        assert BrowserManager._is_allowed("https://api.example.com/v1", ["*.example.com"])

    def test_disallowed_domain(self):
        assert not BrowserManager._is_allowed("https://evil.com/steal", ["example.com"])

    def test_multiple_allowed_domains(self):
        allowed = ["example.com", "trusted.org"]
        assert BrowserManager._is_allowed("https://trusted.org/page", allowed)
        assert not BrowserManager._is_allowed("https://evil.com/page", allowed)

    def test_data_uri_blocked_by_strict_list(self):
        assert not BrowserManager._is_allowed("data:text/html,<h1>hi</h1>", ["example.com"])
