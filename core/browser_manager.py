from __future__ import annotations

import base64
import hashlib
import re
from copy import deepcopy
from typing import TYPE_CHECKING

from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Request,
    Route,
    async_playwright,
)

from core.schemas import ActionRequest, ActionType, AgentState
from core.settings import Settings

if TYPE_CHECKING:
    pass


class BrowserManager:
    """
    Manages a Playwright browser context for a single agent run.

    Slice 1 delivers launch mode only. CDP connect mode is added in Slice 2.

    Domain allowlisting is enforced at BrowserContext level so every page,
    popup, and new tab created within the context is covered automatically.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._is_cdp_connect: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def launch(self, start_url: str) -> None:
        """Spawn a fresh Chromium instance and navigate to start_url."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._settings.browser.headless,
        )
        self._context = await self._browser.new_context(
            viewport={
                "width": self._settings.browser.viewport_width,
                "height": self._settings.browser.viewport_height,
            },
            device_scale_factor=1,
        )
        await self._setup_context_policies()
        self._page = await self._context.new_page()
        await self._page.goto(
            start_url,
            timeout=self._settings.browser.timeout_ms,
        )
        logger.info("Browser launched", url=start_url)

    async def connect(self, cdp_url: str, start_url: str) -> None:
        """
        Attach to an existing Chrome browser via CDP.

        The Chrome process must be started with:
          --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1

        Use scripts/start_chrome_debug.sh to launch safely.
        """
        sec = self._settings.security
        from urllib.parse import urlparse  # noqa: PLC0415

        parsed_cdp = urlparse(cdp_url)
        cdp_hostname = (parsed_cdp.hostname or "").lower()
        allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        if sec.cdp_host:
            allowed_hosts.add(sec.cdp_host.lower())
        if cdp_hostname not in allowed_hosts:
            raise ValueError(
                f"CDP host '{cdp_hostname}' is not allowed. "
                f"Permitted hosts: {allowed_hosts}. "
                "Only localhost CDP connections are permitted. "
                "Use scripts/start_chrome_debug.sh to launch Chrome correctly."
            )

        self._is_cdp_connect = True
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)

        # Use existing context if available; create new one if not
        contexts = self._browser.contexts
        if contexts:
            self._context = contexts[0]
        else:
            self._context = await self._browser.new_context(
                viewport={
                    "width": self._settings.browser.viewport_width,
                    "height": self._settings.browser.viewport_height,
                },
            )

        # Apply same context-level security policy as launch mode
        await self._setup_context_policies()

        pages = self._context.pages
        if pages:
            self._page = pages[0]
        else:
            self._page = await self._context.new_page()

        if start_url and self._page.url != start_url:
            await self._page.goto(start_url, timeout=self._settings.browser.timeout_ms)

        logger.info("Browser connected via CDP", cdp_url=cdp_url, url=self._page.url)

    async def close(self) -> None:
        """Tear down browser resources.

        In CDP connect mode, detach the Playwright connection without closing
        the user's browser. In launch mode, close everything.
        """
        try:
            if self._is_cdp_connect:
                # Detach only — don't close the remote browser
                if self._browser:
                    await self._browser.close()  # disconnects CDP without killing the process
            else:
                if self._context:
                    await self._context.close()
                if self._browser:
                    await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            logger.exception("Error closing browser")
        finally:
            self._context = None
            self._browser = None
            self._playwright = None
            self._page = None
            self._is_cdp_connect = False

    # ------------------------------------------------------------------
    # Domain allowlisting (context-level, covers all pages/popups/iframes)
    # ------------------------------------------------------------------

    async def _setup_context_policies(self) -> None:
        """Register context-wide route interceptor and popup handler."""
        if not self._context:
            return

        allowed = self._settings.security.allowed_domains

        if allowed:
            await self._context.route("**/*", self._make_route_handler(allowed))
            self._context.on("page", self._on_new_page)
            logger.info("Domain allowlist active", allowed_domains=allowed)

    def _make_route_handler(self, allowed_domains: list[str]):
        async def _handler(route: Route, request: Request) -> None:
            url = request.url
            if self._is_allowed(url, allowed_domains):
                await route.continue_()
            else:
                logger.warning("Blocked request to disallowed domain", url=url)
                await route.abort("blockedbyclient")

        return _handler

    def _on_new_page(self, page: Page) -> None:
        """Called when a new tab/popup opens. Apply same policy and adopt it."""
        allowed = self._settings.security.allowed_domains
        import asyncio

        async def _adopt_new_page() -> None:
            try:
                if allowed:
                    await page.route("**/*", self._make_route_handler(allowed))
                await page.wait_for_load_state("domcontentloaded", timeout=self._settings.browser.timeout_ms)
            except Exception:
                logger.debug("New page did not reach domcontentloaded before timeout")
            try:
                await page.bring_to_front()
            except Exception:
                logger.debug("Could not bring new page to front")
            self._page = page
            logger.info("Adopted new page", url=page.url)

        asyncio.create_task(_adopt_new_page())  # noqa: RUF006

    @staticmethod
    def _is_allowed(url: str, allowed_domains: list[str]) -> bool:
        """Return True if URL's hostname matches any entry in the allowlist.

        Empty allowlist means allow all (no restrictions configured).
        Matches are hostname-level, not substring-based:
          - "example.com" matches exactly example.com
          - "*.example.com" matches any subdomain of example.com
        """
        if not allowed_domains:
            return True
        from urllib.parse import urlparse  # noqa: PLC0415

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False
        for domain in allowed_domains:
            domain = domain.lower().strip()
            if domain.startswith("*."):
                suffix = domain[2:]
                if hostname == suffix or hostname.endswith("." + suffix):
                    return True
            else:
                if hostname == domain:
                    return True
        return False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    async def get_state(self) -> AgentState:
        """
        Capture current browser state as AgentState.

        If security.block_password_fields is enabled, password input bounding
        boxes are blacked out before the screenshot is returned. This happens
        before the bytes reach the model or any storage layer.
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call launch() or connect() first.")

        raw_bytes: bytes = await self._page.screenshot(
            type="jpeg",
            quality=self._settings.browser.screenshot_quality,
            full_page=False,
        )

        if self._settings.security.block_password_fields:
            raw_bytes = await self._redact_password_fields(raw_bytes)

        b64 = base64.b64encode(raw_bytes).decode()
        state_hash = hashlib.md5(raw_bytes).hexdigest()
        url = self._page.url

        return AgentState(
            screenshot_b64=b64,
            screenshot_bytes=raw_bytes,
            url=url,
            state_hash=state_hash,
        )

    async def _redact_password_fields(self, raw_bytes: bytes) -> bytes:
        """
        Black out bounding boxes of all input[type=password] elements.

        Preserves the input format (JPEG) to keep the pipeline consistent.
        No-op if no password fields found or if PIL is not available.
        """
        if not self._page:
            return raw_bytes
        try:
            bboxes = await self._page.eval_on_selector_all(
                "input[type=password]",
                """
                elements => elements.map(el => {
                    const r = el.getBoundingClientRect();
                    return { x: r.x, y: r.y, width: r.width, height: r.height };
                })
                """,
            )
        except Exception:
            logger.debug("Could not query password fields (page may be navigating)")
            return raw_bytes

        if not bboxes:
            return raw_bytes

        try:
            import io  # noqa: PLC0415

            from PIL import Image, ImageDraw  # noqa: PLC0415

            img = Image.open(io.BytesIO(raw_bytes))
            draw = ImageDraw.Draw(img)
            for bbox in bboxes:
                x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                if w > 0 and h > 0:
                    draw.rectangle([x, y, x + w, y + h], fill="black")
            buf = io.BytesIO()
            # Preserve JPEG format to match browser capture; use same quality as capture
            quality = self._settings.browser.screenshot_quality
            img.save(buf, format="JPEG", quality=quality)
            redacted = buf.getvalue()
            logger.debug("Password fields redacted", count=len(bboxes))
            return redacted
        except Exception:
            logger.warning("Password field redaction failed; returning original screenshot")
            return raw_bytes

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def execute_action(self, action: ActionRequest) -> bool:
        """
        Dispatch an action to the browser. Returns True on success.

        Supported action_types: click, type, scroll, wait, key_press.
        """
        if not self._page:
            raise RuntimeError("Browser not started. Call launch() first.")

        try:
            match action.action_type:
                case ActionType.click:
                    await self._do_click(action)
                case ActionType.type:
                    await self._do_type(action)
                case ActionType.login:
                    await self._do_login(action)
                case ActionType.scroll:
                    await self._do_scroll(action)
                case ActionType.wait:
                    await self._do_wait(action)
                case ActionType.key_press:
                    await self._do_key_press(action)
                case _:
                    logger.error("Unknown action type", action_type=action.action_type)
                    return False

            logger.debug(
                "Action executed",
                action_type=action.action_type,
                params=self._safe_params(action),
            )
            return True

        except Exception:
            logger.exception(
                "Action failed",
                action_type=action.action_type,
                params=self._safe_params(action),
            )
            return False

    async def _do_click(self, action: ActionRequest) -> None:
        p = action.params
        if "x" in p and "y" in p:
            await self._page.mouse.click(float(p["x"]), float(p["y"]))
        elif "selector" in p:
            selector = str(p["selector"])
            locator = self._page.locator(selector).first
            await locator.wait_for(state="visible", timeout=self._settings.browser.timeout_ms)
            try:
                await locator.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                logger.debug("Could not scroll target into view before click", selector=selector)
            try:
                await locator.click(timeout=self._settings.browser.timeout_ms)
            except Exception:
                # Fallback for overlays/animation race: force click.
                await locator.click(timeout=2000, force=True)
        else:
            raise ValueError("click action requires 'x'+'y' or 'selector'")

    async def _do_type(self, action: ActionRequest) -> None:
        p = action.params
        text = str(p.get("text", ""))
        press_enter = p.get("press_enter") is True
        if "selector" in p:
            await self._page.fill(p["selector"], text)
            if press_enter:
                await self._page.keyboard.press("Enter")
        elif "x" in p and "y" in p:
            await self._page.mouse.click(float(p["x"]), float(p["y"]))
            await self._page.keyboard.type(text)
            if press_enter:
                await self._page.keyboard.press("Enter")
        else:
            if await self.try_fill_focused_input(text, press_enter=press_enter):
                return
            # Fall back to typing when no focused editable control can be filled.
            await self._page.keyboard.type(text)
            if press_enter:
                await self._page.keyboard.press("Enter")

    async def _do_login(self, action: ActionRequest) -> None:
        """Fill login credentials into common username/password fields."""
        if not self._page:
            raise RuntimeError("Browser not started. Call launch() or connect() first.")
        p = action.params
        user_id = str(p.get("user_id", "")).strip()
        password = str(p.get("password", "")).strip()
        if not user_id or not password:
            raise ValueError("login action requires non-empty user_id and password")

        user_selector = str(
            p.get(
                "user_selector",
                "input[type=email], input[name*=user i], input[id*=user i], "
                "input[name*=login i], input[id*=login i], input[type=text]",
            )
        )
        pass_selector = str(
            p.get(
                "password_selector",
                "input[type=password], input[name*=pass i], input[id*=pass i]",
            )
        )
        submit_selector = str(
            p.get(
                "submit_selector",
                "button[type=submit], input[type=submit], button[name*=login i], button[id*=login i]",
            )
        )

        user_field = self._page.locator(user_selector).first
        pass_field = self._page.locator(pass_selector).first
        await user_field.wait_for(state="visible", timeout=self._settings.browser.timeout_ms)
        await pass_field.wait_for(state="visible", timeout=self._settings.browser.timeout_ms)

        await user_field.fill(user_id)
        await pass_field.fill(password)

        submit = self._page.locator(submit_selector).first
        if await submit.count():
            try:
                await submit.click(timeout=2000)
                return
            except Exception:
                logger.debug("Login submit click failed; falling back to Enter key")
        await self._page.keyboard.press("Enter")

    async def _do_scroll(self, action: ActionRequest) -> None:
        p = action.params
        direction = str(p.get("direction", "down"))
        amount = int(p.get("amount", 300))
        delta_y = amount if direction == "down" else -amount
        delta_x = amount if direction == "right" else (-amount if direction == "left" else 0)
        await self._page.mouse.wheel(delta_x, delta_y)

    async def _do_wait(self, action: ActionRequest) -> None:
        duration_ms = int(action.params.get("duration_ms", 1000))
        await self._page.wait_for_timeout(duration_ms)

    async def _do_key_press(self, action: ActionRequest) -> None:
        key = str(action.params.get("key", "Enter"))
        await self._page.keyboard.press(key)

    # ------------------------------------------------------------------
    # Recovery helpers (called by Supervisor via RunSession)
    # ------------------------------------------------------------------

    async def refresh(self) -> None:
        if self._page:
            await self._page.reload(timeout=self._settings.browser.timeout_ms)

    async def go_back(self) -> None:
        if self._page:
            await self._page.go_back(timeout=self._settings.browser.timeout_ms)

    async def has_login_form(self) -> bool:
        """Return True when a visible password field is present on page."""
        if not self._page:
            return False
        try:
            count = await self._page.locator("input[type=password]").count()
            if count == 0:
                return False
            for i in range(count):
                loc = self._page.locator("input[type=password]").nth(i)
                if await loc.is_visible():
                    return True
            return False
        except Exception:
            return False

    async def try_select_option_text(self, option_text: str) -> bool:
        """Try selecting a dropdown/list option by visible text with robust fallbacks.

        Order: native <select> first, then ARIA listbox/menu via get_by_text, then generic text click.
        """
        if not self._page:
            return False
        text = option_text.strip()
        if not text:
            return False

        try:
            # 1. Native <select>: use select_option (most reliable, handles quotes in text)
            select_count = await self._page.locator("select").count()
            for i in range(select_count):
                select_loc = self._page.locator("select").nth(i)
                if not await select_loc.is_visible():
                    continue
                try:
                    await select_loc.scroll_into_view_if_needed(timeout=1500)
                except Exception:
                    pass
                try:
                    await select_loc.select_option(label=text, timeout=2000)
                    logger.info("Dropdown option selected via native select", option=text)
                    return True
                except Exception:
                    try:
                        await select_loc.select_option(value=text, timeout=1500)
                        logger.info("Dropdown option selected via native select (value)", option=text)
                        return True
                    except Exception:
                        continue

            # 2. ARIA listbox/menu: use get_by_text (avoids fragile :has-text() interpolation)
            role_selectors = ["[role='option']", "[role='menuitem']", "li", "button", "a"]
            for role_sel in role_selectors:
                opts = self._page.locator(role_sel).filter(has=self._page.get_by_text(text, exact=False))
                if await opts.count() == 0:
                    continue
                loc = opts.first
                if not await loc.is_visible():
                    continue
                try:
                    await loc.scroll_into_view_if_needed(timeout=1500)
                except Exception:
                    pass
                try:
                    await loc.click(timeout=3000)
                    logger.info("Dropdown option selected via role", option=text, role=role_sel)
                    return True
                except Exception:
                    try:
                        await loc.click(timeout=1500, force=True)
                        logger.info("Dropdown option selected via role (force)", option=text, role=role_sel)
                        return True
                    except Exception:
                        continue

            # 3. Final fallback: get_by_text click
            text_loc = self._page.get_by_text(text, exact=False).first
            if await text_loc.count() > 0 and await text_loc.is_visible():
                try:
                    await text_loc.scroll_into_view_if_needed(timeout=1500)
                except Exception:
                    pass
                await text_loc.click(timeout=3000)
                logger.info("Dropdown option selected via text fallback", option=text)
                return True
        except Exception:
            logger.debug("Dropdown option selection attempt failed", option=text)
            return False

        return False

    async def is_dropdown_open(self) -> bool:
        """Best-effort detection for an open dropdown/listbox/menu on the page."""
        if not self._page:
            return False
        selectors = [
            "[role='listbox']",
            "[role='menu']",
            "select:focus",
            "select[size]",
            "[aria-expanded='true']",
        ]
        try:
            for selector in selectors:
                loc = self._page.locator(selector)
                count = await loc.count()
                if count == 0:
                    continue
                for i in range(count):
                    try:
                        if await loc.nth(i).is_visible():
                            return True
                    except Exception:
                        continue
        except Exception:
            logger.debug("Dropdown open-state detection failed")
        return False

    async def try_submit_search(self) -> bool:
        """Click a visible search submit control using common selectors."""
        if not self._page:
            return False
        selectors = [
            "#nav-search-submit-button",
            "input[type='submit'][aria-label*='Go' i]",
            "button[type='submit'][aria-label*='Search' i]",
            "button[aria-label*='Search' i]",
            "input[type='submit']",
            "button[type='submit']",
        ]
        try:
            for selector in selectors:
                loc = self._page.locator(selector).first
                if await loc.count() == 0:
                    continue
                if not await loc.is_visible():
                    continue
                try:
                    await loc.scroll_into_view_if_needed(timeout=1500)
                except Exception:
                    pass
                try:
                    await loc.click(timeout=2500)
                    logger.info("Search submitted via selector", selector=selector)
                    return True
                except Exception:
                    try:
                        await loc.click(timeout=1500, force=True)
                        logger.info("Search submitted via forced selector click", selector=selector)
                        return True
                    except Exception:
                        continue
        except Exception:
            logger.debug("Search submit helper failed")
        return False

    async def try_fill_focused_input(self, text: str, *, press_enter: bool = False) -> bool:
        """Replace the value of the focused editable control when possible."""
        if not self._page:
            return False
        try:
            focused = self._page.locator(":focus").first
            if await focused.count() == 0:
                return False
            tag_name = await focused.evaluate("el => el.tagName.toLowerCase()")
            is_editable = await focused.evaluate(
                "el => el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el.isContentEditable"
            )
            if not is_editable:
                return False
            if tag_name in ("input", "textarea"):
                await focused.fill(text)
            else:
                await focused.evaluate(
                    """(el, value) => {
                        el.textContent = value;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                    }""",
                    text,
                )
            if press_enter:
                await self._page.keyboard.press("Enter")
            logger.info("Filled focused editable control")
            return True
        except Exception:
            logger.debug("Focused input fill failed")
            return False

    async def get_search_box_selector(self) -> str | None:
        if not self._page:
            return None
        selectors = [
            "#twotabsearchtextbox",
            "input[name='field-keywords']",
            "input[type='search']",
            "input[aria-label*='Search' i]",
        ]
        try:
            for selector in selectors:
                loc = self._page.locator(selector).first
                if await loc.count() == 0:
                    continue
                if await loc.is_visible():
                    return selector
        except Exception:
            logger.debug("Search box selector detection failed")
        return None

    async def is_search_box_focused(self) -> bool:
        if not self._page:
            return False
        try:
            selector = await self.get_search_box_selector()
            if not selector:
                return False
            search_box = self._page.locator(selector).first
            return await search_box.evaluate("el => el === document.activeElement")
        except Exception:
            return False

    async def get_search_box_value(self) -> str:
        if not self._page:
            return ""
        try:
            selector = await self.get_search_box_selector()
            if not selector:
                return ""
            loc = self._page.locator(selector).first
            if await loc.count() == 0:
                return ""
            return str(await loc.input_value())
        except Exception:
            return ""

    async def get_page_title(self) -> str:
        if not self._page:
            return ""
        try:
            return (await self._page.title()).strip()
        except Exception:
            return ""

    async def get_visible_text(self, limit: int = 3000) -> str:
        if not self._page:
            return ""
        try:
            text = await self._page.evaluate(
                """maxLen => {
                    const body = document.body;
                    if (!body) return "";
                    return (body.innerText || "").slice(0, maxLen);
                }""",
                limit,
            )
            return str(text).strip()
        except Exception:
            return ""

    async def get_main_text(self, limit: int = 15000) -> str:
        if not self._page:
            return ""
        try:
            text = await self._page.evaluate(
                """maxLen => {
                    const target = document.querySelector('article, main') || document.body;
                    if (!target) return "";
                    return (target.innerText || "").slice(0, maxLen);
                }""",
                limit,
            )
            return str(text).strip()
        except Exception:
            return ""

    async def extract_candidate_items(self, limit: int = 12) -> list[dict[str, str | None]]:
        """Extract raw content blocks with links from the page.

        Site-agnostic: finds repeated sibling containers (the natural structure
        of any product grid / search results list) and returns their text + URL.
        The blocks are raw — downstream AI identifies which are real products.
        """
        if not self._page:
            return []
        try:
            items = await self._page.evaluate(
                """maxItems => {
                    const normalize = v => (v || "").replace(/\\s+/g, " ").trim();
                    const seen = new WeakSet();
                    const blocks = [];

                    const links = Array.from(document.querySelectorAll("a[href]"));
                    for (const link of links) {
                        if (link.closest("nav, header, footer, [role='navigation'], [role='banner'], [role='contentinfo']")) continue;

                        const href = link.getAttribute("href") || "";
                        if (!href || href.startsWith("#") || href.startsWith("javascript:") || href.startsWith("mailto:")) continue;
                        if (/\\/(signin|login|account|orders|cart|help|wishlist|customer|preferences)\\b/i.test(href)) continue;

                        let card = link;
                        for (let el = link.parentElement; el && el !== document.body; el = el.parentElement) {
                            if (el.parentElement && el.parentElement.children.length >= 3) {
                                card = el;
                                break;
                            }
                        }

                        if (seen.has(card)) continue;
                        seen.add(card);

                        const text = normalize(card.innerText || "");
                        if (text.length < 30) continue;

                        let url;
                        try { url = new URL(href, window.location.href).href; } catch { continue; }

                        blocks.push({
                            title: normalize(link.innerText || "").slice(0, 250),
                            url,
                            snippet: text.slice(0, 500),
                        });

                        if (blocks.length >= maxItems) break;
                    }
                    return blocks;
                }""",
                limit,
            )
            return list(items or [])
        except Exception:
            logger.debug("Candidate extraction failed")
            return []

    async def get_page_structure_hints(self) -> dict[str, int]:
        if not self._page:
            return {}
        try:
            hints = await self._page.evaluate(
                """() => {
                    const count = selector => document.querySelectorAll(selector).length;
                    return {
                        paragraph_count: count("article p, main p, p"),
                        form_count: count("form"),
                        input_count: count("input, textarea, select"),
                        link_count: count("a[href]"),
                        candidate_count: count("[data-asin], article a[href], li a[href], .s-result-item a[href], .product a[href], .card a[href]"),
                    };
                }"""
            )
            return dict(hints or {})
        except Exception:
            logger.debug("Page structure hint extraction failed")
            return {}

    async def leave_current_page(self) -> bool:
        """Close the current tab if it is an extra page, else navigate back."""
        if not self._page:
            return False
        try:
            if self._context and len(self._context.pages) > 1:
                current = self._page
                remaining = [p for p in self._context.pages if p is not current]
                await current.close()
                if remaining:
                    self._page = remaining[-1]
                    try:
                        await self._page.bring_to_front()
                    except Exception:
                        logger.debug("Could not bring previous page to front")
                    logger.info("Closed current page and focused previous tab", url=self._page.url)
                    return True
            await self.go_back()
            logger.info("Navigated back from current page")
            return True
        except Exception:
            logger.debug("leave_current_page helper failed")
            return False

    @staticmethod
    def _safe_params(action: ActionRequest) -> dict:
        """Redact credential fields from logs."""
        params = deepcopy(action.params)
        if action.action_type == ActionType.login:
            if "password" in params:
                params["password"] = "***REDACTED***"
        if params.get("secret") and "text" in params:
            params["text"] = "***REDACTED***"
        return params

    @property
    def page(self) -> Page | None:
        return self._page
